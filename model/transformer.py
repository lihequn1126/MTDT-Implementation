import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import math
import shutil

# ==============================================================================
# --- 配置区域 (Configuration) ---
# ==============================================================================

import argparse

parser = argparse.ArgumentParser(description='Time Series Forecasting')


parser.add_argument('--nwp_path', type=str, default=r"../nwpData/baoxing.csv", help='Path to NWP data')
parser.add_argument('--load_path', type=str, default=r"../LoadData/baoxing.csv", help='Path to Load data')
parser.add_argument('--output_dir', type=str, default=r"../result/baoxing_transformer", help='Output directory')


args = parser.parse_args()


NWP_PATH = args.nwp_path
LOAD_PATH = args.load_path
OUTPUT_DIR = args.output_dir


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Running with:\n NWP: {NWP_PATH}\n Load: {LOAD_PATH}\n Output: {OUTPUT_DIR}")

# 数据参数
SEQ_LEN = 96        # 输入序列长度
PRED_LEN = 96       # 预测序列长度
POINTS_PER_DAY = 96 # 每天的数据点数 (15min分辨率)

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 15       # 早停

# Transformer 模型参数
D_MODEL = 64        # Embedding 维度
NHEAD = 4           # 多头数
NUM_LAYERS = 2      # Encoder 层数
DROPOUT = 0.1

# --- 数据集划分设置 (按天) ---
# 模式1：指定具体天数
FIXED_TRAIN_DAYS = None 
FIXED_VAL_DAYS = None    
# 模式2：按比例自动计算整天数
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
# 剩余归为测试集

# 绘图设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# --- 1. 数据处理 (Data Processing) ---
# ==============================================================================

def load_and_process_data():
    print("🚀 Loading data...")
    if not os.path.exists(NWP_PATH) or not os.path.exists(LOAD_PATH):
        raise FileNotFoundError("数据文件路径不正确")

    df_nwp = pd.read_csv(NWP_PATH)
    df_load = pd.read_csv(LOAD_PATH)
    
    # 时间对齐
    df_nwp['time'] = pd.to_datetime(df_nwp['time'])
    df_load['time'] = pd.to_datetime(df_load['time'])
    df = pd.merge(df_load, df_nwp, on='time', how='inner').sort_values('time').set_index('time')
    
    # 统一列名
    load_col = [c for c in df.columns if 'load' in c.lower()][0]
    df = df.rename(columns={load_col: 'y'})
    
    # 特征工程
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # 简单的 dropna
    df = df.dropna()
    
    # 调整列顺序，y放最后
    cols = [c for c in df.columns if c != 'y'] + ['y']
    df = df[cols]
    day_steps = 96
    df = df.iloc[day_steps * 7:]
    
    return df

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def create_sequences(data, seq_len, pred_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        x = data[i:(i + seq_len), :-1]
        y = data[(i + seq_len):(i + seq_len + pred_len), -1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# ==============================================================================
# --- 2. Transformer 模型定义 ---
# ==============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        # 1. Input Embedding
        self.input_linear = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. Transformer Encoder
        # batch_first=False 是 PyTorch 默认，这里手动 permute 适配
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 4. Output Head
        self.output_linear = nn.Linear(d_model, output_dim) 

    def forward(self, x):
        # x input: [batch, seq, feature]
        
        # Mapping to d_model
        x = self.input_linear(x) 
        
        # Permute to [seq, batch, d_model] for Transformer
        x = x.permute(1, 0, 2) 
        
        # Add PE
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Encoder
        output = self.transformer_encoder(x) # [seq, batch, d_model]
        
        # Take the last time step
        last_output = output[-1, :, :] # [batch, d_model]
        
        # Prediction
        prediction = self.output_linear(last_output) # [batch, output_dim]
        return prediction

# ==============================================================================
# --- 3. 训练与评估逻辑 (Main Logic) ---
# ==============================================================================

def train_and_evaluate():
    # 0. 目录准备
    if os.path.exists(OUTPUT_DIR):
        pass
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dayplot_dir = os.path.join(OUTPUT_DIR, "dayplot")
    os.makedirs(dayplot_dir, exist_ok=True)
    
    # 1. 数据加载与按天切分
    df = load_and_process_data()
    total_rows = len(df)
    total_days = total_rows // POINTS_PER_DAY
    print(f"Total Data: {total_rows} points ({total_days:.2f} days)")
    
    # 计算切分天数
    if FIXED_TRAIN_DAYS is not None and FIXED_VAL_DAYS is not None:
        n_train_days = FIXED_TRAIN_DAYS
        n_val_days = FIXED_VAL_DAYS
    else:
        n_train_days = int(total_days * TRAIN_RATIO)
        n_val_days = int(total_days * VAL_RATIO)
    
    n_test_days = total_days - n_train_days - n_val_days
    if n_test_days <= 0: raise ValueError("Test set empty, adjust ratio.")
    
    print(f"Split Plan (Days): Train={n_train_days}, Val={n_val_days}, Test={n_test_days}")
    
    # 计算索引
    train_end_idx = n_train_days * POINTS_PER_DAY
    val_end_idx = (n_train_days + n_val_days) * POINTS_PER_DAY
    
    # 物理切分 DataFrame
    df_train = df.iloc[:train_end_idx]
    df_val   = df.iloc[train_end_idx:val_end_idx]
    df_test  = df.iloc[val_end_idx:]
    
    # 2. 归一化 (防泄露：只 Fit 训练集)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_train.values)
    val_scaled = scaler.transform(df_val.values)
    test_scaled = scaler.transform(df_test.values)
    
    scaler_y = StandardScaler()
    scaler_y.fit(df_train['y'].values.reshape(-1, 1))
    
    # 3. 制作序列 (带边界拼接)
    def prepare_dataset(curr_scaled, prev_scaled_tail=None):
        if prev_scaled_tail is not None:
            data_combined = np.vstack([prev_scaled_tail, curr_scaled])
        else:
            data_combined = curr_scaled
        return create_sequences(data_combined, SEQ_LEN, PRED_LEN)

    # 训练集
    X_train, y_train = prepare_dataset(train_scaled, None)
    
    # 验证集 (拼接训练集末尾)
    train_tail = train_scaled[-SEQ_LEN:]
    X_val, y_val = prepare_dataset(val_scaled, train_tail)
    
    # 测试集 (拼接验证集末尾)
    val_tail = val_scaled[-SEQ_LEN:]
    X_test, y_test = prepare_dataset(test_scaled, val_tail)
    
    print(f"Sequences: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. 模型训练 (Transformer)
    input_dim = X_train.shape[2]
    model = TransformerModel(
        input_dim=input_dim, 
        output_dim=PRED_LEN,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("🔥 Transformer Training started...")
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience = 0
    
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            output = model(bx)
            loss = criterion(output, by)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        train_losses.append(np.mean(batch_losses))
        
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                val_batch_losses.append(criterion(model(bx), by).item())
        val_loss = np.mean(val_batch_losses)
        val_losses.append(val_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Val Loss: {val_loss:.5f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("🛑 Early stopping.")
                break
    
    # 5. 预测与评估
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    model.eval()
    
    test_preds_list, test_trues_list = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(DEVICE)
            test_preds_list.append(model(bx).cpu().numpy())
            test_trues_list.append(by.numpy())
            
    test_preds = np.concatenate(test_preds_list)
    test_trues = np.concatenate(test_trues_list)
    
    # 反归一化
    test_preds_inv = scaler_y.inverse_transform(test_preds)
    test_trues_inv = scaler_y.inverse_transform(test_trues)
    
    # 指标计算
    mae = mean_absolute_error(test_trues_inv.flatten(), test_preds_inv.flatten())
    rmse = np.sqrt(mean_squared_error(test_trues_inv.flatten(), test_preds_inv.flatten()))
    r2 = r2_score(test_trues_inv.flatten(), test_preds_inv.flatten())
    
    print(f"\n📊 Global Metrics (Transformer): MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    
    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}\nR2: {r2}\n")
    
    # 损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve (Transformer)')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()
    
    # 6. 绘图 (按天拼接)
    print(f"🖼️ Saving daily plots to {dayplot_dir} ...")
    
    stitch_pred = []
    stitch_true = []
    stitch_time = []
    
    # 测试集起始时间
    test_start_time = df_test.index[0]
    
    # 按照 POINTS_PER_DAY 步长循环
    for i in range(0, len(test_preds_inv), POINTS_PER_DAY):
        if i >= len(test_preds_inv): break
        
        y_p = test_preds_inv[i]
        y_t = test_trues_inv[i]
        
        current_day_start = test_start_time + pd.Timedelta(minutes=15*i)
        current_timeline = pd.date_range(start=current_day_start, periods=PRED_LEN, freq='15min')
        
        stitch_pred.extend(y_p)
        stitch_true.extend(y_t)
        stitch_time.extend(current_timeline)
        
        day_rmse = np.sqrt(mean_squared_error(y_t, y_p))
        day_r2 = r2_score(y_t, y_p)
        date_str = str(current_day_start.date())
        
        plt.figure(figsize=(10, 5))
        plt.plot(current_timeline, y_t, label='True', color='blue')
        plt.plot(current_timeline, y_p, label='Pred', color='red', linestyle='--')
        plt.title(f"Date: {date_str} | RMSE: {day_rmse:.2f} | R2: {day_r2:.2f}")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(dayplot_dir, f"{date_str}.png"))
        plt.close()
        
    # 全量结果
    res_df = pd.DataFrame({
        'time': stitch_time,
        'true': stitch_true,
        'pred': stitch_pred
    })
    res_df.to_csv(os.path.join(OUTPUT_DIR, "prediction_result.csv"), index=False)
    
    plt.figure(figsize=(15, 6))
    plt.plot(res_df['time'], res_df['true'], label='True', alpha=0.7)
    plt.plot(res_df['time'], res_df['pred'], label='Pred', alpha=0.7, linestyle='--')
    plt.title(f'Full Test Set Prediction (Transformer) | RMSE: {rmse:.2f}')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "full_prediction.png"))
    plt.close()
    
    print(f"✅ All tasks finished in {OUTPUT_DIR}")

if __name__ == "__main__":
    train_and_evaluate()
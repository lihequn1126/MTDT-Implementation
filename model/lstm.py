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
import shutil

import argparse

parser = argparse.ArgumentParser(description='Time Series Forecasting')


parser.add_argument('--nwp_path', type=str, default=r"../nwpData/baoxing.csv", help='Path to NWP data')
parser.add_argument('--load_path', type=str, default=r"../LoadData/baoxing.csv", help='Path to Load data')
parser.add_argument('--output_dir', type=str, default=r"../result/baoxing_lstm", help='Output directory')


args = parser.parse_args()


NWP_PATH = args.nwp_path
LOAD_PATH = args.load_path
OUTPUT_DIR = args.output_dir


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Running with:\n NWP: {NWP_PATH}\n Load: {LOAD_PATH}\n Output: {OUTPUT_DIR}")

SEQ_LEN = 96       # 输入序列长度 (过去24小时)
PRED_LEN = 96      # 预测序列长度 (未来24小时)
POINTS_PER_DAY = 96 # 每天的数据点数 (15min分辨率)

BATCH_SIZE = 64
EPOCHS = 100
HIDDEN_DIM = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
PATIENCE = 15

# --- 数据集划分设置 ---
# 这里你可以选择两种模式：
# 模式1：指定具体天数 (如果不为 None，则优先使用)
FIXED_TRAIN_DAYS = None  # 例如 300
FIXED_VAL_DAYS = None    # 例如 30
# 模式2：按比例自动计算整天数 (如果上面是 None)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
# (剩余的归为测试集)

# 绘图设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_process_data():
    print("Loading data...")
    # 读取数据
    if not os.path.exists(NWP_PATH) or not os.path.exists(LOAD_PATH):
        raise FileNotFoundError("数据文件路径不正确，请检查 NWP_PATH 和 LOAD_PATH")
        
    df_nwp = pd.read_csv(NWP_PATH)
    df_load = pd.read_csv(LOAD_PATH)
    
    # 时间对齐
    df_nwp['time'] = pd.to_datetime(df_nwp['time'])
    df_load['time'] = pd.to_datetime(df_load['time'])
    
    # Inner Merge 确保只有两者都有的时间点
    df = pd.merge(df_load, df_nwp, on='time', how='inner').sort_values('time').set_index('time')
    
    # 统一列名
    load_col = [c for c in df.columns if 'load' in c.lower()]
    if not load_col:
        raise ValueError("未找到包含 'load' 的列名")
    df = df.rename(columns={load_col[0]: 'y'})
    
    # 特征工程
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # 这里的 lag 特征如果要做，必须在 dropna 之前小心处理，
    # 但为了保证整天切分逻辑简单，这里暂时去掉 shift 导致的大量缺失，
    # 或者确保 dropna 后数据依然是连续完整的 15min 间隔。
    # 简单起见，这里只 dropna 一次
    df = df.dropna()
    
    # 确保列顺序，y 在最后
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
    """
    制作滑窗序列
    data: shape (N, Features)
    return: X (N-seq-pred+1, seq, Feat-1), y (N-seq-pred+1, pred)
    """
    xs, ys = [], []
    # 步长为1，尽可能多地采样训练
    for i in range(len(data) - seq_len - pred_len + 1):
        x = data[i:(i + seq_len), :-1] # 所有特征除了 y
        y = data[(i + seq_len):(i + seq_len + pred_len), -1] # 只有 y
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :] 
        return self.fc(last_out)

def train_and_evaluate():
    if os.path.exists(OUTPUT_DIR):
        # shutil.rmtree(OUTPUT_DIR) # 慎用，防止误删
        pass
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    dayplot_dir = os.path.join(OUTPUT_DIR, "dayplot")
    os.makedirs(dayplot_dir, exist_ok=True)
    
    # 1. 加载原始数据
    df = load_and_process_data()
    total_rows = len(df)
    total_days = total_rows // POINTS_PER_DAY
    
    print(f"Total Data: {total_rows} points ({total_days:.2f} days)")
    
    # 2. 计算按天划分的索引
    if FIXED_TRAIN_DAYS is not None and FIXED_VAL_DAYS is not None:
        n_train_days = FIXED_TRAIN_DAYS
        n_val_days = FIXED_VAL_DAYS
    else:
        n_train_days = int(total_days * TRAIN_RATIO)
        n_val_days = int(total_days * VAL_RATIO)
    
    n_test_days = total_days - n_train_days - n_val_days
    
    if n_test_days <= 0:
        raise ValueError("数据量不足以划分测试集，请调整比例或天数。")
        
    print(f"Split Plan (Days): Train={n_train_days}, Val={n_val_days}, Test={n_test_days}")
    
    # 计算切分点索引
    train_end_idx = n_train_days * POINTS_PER_DAY
    val_end_idx = (n_train_days + n_val_days) * POINTS_PER_DAY
    
    # 切分原始 DataFrame (为了保持时间索引后续画图用)
    df_train = df.iloc[:train_end_idx]
    df_val   = df.iloc[train_end_idx:val_end_idx]
    df_test  = df.iloc[val_end_idx:]
    
    # 3. 归一化 (Fit 只在训练集上进行，防止泄露)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_train.values)
    val_scaled = scaler.transform(df_val.values)
    test_scaled = scaler.transform(df_test.values)
    
    scaler_y = StandardScaler()
    # y是最后一列
    scaler_y.fit(df_train['y'].values.reshape(-1, 1))
    
    # 4. 制作数据集序列
    # 注意：验证集和测试集的第一个序列需要用到前一个数据集末尾的 SEQ_LEN 数据作为输入
    # 否则第一个预测点的输入数据不够
    
    # 辅助函数：拼接前缀并生成序列
    def prepare_dataset(curr_scaled, prev_scaled_tail=None):
        if prev_scaled_tail is not None:
            # 拼接到前面
            data_combined = np.vstack([prev_scaled_tail, curr_scaled])
        else:
            data_combined = curr_scaled
        return create_sequences(data_combined, SEQ_LEN, PRED_LEN)

    # 训练集：直接制作
    X_train, y_train = prepare_dataset(train_scaled, None)
    
    # 验证集：需要训练集最后 SEQ_LEN 个点
    train_tail = train_scaled[-SEQ_LEN:]
    X_val, y_val = prepare_dataset(val_scaled, train_tail)
    
    # 测试集：需要验证集最后 SEQ_LEN 个点
    val_tail = val_scaled[-SEQ_LEN:]
    X_test, y_test = prepare_dataset(test_scaled, val_tail)
    
    print(f"Sequences shape: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # 构造 DataLoader
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. 模型初始化与训练
    input_dim = X_train.shape[2]
    model = LSTMModel(input_dim, HIDDEN_DIM, NUM_LAYERS, PRED_LEN).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Training started...")
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        epoch_train_loss = np.mean(batch_losses)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                pred = model(bx)
                val_batch_losses.append(criterion(pred, by).item())
        
        epoch_val_loss = np.mean(val_batch_losses)
        val_losses.append(epoch_val_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f}")
            
        # 早停机制
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # 保存 Loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()
    
    # 6. 预测与评估
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    model.eval()
    
    test_preds_list, test_trues_list = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(DEVICE)
            pred = model(bx)
            test_preds_list.append(pred.cpu().numpy())
            test_trues_list.append(by.numpy())
            
    test_preds = np.concatenate(test_preds_list) # shape: (N_test, 96)
    test_trues = np.concatenate(test_trues_list)
    
    # 反归一化
    test_preds_inv = scaler_y.inverse_transform(test_preds)
    test_trues_inv = scaler_y.inverse_transform(test_trues)
    
    # 计算全局指标
    mae = mean_absolute_error(test_trues_inv.flatten(), test_preds_inv.flatten())
    rmse = np.sqrt(mean_squared_error(test_trues_inv.flatten(), test_preds_inv.flatten()))
    r2 = r2_score(test_trues_inv.flatten(), test_preds_inv.flatten())
    
    print(f"\nGlobal Metrics on Test Set: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    
    # 保存指标
    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}\nR2: {r2}\n")
    
    # 7. 绘图结果生成
    # 获取测试集的起始时间点
    # df_test 包含完整的天数数据，我们预测也是从 df_test 的第1个点对应的输出开始
    # X_test 的第0个样本，其输入是 df_test 前一天的最后96点，输出正是 df_test 的第1天(前96点)
    # 所以时间轴可以直接从 df_test.index 开始
    
    # 注意：如果 test_preds 数量和 df_test 长度不完全一致（由于滑窗最后不足 96 点的情况），
    # 这里的逻辑：create_sequences 保证了只要够 seq+pred 就能生成。
    # 我们的 prepare_dataset 使得生成的样本数 = len(test_data) - pred_len + 1 (如果步长为1)
    # 为了画出“天”的效果，我们需要按步长采样。
    
    stitch_pred = []
    stitch_true = []
    stitch_time = []
    
    # 测试集总共有 n_test_days 天。
    # 我们每隔 96 (POINTS_PER_DAY) 取一个样本进行拼接
    # 这样刚好覆盖每一天，且互不重叠
    
    test_start_time = df_test.index[0]
    
    # 遍历每一天
    # 能够预测的天数 = len(test_preds) // 96 ?? 
    # 实际上，test_preds 的长度大约是 total_test_points - 96 + 1
    # 我们按照 stride=96 进行索引
    
    for i in range(0, len(test_preds_inv), POINTS_PER_DAY):
        # 确保不越界（虽然通常整天划分应该刚好）
        if i >= len(test_preds_inv): break
        
        y_p = test_preds_inv[i]
        y_t = test_trues_inv[i]
        
        # 计算该天的起始时间
        # 第 i 个样本对应的预测起始时间是 test_start_time + i * 15min
        current_day_start = test_start_time + pd.Timedelta(minutes=15*i)
        current_timeline = pd.date_range(start=current_day_start, periods=PRED_LEN, freq='15min')
        
        stitch_pred.extend(y_p)
        stitch_true.extend(y_t)
        stitch_time.extend(current_timeline)
        
        # 保存日图
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
        
    # 保存全量 CSV
    res_df = pd.DataFrame({
        'time': stitch_time,
        'true': stitch_true,
        'pred': stitch_pred
    })
    res_df.to_csv(os.path.join(OUTPUT_DIR, "prediction_result.csv"), index=False)
    
    # 全量图
    plt.figure(figsize=(15, 6))
    plt.plot(res_df['time'], res_df['true'], label='True', alpha=0.7)
    plt.plot(res_df['time'], res_df['pred'], label='Pred', alpha=0.7, linestyle='--')
    plt.title("Full Test Set Prediction")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "full_prediction.png"))
    plt.close()
    
    print("Done.")

if __name__ == "__main__":
    train_and_evaluate()
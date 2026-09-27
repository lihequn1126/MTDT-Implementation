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

from data_provider import get_data  # 统一无泄漏数据管道

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
# --- 1. 数据处理: 统一使用 data_provider.get_data (history 模式) ---
# ==============================================================================

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
    
    # 1. 统一无泄漏数据管道 (history 模式)
    loaders, scaler_y, test_start_time, info, _ = get_data(
        NWP_PATH, LOAD_PATH, SEQ_LEN, PRED_LEN, BATCH_SIZE,
        TRAIN_RATIO, VAL_RATIO, POINTS_PER_DAY, mode="history")
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    
    # 4. 模型训练 (Transformer)
    input_dim = info['n_features']
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
    
    # 反归一化 (scaler_y 提供标量 mean_/scale_)
    test_preds_inv = test_preds * scaler_y.scale_ + scaler_y.mean_
    test_trues_inv = test_trues * scaler_y.scale_ + scaler_y.mean_
    
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
    
    # 测试集起始时间 (由数据管道提供)
    
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

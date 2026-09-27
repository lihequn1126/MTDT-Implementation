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
import argparse

from data_provider import get_data  # 统一无泄漏数据管道

parser = argparse.ArgumentParser(description='Time Series Forecasting')


parser.add_argument('--nwp_path', type=str, default=r"../nwpData/baoxing.csv", help='Path to NWP data')
parser.add_argument('--load_path', type=str, default=r"../LoadData/baoxing.csv", help='Path to Load data')
parser.add_argument('--output_dir', type=str, default=r"../result/baoxing_dlinear", help='Output directory')


args = parser.parse_args()


NWP_PATH = args.nwp_path
LOAD_PATH = args.load_path
OUTPUT_DIR = args.output_dir


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Running with:\n NWP: {NWP_PATH}\n Load: {LOAD_PATH}\n Output: {OUTPUT_DIR}")


SEQ_LEN = 96        # 输入序列长度
PRED_LEN = 96       # 预测序列长度
POINTS_PER_DAY = 96 

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0005 
PATIENCE = 10

# --- 数据集划分 ---
FIXED_TRAIN_DAYS = None  
FIXED_VAL_DAYS = None    
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1

# 绘图设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 数据处理: 统一使用 data_provider.get_data (history 模式) =================

# ================= DLinear 模型 (无 Dropout) =================

class MovingAverage(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [Batch, Seq, Channels]
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinearModel(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in):
        super(DLinearModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 分解模块
        self.decomposition = SeriesDecomp(kernel_size=25)
        
        # 线性层
        self.Linear_Seasonal = nn.Linear(seq_len * enc_in, pred_len)
        self.Linear_Trend = nn.Linear(seq_len * enc_in, pred_len)
        
        # --- 已移除 Dropout ---
        
        # 初始化
        self.Linear_Seasonal.weight.data.normal_(0, 0.01)
        self.Linear_Trend.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x: [Batch, Seq, Channels]
        
        # 1. 分解
        seasonal_init, trend_init = self.decomposition(x)
        
        # 2. 展平
        batch_size = x.shape[0]
        seasonal_init = seasonal_init.reshape(batch_size, -1)
        trend_init = trend_init.reshape(batch_size, -1)
        
        # 3. 线性映射 (无 Dropout)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        
        # 4. 合并
        x = seasonal_output + trend_output
        return x

# ================= 训练流程 =================

def train_and_evaluate():
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
    
    # 4. 模型
    input_dim = info['n_features']
    model = DLinearModel(seq_len=SEQ_LEN, pred_len=PRED_LEN, enc_in=input_dim).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Training DLinear (No Dropout)...")
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    
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
        
        avg_train_loss = np.mean(batch_losses)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                pred = model(bx)
                val_batch_losses.append(criterion(pred, by).item())
        
        avg_val_loss = np.mean(val_batch_losses)
        val_losses.append(avg_val_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stop at {epoch+1}")
                break
                
    # 5. 绘图
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()
    
    # 6. 评估
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    model.eval()
    
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(DEVICE)
            pred = model(bx)
            preds.append(pred.cpu().numpy())
            trues.append(by.numpy())
            
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    preds_inv = preds * scaler_y.scale_ + scaler_y.mean_
    trues_inv = trues * scaler_y.scale_ + scaler_y.mean_
    
    mae = mean_absolute_error(trues_inv.flatten(), preds_inv.flatten())
    rmse = np.sqrt(mean_squared_error(trues_inv.flatten(), preds_inv.flatten()))
    r2 = r2_score(trues_inv.flatten(), preds_inv.flatten())
    
    print(f"\nFinal Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    
    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}\nR2: {r2}\n")
        
    # 保存结果 CSV
    stitch_pred, stitch_true, stitch_time = [], [], []
    # test_start_time 由统一数据管道提供
    
    for i in range(0, len(preds_inv), POINTS_PER_DAY):
        if i >= len(preds_inv): break
        y_p = preds_inv[i]
        y_t = trues_inv[i]
        curr_time = pd.date_range(start=test_start_time + pd.Timedelta(minutes=15*i), periods=PRED_LEN, freq='15min')
        
        stitch_pred.extend(y_p)
        stitch_true.extend(y_t)
        stitch_time.extend(curr_time)
        
        # Day Plot
        plt.figure(figsize=(10,4))
        plt.plot(curr_time, y_t, label='True')
        plt.plot(curr_time, y_p, label='DLinear', linestyle='--')
        plt.title(f"{str(curr_time[0].date())}")
        plt.legend()
        plt.savefig(os.path.join(dayplot_dir, f"{str(curr_time[0].date())}.png"))
        plt.close()
        
    res_df = pd.DataFrame({'time': stitch_time, 'true': stitch_true, 'pred': stitch_pred})
    res_df.to_csv(os.path.join(OUTPUT_DIR, "prediction_result.csv"), index=False)
    print("Done.")

if __name__ == "__main__":
    train_and_evaluate()

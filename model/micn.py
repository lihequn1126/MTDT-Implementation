import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import math
import random
import argparse

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

parser = argparse.ArgumentParser(description='MICN Config')
parser.add_argument('--nwp_path', type=str, default=r"../nwpData/hanyuan.csv", help='Path to NWP data')
parser.add_argument('--load_path', type=str, default=r"../LoadData/hanyuan.csv", help='Path to Load data')
parser.add_argument('--output_dir', type=str, default=r"../result/hanyuan_micn", help='Output directory')
args, _ = parser.parse_known_args()

class Config:
    NWP_PATH = args.nwp_path
    LOAD_PATH = args.load_path
    OUTPUT_DIR = args.output_dir

    SEQ_LEN = 96       
    PRED_LEN = 96      
    POINTS_PER_DAY = 96 
    
    # MICN 特有参数
    D_MODEL = 64
    D_FF = 128
    DROPOUT = 0.1
    # 卷积核大小，用于捕捉不同尺度的局部特征 (通常设为 [3, 5, 7] 等)
    CONV_KERNEL = [3, 5] 
    DECOMP_KERNEL = 25  # 分解层移动平均窗口大小
    
    BATCH_SIZE = 32
    EPOCHS = 50         
    LEARNING_RATE = 0.001 
    PATIENCE = 10       
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    ENC_IN = 0 

cfg = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeriesDecomp(nn.Module):
    """ 序列分解层：将序列分解为趋势项和季节性项 """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        # x: [B, L, C]
        moving_mean = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        res = x - moving_mean
        return res, moving_mean


class SeasonalPrediction(nn.Module):
    """ MICN 的季节性预测分支 (Multi-scale Local Context) """
    def __init__(self, d_model, kernels, dropout):
        super(SeasonalPrediction, self).__init__()
        self.kernels = kernels
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for k in kernels
        ])
        self.projection = nn.Linear(d_model * len(kernels), d_model)

    def forward(self, x):
        # x: [B, L, D]
        x = x.permute(0, 2, 1) # [B, D, L]
        out = []
        for conv in self.conv_layers:
            out.append(conv(x))
        out = torch.cat(out, dim=1) # [B, D*len(kernels), L]
        out = out.permute(0, 2, 1) # [B, L, D*...]
        return self.projection(out)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.SEQ_LEN
        self.pred_len = configs.PRED_LEN
        
        # 分解层
        self.decomp = SeriesDecomp(configs.DECOMP_KERNEL)
        
        # 嵌入层
        self.enc_embedding = nn.Linear(configs.ENC_IN, configs.D_MODEL)
        
        # MICN 季节性预测 (核心卷积块)
        self.seasonal_mixer = SeasonalPrediction(configs.D_MODEL, configs.CONV_KERNEL, configs.DROPOUT)
        
        # 趋势项预测 (通常使用线性映射)
        self.trend_projection = nn.Linear(configs.SEQ_LEN + configs.PRED_LEN, configs.PRED_LEN)
        
        # 最终输出投影
        self.seasonal_projection = nn.Linear(configs.SEQ_LEN + configs.PRED_LEN, configs.PRED_LEN)
        self.final_projection = nn.Linear(configs.D_MODEL, configs.ENC_IN)

    def forward(self, x_hist, x_fut_nwp):
        # 1. 归一化 (Instance Norm)
        means = x_hist.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_hist, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        
        x_hist = (x_hist - means) / stdev
        
        means_nwp = means[:, :, :-1]
        stdev_nwp = stdev[:, :, :-1]
        x_fut_nwp = (x_fut_nwp - means_nwp) / stdev_nwp
        
        # 2. 构建输入序列
        batch_size = x_hist.shape[0]
        zeros_target = torch.zeros(batch_size, self.pred_len, 1, device=x_hist.device)
        x_fut_combined = torch.cat([x_fut_nwp, zeros_target], dim=2)
        x_full = torch.cat([x_hist, x_fut_combined], dim=1) # [B, L_seq + L_pred, C]
        
        # 3. 分解与特征提取
        seasonal_init, trend_init = self.decomp(x_full)
        
        # 季节性分支
        seasonal_feat = self.enc_embedding(seasonal_init)
        seasonal_feat = self.seasonal_mixer(seasonal_feat) 
        seasonal_feat = self.final_projection(seasonal_feat) # [B, L_full, C]
        
        # 4. 时间维度投影
        seasonal_out = self.seasonal_projection(seasonal_feat.permute(0, 2, 1)).permute(0, 2, 1)
        trend_out = self.trend_projection(trend_init.permute(0, 2, 1)).permute(0, 2, 1)
        
        # 5. 合并
        dec_out = seasonal_out + trend_out
        
        # 6. 反实例归一化
        target_mean = means[:, :, -1:] 
        target_stdev = stdev[:, :, -1:]
        final_out = dec_out[:, :, -1:] * target_stdev + target_mean
        
        return final_out


class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len, self.pred_len = seq_len, pred_len
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_end = s_end + self.pred_len
        return self.data[index:s_end], self.data[s_end:r_end, :-1], self.data[s_end:r_end, -1:]

def load_and_process_data():
    if not os.path.exists(cfg.NWP_PATH) or not os.path.exists(cfg.LOAD_PATH): return None
    df_nwp, df_load = pd.read_csv(cfg.NWP_PATH), pd.read_csv(cfg.LOAD_PATH)
    df_nwp['time'], df_load['time'] = pd.to_datetime(df_nwp['time']), pd.to_datetime(df_load['time'])
    df = pd.merge(df_load, df_nwp, on='time', how='inner').sort_values('time').set_index('time')
    
    target_col = [c for c in df.columns if 'load' in c.lower()][0]
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    cols = [c for c in df.columns if c != target_col] + [target_col]
    df = df[cols].ffill().bfill().iloc[96*7:]
    return df

def create_dataloaders(df):
    total_days = len(df) // cfg.POINTS_PER_DAY
    tr_idx = int(total_days * cfg.TRAIN_RATIO) * cfg.POINTS_PER_DAY
    val_idx = int(total_days * (cfg.TRAIN_RATIO + cfg.VAL_RATIO)) * cfg.POINTS_PER_DAY
    
    scaler = StandardScaler()
    train_vals = scaler.fit_transform(df.iloc[:tr_idx].values)
    val_vals = scaler.transform(df.iloc[tr_idx-cfg.SEQ_LEN:val_idx].values)
    test_vals = scaler.transform(df.iloc[val_idx-cfg.SEQ_LEN:].values)
    
    scaler_y = StandardScaler()
    scaler_y.mean_, scaler_y.scale_ = scaler.mean_[-1], scaler.scale_[-1]
    cfg.ENC_IN = train_vals.shape[1]

    return DataLoader(TimeSeriesDataset(train_vals, cfg.SEQ_LEN, cfg.PRED_LEN), batch_size=cfg.BATCH_SIZE, shuffle=True), \
           DataLoader(TimeSeriesDataset(val_vals, cfg.SEQ_LEN, cfg.PRED_LEN), batch_size=cfg.BATCH_SIZE, shuffle=False), \
           DataLoader(TimeSeriesDataset(test_vals, cfg.SEQ_LEN, cfg.PRED_LEN), batch_size=cfg.BATCH_SIZE, shuffle=False), \
           scaler_y, df.index[val_idx]


def train_and_evaluate():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    dayplot_dir = os.path.join(cfg.OUTPUT_DIR, "dayplot")
    os.makedirs(dayplot_dir, exist_ok=True)
    
    df = load_and_process_data()
    train_loader, val_loader, test_loader, scaler_y, test_start_time = create_dataloaders(df)
    
    model = Model(cfg).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    early_stop = 0
    
    print("Training MICN Model...")
    for epoch in range(cfg.EPOCHS):
        model.train()
        t_l = []
        for x, x_f, y in train_loader:
            x, x_f, y = x.to(DEVICE), x_f.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x, x_f)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            t_l.append(loss.item())
        
        model.eval()
        v_l = []
        with torch.no_grad():
            for x, x_f, y in val_loader:
                out = model(x.to(DEVICE), x_f.to(DEVICE))
                v_l.append(criterion(out, y.to(DEVICE)).item())
        
        avg_v = np.mean(v_l)
        print(f"Epoch {epoch+1} | Train Loss: {np.mean(t_l):.5f} | Val Loss: {avg_v:.5f}")
        
        if avg_v < best_loss:
            best_loss = avg_v
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "best.pth"))
            early_stop = 0
        elif (early_stop := early_stop + 1) >= cfg.PATIENCE: break

    model.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, "best.pth")))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, x_f, y in test_loader:
            out = model(x.to(DEVICE), x_f.to(DEVICE))
            preds.append(out.cpu().numpy())
            trues.append(y.numpy())
            
    preds = np.concatenate(preds, axis=0).squeeze(-1)
    trues = np.concatenate(trues, axis=0).squeeze(-1)
    
    preds = preds * scaler_y.scale_ + scaler_y.mean_
    trues = trues * scaler_y.scale_ + scaler_y.mean_
  
    mae = mean_absolute_error(trues.flatten(), preds.flatten())
    rmse = np.sqrt(mean_squared_error(trues.flatten(), preds.flatten()))
    r2 = r2_score(trues.flatten(), preds.flatten())
    print(f"Test Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    stitch_pred, stitch_true, stitch_time = [], [], []
    for i in range(0, len(preds), cfg.POINTS_PER_DAY):
        y_p, y_t = preds[i], trues[i]
        curr_start = test_start_time + pd.Timedelta(minutes=15 * i)
        curr_time = pd.date_range(start=curr_start, periods=len(y_p), freq='15min')
        stitch_pred.extend(y_p); stitch_true.extend(y_t); stitch_time.extend(curr_time)
        
        plt.figure(figsize=(10, 4))
        plt.plot(curr_time, y_t, label='True'); plt.plot(curr_time, y_p, '--', label='MICN')
        plt.title(f"{curr_start.date()} RMSE: {np.sqrt(mean_squared_error(y_t, y_p)):.2f}")
        plt.legend(); plt.savefig(os.path.join(dayplot_dir, f"{curr_start.date()}.png")); plt.close()
        
    pd.DataFrame({'time': stitch_time, 'true': stitch_true, 'pred': stitch_pred}).to_csv(
        os.path.join(cfg.OUTPUT_DIR, "prediction_result.csv"), index=False)
    
    plt.figure(figsize=(15, 6))
    plt.plot(stitch_time, stitch_true, label='True', alpha=0.7); plt.plot(stitch_time, stitch_pred, '--', label='MICN', alpha=0.7)
    plt.title("Full Test Set Prediction (MICN)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "full_prediction.png")); plt.show()

if __name__ == "__main__":
    train_and_evaluate()
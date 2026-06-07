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

parser = argparse.ArgumentParser(description='TimesNet Config')
parser.add_argument('--nwp_path', type=str, default=r"../nwpData/hanyuan.csv", help='Path to NWP data')
parser.add_argument('--load_path', type=str, default=r"../LoadData/hanyuan.csv", help='Path to Load data')
parser.add_argument('--output_dir', type=str, default=r"../result/hanyuan_timesnet", help='Output directory')
args, _ = parser.parse_known_args()

class Config:
    NWP_PATH = args.nwp_path
    LOAD_PATH = args.load_path
    OUTPUT_DIR = args.output_dir

    SEQ_LEN = 96       
    PRED_LEN = 96      
    POINTS_PER_DAY = 96 
    
    # TimesNet 特有参数
    TOP_K = 5           # FFT 提取的前 K 个主周期
    D_MODEL = 64        # 模型隐藏层维度
    D_FF = 128          # FFN 维度
    E_LAYERS = 2        # TimesBlock 的层数
    NUM_KERNELS = 6     # Inception 卷积核数量
    DROPOUT = 0.1
    
    BATCH_SIZE = 32
    EPOCHS = 50         
    LEARNING_RATE = 0.001 
    PATIENCE = 10       
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    ENC_IN = 0 # 动态设置

cfg = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def FFT_for_Period(x, k=2):
    # x: [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # 取平均幅值，找出最显著的周期
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0 # 排除直流分量
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list] # 返回周期长度和对应的权重

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.SEQ_LEN + configs.PRED_LEN
        self.k = configs.TOP_K
        # 参数化 Inception
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.D_MODEL, configs.D_FF, num_kernels=configs.NUM_KERNELS),
            nn.GELU(),
            Inception_Block_V1(configs.D_FF, configs.D_MODEL, num_kernels=configs.NUM_KERNELS)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        
        res = []
        for i in range(self.k):
            period = period_list[i]
            # 补齐长度以适应 2D 重塑
            if self.seq_len % period != 0:
                length = (((self.seq_len // period) + 1) * period)
                padding = torch.zeros([B, (length - self.seq_len), N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x
            
            # 1D -> 2D: [B, N, T] -> [B, N, Period, T/Period]
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D 卷积
            out = self.conv(out)
            # 2D -> 1D
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])
        
        res = torch.stack(res, dim=-1)
        # 自适应聚合
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.SEQ_LEN
        self.pred_len = configs.PRED_LEN
        
        self.model_optim = nn.ModuleList([TimesBlock(configs) for _ in range(configs.E_LAYERS)])
        self.enc_embedding = nn.Linear(configs.ENC_IN, configs.D_MODEL)
        self.predict_linear = nn.Linear(configs.SEQ_LEN, configs.PRED_LEN + configs.SEQ_LEN)
        self.projection = nn.Linear(configs.D_MODEL, configs.ENC_IN)

    def forward(self, x_hist, x_fut_nwp):
        # 1. Instance Normalization
        means = x_hist.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_hist, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_hist = (x_hist - means) / stdev
        
        # NWP 归一化
        means_nwp = means[:, :, :-1]
        stdev_nwp = stdev[:, :, :-1]
        x_fut_nwp = (x_fut_nwp - means_nwp) / stdev_nwp
 
        # 构建完整输入
        batch_size = x_hist.shape[0]
        zeros_target = torch.zeros(batch_size, self.pred_len, 1, device=x_hist.device)
        x_fut_combined = torch.cat([x_fut_nwp, zeros_target], dim=2)
        x_full = torch.cat([x_hist, x_fut_combined], dim=1) # [B, Seq+Pred, C]
        
        # 嵌入
        enc_out = self.enc_embedding(x_full) # [B, T, D]
  
        # TimesNet 核心块
        for block in self.model_optim:
            enc_out = block(enc_out)
    
        # 投影回原始空间
        dec_out = self.projection(enc_out) # [B, T, C]
        
        # 取出预测长度部分
        dec_out = dec_out[:, -self.pred_len:, :]
        
        # 修复后的反归一化 (Target列)
        target_mean = means[:, :, -1:]   # [B, 1, 1]
        target_stdev = stdev[:, :, -1:] # [B, 1, 1]
        
        final_out = dec_out[:, :, -1:] * target_stdev + target_mean
        return final_out

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_end = s_end + self.pred_len
        return self.data[index:s_end], self.data[s_end:r_end, :-1], self.data[s_end:r_end, -1:]

def load_and_process_data():
    if not os.path.exists(cfg.NWP_PATH) or not os.path.exists(cfg.LOAD_PATH): return None
    df_nwp = pd.read_csv(cfg.NWP_PATH)
    df_load = pd.read_csv(cfg.LOAD_PATH)
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
    
    print("Training TimesNet...")
    for epoch in range(cfg.EPOCHS):
        model.train()
        t_l = []
        for x, x_f, y in train_loader:
            x, x_f, y = x.to(DEVICE), x_f.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x, x_f)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
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

    # --- 评估 ---
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
    
    # =====================================================================
    # 核心修复点：在此处进行全局反归一化 (还原为真实的量纲)
    preds = preds * scaler_y.scale_ + scaler_y.mean_
    trues = trues * scaler_y.scale_ + scaler_y.mean_
    # =====================================================================

    # 指标
    mae = mean_absolute_error(trues.flatten(), preds.flatten())
    rmse = np.sqrt(mean_squared_error(trues.flatten(), preds.flatten()))
    r2 = r2_score(trues.flatten(), preds.flatten())
    print(f"Test Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    with open(os.path.join(cfg.OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}\nR2: {r2}\n")

    # 缝合与绘图
    stitch_pred, stitch_true, stitch_time = [], [], []
    for i in range(0, len(preds), cfg.POINTS_PER_DAY):
        y_p, y_t = preds[i], trues[i]
        curr_start = test_start_time + pd.Timedelta(minutes=15 * i)
        curr_time = pd.date_range(start=curr_start, periods=len(y_p), freq='15min')
        stitch_pred.extend(y_p); stitch_true.extend(y_t); stitch_time.extend(curr_time)
        
        # 分日绘图
        plt.figure(figsize=(10, 4))
        plt.plot(curr_time, y_t, label='True'); plt.plot(curr_time, y_p, '--', label='TimesNet')
        plt.title(f"{curr_start.date()} RMSE: {np.sqrt(mean_squared_error(y_t, y_p)):.2f}")
        plt.legend(); plt.savefig(os.path.join(dayplot_dir, f"{curr_start.date()}.png")); plt.close()
        
    pd.DataFrame({'time': stitch_time, 'true': stitch_true, 'pred': stitch_pred}).to_csv(
        os.path.join(cfg.OUTPUT_DIR, "prediction_result.csv"), index=False)
    
    plt.figure(figsize=(15, 6))
    plt.plot(stitch_time, stitch_true, label='True', alpha=0.7)
    plt.plot(stitch_time, stitch_pred, '--', label='TimesNet', alpha=0.7)
    plt.title("Full Test Set Prediction (TimesNet)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "full_prediction.png")); plt.show()

if __name__ == "__main__":
    train_and_evaluate()
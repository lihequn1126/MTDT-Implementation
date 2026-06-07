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

parser = argparse.ArgumentParser(description='PatchTST Config')
parser.add_argument('--nwp_path', type=str, default=r"../nwpData/hanyuan.csv", help='Path to NWP data')
parser.add_argument('--load_path', type=str, default=r"../LoadData/hanyuan.csv", help='Path to Load data')
parser.add_argument('--output_dir', type=str, default=r"../result/hanyuan_patchtst", help='Output directory')
args, _ = parser.parse_known_args()

class Config:
    NWP_PATH = args.nwp_path
    LOAD_PATH = args.load_path
    OUTPUT_DIR = args.output_dir

    SEQ_LEN = 96       
    PRED_LEN = 96      
    POINTS_PER_DAY = 96 
    
    # PatchTST 特有参数
    PATCH_LEN = 16      
    STRIDE = 8          
    D_MODEL = 128       
    N_HEADS = 8         
    E_LAYERS = 3        
    D_FF = 512          
    DROPOUT = 0.1      
    ACTIVATION = 'gelu' 
    
    BATCH_SIZE = 32
    EPOCHS = 50         
    LEARNING_RATE = 0.0005 
    PATIENCE = 10       

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    ENC_IN = 0 # 动态设置

cfg = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch * Vars, Num_Patches, Patch_Len]
        x = self.value_embedding(x) 
        return self.dropout(x)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.patch_len = configs.PATCH_LEN
        self.stride = configs.STRIDE
        self.pred_len = configs.PRED_LEN
        self.full_seq_len = configs.SEQ_LEN + configs.PRED_LEN
        
        # 计算 Patch 数量
        self.num_patch = (max(self.full_seq_len, self.patch_len) - self.patch_len) // self.stride + 1
        
        # Embedding
        self.enc_embedding = PatchEmbedding(configs.D_MODEL, self.patch_len, self.stride, configs.DROPOUT)
        
        # Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patch, configs.D_MODEL))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.D_MODEL, nhead=configs.N_HEADS, dim_feedforward=configs.D_FF, 
            dropout=configs.DROPOUT, activation=configs.ACTIVATION, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=configs.E_LAYERS)
        
        # Output Head
        self.head = nn.Linear(configs.D_MODEL * self.num_patch, configs.PRED_LEN)

    def forward(self, x_hist, x_fut_nwp):
        # 1. Instance Normalization
        means = x_hist.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_hist, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_hist = (x_hist - means) / stdev
        
        # 使用对应的通道均值对未来 NWP 进行归一化
        means_nwp = means[:, :, :-1]
        stdev_nwp = stdev[:, :, :-1]
        x_fut_nwp = (x_fut_nwp - means_nwp) / stdev_nwp
        
        # 2. 合并历史与未来已知信息
        batch_size = x_hist.shape[0]
        zeros_target = torch.zeros(batch_size, self.pred_len, 1, device=x_hist.device)
        x_fut_combined = torch.cat([x_fut_nwp, zeros_target], dim=2)
        x_full = torch.cat([x_hist, x_fut_combined], dim=1) # [B, L_full, C]
        
        # 3. 通道独立 (Channel Independence)
        B, L, C = x_full.shape
        x = x_full.permute(0, 2, 1).reshape(B * C, L)
        
        # 4. Patching: [B*C, Num_Patches, Patch_Len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # 5. Embedding & Encoder
        enc_out = self.enc_embedding(x) + self.pos_embedding
        enc_out = self.encoder(enc_out) # [B*C, N, D]
        
        # 6. Linear Head
        enc_out = enc_out.reshape(B * C, -1)
        dec_out = self.head(enc_out) # [B*C, Pred_Len]
        
        # 7. Reshape Back & Denormalization
        dec_out = dec_out.reshape(B, C, -1).permute(0, 2, 1) # [B, Pred_Len, C]
        
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
    if not os.path.exists(cfg.NWP_PATH) or not os.path.exists(cfg.LOAD_PATH):
        return None
    df_nwp = pd.read_csv(cfg.NWP_PATH)
    df_load = pd.read_csv(cfg.LOAD_PATH)
    df_nwp['time'] = pd.to_datetime(df_nwp['time'])
    df_load['time'] = pd.to_datetime(df_load['time'])
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

    train_loader = DataLoader(TimeSeriesDataset(train_vals, cfg.SEQ_LEN, cfg.PRED_LEN), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(val_vals, cfg.SEQ_LEN, cfg.PRED_LEN), batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TimeSeriesDataset(test_vals, cfg.SEQ_LEN, cfg.PRED_LEN), batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler_y, df.index[val_idx]


def train_and_evaluate():
    if not os.path.exists(cfg.OUTPUT_DIR): os.makedirs(cfg.OUTPUT_DIR)
    dayplot_dir = os.path.join(cfg.OUTPUT_DIR, "dayplot")
    if not os.path.exists(dayplot_dir): os.makedirs(dayplot_dir)
    
    df = load_and_process_data()
    if df is None: return
    train_loader, val_loader, test_loader, scaler_y, test_start_time = create_dataloaders(df)
    
    model = Model(cfg).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    early_stop = 0
    
    # 训练循环
    print("Training PatchTST Model...")
    for epoch in range(cfg.EPOCHS):
        model.train()
        train_l = []
        for x, x_f, y in train_loader:
            x, x_f, y = x.to(DEVICE), x_f.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x, x_f)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_l.append(loss.item())
        
        model.eval()
        val_l = []
        with torch.no_grad():
            for x, x_f, y in val_loader:
                out = model(x.to(DEVICE), x_f.to(DEVICE))
                val_l.append(criterion(out, y.to(DEVICE)).item())
        
        avg_val = np.mean(val_l)
        print(f"Epoch {epoch+1} | Train Loss: {np.mean(train_l):.5f} | Val Loss: {avg_val:.5f}")
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "best_model.pth"))
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= cfg.PATIENCE: break

    # --- 评估阶段 ---
    model.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, "best_model.pth")))
    model.eval()
    preds, trues = [], []
    
    with torch.no_grad():
        for x, x_f, y in test_loader:
            out = model(x.to(DEVICE), x_f.to(DEVICE))
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
            
    preds = np.concatenate(preds, axis=0).squeeze(-1) # [N, Pred_Len]
    trues = np.concatenate(trues, axis=0).squeeze(-1) # [N, Pred_Len]
    
    # =====================================================================
    # 核心修复点：在此处进行全局反归一化 (还原为真实的量纲)
    preds = preds * scaler_y.scale_ + scaler_y.mean_
    trues = trues * scaler_y.scale_ + scaler_y.mean_
    # =====================================================================
    
    # 全局指标计算
    mae = mean_absolute_error(trues.flatten(), preds.flatten())
    rmse = np.sqrt(mean_squared_error(trues.flatten(), preds.flatten()))
    r2 = r2_score(trues.flatten(), preds.flatten())
    
    print(f"Test Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    with open(os.path.join(cfg.OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}\nR2: {r2}\n")

    # 缝合逻辑与分日绘图
    stitch_pred, stitch_true, stitch_time = [], [], []
    num_samples = len(preds)
    
    for i in range(0, num_samples, cfg.POINTS_PER_DAY):
        y_p = preds[i]
        y_t = trues[i]
        
        current_start = test_start_time + pd.Timedelta(minutes=15 * i)
        current_timeline = pd.date_range(start=current_start, periods=len(y_p), freq='15min')
        
        stitch_pred.extend(y_p)
        stitch_true.extend(y_t)
        stitch_time.extend(current_timeline)

        # 单日绘图
        day_rmse = np.sqrt(mean_squared_error(y_t, y_p))
        date_str = str(current_start.date())
        plt.figure(figsize=(10, 5))
        plt.plot(current_timeline, y_t, label='True', color='blue')
        plt.plot(current_timeline, y_p, label='PatchTST', color='red', linestyle='--')
        plt.title(f"Date: {date_str} | RMSE: {day_rmse:.2f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(dayplot_dir, f"{date_str}.png"))
        plt.close()
 
    res_df = pd.DataFrame({'time': stitch_time, 'true': stitch_true, 'pred': stitch_pred})
    res_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "prediction_result.csv"), index=False)

    plt.figure(figsize=(15, 6))
    plt.plot(res_df['time'], res_df['true'], label='True', alpha=0.7)
    plt.plot(res_df['time'], res_df['pred'], label='PatchTST', alpha=0.7, linestyle='--')
    plt.title("Full Test Set Prediction (PatchTST)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "full_prediction.png"))
    plt.close()
    print("All results and plots saved.")

if __name__ == "__main__":
    train_and_evaluate()
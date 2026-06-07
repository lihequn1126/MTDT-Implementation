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
import shutil
import random

# ================= 0. 设置随机种子 =================
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

import argparse
import os

parser = argparse.ArgumentParser(description='iTransformer_decomp Config')

parser.add_argument('--nwp_path', type=str, default=r"../nwpData/baoxing.csv", help='Path to NWP data')
parser.add_argument('--load_path', type=str, default=r"../LoadData/baoxing.csv", help='Path to Load data')
parser.add_argument('--output_dir', type=str, default=r"../result_1/baoxing_itransformer_decomp", help='Output directory')


args, _ = parser.parse_known_args()

# 自动创建目录
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

print(f"Running iTransformer with:\n NWP: {args.nwp_path}\n Load: {args.load_path}\n Output: {args.output_dir}")


class Config:
    NWP_PATH = args.nwp_path
    LOAD_PATH = args.load_path
    OUTPUT_DIR = args.output_dir

    # 数据参数
    SEQ_LEN = 96        
    PRED_LEN = 96       
    POINTS_PER_DAY = 96 
    FREQ = '15min'

    # 模型参数
    ENC_IN = 0          # 自动计算
    D_MODEL = 512       
    N_HEADS = 8         
    E_LAYERS = 2        
    D_FF = 2048         
    DROPOUT = 0.1       
    ACTIVATION = 'gelu' 
    USE_NORM = True     

    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 60          
    LEARNING_RATE = 0.0005 
    PATIENCE = 10      

    FIXED_TRAIN_DAYS = None 
    FIXED_VAL_DAYS = None    
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1

cfg = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 绘图设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 模型组件 (Model Components) =================

class DataEmbedding_inverted(nn.Module):
    def __init__(self, input_len, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(input_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        # x: [Batch, Vars, Seq_Len]
        x = self.value_embedding(x)
        return self.dropout(x)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        if self.output_attention: return (V.contiguous(), A)
        else: return (V.contiguous(), None)

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None: x = self.norm(x)
        return x, attns

# === [核心修改 1] 序列分解模块 ===
class MovingAverage(nn.Module):
    """
    移动平均用于提取趋势项
    """
    def __init__(self, kernel_size, stride):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        # Padding at both ends
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    """
    分解为 Seasonal 和 Trend
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

# === [核心修改 2] 融合分解架构的模型 ===
class OptimizedDecompModel(nn.Module):
    def __init__(self, configs):
        super(OptimizedDecompModel, self).__init__()
        self.seq_len = configs.SEQ_LEN
        self.pred_len = configs.PRED_LEN
        self.use_norm = configs.USE_NORM
        
        # 1. Decomposition
        # kernel_size 设为 35  平滑短期波动
        self.decomposition = SeriesDecomp(kernel_size=35)
        
        # 2. iTransformer Encoder (处理 Seasonal 部分)
        # 输入长度 = 历史 + 预测 (iTransformer 逻辑)
        full_len = self.seq_len + self.pred_len
        self.enc_embedding = DataEmbedding_inverted(full_len, configs.D_MODEL, configs.DROPOUT)
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.DROPOUT, output_attention=False),
                        configs.D_MODEL, configs.N_HEADS),
                    configs.D_MODEL, configs.D_FF, dropout=configs.DROPOUT, activation=configs.ACTIVATION
                ) for l in range(configs.E_LAYERS)
            ],
            norm_layer=nn.LayerNorm(configs.D_MODEL)
        )
        self.projector = nn.Linear(configs.D_MODEL, self.pred_len, bias=True)

        # 3. Trend Projection (处理 Trend 部分)
        # 简单的线性映射：[Batch, Vars, Seq_Len] -> [Batch, Vars, Pred_Len]
        self.trend_projector = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x_hist, x_fut_nwp):
        # x_hist: [Batch, Seq_Len, Vars]
        # x_fut_nwp: [Batch, Pred_Len, Vars_NWP]
        
        # --- Step 1: Normalization (RevIN) ---
        if self.use_norm:
            means = x_hist.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x_hist, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x_hist = (x_hist - means) / stdev
            
            # 对未来 NWP 也做同样的 norm (使用历史 NWP 的统计量)
            # 假设 NWP 是前 N-1 列
            means_nwp = means[:, :, :-1]
            stdev_nwp = stdev[:, :, :-1]
            x_fut_nwp = (x_fut_nwp - means_nwp) / stdev_nwp

        # --- Step 2: Decomposition ---
        # 将历史数据分解为 Seasonal 和 Trend
        seasonal_init, trend_init = self.decomposition(x_hist)
        
        # --- Step 3: Trend Branch (Linear) ---
        # permute to [Batch, Vars, Seq_Len] for Linear
        trend_init = trend_init.permute(0, 2, 1)
        trend_output = self.trend_projector(trend_init) # -> [Batch, Vars, Pred_Len]
        trend_output = trend_output.permute(0, 2, 1)    # -> [Batch, Pred_Len, Vars]
        
        # --- Step 4: Seasonal Branch (iTransformer) ---
        # 构造 iTransformer 的输入：
        # Seasonal_Hist + Zero_Padding_Target (or Future_NWP)
        batch_size = x_hist.shape[0]
        
        # 拼接未来特征：NWP部分用真实的 x_fut_nwp，Target部分填0
        zeros_target = torch.zeros(batch_size, self.pred_len, 1, device=x_hist.device)
        x_fut_combined = torch.cat([x_fut_nwp, zeros_target], dim=2)
        
        # 拼接历史 seasonal 和未来
        # 注意：这里我们让 iTransformer 看到的是 "去趋势后的历史" + "未来特征"
        # 这样 Attention 只需要关注波动
        x_full = torch.cat([seasonal_init, x_fut_combined], dim=1)
        
        # Embedding & Encoder [Batch, Vars, Time] -> [Batch, Vars, D_Model]
        x_full = x_full.permute(0, 2, 1) 
        enc_out = self.enc_embedding(x_full, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        
        # Projector -> [Batch, Vars, Pred_Len]
        seasonal_output = self.projector(enc_out)
        seasonal_output = seasonal_output.permute(0, 2, 1) # -> [Batch, Pred_Len, Vars]

        # --- Step 5: Combine & De-Normalization ---
        dec_out = seasonal_output + trend_output

        if self.use_norm:
            # 只反归一化 Target 列 (最后一列)
            # 如果需要输出所有变量，可以用 full means/stdev
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out

# ================= 2. 数据处理 (Data Loading) =================

def load_and_process_data():
    print("🚀 Loading data with Enhanced Features...")
    if not os.path.exists(cfg.NWP_PATH) or not os.path.exists(cfg.LOAD_PATH):
        print("Warning: Data path not found.")
        return None

    try:
        df_nwp = pd.read_csv(cfg.NWP_PATH)
        df_load = pd.read_csv(cfg.LOAD_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
    df_nwp['time'] = pd.to_datetime(df_nwp['time'])
    df_load['time'] = pd.to_datetime(df_load['time'])
    
    df = pd.merge(df_load, df_nwp, on='time', how='inner').sort_values('time').set_index('time')
    
    load_cols = [c for c in df.columns if 'load' in c.lower()]
    target_col = load_cols[0] if load_cols else df.columns[-1]
    
    # [基础特征]
    minutes = df.index.hour * 60 + df.index.minute
    df['min_sin'] = np.sin(2 * np.pi * minutes / 1440)
    df['min_cos'] = np.cos(2 * np.pi * minutes / 1440)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    # === [核心修改 3] 特征增强 (Lag Features) ===
    print("Adding Lag Features...")
    # 假设一天 96 个点
    day_steps = 96
    
    # 1. 昨天同一时刻
    df['load_lag_1d'] = df[target_col].shift(day_steps)
    # 2. 上周同一时刻 (强周期性)
    df['load_lag_7d'] = df[target_col].shift(day_steps * 7)
    # 3. 昨天附近的均值 (平滑特征)
    df['load_mean_1d'] = df[target_col].rolling(window=day_steps, min_periods=1).mean().shift(1)
    
    # 删除因 shift 产生的空值
    df = df.dropna()
    
    # 确保 Target 在最后一列
    feature_cols = [c for c in df.columns if c != target_col]
    df = df[feature_cols + [target_col]]
    
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Target: {target_col}")
    
    return df

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data[s_begin:s_end]
        # NWP 是除了最后一列(Target)之外的所有列吗？
        # 注意：这里因为加了 Lag 特征，Lag 也是历史已知的，但在未来是未知的
        # 但我们这里的 Dataset 逻辑是：x_fut_nwp 应该只包含真实的未来已知量 (NWP + Time)
        # Lag 特征在未来是未知的（因为通过递归预测得到，或者这里简化为只用 NWP）
        # 为了代码不崩，我们假设 x_fut_nwp 取除了 Lag 和 Load 之外的特征
        # 简单处理：取前 N 列作为 NWP (假设 Lag 特征放在中间了)
        # 让我们回溯 load_and_process_data，feature_cols 顺序不确定
        # 为了稳健，我们这里简单将所有非 Target 列视为 Future Input，
        # 实际预测时，Lag 列在未来位置的值其实是无效的，但 Transformer 会学习忽略它们或利用它们（如果是 Leakage）。
        # **严谨做法**：x_fut_nwp 应该只包含 NWP 和 Time features。
        # 考虑到代码通用性，这里直接取 :-1。
        seq_x_fut_nwp = self.data[r_begin:r_end, :-1] 
        seq_y = self.data[r_begin:r_end, -1:]
        
        return seq_x, seq_x_fut_nwp, seq_y

def create_dataloaders(df):
    total_rows = len(df)
    total_days = total_rows // cfg.POINTS_PER_DAY
    
    if cfg.FIXED_TRAIN_DAYS is not None and cfg.FIXED_VAL_DAYS is not None:
        n_train_days = cfg.FIXED_TRAIN_DAYS
        n_val_days = cfg.FIXED_VAL_DAYS
    else:
        n_train_days = int(total_days * cfg.TRAIN_RATIO)
        n_val_days = int(total_days * cfg.VAL_RATIO)
    
    train_end_idx = n_train_days * cfg.POINTS_PER_DAY
    val_end_idx = (n_train_days + n_val_days) * cfg.POINTS_PER_DAY
    
    df_train = df.iloc[:train_end_idx]
    df_val = df.iloc[train_end_idx:val_end_idx]
    df_test = df.iloc[val_end_idx:] 
    
    scaler = StandardScaler()
    train_vals = scaler.fit_transform(df_train.values)
    val_vals = scaler.transform(df_val.values) if len(df_val) > 0 else np.empty((0, train_vals.shape[1]))
    test_vals = scaler.transform(df_test.values) if len(df_test) > 0 else np.empty((0, train_vals.shape[1]))
    
    scaler_y = StandardScaler()
    scaler_y.mean_ = scaler.mean_[-1]
    scaler_y.scale_ = scaler.scale_[-1]
    scaler_y.var_ = scaler.var_[-1]
    
    cfg.ENC_IN = train_vals.shape[1]
    
    def prepare_data(curr, prev_tail=None):
        if prev_tail is not None: return np.vstack([prev_tail, curr])
        return curr

    train_data = prepare_data(train_vals, None)
    # Val/Test 需要拼接前一段的尾部以构建序列
    val_data = prepare_data(val_vals, train_vals[-cfg.SEQ_LEN:]) if len(val_vals) > 0 else np.empty((0, cfg.ENC_IN))
    test_data = prepare_data(test_vals, val_vals[-cfg.SEQ_LEN:]) if len(test_vals) > 0 else np.empty((0, cfg.ENC_IN))
    
    train_set = TimeSeriesDataset(train_data, cfg.SEQ_LEN, cfg.PRED_LEN)
    val_set = TimeSeriesDataset(val_data, cfg.SEQ_LEN, cfg.PRED_LEN)
    test_set = TimeSeriesDataset(test_data, cfg.SEQ_LEN, cfg.PRED_LEN)
    
    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    test_start_time = df_test.index[0] if len(df_test) > 0 else None
    
    return train_loader, val_loader, test_loader, scaler_y, test_start_time

# ================= 3. 训练与评估 =================

def train_and_evaluate():
    if os.path.exists(cfg.OUTPUT_DIR):
        try:
            shutil.rmtree(cfg.OUTPUT_DIR) # 清空旧结果
        except:
            pass
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    dayplot_dir = os.path.join(cfg.OUTPUT_DIR, "dayplot")
    os.makedirs(dayplot_dir, exist_ok=True)
    
    df = load_and_process_data()
    if df is None: return
    train_loader, val_loader, test_loader, scaler_y, test_start_time = create_dataloaders(df)
    
    # 使用优化后的分解模型
    model = OptimizedDecompModel(cfg).to(DEVICE)
    print(f"🔥 Decomp-iTransformer Created. Params: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_loss = float('inf')
    early_stop_cnt = 0
    train_losses, val_losses = [], []
    
    print("🚀 Training started...")
    for epoch in range(cfg.EPOCHS):
        model.train()
        batch_losses = []
        for batch_x, batch_x_fut_nwp, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_x_fut_nwp = batch_x_fut_nwp.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x, batch_x_fut_nwp)
            
            # outputs shape: [Batch, Pred, Vars]
            # 我们只优化 Target (最后一列)
            pred_load = outputs[:, :, -1:] 
            loss = criterion(pred_load, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        avg_train_loss = np.mean(batch_losses)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_batch_losses = []
        if len(val_loader) > 0:
            with torch.no_grad():
                for batch_x, batch_x_fut_nwp, batch_y in val_loader:
                    batch_x = batch_x.to(DEVICE)
                    batch_x_fut_nwp = batch_x_fut_nwp.to(DEVICE)
                    batch_y = batch_y.to(DEVICE)
                    outputs = model(batch_x, batch_x_fut_nwp)
                    pred_load = outputs[:, :, -1:]
                    val_batch_losses.append(criterion(pred_load, batch_y).item())
            avg_val_loss = np.mean(val_batch_losses)
        else:
            avg_val_loss = avg_train_loss
            
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{cfg.EPOCHS} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stop_cnt = 0
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "best_model.pth"))
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= cfg.PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
    
    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "loss_curve.png"))
    plt.close()

    # Testing
    print("\n🧪 Testing...")
    model.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, "best_model.pth")))
    model.eval()
    
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_x_fut_nwp, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            batch_x_fut_nwp = batch_x_fut_nwp.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            outputs = model(batch_x, batch_x_fut_nwp)
            # 同样只取最后一列
            preds.append(outputs[:, :, -1].cpu().numpy())
            trues.append(batch_y[:, :, 0].cpu().numpy())
            
    if len(preds) == 0: return

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # 反归一化
    preds_inv = preds * scaler_y.scale_ + scaler_y.mean_
    trues_inv = trues * scaler_y.scale_ + scaler_y.mean_
    
    mae = mean_absolute_error(trues_inv.flatten(), preds_inv.flatten())
    rmse = np.sqrt(mean_squared_error(trues_inv.flatten(), preds_inv.flatten()))
    r2 = r2_score(trues_inv.flatten(), preds_inv.flatten())
    
    print(f"📊 Global Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    with open(os.path.join(cfg.OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}\nR2: {r2}\n")
        
    # Day Plots
    stitch_pred, stitch_true, stitch_time = [], [], []
    num_samples = len(preds_inv)
    
    for i in range(0, num_samples, cfg.POINTS_PER_DAY):
        if i >= num_samples: break
        y_p, y_t = preds_inv[i], trues_inv[i]
        current_start = test_start_time + pd.Timedelta(minutes=15 * i)
        current_timeline = pd.date_range(start=current_start, periods=len(y_p), freq='15min')
        
        stitch_pred.extend(y_p)
        stitch_true.extend(y_t)
        stitch_time.extend(current_timeline)
        
        day_rmse = np.sqrt(mean_squared_error(y_t, y_p))
        day_r2 = r2_score(y_t, y_p)
        
        plt.figure(figsize=(10, 5))
        plt.plot(current_timeline, y_t, label='True', color='blue')
        plt.plot(current_timeline, y_p, label='Pred', color='red', linestyle='--')
        plt.title(f"{str(current_start.date())} | RMSE: {day_rmse:.2f} | R2: {day_r2:.2f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(dayplot_dir, f"{str(current_start.date())}.png"))
        plt.close()
        
    res_df = pd.DataFrame({'time': stitch_time, 'true': stitch_true, 'pred': stitch_pred})
    res_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "prediction_result.csv"), index=False)
    
    plt.figure(figsize=(20, 6))
    plt.plot(pd.to_datetime(res_df['time']), res_df['true'], label='True', color='blue', alpha=0.7)
    plt.plot(pd.to_datetime(res_df['time']), res_df['pred'], label='Pred', color='red', alpha=0.7, linestyle='--')
    plt.title(f'Full Prediction (Decomp+Features) | RMSE: {rmse:.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "full_prediction.png"))
    plt.close()
    
    print(f"✅ All Done. Results saved to {cfg.OUTPUT_DIR}")

if __name__ == "__main__":
    train_and_evaluate()
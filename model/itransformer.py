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

parser = argparse.ArgumentParser(description='iTransformer Config')

parser.add_argument('--nwp_path', type=str, default=r"../nwpData/baoxing.csv", help='Path to NWP data')
parser.add_argument('--load_path', type=str, default=r"../LoadData/baoxing.csv", help='Path to Load data')
parser.add_argument('--output_dir', type=str, default=r"../result/baoxing_itransformer", help='Output directory')


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
    SEQ_LEN = 96        # 历史窗口长度
    PRED_LEN = 96       # 预测长度
    POINTS_PER_DAY = 96 # 每天的数据点数 (15min分辨率)
    FREQ = '15min'

    # iTransformer 模型参数
    ENC_IN = 0          # 输入特征数 (自动计算)
    D_MODEL = 512       # Embedding 维度
    N_HEADS = 8         # 多头注意力头数
    E_LAYERS = 2        # Encoder 层数
    D_FF = 2048         # FFN 维度
    DROPOUT = 0.1       # Dropout 比率
    ACTIVATION = 'gelu' # 激活函数
    USE_NORM = True     # 是否使用 Instance Normalization

    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 50         
    LEARNING_RATE = 0.0005 
    PATIENCE = 10       

    # --- 数据集划分设置 (按天) ---
    FIXED_TRAIN_DAYS = None 
    FIXED_VAL_DAYS = None    
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1

cfg = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 绘图设置
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. iTransformer 模型组件 =================

class DataEmbedding_inverted(nn.Module):
    """
    iTransformer 的核心 Embedding 层：
    将 时间维度 投影到 Embedding 维度 (D_Model)。
    修改：输入维度变为 seq_len + pred_len，以包含未来信息
    """
    def __init__(self, input_len, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(input_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        # x: [Batch, Variates, Input_Len]
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

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

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
        if self.norm is not None:
            x = self.norm(x)
        return x, attns

class Model(nn.Module):
    """
    iTransformer 主模型类 (支持未来特征融合)
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.SEQ_LEN
        self.pred_len = configs.PRED_LEN
        self.output_attention = False 
        self.use_norm = configs.USE_NORM
        
        # 核心修改：Embedding 接受的历史长度为 Seq + Pred
        # 我们将把 [历史数据, 未来NWP(Target补0)] 拼在一起输入
        full_len = self.seq_len + self.pred_len
        self.enc_embedding = DataEmbedding_inverted(full_len, configs.D_MODEL, configs.DROPOUT)
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.DROPOUT, output_attention=False),
                        configs.D_MODEL, configs.N_HEADS),
                    configs.D_MODEL,
                    configs.D_FF,
                    dropout=configs.DROPOUT,
                    activation=configs.ACTIVATION
                ) for l in range(configs.E_LAYERS)
            ],
            norm_layer=nn.LayerNorm(configs.D_MODEL)
        )

        self.projector = nn.Linear(configs.D_MODEL, configs.PRED_LEN, bias=True)

    def forward(self, x_hist, x_fut_nwp):
        """
        x_hist: [Batch, Seq_Len, N_vars] (历史所有变量，包含Load)
        x_fut_nwp: [Batch, Pred_Len, N_vars - 1] (未来NWP变量，不包含Load)
        """
        
        # 1. Normalization (基于历史数据计算统计量)
        if self.use_norm:
            # [B, Seq, N] -> mean: [B, 1, N]
            means = x_hist.mean(1, keepdim=True).detach()
            stdev = torch.sqrt(torch.var(x_hist, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            
            # 归一化历史数据
            x_hist = (x_hist - means) / stdev
            
            # 归一化未来NWP数据 (使用历史的统计量，防止泄露)
            # x_fut_nwp 只有 N-1 列，对应 x_hist 的前 N-1 列 (假设Target在最后)
            means_nwp = means[:, :, :-1]
            stdev_nwp = stdev[:, :, :-1]
            x_fut_nwp = (x_fut_nwp - means_nwp) / stdev_nwp

        # 2. 构造全长序列输入
        # 目标：构建 [B, Seq + Pred, N]
        
        # 2.1 构造未来的 Load 部分 (用 0 填充，因为我们不知道未来 Load)
        batch_size = x_hist.shape[0]
        zeros_target = torch.zeros(batch_size, self.pred_len, 1, device=x_hist.device)
        
        # 2.2 拼接未来部分 [B, Pred, N]
        x_fut_combined = torch.cat([x_fut_nwp, zeros_target], dim=2)
        
        # 2.3 拼接历史和未来 [B, Seq + Pred, N]
        x_full = torch.cat([x_hist, x_fut_combined], dim=1)

        # 3. iTransformer 处理
        # 转置: [B, Seq+Pred, N] -> [B, N, Seq+Pred]
        x_full = x_full.permute(0, 2, 1)

        # Embedding: [B, N, Seq+Pred] -> [B, N, D_Model]
        # 这里模型会同时看到：历史Load, 历史NWP, 未来NWP(真实值), 未来Load(0值)
        # Attention 机制会让 Load 变量关注到 NWP 变量的未来信息
        enc_out = self.enc_embedding(x_full, None)

        # Encoder: Attention across Variates [B, N, D_Model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Projection: [B, N, Pred_Len]
        dec_out = self.projector(enc_out)

        # 转置回: [B, Pred_Len, N]
        dec_out = dec_out.permute(0, 2, 1)

        # 4. De-Normalization
        if self.use_norm:
            # 使用之前计算的 Load 的统计量 (最后一列)
            target_mean = means[:, 0, -1].unsqueeze(1).repeat(1, self.pred_len) # [B, Pred]
            target_stdev = stdev[:, 0, -1].unsqueeze(1).repeat(1, self.pred_len) # [B, Pred]
            
            # dec_out 目前是所有变量的预测，我们只取最后一列 Target
            # 或者对所有列反归一化。这里为了保持维度统一，先取 Target
            # 但 Model 输出是 [B, Pred, N]，我们先只反归一化输出
            
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out

# ================= 2. 数据处理 =================

def load_and_process_data():
    print("🚀 Loading data...")
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
    
    # Merge
    df = pd.merge(df_load, df_nwp, on='time', how='inner').sort_values('time').set_index('time')
    
    # 统一 Load 列名并确保在最后
    load_cols = [c for c in df.columns if 'load' in c.lower()]
    if not load_cols: 
        target_col = df.columns[-1]
    else:
        target_col = load_cols[0]
        
    # 特征工程 (可选)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    # 重新排列: [特征..., Load]
    feature_cols = [c for c in df.columns if c != target_col]
    df = df[feature_cols + [target_col]]
    
    # 填充缺失值
    df = df.ffill().bfill()
    day_steps = 96
    df = df.iloc[day_steps * 7:]
    
    print(f"Features: {feature_cols}")
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
        
        # 1. 历史序列 [Seq, N]
        seq_x = self.data[s_begin:s_end]
        
        # 2. 未来天气序列 [Pred, N-1] (假设最后一列是Target)
        # 注意：这里我们取未来的 NWP 特征
        seq_x_fut_nwp = self.data[r_begin:r_end, :-1]
        
        # 3. 未来标签 [Pred, 1] (只取最后一列)
        seq_y = self.data[r_begin:r_end, -1:]
        
        return seq_x, seq_x_fut_nwp, seq_y

def create_dataloaders(df):
    total_rows = len(df)
    total_days = total_rows // cfg.POINTS_PER_DAY
    print(f"Total Data: {total_rows} points ({total_days:.2f} days)")

    # 1. 计算按天切分索引
    if cfg.FIXED_TRAIN_DAYS is not None and cfg.FIXED_VAL_DAYS is not None:
        n_train_days = cfg.FIXED_TRAIN_DAYS
        n_val_days = cfg.FIXED_VAL_DAYS
    else:
        n_train_days = int(total_days * cfg.TRAIN_RATIO)
        n_val_days = int(total_days * cfg.VAL_RATIO)
    
    n_test_days = total_days - n_train_days - n_val_days
    if n_test_days <= 0: n_test_days = 0 
    
    print(f"Split Plan (Days): Train={n_train_days}, Val={n_val_days}, Test={n_test_days}")

    train_end_idx = n_train_days * cfg.POINTS_PER_DAY
    val_end_idx = (n_train_days + n_val_days) * cfg.POINTS_PER_DAY
    
    df_train = df.iloc[:train_end_idx]
    df_val = df.iloc[train_end_idx:val_end_idx]
    df_test = df.iloc[val_end_idx:] 
    
    # 2. 归一化 (仅在 Train Fit)
    scaler = StandardScaler()
    train_vals = scaler.fit_transform(df_train.values)
    
    if len(df_val) > 0: val_vals = scaler.transform(df_val.values)
    else: val_vals = np.empty((0, train_vals.shape[1]))

    if len(df_test) > 0: test_vals = scaler.transform(df_test.values)
    else: test_vals = np.empty((0, train_vals.shape[1]))
    
    # 记录 y 的 scaler 用于反归一化 (最后一列)
    scaler_y = StandardScaler()
    scaler_y.mean_ = scaler.mean_[-1]
    scaler_y.scale_ = scaler.scale_[-1]
    scaler_y.var_ = scaler.var_[-1]
    
    # 设置 Config 中的通道数
    cfg.ENC_IN = train_vals.shape[1]
    
    # 3. 数据集构造 (含 Lookback 处理)
    def prepare_data(curr, prev_tail=None):
        if prev_tail is not None:
            combined = np.vstack([prev_tail, curr])
        else:
            combined = curr
        return combined

    train_data = prepare_data(train_vals, None)
    
    train_tail = train_vals[-cfg.SEQ_LEN:] if len(train_vals) > 0 else None
    val_data = prepare_data(val_vals, train_tail) if len(val_vals) > 0 else np.empty((0, cfg.ENC_IN))
    
    val_tail = val_vals[-cfg.SEQ_LEN:] if len(val_vals) > 0 else None
    test_data = prepare_data(test_vals, val_tail) if len(test_vals) > 0 else np.empty((0, cfg.ENC_IN))
    
    train_set = TimeSeriesDataset(train_data, cfg.SEQ_LEN, cfg.PRED_LEN)
    val_set = TimeSeriesDataset(val_data, cfg.SEQ_LEN, cfg.PRED_LEN) if len(val_data) > 0 else []
    test_set = TimeSeriesDataset(test_data, cfg.SEQ_LEN, cfg.PRED_LEN) if len(test_data) > 0 else []
    
    print(f"Samples: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    # 记录测试集开始时间
    test_start_time = df_test.index[0] if len(df_test) > 0 else None
    
    return train_loader, val_loader, test_loader, scaler_y, test_start_time

# ================= 3. 训练流程 =================

def train_and_evaluate():
    # 0. 准备
    if os.path.exists(cfg.OUTPUT_DIR):
        # shutil.rmtree(cfg.OUTPUT_DIR) 
        pass
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    dayplot_dir = os.path.join(cfg.OUTPUT_DIR, "dayplot")
    os.makedirs(dayplot_dir, exist_ok=True)
    
    # 1. 数据
    df = load_and_process_data()
    if df is None: return
    train_loader, val_loader, test_loader, scaler_y, test_start_time = create_dataloaders(df)
    
    # 2. 模型
    model = Model(cfg).to(DEVICE)
    print(f"🔥 iTransformer Model Created. Params: {sum(p.numel() for p in model.parameters())}")
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # 3. 训练
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
            
            # Forward 传入历史和未来NWP
            outputs = model(batch_x, batch_x_fut_nwp)
            
            # outputs: [B, Pred, N], batch_y: [B, Pred, 1]
            # 我们只需要比较 Load (最后一列)
            pred_load = outputs[:, :, -1:] 
            
            loss = criterion(pred_load, batch_y)
            
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        avg_train_loss = np.mean(batch_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
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
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{cfg.EPOCHS} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")
            
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stop_cnt = 0
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, "best_model.pth"))
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= cfg.PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break
                
    # 4. 绘图 Loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "loss_curve.png"))
    plt.close()
    
    # 5. 测试
    print("\n🧪 Testing...")
    if not os.path.exists(os.path.join(cfg.OUTPUT_DIR, "best_model.pth")):
        return

    model.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, "best_model.pth")))
    model.eval()
    
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_x_fut_nwp, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            batch_x_fut_nwp = batch_x_fut_nwp.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            outputs = model(batch_x, batch_x_fut_nwp)
            
            # 取最后一列 (Load)
            preds.append(outputs[:, :, -1].cpu().numpy())
            # batch_y 本身就是最后一列
            trues.append(batch_y[:, :, 0].cpu().numpy())
            
    if len(preds) == 0: return

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # 反归一化
    preds_inv = preds * scaler_y.scale_ + scaler_y.mean_
    trues_inv = trues * scaler_y.scale_ + scaler_y.mean_
    
    # Metrics
    mae = mean_absolute_error(trues_inv.flatten(), preds_inv.flatten())
    rmse = np.sqrt(mean_squared_error(trues_inv.flatten(), preds_inv.flatten()))
    r2 = r2_score(trues_inv.flatten(), preds_inv.flatten())
    
    print(f"📊 Global Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    with open(os.path.join(cfg.OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}\nR2: {r2}\n")
        
    # 6. 生成日图和结果
    stitch_pred = []
    stitch_true = []
    stitch_time = []
    
    num_samples = len(preds_inv)
    
    for i in range(0, num_samples, cfg.POINTS_PER_DAY):
        if i >= num_samples: break
        
        y_p = preds_inv[i]
        y_t = trues_inv[i]
        
        current_start = test_start_time + pd.Timedelta(minutes=15 * i)
        current_timeline = pd.date_range(start=current_start, periods=len(y_p), freq='15min')
        
        stitch_pred.extend(y_p)
        stitch_true.extend(y_t)
        stitch_time.extend(current_timeline)
        
        # Day Plot
        day_rmse = np.sqrt(mean_squared_error(y_t, y_p))
        day_r2 = r2_score(y_t, y_p)
        date_str = str(current_start.date())
        
        plt.figure(figsize=(10, 5))
        plt.plot(current_timeline, y_t, label='True', color='blue')
        plt.plot(current_timeline, y_p, label='iTransformer', color='red', linestyle='--')
        plt.title(f"Date: {date_str} | RMSE: {day_rmse:.2f} | R2: {day_r2:.2f}")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(dayplot_dir, f"{date_str}.png"))
        plt.close()
        
    # Full CSV
    res_df = pd.DataFrame({
        'time': stitch_time,
        'true': stitch_true,
        'pred': stitch_pred
    })
    res_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "prediction_result.csv"), index=False)
    
    # Full Plot
    plt.figure(figsize=(15, 6))
    plt.plot(pd.to_datetime(res_df['time']), res_df['true'], label='True', alpha=0.7)
    plt.plot(pd.to_datetime(res_df['time']), res_df['pred'], label='Pred', alpha=0.7, linestyle='--')
    plt.title("Full Test Set Prediction (iTransformer)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "full_prediction.png"))
    plt.close()
    
    print(f"✅ All Done. Results saved to {cfg.OUTPUT_DIR}")

if __name__ == "__main__":
    train_and_evaluate()
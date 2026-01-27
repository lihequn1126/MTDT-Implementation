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

import argparse

parser = argparse.ArgumentParser(description='Time Series Forecasting')


parser.add_argument('--nwp_path', type=str, default=r"../nwpData/hanyuan.csv", help='Path to NWP data')
parser.add_argument('--load_path', type=str, default=r"../LoadData/hanyuan.csv", help='Path to Load data')
parser.add_argument('--output_dir', type=str, default=r"../result/hanyuan_autoformer", help='Output directory')


args = parser.parse_args()


NWP_PATH = args.nwp_path
LOAD_PATH = args.load_path
OUTPUT_DIR = args.output_dir


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Running with:\n NWP: {NWP_PATH}\n Load: {LOAD_PATH}\n Output: {OUTPUT_DIR}")

SEQ_LEN = 96        
LABEL_LEN = 48     
PRED_LEN = 96       
POINTS_PER_DAY = 96 

BATCH_SIZE = 32     
EPOCHS = 100
LEARNING_RATE = 0.001 
PATIENCE = 15

d_model = 64        
d_ff = 128          
n_heads = 4      
e_layers = 2        
d_layers = 1        
dropout = 0.05    

FIXED_TRAIN_DAYS = None  
FIXED_VAL_DAYS = None    
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_process_data():
    print("Loading data...")
    if not os.path.exists(NWP_PATH) or not os.path.exists(LOAD_PATH):
        raise FileNotFoundError("数据文件路径不正确")
        
    df_nwp = pd.read_csv(NWP_PATH)
    df_load = pd.read_csv(LOAD_PATH)
    
    df_nwp['time'] = pd.to_datetime(df_nwp['time'])
    df_load['time'] = pd.to_datetime(df_load['time'])
    
    df = pd.merge(df_load, df_nwp, on='time', how='inner').sort_values('time').set_index('time')
    
    load_col = [c for c in df.columns if 'load' in c.lower()]
    if not load_col: raise ValueError("未找到 load 列")
    df = df.rename(columns={load_col[0]: 'y'})
    
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    df = df.dropna()
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
        x = data[i:(i + seq_len), :] # 输入包含 y
        y = data[(i + seq_len):(i + seq_len + pred_len), -1] 
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.kernel_size = kernel_size

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_avg = self.moving_avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        x_res = x - x_avg
        return x_res, x_avg

class AutoCorrelation(nn.Module):
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        head = values.shape[2]
        channel = values.shape[3]
        length = values.shape[1]
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        tmp_corr = torch.softmax(weights, dim=-1)
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), 1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, length, head, channel))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * k_fft.conj()
        corr = torch.fft.irfft(res, dim=-1)

        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)

class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None, d_values=None):
        super(AutoCorrelationLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_correlation = correlation
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
        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

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
        for attn_layer in self.attn_layers:
            x, _ = attn_layer(x, attn_mask=attn_mask)
        if self.norm is not None: x = self.norm(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='replicate', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        x = x + self.dropout(self.self_attention(x, x, x, x_mask)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, cross_mask)[0])
        x = self.norm2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x = self.norm3(x + y)
        residual = self.projection(x.transpose(1, 2)).transpose(1, 2)
        return x, residual

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, trend=trend)
            trend = trend + residual
        if self.norm is not None: x = self.norm(x)
        if self.projection is not None: trend = trend + self.projection(x)
        return trend

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = self.value_embedding(x)
        return self.dropout(x)

class Autoformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len,
                 factor=1, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
                 dropout=0.1, activation='gelu', output_attention=False):
        super(Autoformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        kernel_size = 25
        self.decomp = SeriesDecomp(kernel_size)

        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model, c_out, d_ff, dropout=dropout, activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

    def forward(self, x_enc):
        seasonal_init, trend_init = self.decomp(x_enc)
        trend_part = trend_init[:, -self.label_len:, :]
        mean_trend = trend_part.mean(dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        trend_init = torch.cat([trend_part, mean_trend], dim=1)
        seasonal_part = seasonal_init[:, -self.label_len:, :]
        zeros_seasonal = torch.zeros(seasonal_part.shape[0], self.pred_len, seasonal_part.shape[2]).to(seasonal_part.device)
        seasonal_init = torch.cat([seasonal_part, zeros_seasonal], dim=1)
        x_dec = seasonal_init
        enc_out = self.enc_embedding(x_enc)
        enc_out = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)

        return dec_out[:, -self.pred_len:, :]
def train_and_evaluate():
    if os.path.exists(OUTPUT_DIR):
        pass
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dayplot_dir = os.path.join(OUTPUT_DIR, "dayplot")
    os.makedirs(dayplot_dir, exist_ok=True)
    df = load_and_process_data()
    total_rows = len(df)
    total_days = total_rows // POINTS_PER_DAY
    
    if FIXED_TRAIN_DAYS is not None:
        n_train_days = FIXED_TRAIN_DAYS
        n_val_days = FIXED_VAL_DAYS
    else:
        n_train_days = int(total_days * TRAIN_RATIO)
        n_val_days = int(total_days * VAL_RATIO)
    
    train_end_idx = n_train_days * POINTS_PER_DAY
    val_end_idx = (n_train_days + n_val_days) * POINTS_PER_DAY
    
    df_train = df.iloc[:train_end_idx]
    df_val   = df.iloc[train_end_idx:val_end_idx]
    df_test  = df.iloc[val_end_idx:]
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(df_train.values)
    val_scaled = scaler.transform(df_val.values)
    test_scaled = scaler.transform(df_test.values)
    
    scaler_y = StandardScaler()
    scaler_y.fit(df_train['y'].values.reshape(-1, 1))
    def prepare_dataset(curr_scaled, prev_scaled_tail=None):
        if prev_scaled_tail is not None:
            data_combined = np.vstack([prev_scaled_tail, curr_scaled])
        else:
            data_combined = curr_scaled
        return create_sequences(data_combined, SEQ_LEN, PRED_LEN)

    X_train, y_train = prepare_dataset(train_scaled, None)
    train_tail = train_scaled[-SEQ_LEN:]
    X_val, y_val = prepare_dataset(val_scaled, train_tail)
    val_tail = val_scaled[-SEQ_LEN:]
    X_test, y_test = prepare_dataset(test_scaled, val_tail)
    
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    input_dim = X_train.shape[2] 
    model = Autoformer(
        enc_in=input_dim, 
        dec_in=input_dim, 
        c_out=input_dim, 
        seq_len=SEQ_LEN, 
        label_len=LABEL_LEN, 
        pred_len=PRED_LEN,
        d_model=d_model,  
        d_ff=d_ff,       
        n_heads=n_heads, 
        e_layers=e_layers,
        d_layers=d_layers,
        dropout=dropout
    ).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(bx)
            pred_load = outputs[:, :, -1]
            loss = criterion(pred_load, by)
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
                outputs = model(bx)
                pred_load = outputs[:, :, -1]
                val_batch_losses.append(criterion(pred_load, by).item())
        
        avg_val_loss = np.mean(val_batch_losses)
        val_losses.append(avg_val_loss)
        
        if (epoch+1) % 1 == 0:
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
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()
    
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    model.eval()
    
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(DEVICE)
            outputs = model(bx)
            pred_load = outputs[:, :, -1]
            preds.append(pred_load.cpu().numpy())
            trues.append(by.numpy())
            
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    
    preds_inv = scaler_y.inverse_transform(preds)
    trues_inv = scaler_y.inverse_transform(trues)
    
    mae = mean_absolute_error(trues_inv.flatten(), preds_inv.flatten())
    rmse = np.sqrt(mean_squared_error(trues_inv.flatten(), preds_inv.flatten()))
    r2 = r2_score(trues_inv.flatten(), preds_inv.flatten())
    
    print(f"\nFinal Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    
    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}\nR2: {r2}\n")
        
    stitch_pred, stitch_true, stitch_time = [], [], []
    test_start_time = df_test.index[0]
    
    for i in range(0, len(preds_inv), POINTS_PER_DAY):
        if i >= len(preds_inv): break
        y_p = preds_inv[i]
        y_t = trues_inv[i]
        curr_time = pd.date_range(start=test_start_time + pd.Timedelta(minutes=15*i), periods=PRED_LEN, freq='15min')
        
        stitch_pred.extend(y_p)
        stitch_true.extend(y_t)
        stitch_time.extend(curr_time)
        
        plt.figure(figsize=(10,4))
        plt.plot(curr_time, y_t, label='True')
        plt.plot(curr_time, y_p, label='Autoformer', linestyle='--')
        plt.title(f"{str(curr_time[0].date())}")
        plt.legend()
        plt.savefig(os.path.join(dayplot_dir, f"{str(curr_time[0].date())}.png"))
        plt.close()
        
    res_df = pd.DataFrame({'time': stitch_time, 'true': stitch_true, 'pred': stitch_pred})
    res_df.to_csv(os.path.join(OUTPUT_DIR, "prediction_result.csv"), index=False)
if __name__ == "__main__":
    train_and_evaluate()
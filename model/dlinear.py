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

parser = argparse.ArgumentParser(description='Time Series Forecasting')


parser.add_argument('--nwp_path', type=str, default=r"../nwpData/hanyuan.csv", help='Path to NWP data')
parser.add_argument('--load_path', type=str, default=r"../LoadData/hanyuan.csv", help='Path to Load data')
parser.add_argument('--output_dir', type=str, default=r"../result/hanyuan_dlinear", help='Output directory')


args = parser.parse_args()


NWP_PATH = args.nwp_path
LOAD_PATH = args.load_path
OUTPUT_DIR = args.output_dir


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

SEQ_LEN = 96        
PRED_LEN = 96      
POINTS_PER_DAY = 96 

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0005 
PATIENCE = 10

FIXED_TRAIN_DAYS = None  
FIXED_VAL_DAYS = None    
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_process_data():
    if not os.path.exists(NWP_PATH) or not os.path.exists(LOAD_PATH):
        raise FileNotFoundError("路径错误")
        
    df_nwp = pd.read_csv(NWP_PATH)
    df_load = pd.read_csv(LOAD_PATH)
    
    df_nwp['time'] = pd.to_datetime(df_nwp['time'])
    df_load['time'] = pd.to_datetime(df_load['time'])
    
    df = pd.merge(df_load, df_nwp, on='time', how='inner').sort_values('time').set_index('time')
    
    load_col = [c for c in df.columns if 'load' in c.lower()]
    if not load_col: raise ValueError("无 Load 列")
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
        x = data[i:(i + seq_len), :] 
        y = data[(i + seq_len):(i + seq_len + pred_len), -1] 
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class MovingAverage(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
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
        self.decomposition = SeriesDecomp(kernel_size=25)
        self.Linear_Seasonal = nn.Linear(seq_len * enc_in, pred_len)
        self.Linear_Trend = nn.Linear(seq_len * enc_in, pred_len)
        self.Linear_Seasonal.weight.data.normal_(0, 0.01)
        self.Linear_Trend.weight.data.normal_(0, 0.01)

    def forward(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        batch_size = x.shape[0]
        seasonal_init = seasonal_init.reshape(batch_size, -1)
        trend_init = trend_init.reshape(batch_size, -1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x

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
    
    print(f"Train Shape: {X_train.shape}")
    
    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    input_dim = X_train.shape[2] 
    model = DLinearModel(seq_len=SEQ_LEN, pred_len=PRED_LEN, enc_in=input_dim).to(DEVICE)
    
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
            pred = model(bx)
            preds.append(pred.cpu().numpy())
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
        plt.plot(curr_time, y_p, label='DLinear', linestyle='--')
        plt.title(f"{str(curr_time[0].date())}")
        plt.legend()
        plt.savefig(os.path.join(dayplot_dir, f"{str(curr_time[0].date())}.png"))
        plt.close()
        
    res_df = pd.DataFrame({'time': stitch_time, 'true': stitch_true, 'pred': stitch_pred})
    res_df.to_csv(os.path.join(OUTPUT_DIR, "prediction_result.csv"), index=False)
if __name__ == "__main__":
    train_and_evaluate()
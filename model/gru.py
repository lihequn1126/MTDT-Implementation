import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import shutil

import argparse

parser = argparse.ArgumentParser(description='Time Series Forecasting')


parser.add_argument('--nwp_path', type=str, default=r"../nwpData/hanyuan.csv", help='Path to NWP data')
parser.add_argument('--load_path', type=str, default=r"../LoadData/hanyuan.csv", help='Path to Load data')
parser.add_argument('--output_dir', type=str, default=r"../result/hanyuan_gru", help='Output directory')


args = parser.parse_args()


NWP_PATH = args.nwp_path
LOAD_PATH = args.load_path
OUTPUT_DIR = args.output_dir


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
SEQ_LEN = 96       
PRED_LEN = 96      
POINTS_PER_DAY = 96 

BATCH_SIZE = 64
EPOCHS = 100
HIDDEN_DIM = 64   
NUM_LAYERS = 2  
LEARNING_RATE = 0.001
PATIENCE = 15
DROPOUT = 0.2

FIXED_TRAIN_DAYS = None 
FIXED_VAL_DAYS = None   
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.3):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
    
    def forward(self, x_enc):
        output, hidden = self.gru(x_enc)
        return output, hidden

class GRUDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, decoder_input_dim, output_dim, dropout=0.3):
        super(GRUDecoder, self).__init__()
        self.gru = nn.GRU(
            input_size=decoder_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_dec, hidden_state):
        decoder_output, _ = self.gru(x_dec, hidden_state)
        pred = self.fc(decoder_output)
        return pred.squeeze(-1)

class Seq2SeqGRU(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqGRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x_enc, x_dec):
        _, encoder_hidden = self.encoder(x_enc)
        predictions = self.decoder(x_dec, encoder_hidden)
        
        return predictions

def load_and_process_data():
    if not os.path.exists(NWP_PATH) or not os.path.exists(LOAD_PATH):
        raise FileNotFoundError("数据文件路径不正确")
        
    df_nwp = pd.read_csv(NWP_PATH)
    df_load = pd.read_csv(LOAD_PATH)
    
    df_nwp['time'] = pd.to_datetime(df_nwp['time'])
    df_load['time'] = pd.to_datetime(df_load['time'])
    
    df = pd.merge(df_load, df_nwp, on='time', how='inner').sort_values('time').set_index('time')
    
    load_col = [c for c in df.columns if 'load' in c.lower()]
    if not load_col: raise ValueError("未找到包含 'load' 的列名")
    df = df.rename(columns={load_col[0]: 'y'})
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    df = df.dropna()
    day_steps = 96
    df = df.iloc[day_steps * 7:]
    cols = [c for c in df.columns if c != 'y'] + ['y']
    df = df[cols]
    
    return df

class TimeSeriesDataset(Dataset):
    def __init__(self, X_enc, X_dec, y):
        self.X_enc = torch.tensor(X_enc, dtype=torch.float32)
        self.X_dec = torch.tensor(X_dec, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X_enc)
    def __getitem__(self, idx): return self.X_enc[idx], self.X_dec[idx], self.y[idx]

def create_sequences(data, seq_len, pred_len):
    xs_enc, xs_dec, ys = [], [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        x_enc = data[i : (i + seq_len), :] 
        x_dec = data[(i + seq_len) : (i + seq_len + pred_len), :-1]
        y = data[(i + seq_len) : (i + seq_len + pred_len), -1]
        
        xs_enc.append(x_enc)
        xs_dec.append(x_dec)
        ys.append(y)
        
    return np.array(xs_enc), np.array(xs_dec), np.array(ys)

def train_and_evaluate():
    if os.path.exists(OUTPUT_DIR):
        pass
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dayplot_dir = os.path.join(OUTPUT_DIR, "dayplot")
    os.makedirs(dayplot_dir, exist_ok=True)
    df = load_and_process_data()
    total_rows = len(df)
    total_days = total_rows // POINTS_PER_DAY
    print(f"Total Data: {total_rows} points ({total_days:.2f} days)")
    
    if FIXED_TRAIN_DAYS is not None and FIXED_VAL_DAYS is not None:
        n_train_days = FIXED_TRAIN_DAYS
        n_val_days = FIXED_VAL_DAYS
    else:
        n_train_days = int(total_days * TRAIN_RATIO)
        n_val_days = int(total_days * VAL_RATIO)
    n_test_days = total_days - n_train_days - n_val_days
    
    print(f"Split: Train={n_train_days}, Val={n_val_days}, Test={n_test_days}")
    
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

    X_enc_train, X_dec_train, y_train = prepare_dataset(train_scaled, None)
    
    train_tail = train_scaled[-SEQ_LEN:]
    X_enc_val, X_dec_val, y_val = prepare_dataset(val_scaled, train_tail)
    
    val_tail = val_scaled[-SEQ_LEN:]
    X_enc_test, X_dec_test, y_test = prepare_dataset(test_scaled, val_tail)
    
    print(f"Train Shape: Enc={X_enc_train.shape}, Dec={X_dec_train.shape}")
    
    train_loader = DataLoader(TimeSeriesDataset(X_enc_train, X_dec_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TimeSeriesDataset(X_enc_val, X_dec_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TimeSeriesDataset(X_enc_test, X_dec_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    input_dim = X_enc_train.shape[2] 
    decoder_input_dim = X_dec_train.shape[2]
    
    encoder = GRUEncoder(input_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = GRUDecoder(HIDDEN_DIM, NUM_LAYERS, decoder_input_dim, 1, DROPOUT)
    model = Seq2SeqGRU(encoder, decoder).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        for enc_in, dec_in, target in train_loader:
            enc_in, dec_in, target = enc_in.to(DEVICE), dec_in.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            pred = model(enc_in, dec_in)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        epoch_train_loss = np.mean(batch_losses)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for enc_in, dec_in, target in val_loader:
                enc_in, dec_in, target = enc_in.to(DEVICE), dec_in.to(DEVICE), target.to(DEVICE)
                pred = model(enc_in, dec_in)
                val_batch_losses.append(criterion(pred, target).item())
        
        epoch_val_loss = np.mean(val_batch_losses)
        val_losses.append(epoch_val_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f}")
            
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve (Seq2Seq GRU)')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()
    
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    model.eval()
    
    test_preds_list, test_trues_list = [], []
    
    with torch.no_grad():
        for enc_in, dec_in, target in test_loader:
            enc_in, dec_in = enc_in.to(DEVICE), dec_in.to(DEVICE)
            pred = model(enc_in, dec_in)
            
            test_preds_list.append(pred.cpu().numpy())
            test_trues_list.append(target.numpy())
            
    test_preds = np.concatenate(test_preds_list)
    test_trues = np.concatenate(test_trues_list)
    test_preds_inv = scaler_y.inverse_transform(test_preds)
    test_trues_inv = scaler_y.inverse_transform(test_trues)
    
    mae = mean_absolute_error(test_trues_inv.flatten(), test_preds_inv.flatten())
    rmse = np.sqrt(mean_squared_error(test_trues_inv.flatten(), test_preds_inv.flatten()))
    r2 = r2_score(test_trues_inv.flatten(), test_preds_inv.flatten())
    
    print(f"\nGlobal Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    
    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}\nR2: {r2}\n")
    
    stitch_pred, stitch_true, stitch_time = [], [], []
    test_start_time = df_test.index[0]
    
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
        plt.plot(current_timeline, y_p, label='Pred', color='green', linestyle='--')
        plt.title(f"Date: {date_str} | RMSE: {day_rmse:.2f} | R2: {day_r2:.2f}")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(dayplot_dir, f"{date_str}.png"))
        plt.close()
        
    res_df = pd.DataFrame({'time': stitch_time, 'true': stitch_true, 'pred': stitch_pred})
    res_df.to_csv(os.path.join(OUTPUT_DIR, "prediction_result.csv"), index=False)
    
    plt.figure(figsize=(15, 6))
    plt.plot(res_df['time'], res_df['true'], label='True', alpha=0.7)
    plt.plot(res_df['time'], res_df['pred'], label='Pred', alpha=0.7, linestyle='--')
    plt.title("Full Test Set Prediction (Seq2Seq GRU)")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "full_prediction.png"))
    plt.close()
if __name__ == "__main__":
    train_and_evaluate()
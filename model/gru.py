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

from data_provider import get_data  # 统一无泄漏数据管道

parser = argparse.ArgumentParser(description='Time Series Forecasting')


parser.add_argument('--nwp_path', type=str, default=r"../nwpData/baoxing.csv", help='Path to NWP data')
parser.add_argument('--load_path', type=str, default=r"../LoadData/baoxing.csv", help='Path to Load data')
parser.add_argument('--output_dir', type=str, default=r"../result/baoxing_gru", help='Output directory')


args = parser.parse_args()


NWP_PATH = args.nwp_path
LOAD_PATH = args.load_path
OUTPUT_DIR = args.output_dir


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Running with:\n NWP: {NWP_PATH}\n Load: {LOAD_PATH}\n Output: {OUTPUT_DIR}")
SEQ_LEN = 96       # Encoder 输入长度 (过去24小时)
PRED_LEN = 96      # Decoder 预测长度 (未来24小时)
POINTS_PER_DAY = 96 

BATCH_SIZE = 64
EPOCHS = 100
HIDDEN_DIM = 64    # 隐藏层维度
NUM_LAYERS = 2     # GRU层数
LEARNING_RATE = 0.001
PATIENCE = 15
DROPOUT = 0.2

# 数据集划分
FIXED_TRAIN_DAYS = None 
FIXED_VAL_DAYS = None   
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. 模型定义 (标准 Seq2Seq GRU)
# ==============================================================================

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
        # x_enc: [batch, seq_len, input_dim]
        # output: [batch, seq_len, hidden_dim]
        # hidden: [num_layers, batch, hidden_dim]
        output, hidden = self.gru(x_enc)
        return output, hidden

class GRUDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, decoder_input_dim, output_dim, dropout=0.3):
        super(GRUDecoder, self).__init__()
        
        # 标准 GRU Decoder，不拼接 Context Vector
        self.gru = nn.GRU(
            input_size=decoder_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_dec, hidden_state):
        # x_dec: [batch, pred_len, decoder_input_dim] (未来已知的特征)
        # hidden_state: [num_layers, batch, hidden_dim] (来自 Encoder 的最后状态)
        
        # 因为没有 Attention 依赖每一步的计算，我们可以直接将整个序列输入 GRU
        # 这比手动循环更快
        decoder_output, _ = self.gru(x_dec, hidden_state)
        
        # decoder_output: [batch, pred_len, hidden_dim]
        pred = self.fc(decoder_output)
        
        # pred: [batch, pred_len, 1] -> squeeze -> [batch, pred_len]
        return pred.squeeze(-1)

class Seq2SeqGRU(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqGRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x_enc, x_dec):
        # 1. Encoder 编码历史信息
        _, encoder_hidden = self.encoder(x_enc)
        
        # 2. 将 Encoder 的最终隐藏状态作为 Decoder 的初始隐藏状态
        # Decoder 根据未来特征 (x_dec) 和历史记忆 (encoder_hidden) 进行预测
        predictions = self.decoder(x_dec, encoder_hidden)
        
        return predictions

# ==============================================================================
# 2. 数据处理: 统一使用 data_provider.get_data (future 模式)
# ==============================================================================

# ==============================================================================
# 3. 训练与评估流程 (移除 Attention 相关部分)
# ==============================================================================

def train_and_evaluate():
    if os.path.exists(OUTPUT_DIR):
        # shutil.rmtree(OUTPUT_DIR) 
        pass
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dayplot_dir = os.path.join(OUTPUT_DIR, "dayplot")
    os.makedirs(dayplot_dir, exist_ok=True)
    
    # 1. 统一无泄漏数据管道 (future 模式: decoder 只吃未来已知协变量 NWP+日历)
    loaders, scaler_y, test_start_time, info, _ = get_data(
        NWP_PATH, LOAD_PATH, SEQ_LEN, PRED_LEN, BATCH_SIZE,
        TRAIN_RATIO, VAL_RATIO, POINTS_PER_DAY, mode="future")
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    
    # 2. 模型初始化
    input_dim = info['n_features']          # encoder 输入 = 全部特征列
    decoder_input_dim = info['n_known']     # decoder 输入 = 未来已知协变量 (NWP + 日历)
    
    encoder = GRUEncoder(input_dim, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
    decoder = GRUDecoder(HIDDEN_DIM, NUM_LAYERS, decoder_input_dim, 1, DROPOUT)
    model = Seq2SeqGRU(encoder, decoder).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. 训练循环
    print("Training started...")
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        for enc_in, dec_in, target in train_loader:
            enc_in, dec_in, target = enc_in.to(DEVICE), dec_in.to(DEVICE), target.to(DEVICE)
            target = target.squeeze(-1)  # [B,H,1] -> [B,H]
            optimizer.zero_grad()
            
            # Forward
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
                target = target.squeeze(-1)  # [B,H,1] -> [B,H]
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
    
    # Loss 图
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve (Seq2Seq GRU)')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()
    
    # 4. 测试与绘图
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    model.eval()
    
    test_preds_list, test_trues_list = [], []
    
    with torch.no_grad():
        for enc_in, dec_in, target in test_loader:
            enc_in, dec_in = enc_in.to(DEVICE), dec_in.to(DEVICE)
            pred = model(enc_in, dec_in)
            
            test_preds_list.append(pred.cpu().numpy())
            test_trues_list.append(target.squeeze(-1).numpy())
            
    test_preds = np.concatenate(test_preds_list)
    test_trues = np.concatenate(test_trues_list)
    
    # 反归一化 (scaler_y 提供标量 mean_/scale_)
    test_preds_inv = test_preds * scaler_y.scale_ + scaler_y.mean_
    test_trues_inv = test_trues * scaler_y.scale_ + scaler_y.mean_
    
    mae = mean_absolute_error(test_trues_inv.flatten(), test_preds_inv.flatten())
    rmse = np.sqrt(mean_squared_error(test_trues_inv.flatten(), test_preds_inv.flatten()))
    r2 = r2_score(test_trues_inv.flatten(), test_preds_inv.flatten())
    
    print(f"\nGlobal Metrics: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    
    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"MAE: {mae}\nRMSE: {rmse}\nR2: {r2}\n")
    
    stitch_pred, stitch_true, stitch_time = [], [], []
    # test_start_time 由统一数据管道提供
    
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
    
    print("Done.")

if __name__ == "__main__":
    train_and_evaluate()

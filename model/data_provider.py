"""
Unified, leakage-free data pipeline shared by the main model and all baselines.

Design goals
------------
1. **No leakage**: only genuinely-known-in-the-future covariates (NWP + calendar)
   are exposed for the prediction window. Lag / rolling-mean features (which are
   derived from the target, and whose future values would leak ground truth) are
   kept **history-only** and are masked to zero in the future block.
2. **Explicit column roles**: every column is tagged as one of
   {nwp, calendar, lag, target}. Column order is canonical and fixed:

        [ NWP ... , calendar ... , lag ... , TARGET ]
        |<---- known future ---->|<-- unknown future -->|
        |<------------------ n_known ----------------->|

   The first ``n_known`` columns (NWP + calendar) are known in the future.
   Everything after them (lag features + target) is unknown in the future and
   must be zero-padded (in normalized space) by the model.
3. **Two access modes** so both model families can share the exact same
   features / split / scaling:
     - mode="future" : yields (seq_x[L,C], seq_x_fut_known[H,n_known], seq_y[H,1])
                       for models that consume future covariates (iTransformer-style).
     - mode="history": yields (seq_x[L,C], seq_y[H,1]) for history-only models
                       (DLinear / LSTM / Transformer / Autoformer ...).

The key anti-leakage fix vs. the previous code: the old datasets returned
``data[r_begin:r_end, :-1]`` as the "future NWP", which silently included the
future values of ``load_lag_*`` and especially ``load_mean_1d`` (a trailing
rolling mean of the target). For horizon steps >= 1 that rolling mean contains
future ground-truth load, i.e. target leakage. Here the future block only ever
contains NWP + calendar.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# 1. DataFrame construction with explicit column roles
# --------------------------------------------------------------------------- #
def build_dataframe(nwp_path, load_path, add_lag=True, use_nwp=True, use_calendar=True):
    """Merge load + NWP, engineer features, and return (df, roles).

    Parameters
    ----------
    nwp_path, load_path : str
        CSV paths. NWP csv already contains the (region-specific, PCC-screened)
        meteorological columns; load csv contains ``time, Load``.
    add_lag : bool
        If False, lag/rolling features are omitted entirely (used e.g. for the
        "w/o lag" ablation). They are always history-only regardless.
    use_nwp : bool
        If False, all meteorological (NWP) columns are dropped (used for the
        "w/o NWP" ablation). ``n_known`` then contains calendar only.
    use_calendar : bool
        If False, calendar (min/day sin-cos) features are omitted (used for the
        "w/o calendar" ablation).

    Returns
    -------
    df : pd.DataFrame  (index = time, canonical column order, target last)
    roles : dict with keys
        nwp_cols, cal_cols, lag_cols, target_col,
        known_cols (=nwp+cal), n_known, n_features (=C), feature_cols (all but target)
    """
    df_nwp = pd.read_csv(nwp_path)
    df_load = pd.read_csv(load_path)
    df_nwp['time'] = pd.to_datetime(df_nwp['time'])
    df_load['time'] = pd.to_datetime(df_load['time'])

    df = (pd.merge(df_load, df_nwp, on='time', how='inner')
            .sort_values('time')
            .set_index('time'))

    # target
    load_cols = [c for c in df.columns if 'load' in c.lower()]
    target_col = load_cols[0] if load_cols else df.columns[-1]

    # NWP columns = every merged column that is neither the target nor time
    nwp_cols = [c for c in df.columns if c != target_col]
    if not use_nwp:
        # drop meteorological inputs entirely ("w/o NWP" ablation)
        df = df.drop(columns=nwp_cols)
        nwp_cols = []

    # calendar features (deterministic -> KNOWN in the future)
    cal_cols = []
    if use_calendar:
        minutes = df.index.hour * 60 + df.index.minute
        df['min_sin'] = np.sin(2 * np.pi * minutes / 1440)
        df['min_cos'] = np.cos(2 * np.pi * minutes / 1440)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        cal_cols = ['min_sin', 'min_cos', 'day_sin', 'day_cos']

    # lag / rolling features (derived from target -> HISTORY ONLY / unknown future)
    lag_cols = []
    if add_lag:
        day_steps = 96
        df['load_lag_1d'] = df[target_col].shift(day_steps)
        df['load_lag_7d'] = df[target_col].shift(day_steps * 7)
        # trailing rolling mean of yesterday; shift(1) keeps it strictly causal
        # w.r.t. the *reference* time, but its FUTURE values still embed future
        # target -> therefore it is treated as unknown-future and never exposed.
        df['load_mean_1d'] = (df[target_col]
                              .rolling(window=day_steps, min_periods=1)
                              .mean()
                              .shift(1))
        lag_cols = ['load_lag_1d', 'load_lag_7d', 'load_mean_1d']

    df = df.dropna()

    # canonical order: [nwp, calendar, lag, target]
    known_cols = nwp_cols + cal_cols          # known in the future
    feature_cols = known_cols + lag_cols      # all non-target features
    df = df[feature_cols + [target_col]]

    roles = {
        'nwp_cols': nwp_cols,
        'cal_cols': cal_cols,
        'lag_cols': lag_cols,
        'target_col': target_col,
        'known_cols': known_cols,
        'feature_cols': feature_cols,
        'n_known': len(known_cols),
        'n_features': len(feature_cols) + 1,   # C = features + target
    }
    return df, roles


# --------------------------------------------------------------------------- #
# 2. Datasets
# --------------------------------------------------------------------------- #
class FutureCovDataset(Dataset):
    """Yields (seq_x[L,C], seq_x_fut_known[H,n_known], seq_y[H,1]).

    Only the ``n_known`` known-future columns are provided for the prediction
    window. Models are responsible for zero-padding the remaining
    ``C - n_known`` columns (lag + target) in normalized space.
    """

    def __init__(self, data, seq_len, pred_len, n_known):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_known = n_known

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_end = s_end + self.pred_len
        seq_x = self.data[index:s_end]                       # [L, C]
        seq_x_fut_known = self.data[s_end:r_end, :self.n_known]  # [H, n_known]  (NWP+calendar only)
        seq_y = self.data[s_end:r_end, -1:]                  # [H, 1]
        return seq_x, seq_x_fut_known, seq_y


class HistoryOnlyDataset(Dataset):
    """Yields (seq_x[L,C], seq_y[H]) for models that only consume history.

    Note: ``seq_y`` is 1-D (length H) to match the classical baselines
    (DLinear / LSTM / Transformer / Autoformer) whose heads output [B, H].
    ``seq_x`` contains ALL C columns (incl. load history + NWP + calendar +
    lag), so these models now receive the same feature set as the main model.
    """

    def __init__(self, data, seq_len, pred_len):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_end = index + self.seq_len
        r_end = s_end + self.pred_len
        seq_x = self.data[index:s_end]         # [L, C]
        seq_y = self.data[s_end:r_end, -1]     # [H]
        return seq_x, seq_y


# --------------------------------------------------------------------------- #
# 3. Dataloader factory (shared split + scaling)
# --------------------------------------------------------------------------- #
def make_dataloaders(df, roles, seq_len, pred_len,
                     batch_size=32, train_ratio=0.7, val_ratio=0.1,
                     points_per_day=96, mode="future"):
    """Chronological day-aligned 70/10/20 split, StandardScaler fit on train only.

    Returns
    -------
    loaders : dict(train, val, test)
    scaler_y : StandardScaler-like with .scale_/.mean_ for the target column
    test_start_time : pd.Timestamp | None
    info : dict(n_features, n_known, feature_names)
    """
    assert mode in ("future", "history")

    total_days = len(df) // points_per_day
    n_train_days = int(total_days * train_ratio)
    n_val_days = int(total_days * val_ratio)
    train_end = n_train_days * points_per_day
    val_end = (n_train_days + n_val_days) * points_per_day

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    scaler = StandardScaler()
    train_vals = scaler.fit_transform(df_train.values)
    val_vals = (scaler.transform(df_val.values)
                if len(df_val) > 0 else np.empty((0, train_vals.shape[1])))
    test_vals = (scaler.transform(df_test.values)
                 if len(df_test) > 0 else np.empty((0, train_vals.shape[1])))

    # target de-normalization stats (last column)
    scaler_y = StandardScaler()
    scaler_y.mean_ = scaler.mean_[-1]
    scaler_y.scale_ = scaler.scale_[-1]
    scaler_y.var_ = scaler.var_[-1]

    # prepend previous split's tail so val/test can form full windows
    def _prepend(curr, prev_tail):
        if prev_tail is not None and len(curr) > 0:
            return np.vstack([prev_tail, curr])
        return curr

    train_data = train_vals
    val_data = (_prepend(val_vals, train_vals[-seq_len:])
                if len(val_vals) > 0 else val_vals)
    test_data = (_prepend(test_vals, val_vals[-seq_len:])
                 if len(test_vals) > 0 and len(val_vals) > 0 else test_vals)

    n_known = roles['n_known']

    def _make_set(data):
        if len(data) < seq_len + pred_len:
            return None
        if mode == "future":
            return FutureCovDataset(data, seq_len, pred_len, n_known)
        return HistoryOnlyDataset(data, seq_len, pred_len)

    train_set = _make_set(train_data)
    val_set = _make_set(val_data)
    test_set = _make_set(test_data)

    loaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            drop_last=True) if train_set else None,
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False)
                if val_set else None,
        'test': DataLoader(test_set, batch_size=batch_size, shuffle=False)
                if test_set else None,
    }

    test_start_time = df_test.index[0] if len(df_test) > 0 else None
    info = {
        'n_features': roles['n_features'],
        'n_known': roles['n_known'],
        'feature_names': roles['feature_cols'],
    }
    return loaders, scaler_y, test_start_time, info


# --------------------------------------------------------------------------- #
# 4. Convenience one-shot loader
# --------------------------------------------------------------------------- #
def get_data(nwp_path, load_path, seq_len=96, pred_len=96,
             batch_size=32, train_ratio=0.7, val_ratio=0.1,
             points_per_day=96, mode="future", add_lag=True,
             use_nwp=True, use_calendar=True):
    """One-call helper: build df + dataloaders. Returns
    (loaders, scaler_y, test_start_time, info, roles)."""
    df, roles = build_dataframe(nwp_path, load_path, add_lag=add_lag,
                                use_nwp=use_nwp, use_calendar=use_calendar)
    loaders, scaler_y, test_start_time, info = make_dataloaders(
        df, roles, seq_len, pred_len, batch_size,
        train_ratio, val_ratio, points_per_day, mode)
    return loaders, scaler_y, test_start_time, info, roles

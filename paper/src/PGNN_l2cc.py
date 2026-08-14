
#!/usr/bin/env python3
import os, re, math, json, copy, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix, csr_matrix

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ------------------------- CONFIG -------------------------
EPOCHS   = 600
PATIENCE = 30
LR_MAX   = 0.01
LR_MIN   = 1e-4
TRAIN_SIZE = 0.80
K_FOLDS = 5  # 1 = normal 80/20 holdout; any integer > 1 activates K-fold CV
WD      = 1e-4
HID     = 128
DROPOUT = 0.20
SEED    = 42
ADD_SELF_LOOPS = True
INNER_ITERS    = 5
MU_BACKOFF     = 0.5

EVAL_EVERY = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


RESULTS_ROOT = "PGNN_ALL_RESULTS_L2_cc_AgeResults"
os.makedirs(RESULTS_ROOT, exist_ok=True)
MASTER_CSV = os.path.join(RESULTS_ROOT, "master_runs.csv")

TARGETS = ["Age"]

KNN_LIST = [2, 5, 10, 15, 20, 25, 30, 35, 40]

HIGH_P_VALUES = [2.0, 3.0, 5.0, 10.0, 1e2, 1e4, 1e6]
K_CHOICES_BY_P = {
    2.0: [5, 10],
    3.0: [10, 15],
    4.0: [10, 20],
    5.0: [15, 30],
    6.0: [20, 40],
    7.0: [20, 50],
    8.0: [30, 60],
    9.0: [30, 60],
    10.0: [40, 80],
    100.0: [100, 200],
    10000.0: [400, 800],
    1000000.0: [1600, 3200],
}

RUNS = [
    {"edge_metrics": ["l2"], "fw_source": None,  "fw_power": 0.0,
     "fw_abs": False, "fw_norm": False, "fw_label": "none"},

    {"edge_metrics": ["l2"], "fw_source": "alm", "fw_power": 1.0,
     "fw_abs": True, "fw_norm": True, "fw_label": "alm_c"},
    {"edge_metrics": ["l2"], "fw_source": "alm", "fw_power": 2.0,
     "fw_abs": True, "fw_norm": True, "fw_label": "alm_c2"},

    {"edge_metrics": ["l2"], "fw_source": "bmd", "fw_power": 1.0,
     "fw_abs": True, "fw_norm": True, "fw_label": "bmd_c"},
    {"edge_metrics": ["l2"], "fw_source": "bmd", "fw_power": 2.0,
     "fw_abs": True, "fw_norm": True, "fw_label": "bmd_c2"},

    {"edge_metrics": ["l2"], "fw_source": "bfp", "fw_power": 1.0,
     "fw_abs": True, "fw_norm": True, "fw_label": "bfp_c"},
    {"edge_metrics": ["l2"], "fw_source": "bfp", "fw_power": 2.0,
     "fw_abs": True, "fw_norm": True, "fw_label": "bfp_c2"},

    {"edge_metrics": ["l2"], "fw_source": "age", "fw_power": 1.0,
     "fw_abs": True, "fw_norm": True, "fw_label": "age_c"},
    {"edge_metrics": ["l2"], "fw_source": "age", "fw_power": 2.0,
     "fw_abs": True, "fw_norm": True, "fw_label": "age_c2"},
]


# ------------------------- Data wrapper -------------------------
try:
    from torch_geometric.data import Data as _PyGData

    class Data(_PyGData):
        pass

except Exception:
    class Data:
        def __init__(self, x, edge_index, y):
            self.x = x
            self.edge_index = edge_index
            self.y = y


# ------------------------- Utils -------------------------
def human_time(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _slugify(name: str) -> str:
    s = os.path.splitext(os.path.basename(name))[0]
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def append_csv_row_locked(row: dict, csv_path: str):
    """
    Appends one row to a CSV file.
    Uses Linux file lock when available, useful for SLURM parallel jobs.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_row = pd.DataFrame([row])

    try:
        import fcntl

        lock_path = csv_path + ".lock"
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)

            write_header = not os.path.exists(csv_path)
            df_row.to_csv(csv_path, mode="a", header=write_header, index=False)

            fcntl.flock(lock_file, fcntl.LOCK_UN)

    except Exception:
        write_header = not os.path.exists(csv_path)
        df_row.to_csv(csv_path, mode="a", header=write_header, index=False)


def append_master_row(row: dict, master_csv_path: str = MASTER_CSV):
    append_csv_row_locked(row, master_csv_path)


def mu_for(p: float, K: int) -> float:
    if p <= 10.0:
        anchors = {
            2.0: 0.02,
            3.0: 0.01,
            4.0: 0.005,
            5.0: 0.003,
            6.0: 0.002,
            7.0: 0.001,
            8.0: 0.0005,
            9.0: 0.0002,
            10.0: 0.0001,
        }

        keys = sorted(anchors.keys())
        lo = max([k for k in keys if k <= p], default=2.0)
        hi = min([k for k in keys if k >= p], default=10.0)

        if lo == hi:
            base = anchors[lo]
        else:
            t = (p - lo) / (hi - lo)
            base = anchors[lo] * (1 - t) + anchors[hi] * t

    else:
        alpha = 2.5
        base = 1e-4 * (10.0 / float(p)) ** alpha

    k_ref = 10.0
    k_factor = (k_ref / float(max(K, 1))) ** 0.5
    return base * k_factor


def one_cycle_lr(t, T, lr_max=LR_MAX, lr_min=LR_MIN):
    half = T // 2

    if t <= half:
        pct = t / max(half, 1)
        cos = (1 - math.cos(math.pi * pct)) / 2
        return lr_min + (lr_max - lr_min) * cos
    else:
        pct = (t - half) / max(T - half, 1)
        cos = (1 + math.cos(math.pi * pct)) / 2
        return lr_min + (lr_max - lr_min) * cos


def total_train_configs(num_csvs: int, k_folds: int = K_FOLDS) -> int:
    per_target = sum(len(K_CHOICES_BY_P[p]) for p in HIGH_P_VALUES)
    folds = max(int(k_folds), 1)
    return (
        num_csvs
        * folds
        * len(KNN_LIST)
        * len(RUNS)
        * len(TARGETS)
        * per_target
    )


def metrics_row(final, target, knn_k, metric_tag, fw_tag, K, p, model_mu, model_epoch, model_lr):
    return {
        "target": target,
        "kNN_graph": int(knn_k),
        "edge_metrics": metric_tag,
        "fw_source": fw_tag,
        "K": int(K),
        "p": float(p),
        "mu_used": float(model_mu),
        "RMSE_scaled": float(final["rmse_scaled"][0]),
        "RelRMSE_scaled_pct": float(final["rel_rmse_pct"][0]),
        "RMSE_orig": float(final["rmse_orig"][0]),
        "RelRMSE_orig_pct": float(final["rel_rmse_orig_pct"][0]),
        "R2": float(final["r2"][0]),
        "best_epoch": int(model_epoch),
        "best_lr": float(model_lr),
    }



def summarize_kfold_master(master_csv_path: str = MASTER_CSV,
                           active_k_folds: int = K_FOLDS):
    """
    Create k-fold aggregate CSV files from fold-level rows in master_runs.csv.

    Files created when active_k_folds > 1:
      1. kfold_average_metrics.csv
         One row per dataset/target/configuration with the mean across folds.
      2. kfold_summary_mean_std.csv
         Mean and standard deviation across folds.

    Only rows whose k_folds value equals active_k_folds are included.
    """
    if int(active_k_folds) <= 1:
        return None, None

    if not os.path.exists(master_csv_path):
        print(f"[WARN] Cannot create k-fold averages: {master_csv_path} does not exist.")
        return None, None

    df = pd.read_csv(master_csv_path)

    required_cols = {"fold", "k_folds", "csv_path"}
    missing = required_cols.difference(df.columns)
    if missing:
        print(f"[WARN] Cannot create k-fold averages. Missing columns: {sorted(missing)}")
        return None, None

    # Use only rows from the requested k-fold setting.
    df["k_folds"] = pd.to_numeric(df["k_folds"], errors="coerce")
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    df = df[df["k_folds"] == int(active_k_folds)].copy()

    if df.empty:
        print(f"[WARN] No rows found for k_folds={active_k_folds}.")
        return None, None

    group_cols = [
        "csv_path", "target", "kNN_graph",
        "edge_metrics", "fw_source", "K", "p"
    ]

    metric_cols = [
        "RMSE_scaled",
        "RelRMSE_scaled_pct",
        "RMSE_orig",
        "RelRMSE_orig_pct",
        "R2",
        "best_epoch",
        "best_lr",
        "mu_used",
        "run_time_sec",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # If the same job/configuration was accidentally rerun, retain the latest
    # row for each fold so a fold is not counted more than once.
    dedup_cols = group_cols + ["k_folds", "fold"]
    df = df.drop_duplicates(subset=dedup_cols, keep="last")

    grouped = df.groupby(group_cols, dropna=False)

    mean_df = grouped[metric_cols].mean().reset_index()
    count_df = grouped["fold"].nunique().reset_index(name="folds_completed")
    mean_df = mean_df.merge(count_df, on=group_cols, how="left")
    mean_df.insert(len(group_cols), "k_folds", int(active_k_folds))

    # Make average columns unmistakable.
    mean_df = mean_df.rename(
        columns={c: f"{c}_mean" for c in metric_cols}
    )

    average_path = os.path.join(
        RESULTS_ROOT,
        f"kfold{int(active_k_folds)}_average_metrics.csv"
    )
    mean_df.to_csv(average_path, index=False)

    mean_std_df = grouped[metric_cols].agg(["mean", "std"]).reset_index()
    mean_std_df.columns = [
        "_".join(str(x) for x in col if str(x) != "").rstrip("_")
        if isinstance(col, tuple) else str(col)
        for col in mean_std_df.columns
    ]
    mean_std_df = mean_std_df.merge(count_df, on=group_cols, how="left")
    mean_std_df.insert(len(group_cols), "k_folds", int(active_k_folds))

    mean_std_path = os.path.join(
        RESULTS_ROOT,
        f"kfold{int(active_k_folds)}_summary_mean_std.csv"
    )
    mean_std_df.to_csv(mean_std_path, index=False)

    incomplete = mean_df[mean_df["folds_completed"] < int(active_k_folds)]
    if not incomplete.empty:
        print(
            f"[WARN] {len(incomplete)} configurations have fewer than "
            f"{active_k_folds} completed folds. Check 'folds_completed'."
        )

    return average_path, mean_std_path

# ------------------------- Graph builders -------------------------
def _mutualize(A: csr_matrix) -> coo_matrix:
    return coo_matrix(A.minimum(A.T))


def add_self_loops_with_weights(edge_index: torch.Tensor,
                                edge_weight: torch.Tensor,
                                n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if not ADD_SELF_LOOPS:
        return edge_index, edge_weight

    device = edge_weight.device
    loops = torch.arange(n, dtype=torch.long, device=device)
    loop_index = torch.stack([loops, loops], dim=0)
    loop_weight = torch.ones(n, dtype=torch.float32, device=device)

    new_edge_index = torch.cat([edge_index, loop_index], dim=1)
    new_edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    return new_edge_index, new_edge_weight


def beta_norm(c: np.ndarray, beta: float = 2.0) -> float:
    c = np.asarray(c, dtype=float)

    if beta == np.inf:
        val = np.max(np.abs(c))
    else:
        val = (np.abs(c) ** beta).sum() ** (1.0 / beta)

    return float(max(val, 1e-12))


def _knn_edge_index_l2_weighted(
    X: np.ndarray,
    k: int,
    fw_used: Optional[np.ndarray] = None,
    beta: float = 2.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    n, d = X.shape

    k_eff = min(k, max(n - 1, 1))

    A = kneighbors_graph(
        X,
        k_eff,
        mode="connectivity",
        metric="euclidean",
        include_self=False
    )

    coo_mut = _mutualize(A.tocsr())
    row = coo_mut.row
    col = coo_mut.col

    if row.size == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)
        return edge_index, edge_weight

    if fw_used is not None:
        c = np.asarray(fw_used, dtype=float)
        if c.shape[0] != d:
            raise ValueError(f"fw_used length {c.shape[0]} != feature dim {d}")
    else:
        c = np.ones(d, dtype=float)

    diff = X[row] - X[col]
    num = (diff ** 2) * c[None, :]
    num = num.sum(axis=1)
    num = np.sqrt(np.maximum(num, 1e-12))

    c_norm = beta_norm(c, beta=beta)
    d_ij = num / c_norm

    mean_d = float(d_ij.mean()) if d_ij.size > 0 else 1.0
    alpha = 1.0 if mean_d <= 1e-8 else 1.0 / mean_d

    w_ij = np.exp(-alpha * d_ij).astype(np.float32)

    edge_index = torch.tensor(np.vstack((row, col)), dtype=torch.long)
    edge_weight = torch.tensor(w_ij, dtype=torch.float32)

    edge_index, edge_weight = add_self_loops_with_weights(edge_index, edge_weight, n)

    return edge_index, edge_weight


def build_edge_graphs_by_names(
    X: np.ndarray,
    k: int,
    names: List[str],
    fw_used: Optional[np.ndarray] = None
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    name_map = {
        "l2": _knn_edge_index_l2_weighted,
    }

    eis = []
    ews = []

    for nm in names:
        if nm not in name_map:
            raise ValueError(
                f"Unknown graph metric '{nm}'. Choose from: {list(name_map.keys())}"
            )

        ei, ew = name_map[nm](X, k, fw_used=fw_used)
        eis.append(ei)
        ews.append(ew)

    return eis, ews


# ------------------------- Feature weighting -------------------------
def compute_feature_weights_by_label(X_train: np.ndarray,
                                     y_train_1d: np.ndarray,
                                     power: float = 1.0,
                                     use_abs: bool = True,
                                     normalize: bool = True) -> np.ndarray:
    coefs = np.array([
        np.corrcoef(X_train[:, j], y_train_1d)[0, 1]
        for j in range(X_train.shape[1])
    ])

    coefs = np.nan_to_num(coefs, nan=0.0)

    if use_abs:
        w = np.abs(coefs)
    else:
        w = coefs

    if power != 1.0:
        if use_abs:
            w = w ** power
        else:
            w = np.sign(w) * (np.abs(w) ** power)

    if normalize:
        mean_w = w.mean() if w.size > 0 else 1.0
        w = w / (mean_w + 1e-12)

    return w


# ------------------------- Data loading -------------------------

def make_cv_splits(n_samples: int,
                   train_size: float = TRAIN_SIZE,
                   seed: int = SEED,
                   k_folds: int = K_FOLDS):
    """
    Returns (fold_id, train_idx, val_idx) tuples.
    If k_folds <= 1, this reproduces the original single train/validation split.
    If k_folds > 1, this uses shuffled KFold cross-validation.
    """
    indices = np.arange(n_samples)

    if k_folds is None or int(k_folds) <= 1:
        train_idx, val_idx = train_test_split(
            indices,
            train_size=train_size,
            random_state=seed,
            shuffle=True
        )
        return [(1, train_idx, val_idx)]

    if int(k_folds) > n_samples:
        raise ValueError(f"k_folds={k_folds} cannot be greater than n_samples={n_samples}")

    splitter = KFold(n_splits=int(k_folds), shuffle=True, random_state=seed)
    return [
        (fold_id, train_idx, val_idx)
        for fold_id, (train_idx, val_idx) in enumerate(splitter.split(indices), start=1)
    ]

def load_penn_for_target(csv_path: str,
                         k: int,
                         target: str = "alm",
                         fw_source: Optional[str] = None,
                         fw_power: float = 1.0,
                         fw_abs: bool = True,
                         fw_norm: bool = True,
                         train_size: float = TRAIN_SIZE,
                         seed: int = SEED,
                         train_idx: Optional[np.ndarray] = None,
                         val_idx: Optional[np.ndarray] = None):
    df = pd.read_csv(csv_path)
    name_map = {c.lower(): c for c in df.columns}

    target = target.lower()

    if target not in name_map:
        raise ValueError(f"CSV must contain '{target}' column case-insensitive.")

    col_y = name_map[target]

    drop_cols = ["0", "PPT ID", "Site", "Gender", "Race"]
    drop_in_df = [c for c in drop_cols if c in df.columns]

    features = df.drop(columns=drop_in_df + [col_y], errors="ignore")
    labels = df[[col_y]].copy()

    X_df = pd.get_dummies(features, drop_first=False)
    feature_columns = X_df.columns.tolist()

    for col in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = X_df[col].fillna(X_df[col].mean())
        else:
            X_df[col] = X_df[col].fillna(0)

    if pd.api.types.is_numeric_dtype(labels[col_y]):
        labels[col_y] = labels[col_y].fillna(labels[col_y].mean())

    if train_idx is None or val_idx is None:
        all_idx = np.arange(len(X_df))
        train_idx, val_idx = train_test_split(
            all_idx,
            train_size=train_size,
            random_state=seed,
            shuffle=True
        )

    X_all = X_df.values
    y_all = labels.values

    X_train_raw = X_all[train_idx]
    X_val_raw = X_all[val_idx]
    y_train_raw = y_all[train_idx]
    y_val_raw = y_all[val_idx]

    scaler_x = StandardScaler().fit(X_train_raw)
    scaler_y = StandardScaler().fit(y_train_raw)

    Xt = scaler_x.transform(X_train_raw)
    Xv = scaler_x.transform(X_val_raw)

    yt = scaler_y.transform(y_train_raw)
    yv = scaler_y.transform(y_val_raw)

    fw_used = None

    if fw_source is not None:
        src = fw_source.lower()

        if src not in ["alm", "bmd", "bfp", "age"]:
            raise ValueError("fw_source must be one of {alm,bmd,bfp,age}.")

        if src not in name_map:
            raise ValueError(f"fw_source '{src}' not found in CSV.")

        src_y = df[[name_map[src]]].copy()
        src_y = src_y.fillna(src_y.mean())

        src_all = src_y.values
        src_train_raw = src_all[train_idx]

        src_scaler = StandardScaler().fit(src_train_raw)
        src_train = src_scaler.transform(src_train_raw)[:, 0]

        fw = compute_feature_weights_by_label(
            Xt,
            src_train,
            power=fw_power,
            use_abs=fw_abs,
            normalize=fw_norm
        )

        fw_used = fw.copy()

    return Xt, yt, Xv, yv, scaler_x, scaler_y, fw_used, feature_columns


# ------------------------- Model -------------------------
class pLaplacianConv(nn.Module):
    def __init__(self, K=5, p=2.0, mu=0.01, inner_iters=INNER_ITERS):
        super().__init__()
        self.K = K
        self.p = p
        self.mu = mu
        self.inner_iters = inner_iters

    def forward_one_graph(self,
                          h: torch.Tensor,
                          edge_index: torch.Tensor,
                          edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        row, col = edge_index
        h0 = h

        for _ in range(self.inner_iters):
            diff = h[row] - h[col]
            norm_diff = torch.norm(diff, dim=1).clamp(min=1e-6)

            max_norm = norm_diff.max().clamp(min=1e-6)
            plap_w = (norm_diff / max_norm).pow(self.p - 2)

            if edge_weight is not None:
                w = plap_w * edge_weight
            else:
                w = plap_w

            agg = torch.zeros_like(h)
            message = w.unsqueeze(1) * (h[col] - h[row])
            agg.index_add_(0, row, message)

            deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
            deg.index_add_(0, row, torch.ones_like(row, dtype=h.dtype))
            agg = agg / deg.clamp_min(1.0).unsqueeze(1)

            h = h + (-self.mu) * agg

        return 0.5 * h + 0.5 * h0

    def forward(self,
                h: torch.Tensor,
                edge_indices: List[torch.Tensor],
                edge_weights: List[torch.Tensor],
                mu: Optional[float] = None) -> torch.Tensor:
        mu_prev = self.mu

        if mu is not None:
            self.mu = float(mu)

        outs = [
            self.forward_one_graph(h, ei, ew)
            for ei, ew in zip(edge_indices, edge_weights)
        ]

        out = torch.stack(outs, dim=0).mean(dim=0)

        self.mu = mu_prev

        return out


class PGNNRegressor(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, K=5, p=2.0, mu=0.01, dropout=0.3):
        super().__init__()

        self.lin1 = nn.Linear(in_dim, hid_dim)
        self.bn1  = nn.BatchNorm1d(hid_dim)

        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.bn2  = nn.BatchNorm1d(hid_dim)

        self.drop = nn.Dropout(dropout)
        self.pconv = pLaplacianConv(K, p, mu)
        self.out = nn.Linear(hid_dim, out_dim)

    def encode(self, X):
        h = F.relu(self.bn1(self.lin1(X)))
        h = self.drop(h)

        h = F.relu(self.bn2(self.lin2(h)))
        h = self.drop(h)

        return h

    def forward(self, data, edge_indices, edge_weights, mu=None):
        h = self.encode(data.x)
        h = self.pconv(h, edge_indices, edge_weights, mu=mu)
        return self.out(h)


# ------------------------- Evaluation -------------------------
def inverse_transform_y(scaler_y, Y_scaled: np.ndarray, target_idx: Optional[int] = None) -> np.ndarray:
    if target_idx is None:
        return scaler_y.inverse_transform(Y_scaled)

    mean = float(scaler_y.mean_[target_idx])
    scale = float(scaler_y.scale_[target_idx])

    return Y_scaled * scale + mean


def evaluate(model,
             Xv,
             yv,
             edge_indices,
             edge_weights,
             scaler_y,
             target_idx: Optional[int] = 0,
             sample_weights: Optional[np.ndarray] = None) -> dict:
    with torch.no_grad():
        model.eval()

        device = next(model.parameters()).device
        Xv_t = torch.tensor(Xv, dtype=torch.float32, device=device)

        h = model.encode(Xv_t)
        val_out = model.pconv(h, edge_indices, edge_weights)
        yp_scaled = model.out(val_out).detach().cpu().numpy()

    if yv.ndim == 1:
        y_true_scaled = yv.reshape(-1, 1)
    else:
        y_true_scaled = yv

    if yp_scaled.ndim == 1:
        yp_scaled = yp_scaled.reshape(-1, 1)

    yp_orig = inverse_transform_y(
        scaler_y,
        yp_scaled,
        target_idx=target_idx
    )

    y_true_or = inverse_transform_y(
        scaler_y,
        y_true_scaled,
        target_idx=target_idx
    )

    sq_err_scaled = (y_true_scaled - yp_scaled) ** 2
    mse_scaled = sq_err_scaled.mean(axis=0)
    rmse_scaled = np.sqrt(mse_scaled)

    rms_true_sc = np.sqrt((y_true_scaled ** 2).mean(axis=0)).clip(min=1e-12)
    rel_rmse_pct = 100.0 * rmse_scaled / rms_true_sc

    sq_err_orig = (y_true_or - yp_orig) ** 2
    mse_orig = sq_err_orig.mean(axis=0)
    rmse_orig = np.sqrt(mse_orig)

    rms_true_or = np.sqrt((y_true_or ** 2).mean(axis=0)).clip(min=1e-12)
    rel_rmse_orig_pct = 100.0 * rmse_orig / rms_true_or

    from sklearn.metrics import r2_score as _r2
    r2 = _r2(y_true_scaled, yp_scaled, multioutput="raw_values")

    return {
        "rmse_scaled": rmse_scaled,
        "rel_rmse_pct": rel_rmse_pct,
        "rmse_orig": rmse_orig,
        "rel_rmse_orig_pct": rel_rmse_orig_pct,
        "r2": r2,
        "yp_scaled": yp_scaled,
    }


# ------------------------- Sweep -------------------------
def sweep_high_p(csv_path: str, pbar=None, k_folds: int = K_FOLDS):
    base = _slugify(csv_path)
    train_pct = int(round(TRAIN_SIZE * 100))

    dataset_start = time.perf_counter()
    n_samples = len(pd.read_csv(csv_path))
    cv_splits = make_cv_splits(
        n_samples,
        train_size=TRAIN_SIZE,
        seed=SEED,
        k_folds=k_folds
    )

    for fold_id, train_idx, val_idx in cv_splits:
        fold_tag = f"fold{fold_id}" if int(k_folds) > 1 else "holdout"
        print(f"\n--- {base}: {fold_tag} | train={len(train_idx)} | val={len(val_idx)} ---")

        for knn_k in KNN_LIST:
            for run in RUNS:
                edge_metrics = run["edge_metrics"]
                fw_source    = run["fw_source"]
                fw_power     = run["fw_power"]
                fw_abs       = run["fw_abs"]
                fw_norm      = run["fw_norm"]
                fw_label     = run["fw_label"]

                for target in TARGETS:
                    load_start = time.perf_counter()

                    Xt, yt, Xv, yv, scaler_x, scaler_y, fw_used, feature_columns = load_penn_for_target(
                        csv_path,
                        k=knn_k,
                        target=target,
                        fw_source=fw_source,
                        fw_power=fw_power,
                        fw_abs=fw_abs,
                        fw_norm=fw_norm,
                        train_size=TRAIN_SIZE,
                        seed=SEED,
                        train_idx=train_idx,
                        val_idx=val_idx
                    )

                    eis_train, ews_train = build_edge_graphs_by_names(
                        Xt,
                        knn_k,
                        edge_metrics,
                        fw_used=fw_used
                    )

                    eis_val, ews_val = build_edge_graphs_by_names(
                        Xv,
                        knn_k,
                        edge_metrics,
                        fw_used=fw_used
                    )

                    eis_train = [ei.to(DEVICE) for ei in eis_train]
                    ews_train = [ew.to(DEVICE) for ew in ews_train]
                    eis_val   = [ei.to(DEVICE) for ei in eis_val]
                    ews_val   = [ew.to(DEVICE) for ew in ews_val]

                    metric_tag = "+".join(edge_metrics)
                    fw_tag = fw_label

                    out_root = os.path.join(
                        RESULTS_ROOT,
                        f"{base}_{fold_tag}_pgnn_target-{target.upper()}_{metric_tag}_fw-{fw_tag}_{train_pct}pct_knn{knn_k}"
                    )

                    os.makedirs(out_root, exist_ok=True)

                    task_dir = os.path.join(out_root, target.lower())
                    os.makedirs(task_dir, exist_ok=True)

                    if fw_used is not None and fw_source in ["alm", "bmd", "bfp", "age"]:
                        fw_save_path = os.path.join(
                            task_dir,
                            f"feature_weights_{fw_source}_power{int(fw_power)}_knn{knn_k}.npy"
                        )
                        np.save(fw_save_path, fw_used)

                    out_dim = 1
                    y_train = yt[:, 0:1]
                    y_val = yv[:, 0:1]

                    data_train = Data(
                        x=torch.tensor(Xt, dtype=torch.float32, device=DEVICE),
                        edge_index=None,
                        y=torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
                    )

                    load_elapsed = time.perf_counter() - load_start

                    for p in HIGH_P_VALUES:
                        for K in K_CHOICES_BY_P[p]:
                            run_start = time.perf_counter()

                            desc = (
                                f"{base} | {fold_tag} | {target} | knn={knn_k} | fw={fw_tag} | "
                                f"p={p:g} | K={K}"
                            )

                            if pbar is not None:
                                pbar.set_description(desc)

                            mu = mu_for(float(p), int(K))

                            model = PGNNRegressor(
                                Xt.shape[1],
                                HID,
                                out_dim,
                                K=K,
                                p=p,
                                mu=mu,
                                dropout=DROPOUT
                            ).to(DEVICE)

                            opt = torch.optim.AdamW(
                                model.parameters(),
                                lr=LR_MIN,
                                weight_decay=WD
                            )

                            loss_fn = nn.MSELoss(reduction="mean")

                            best_state = None
                            best_rel = np.full(out_dim, np.inf)
                            epochs_no_improve = 0

                            T = max(EPOCHS, 50)
                            current_mu = mu

                            for epoch in range(1, T + 1):
                                for g in opt.param_groups:
                                    g["lr"] = one_cycle_lr(epoch, T, LR_MAX, LR_MIN)

                                model.train()
                                opt.zero_grad()

                                out = model(
                                    data_train,
                                    eis_train,
                                    ews_train,
                                    mu=current_mu
                                )

                                loss = loss_fn(out, data_train.y)
                                loss.backward()

                                torch.nn.utils.clip_grad_norm_(
                                    model.parameters(),
                                    max_norm=1.0
                                )

                                opt.step()

                                if epoch % 10 == 0:
                                    eval_now = evaluate(
                                        model,
                                        Xt,
                                        y_train,
                                        eis_train,
                                        ews_train,
                                        scaler_y,
                                        target_idx=0
                                    )

                                    if np.median(eval_now["rel_rmse_pct"]) > 200:
                                        current_mu *= MU_BACKOFF

                                        for g in opt.param_groups:
                                            g["lr"] = max(g["lr"] * 0.5, LR_MIN)

                                if epoch % EVAL_EVERY == 0 or epoch == 1:
                                    metrics = evaluate(
                                        model,
                                        Xv,
                                        y_val,
                                        eis_val,
                                        ews_val,
                                        scaler_y,
                                        target_idx=0
                                    )

                                    rel = metrics["rel_rmse_orig_pct"]
                                    improved = (rel < best_rel).any()

                                    if improved:
                                        best_rel = np.minimum(best_rel, rel)

                                        best_state = {
                                            "model": copy.deepcopy(model.state_dict()),
                                            "metrics": metrics,
                                            "epoch": epoch,
                                            "mu": current_mu,
                                            "lr": opt.param_groups[0]["lr"],
                                        }

                                        epochs_no_improve = 0

                                    else:
                                        epochs_no_improve += EVAL_EVERY

                                        if epochs_no_improve >= PATIENCE:
                                            break

                            if best_state is not None:
                                model.load_state_dict(best_state["model"])

                            final = evaluate(
                                model,
                                Xv,
                                y_val,
                                eis_val,
                                ews_val,
                                scaler_y,
                                target_idx=0
                            )

                            model_mu = best_state["mu"] if best_state else mu
                            model_epoch = best_state["epoch"] if best_state else -1
                            model_lr = best_state["lr"] if best_state else LR_MIN

                            ckpt_path = os.path.join(
                                task_dir,
                                f"best_model_knn{knn_k}_{metric_tag}_fw-{fw_tag}_K{K}_p{p}_mu{model_mu:.3g}.pt"
                            )

                            torch.save(
                                {
                                    "state_dict": model.state_dict(),
                                    "config": {
                                        "csv_path": csv_path,
                                        "target": target,
                                        "knn_k": knn_k,
                                        "edge_metrics": edge_metrics,
                                        "fw_source": fw_source,
                                        "fw_power": fw_power,
                                        "fw_abs": fw_abs,
                                        "fw_norm": fw_norm,
                                        "K": K,
                                        "p": p,
                                        "mu": model_mu,
                                        "epoch": model_epoch,
                                        "lr": model_lr,
                                        "train_size": TRAIN_SIZE,
                                        "seed": SEED,
                                        "device": str(DEVICE),
                                    },
                                    "preprocess": {
                                        "scaler_x": scaler_x,
                                        "scaler_y": scaler_y,
                                        "fw_used": fw_used,
                                        "feature_columns": feature_columns,
                                    },
                                    "metrics": final,
                                },
                                ckpt_path
                            )

                            results_csv_path = os.path.join(
                                task_dir,
                                f"results_knn{knn_k}_{metric_tag}_fw-{fw_tag}_K{K}_p{p}_mu{model_mu:.3g}.csv"
                            )

                            row = metrics_row(
                                final,
                                target,
                                knn_k,
                                metric_tag,
                                fw_tag,
                                K,
                                p,
                                model_mu,
                                model_epoch,
                                model_lr
                            )

                            run_elapsed = time.perf_counter() - run_start

                            pd.DataFrame([{
                                "fold": int(fold_id),
                                "k_folds": int(k_folds),
                                **row,
                                "run_time_sec": float(run_elapsed),
                                "run_time_human": human_time(run_elapsed),
                                "ckpt_path": ckpt_path,
                            }]).to_csv(results_csv_path, index=False)

                            append_master_row({
                                "results_root": RESULTS_ROOT,
                                "csv_path": csv_path,
                                "fold": int(fold_id),
                                "k_folds": int(k_folds),
                                **row,
                                "load_graph_time_sec": float(load_elapsed),
                                "run_time_sec": float(run_elapsed),
                                "run_time_human": human_time(run_elapsed),
                                "results_csv": results_csv_path,
                                "ckpt_path": ckpt_path,
                            })

                            if pbar is not None:
                                pbar.update(1)

                            del model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

    dataset_elapsed = time.perf_counter() - dataset_start
    print(f"Dataset completed: {csv_path}")
    print(f"Dataset time: {human_time(dataset_elapsed)}")


# ------------------------- Main -------------------------
def main():
    global TARGETS, KNN_LIST, RUNS, K_FOLDS

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, default=None,
                        help="Run only one CSV path.")

    parser.add_argument("--target", type=str, default=None,
                        help="Run only one target: Age, ALM, BMD, or BFP.")

    parser.add_argument("--knn", type=int, default=None,
                        help="Run only one kNN graph value.")

    parser.add_argument("--run_id", type=int, default=None,
                        help="Run only one RUNS index: 0 to 8.")

    parser.add_argument("--kfolds", type=int, default=K_FOLDS,
                        help="Number of folds for cross-validation. Use 1 for the original holdout split.")

    args = parser.parse_args()

    csvs = [
        "/scratch/gsunka1/TDA_guoji/male.csv",
        "/scratch/gsunka1/TDA_guoji/female.csv",
        "/scratch/gsunka1/TDA_guoji/penn_data.csv",
    ]

    original_run_id = args.run_id
    K_FOLDS = max(int(args.kfolds), 1)
    original_target = args.target
    original_knn = args.knn
    original_csv = args.csv

    if args.csv is not None:
        csvs = [args.csv]

    if args.target is not None:
        TARGETS = [args.target]

    if args.knn is not None:
        KNN_LIST = [args.knn]

    if args.run_id is not None:
        if args.run_id < 0 or args.run_id >= len(RUNS):
            raise ValueError(f"--run_id must be between 0 and {len(RUNS)-1}")
        RUNS = [RUNS[args.run_id]]

    total_configs = total_train_configs(num_csvs=len(csvs), k_folds=K_FOLDS)

    print("=" * 80)
    print("PGNN sweep started")
    print("Device:", DEVICE)
    print("Number of CSV files:", len(csvs))
    print("Targets:", TARGETS)
    print("KNN_LIST:", KNN_LIST)
    print("Number of RUNS:", len(RUNS))
    print("K_FOLDS:", K_FOLDS)
    print("HIGH_P_VALUES:", HIGH_P_VALUES)
    print("Total training configurations:", total_configs)
    print("=" * 80)

    overall_start = time.perf_counter()

    if tqdm is not None:
        with tqdm(total=total_configs, ncols=120, unit="run") as pbar:
            for path in csvs:
                print(f"\n=== Running PGNN sweep for: {path} ===")
                sweep_high_p(path, pbar=pbar, k_folds=K_FOLDS)
                print(f"Finished: {path}")
    else:
        print("tqdm not installed. Running without progress bar.")
        for path in csvs:
            print(f"\n=== Running PGNN sweep for: {path} ===")
            sweep_high_p(path, pbar=None, k_folds=K_FOLDS)
            print(f"Finished: {path}")

    overall_elapsed = time.perf_counter() - overall_start

    first_csv_slug = _slugify(csvs[0])
    target_tag = TARGETS[0] if len(TARGETS) == 1 else "alltargets"
    knn_tag = str(KNN_LIST[0]) if len(KNN_LIST) == 1 else "allknn"
    run_tag = str(original_run_id) if original_run_id is not None else "allruns"
    fold_tag = str(K_FOLDS) if K_FOLDS > 1 else "holdout"

    runtime_csv_path = os.path.join(
        RESULTS_ROOT,
        f"runtime_summary_{first_csv_slug}_{target_tag}_knn{knn_tag}_run{run_tag}_kfold{fold_tag}.csv"
    )

    all_runtime_csv_path = os.path.join(
        RESULTS_ROOT,
        "all_runtime_summary.csv"
    )

    runtime_row = {
        "results_root": RESULTS_ROOT,
        "csv_argument": original_csv,
        "target_argument": original_target,
        "knn_argument": original_knn,
        "run_id_argument": original_run_id,
        "csvs_run": ";".join(csvs),
        "targets_run": ";".join(TARGETS),
        "knn_list_run": ";".join(map(str, KNN_LIST)),
        "num_runs_group": len(RUNS),
        "k_folds": K_FOLDS,
        "high_p_values": ";".join(map(str, HIGH_P_VALUES)),
        "total_training_configurations_this_job": total_configs,
        "total_elapsed_time": human_time(overall_elapsed),
        "total_elapsed_seconds": round(overall_elapsed, 2),
        "device": str(DEVICE),
        "master_csv": MASTER_CSV,
    }

    pd.DataFrame([runtime_row]).to_csv(runtime_csv_path, index=False)
    append_csv_row_locked(runtime_row, all_runtime_csv_path)

    kfold_average_path, kfold_summary_path = summarize_kfold_master(
        MASTER_CSV,
        active_k_folds=K_FOLDS
    )

    print("=" * 80)
    print("PGNN sweep completed")
    print("Total elapsed time:", human_time(overall_elapsed))
    print("Total elapsed seconds:", round(overall_elapsed, 2))
    print("Results saved in:", RESULTS_ROOT)
    print("Master CSV:", MASTER_CSV)
    print("Runtime summary CSV:", runtime_csv_path)
    print("All runtime summary CSV:", all_runtime_csv_path)
    if kfold_average_path is not None:
        print("K-fold average CSV:", kfold_average_path)
    if kfold_summary_path is not None:
        print("K-fold mean/std CSV:", kfold_summary_path)
    print("=" * 80)


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    main()
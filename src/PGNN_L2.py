
#!/usr/bin/env python3
import os, re, math, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix, csr_matrix

# ------------------------- CONFIG -------------------------
EPOCHS   = 600
PATIENCE = 30
LR_MAX   = 0.01
LR_MIN   = 1e-4
TRAIN_SIZE = 0.80
WD      = 1e-4
HID     = 128
DROPOUT = 0.20
SEED    = 42
ADD_SELF_LOOPS = True
INNER_ITERS    = 5
MU_BACKOFF     = 0.5

RESULTS_ROOT = "PGNN_RESULTS_L2"
os.makedirs(RESULTS_ROOT, exist_ok=True)

# Master index of *all* runs (appended as you go)
MASTER_CSV = os.path.join(RESULTS_ROOT, "master_runs.csv")

TARGETS = ["ALM", "BMD", "BFP"]
KNN_LIST = [2, 5, 10, 15, 20, 25, 30, 35, 40]

HIGH_P_VALUES = [2.0, 3.0, 5.0, 10.0, 1e2, 1e4, 1e6]
K_CHOICES_BY_P = {
    2.0:[5,10],     3.0:[10,15],
    4.0:[10,20],    5.0:[15,30],
    6.0:[20,40],    7.0:[20,50],
    8.0:[30,60],    9.0:[30,60],
    10.0:[40,80],
    100.0:[100,200],
    10000.0:[400,800],
    1000000.0:[1600,3200],
}

EDGE_METRIC = "l2"
FW_LABEL = "none"   # for folder naming only

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
def _slugify(name: str) -> str:
    s = os.path.splitext(os.path.basename(name))[0]
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def append_master_row(row: dict, master_csv_path: str = MASTER_CSV):
    """
    Append a single run record to the master CSV under RESULTS_ROOT.
    Writes header only if the file does not exist yet.
    """
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(master_csv_path)
    df_row.to_csv(master_csv_path, mode="a", header=write_header, index=False)

def mu_for(p: float, K: int) -> float:
    if p <= 10.0:
        anchors = {
            2.0: 0.02,  3.0: 0.01,   4.0: 0.005,   5.0: 0.003,
            6.0: 0.002, 7.0: 0.001,
            8.0: 0.0005, 9.0: 0.0002,
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
        base = 1e-4 * (10.0 / float(p))**alpha

    k_ref = 10.0
    k_factor = (k_ref / float(max(K, 1)))**0.5
    return base * k_factor

# ------------------------- Graph builders (L2, unweighted) -------------------------
def _mutualize(A: csr_matrix) -> coo_matrix:
    return coo_matrix(A.minimum(A.T))

def add_self_loops(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    if not ADD_SELF_LOOPS:
        return edge_index
    device = edge_index.device
    loops = torch.arange(n, dtype=torch.long, device=device)
    loop_index = torch.stack([loops, loops], dim=0)
    return torch.cat([edge_index, loop_index], dim=1)

def _knn_edge_index_l2_unweighted(X: np.ndarray, k: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    kNN connectivity graph (L2 metric), mutualized, no edge weights (edge_weight=None).
    """
    n, _ = X.shape
    A = kneighbors_graph(X, k, mode='connectivity', metric='euclidean', include_self=False)
    coo_mut = _mutualize(A.tocsr())
    row = coo_mut.row
    col = coo_mut.col

    if row.size == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_index = add_self_loops(edge_index, n)
        return edge_index, None

    edge_index = torch.tensor(np.vstack((row, col)), dtype=torch.long)
    edge_index = add_self_loops(edge_index, n)
    return edge_index, None

# ------------------------- Data loading (no fw weights) -------------------------
def load_penn_for_target(csv_path: str,
                         target: str = "alm",
                         train_size: float = TRAIN_SIZE,
                         seed: int = SEED):
    df = pd.read_csv(csv_path)
    name_map = {c.lower(): c for c in df.columns}

    target_l = target.lower()
    if target_l not in name_map:
        raise ValueError(f"CSV must contain '{target}' column (case-insensitive).")

    col_y = name_map[target_l]
    drop_cols = ['0', 'PPT ID', 'Site', 'Gender', 'Race']
    drop_in_df = [c for c in drop_cols if c in df.columns]

    # Always remove ALL supervised targets from X, regardless of which target we predict.
    all_target_cols = []
    for t in ["alm", "bmd", "bfp"]:
        if t in name_map:
            all_target_cols.append(name_map[t])

    features = df.drop(columns=drop_in_df + all_target_cols, errors='ignore')
    labels = df[[col_y]].copy()

    X_df = pd.get_dummies(features, drop_first=False)
    feature_columns = X_df.columns.tolist()

    # fill missing
    for col in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = X_df[col].fillna(X_df[col].mean())
        else:
            X_df[col] = X_df[col].fillna(0)

    if pd.api.types.is_numeric_dtype(labels[col_y]):
        labels[col_y] = labels[col_y].fillna(labels[col_y].mean())

    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_df.values, labels.values, train_size=train_size, random_state=seed
    )

    scaler_x = StandardScaler().fit(X_train_raw)
    scaler_y = StandardScaler().fit(y_train_raw)

    Xt = scaler_x.transform(X_train_raw)
    Xv = scaler_x.transform(X_val_raw)
    yt = scaler_y.transform(y_train_raw)
    yv = scaler_y.transform(y_val_raw)

    return Xt, yt, Xv, yv, scaler_x, scaler_y, feature_columns

# ------------------------- Model -------------------------
class pLaplacianConv(nn.Module):
    def __init__(self, K=5, p=2.0, mu=0.01, inner_iters=INNER_ITERS):
        super().__init__()
        self.K = K
        self.p = p
        self.mu = mu
        self.inner_iters = inner_iters

    def forward_one_graph(self, h: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None):
        row, col = edge_index
        h0 = h
        for _ in range(self.inner_iters):
            diff = h[row] - h[col]
            norm_diff = torch.norm(diff, dim=1).clamp(min=1e-6)
            plap_w = (norm_diff / norm_diff.max()).pow(self.p - 2)

            w = plap_w if edge_weight is None else (plap_w * edge_weight)

            msg = (h[col] - h[row]) * w.unsqueeze(1)
            agg = torch.zeros_like(h)
            agg.index_add_(0, row, msg)

            deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
            deg.index_add_(0, row, torch.ones_like(row, dtype=h.dtype))
            agg = agg / deg.clamp_min(1.0).unsqueeze(1)

            h = h + (-self.mu) * agg
        return 0.5 * h + 0.5 * h0

    def forward(self, h, edge_indices, edge_weights, mu: Optional[float] = None):
        mu_prev = self.mu
        if mu is not None:
            self.mu = float(mu)

        outs = [self.forward_one_graph(h, ei, ew) for ei, ew in zip(edge_indices, edge_weights)]
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
        self.out   = nn.Linear(hid_dim, out_dim)

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

# ------------------------- Train/Eval helpers -------------------------
def one_cycle_lr(t, T, lr_max=LR_MAX, lr_min=LR_MIN):
    half = T // 2
    if t <= half:
        cos = (1 + math.cos(math.pi * (1 - t/max(half,1)))) / 2
        return lr_min + (lr_max - lr_min) * cos
    else:
        cos = (1 + math.cos(math.pi * ((t-half)/max(T-half,1)))) / 2
        return lr_min + (lr_max - lr_min) * cos

def inverse_transform_y(scaler_y, Y_scaled: np.ndarray, target_idx: Optional[int] = None) -> np.ndarray:
    if target_idx is None:
        return scaler_y.inverse_transform(Y_scaled)
    mean  = float(scaler_y.mean_[target_idx])
    scale = float(scaler_y.scale_[target_idx])
    return Y_scaled * scale + mean

def evaluate(model, Xv, yv, edge_indices, edge_weights, scaler_y, target_idx: int = 0) -> dict:
    with torch.no_grad():
        model.eval()
        Xv_t = torch.tensor(Xv, dtype=torch.float32)
        h = model.encode(Xv_t)
        val_out = model.pconv(h, edge_indices, edge_weights)
        yp_scaled = model.out(val_out).cpu().numpy()

    y_true_scaled = yv.reshape(-1, 1) if yv.ndim == 1 else yv
    yp_scaled = yp_scaled.reshape(-1, 1) if yp_scaled.ndim == 1 else yp_scaled

    yp_orig   = inverse_transform_y(scaler_y, yp_scaled, target_idx=target_idx)
    y_true_or = inverse_transform_y(scaler_y, y_true_scaled, target_idx=target_idx)

    sq_err_scaled = (y_true_scaled - yp_scaled) ** 2
    rmse_scaled   = np.sqrt(sq_err_scaled.mean(axis=0))
    rms_true_sc   = np.sqrt((y_true_scaled ** 2).mean(axis=0)).clip(min=1e-12)
    rel_rmse_pct  = 100.0 * rmse_scaled / rms_true_sc

    sq_err_orig       = (y_true_or - yp_orig) ** 2
    rmse_orig         = np.sqrt(sq_err_orig.mean(axis=0))
    rms_true_or       = np.sqrt((y_true_or ** 2).mean(axis=0)).clip(min=1e-12)
    rel_rmse_orig_pct = 100.0 * rmse_orig / rms_true_or

    from sklearn.metrics import r2_score as _r2
    r2 = _r2(y_true_scaled, yp_scaled, multioutput='raw_values')

    return {
        "rmse_scaled": rmse_scaled,
        "rel_rmse_pct": rel_rmse_pct,
        "rmse_orig": rmse_orig,
        "rel_rmse_orig_pct": rel_rmse_orig_pct,
        "r2": r2,
    }

def mse_on_split(model, X, y, edge_indices, edge_weights, loss_fn, mu=None) -> float:
    """
    MSE on a split in eval mode (dropout off). Comparable between train/val.
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        h = model.encode(X_t)
        h = model.pconv(h, edge_indices, edge_weights, mu=mu)
        pred = model.out(h)

        return float(loss_fn(pred, y_t).item())

# ------------------------- Sweep (L2 only, no weights) -------------------------
def sweep_high_p(csv_path: str):
    base = _slugify(csv_path)
    train_pct = int(round(TRAIN_SIZE * 100))

    for knn_k in KNN_LIST:
        for target in TARGETS:
            Xt, yt, Xv, yv, scaler_x, scaler_y, feature_columns = load_penn_for_target(
                csv_path, target=target, train_size=TRAIN_SIZE, seed=SEED
            )

            ei_train, ew_train = _knn_edge_index_l2_unweighted(Xt, knn_k)
            ei_val,   ew_val   = _knn_edge_index_l2_unweighted(Xv, knn_k)

            eis_train, ews_train = [ei_train], [ew_train]
            eis_val,   ews_val   = [ei_val],   [ew_val]

            out_root = os.path.join(
                RESULTS_ROOT,
                f"{base}_pgnn_target-{target.upper()}_{EDGE_METRIC}_fw-{FW_LABEL}_{train_pct}pct_knn{knn_k}"
            )
            os.makedirs(out_root, exist_ok=True)
            task_dir = os.path.join(out_root, target.lower())
            os.makedirs(task_dir, exist_ok=True)

            out_dim = 1
            y_train = yt[:, 0:1]
            y_val   = yv[:, 0:1]

            data_train = Data(
                x=torch.tensor(Xt, dtype=torch.float32),
                edge_index=None,
                y=torch.tensor(y_train, dtype=torch.float32)
            )

            for p in HIGH_P_VALUES:
                for K in K_CHOICES_BY_P[p]:
                    mu_init = mu_for(float(p), int(K))

                    model = PGNNRegressor(
                        Xt.shape[1], HID, out_dim,
                        K=K, p=p, mu=mu_init, dropout=DROPOUT
                    )
                    opt = torch.optim.Adam(model.parameters(), lr=LR_MIN, weight_decay=WD)
                    loss_fn = nn.MSELoss(reduction='mean')

                    best_state = None
                    best_rel   = np.full(out_dim, np.inf)
                    epochs_no_improve = 0

                    T = max(EPOCHS, 50)
                    current_mu = mu_init

                    # per-epoch history for this (target, knn_k, p, K)
                    history_rows = []

                    for epoch in range(1, T+1):
                        for g in opt.param_groups:
                            g['lr'] = one_cycle_lr(epoch, T, LR_MAX, LR_MIN)

                        model.train()
                        opt.zero_grad()
                        out = model(data_train, eis_train, ews_train, mu=current_mu)
                        loss = loss_fn(out, data_train.y)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        opt.step()

                        # log train/val losses every epoch
                        train_mse_step = float(loss.item())
                        train_mse_eval = mse_on_split(model, Xt, y_train, eis_train, ews_train, loss_fn, mu=current_mu)
                        val_mse        = mse_on_split(model, Xv, y_val,   eis_val,   ews_val,   loss_fn, mu=current_mu)

                        history_rows.append({
                            "epoch": epoch,
                            "lr": float(opt.param_groups[0]["lr"]),
                            "mu": float(current_mu),
                            "train_mse_step": train_mse_step,
                            "train_mse_eval": train_mse_eval,
                            "val_mse": val_mse,
                        })

                        # existing mu-backoff heuristic
                        if epoch % 5 == 0:
                            eval_now = evaluate(model, Xt, y_train, eis_train, ews_train, scaler_y, target_idx=0)
                            if np.median(eval_now["rel_rmse_pct"]) > 200:
                                current_mu *= MU_BACKOFF
                                for g in opt.param_groups:
                                    g['lr'] = max(g['lr']*0.5, LR_MIN)

                        # early stopping criterion uses val RelRMSE_orig_pct
                        metrics = evaluate(model, Xv, y_val, eis_val, ews_val, scaler_y, target_idx=0)
                        rel = metrics["rel_rmse_orig_pct"]
                        improved = (rel < best_rel).any()

                        if improved:
                            best_rel = np.minimum(best_rel, rel)
                            best_state = {
                                "model": copy.deepcopy(model.state_dict()),
                                "epoch": epoch,
                                "mu": current_mu,
                                "lr": opt.param_groups[0]['lr'],
                            }
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                            if epochs_no_improve >= PATIENCE:
                                break

                    # save per-epoch history CSV for this run
                    history_path = os.path.join(
                        task_dir,
                        f"history_knn{knn_k}_{EDGE_METRIC}_fw-{FW_LABEL}_K{K}_p{p}_mu0{mu_init:.3g}.csv"
                    )
                    pd.DataFrame(history_rows).to_csv(history_path, index=False)

                    # restore best model (if any)
                    if best_state is not None:
                        model.load_state_dict(best_state["model"])

                    final = evaluate(model, Xv, y_val, eis_val, ews_val, scaler_y, target_idx=0)

                    model_mu = best_state["mu"] if best_state else mu_init
                    model_epoch = best_state["epoch"] if best_state else -1
                    model_lr = best_state["lr"] if best_state else LR_MIN

                    ckpt_path = os.path.join(
                        task_dir,
                        f"best_model_knn{knn_k}_{EDGE_METRIC}_fw-{FW_LABEL}_K{K}_p{p}_mu{model_mu:.3g}.pt"
                    )
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "config": {
                                "csv_path": csv_path,
                                "target": target,
                                "knn_k": knn_k,
                                "edge_metrics": [EDGE_METRIC],
                                "fw_source": None,
                                "K": K,
                                "p": p,
                                "mu": model_mu,
                                "epoch": model_epoch,
                                "lr": model_lr,
                                "train_size": TRAIN_SIZE,
                                "seed": SEED,
                            },
                            "preprocess": {
                                "scaler_x": scaler_x,
                                "scaler_y": scaler_y,
                                "feature_columns": feature_columns,
                            },
                            "metrics": final,
                            "history_csv": history_path,
                        },
                        ckpt_path
                    )

                    results_csv_path = os.path.join(
                        task_dir,
                        f"results_knn{knn_k}_{EDGE_METRIC}_fw-{FW_LABEL}_K{K}_p{p}_mu{model_mu:.3g}.csv"
                    )
                    pd.DataFrame({
                        "target":           [target],
                        "kNN_graph":        [knn_k],
                        "edge_metrics":     [EDGE_METRIC],
                        "fw_source":        [FW_LABEL],
                        "K":                [K],
                        "p":                [p],
                        "mu_init":          [mu_init],
                        "mu_used":          [model_mu],
                        "RelRMSE_orig_pct": [final["rel_rmse_orig_pct"][0]],
                        "R2":               [final["r2"][0]],
                        "best_epoch":       [model_epoch],
                        "best_lr":          [model_lr],
                        "history_csv":      [history_path],
                        "ckpt_path":        [ckpt_path],
                    }).to_csv(results_csv_path, index=False)

                    # append to the master index
                    append_master_row({
                        "results_root": RESULTS_ROOT,
                        "csv_path": csv_path,
                        "target": target,
                        "knn_k": int(knn_k),
                        "p": float(p),
                        "K": int(K),
                        "mu_init": float(mu_init),
                        "mu_used": float(model_mu),
                        "best_epoch": int(model_epoch),
                        "best_lr": float(model_lr),
                        "RelRMSE_orig_pct": float(final["rel_rmse_orig_pct"][0]),
                        "R2": float(final["r2"][0]),
                        "history_csv": history_path,
                        "results_csv": results_csv_path,
                        "ckpt_path": ckpt_path,
                    })

# ------------------------- Main -------------------------
def main():
    csvs = [
        "/male.csv",
        "/female.csv",
        "/penn_data.csv"
    ]
    for path in csvs:
        print(f"\n=== Running PGNN sweep for: {path} ===")
        sweep_high_p(path)
        print(f"Finished: {path}")

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    main()

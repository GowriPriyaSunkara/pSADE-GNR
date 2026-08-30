


#!/usr/bin/env python3
import os, re, math, copy, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix, csr_matrix
from tqdm.auto import tqdm

# ------------------------- DEVICE -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- CONFIG -------------------------
EPOCHS   = 600
PATIENCE = 30
LR_MAX   = 0.01
LR_MIN   = 1e-4
TRAIN_SIZE = 0.80  # retained for compatibility; K-fold uses all samples
N_SPLITS = 5
WD      = 1e-4
HID     = 128
DROPOUT = 0.20
SEED    = 42
ADD_SELF_LOOPS = True
INNER_ITERS    = 5
MU_BACKOFF     = 0.5

RESULTS_ROOT = "PGNN_July13_L2_AgeResults"
os.makedirs(RESULTS_ROOT, exist_ok=True)

# Master index of *all* runs (appended as you go)
MASTER_CSV = os.path.join(RESULTS_ROOT, "master_runs.csv")

#TARGETS = ["ALM", "BMD", "BFP", "Age"]

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

# ------------------------- Data loading for K-fold -------------------------
def load_penn_full_for_target(csv_path: str, target: str = "alm"):
    """
    Load and prepare the complete dataset. The actual train/validation split
    and scaler fitting are performed separately inside each K-fold iteration.
    """
    df = pd.read_csv(csv_path)
    name_map = {c.lower(): c for c in df.columns}

    target_l = target.lower()
    if target_l not in name_map:
        raise ValueError(f"CSV must contain '{target}' column (case-insensitive).")

    col_y = name_map[target_l]
    drop_cols = ['0', 'PPT ID', 'Site', 'Gender', 'Race']
    drop_in_df = [c for c in drop_cols if c in df.columns]

    # Prevent target leakage: remove every supervised target from X.
    all_target_cols = []
    for t in ["alm", "bmd", "bfp", "age"]:
        if t in name_map:
            all_target_cols.append(name_map[t])

    features = df.drop(columns=drop_in_df + all_target_cols, errors='ignore')
    labels = df[[col_y]].copy()

    X_df = pd.get_dummies(features, drop_first=False)
    feature_columns = X_df.columns.tolist()

    # Preserve the original missing-value behavior.
    for col in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = X_df[col].fillna(X_df[col].mean())
        else:
            X_df[col] = X_df[col].fillna(0)

    if pd.api.types.is_numeric_dtype(labels[col_y]):
        labels[col_y] = labels[col_y].fillna(labels[col_y].mean())

    X_all = X_df.values.astype(np.float32)
    y_all = labels.values.astype(np.float32)
    return X_all, y_all, feature_columns


def make_fold_data(X_all, y_all, train_idx, val_idx):
    """
    Fit StandardScaler only on the training portion of the current fold,
    then transform both training and validation portions.
    """
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

    return Xt, yt, Xv, yv, scaler_x, scaler_y

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

            msg = (h[row] - h[col]) * w.unsqueeze(1)
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
        Xv_t = torch.tensor(Xv, dtype=torch.float32, device=DEVICE)
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
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)

        h = model.encode(X_t)
        h = model.pconv(h, edge_indices, edge_weights, mu=mu)
        pred = model.out(h)

        return float(loss_fn(pred, y_t).item())

# ------------------------- 5-fold sweep (L2 only, no weights) -------------------------
def sweep_high_p(csv_path: str, overall_bar=None):
    base = _slugify(csv_path)
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    dataset_summary_rows = []

    for target in TARGETS:
        X_all, y_all, feature_columns = load_penn_full_for_target(
            csv_path, target=target
        )

        fold_splits = list(kfold.split(X_all))

        for knn_k in KNN_LIST:
            for p in HIGH_P_VALUES:
                for K in K_CHOICES_BY_P[p]:
                    combination_fold_rows = []

                    for fold_no, (train_idx, val_idx) in enumerate(fold_splits, start=1):
                        Xt, yt, Xv, yv, scaler_x, scaler_y = make_fold_data(
                            X_all, y_all, train_idx, val_idx
                        )

                        # Build independent graphs for the current fold.
                        ei_train, ew_train = _knn_edge_index_l2_unweighted(Xt, knn_k)
                        ei_val, ew_val = _knn_edge_index_l2_unweighted(Xv, knn_k)

                        ei_train = ei_train.to(DEVICE)
                        ei_val = ei_val.to(DEVICE)
                        if ew_train is not None:
                            ew_train = ew_train.to(DEVICE)
                        if ew_val is not None:
                            ew_val = ew_val.to(DEVICE)

                        eis_train, ews_train = [ei_train], [ew_train]
                        eis_val, ews_val = [ei_val], [ew_val]

                        out_root = os.path.join(
                            RESULTS_ROOT,
                            f"{base}_pgnn_target-{target.upper()}_{EDGE_METRIC}_"
                            f"fw-{FW_LABEL}_{N_SPLITS}fold_knn{knn_k}"
                        )
                        task_dir = os.path.join(out_root, target.lower(), f"fold_{fold_no}")
                        os.makedirs(task_dir, exist_ok=True)

                        y_train = yt[:, 0:1]
                        y_val = yv[:, 0:1]

                        data_train = Data(
                            x=torch.tensor(Xt, dtype=torch.float32, device=DEVICE),
                            edge_index=None,
                            y=torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
                        )

                        mu_init = mu_for(float(p), int(K))

                        # Reset seeds before every fold/run for reproducibility.
                        run_seed = SEED + fold_no
                        np.random.seed(run_seed)
                        torch.manual_seed(run_seed)

                        model = PGNNRegressor(
                            Xt.shape[1], HID, 1,
                            K=K, p=p, mu=mu_init, dropout=DROPOUT
                        ).to(DEVICE)
                        opt = torch.optim.AdamW(
                            model.parameters(), lr=LR_MIN, weight_decay=WD
                        )
                        loss_fn = nn.MSELoss(reduction='mean')

                        best_state = None
                        best_rel = np.full(1, np.inf)
                        epochs_no_improve = 0
                        current_mu = mu_init
                        T = max(EPOCHS, 50)
                        history_rows = []

                        epoch_bar = tqdm(
                            range(1, T + 1),
                            desc=(
                                f"{base} | {target} | kNN={knn_k} | "
                                f"p={p:g} | K={K} | fold={fold_no}/{N_SPLITS}"
                            ),
                            leave=False,
                            dynamic_ncols=True
                        )

                        for epoch in epoch_bar:
                            for g in opt.param_groups:
                                g['lr'] = one_cycle_lr(
                                    epoch, T, LR_MAX, LR_MIN
                                )

                            model.train()
                            opt.zero_grad()
                            out = model(
                                data_train, eis_train, ews_train,
                                mu=current_mu
                            )
                            loss = loss_fn(out, data_train.y)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_norm=1.0
                            )
                            opt.step()

                            train_mse_step = float(loss.item())
                            train_mse_eval = mse_on_split(
                                model, Xt, y_train,
                                eis_train, ews_train,
                                loss_fn, mu=current_mu
                            )
                            val_mse = mse_on_split(
                                model, Xv, y_val,
                                eis_val, ews_val,
                                loss_fn, mu=current_mu
                            )

                            history_rows.append({
                                "fold": fold_no,
                                "epoch": epoch,
                                "lr": float(opt.param_groups[0]["lr"]),
                                "mu": float(current_mu),
                                "train_mse_step": train_mse_step,
                                "train_mse_eval": train_mse_eval,
                                "val_mse": val_mse,
                            })

                            if epoch % 5 == 0:
                                eval_now = evaluate(
                                    model, Xt, y_train,
                                    eis_train, ews_train,
                                    scaler_y, target_idx=0
                                )
                                if np.median(eval_now["rel_rmse_pct"]) > 200:
                                    current_mu *= MU_BACKOFF
                                    for g in opt.param_groups:
                                        g['lr'] = max(
                                            g['lr'] * 0.5, LR_MIN
                                        )

                            metrics = evaluate(
                                model, Xv, y_val,
                                eis_val, ews_val,
                                scaler_y, target_idx=0
                            )
                            rel = metrics["rel_rmse_orig_pct"]
                            improved = (rel < best_rel).any()

                            epoch_bar.set_postfix(
                                loss=f"{train_mse_step:.4f}",
                                val=f"{val_mse:.4f}",
                                best_rel=f"{best_rel[0]:.3f}"
                                if np.isfinite(best_rel[0]) else "inf"
                            )

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

                        epoch_bar.close()

                        history_path = os.path.join(
                            task_dir,
                            f"history_fold{fold_no}_knn{knn_k}_{EDGE_METRIC}_"
                            f"fw-{FW_LABEL}_K{K}_p{p}_mu0{mu_init:.3g}.csv"
                        )
                        pd.DataFrame(history_rows).to_csv(
                            history_path, index=False
                        )

                        if best_state is not None:
                            model.load_state_dict(best_state["model"])

                        final = evaluate(
                            model, Xv, y_val,
                            eis_val, ews_val,
                            scaler_y, target_idx=0
                        )

                        model_mu = (
                            best_state["mu"] if best_state else mu_init
                        )
                        model_epoch = (
                            best_state["epoch"] if best_state else -1
                        )
                        model_lr = (
                            best_state["lr"] if best_state else LR_MIN
                        )

                        ckpt_path = os.path.join(
                            task_dir,
                            f"best_model_fold{fold_no}_knn{knn_k}_"
                            f"{EDGE_METRIC}_fw-{FW_LABEL}_K{K}_p{p}_"
                            f"mu{model_mu:.3g}.pt"
                        )
                        torch.save(
                            {
                                "state_dict": model.state_dict(),
                                "config": {
                                    "csv_path": csv_path,
                                    "target": target,
                                    "fold": fold_no,
                                    "n_splits": N_SPLITS,
                                    "train_indices": train_idx,
                                    "val_indices": val_idx,
                                    "knn_k": knn_k,
                                    "edge_metrics": [EDGE_METRIC],
                                    "fw_source": None,
                                    "K": K,
                                    "p": p,
                                    "mu": model_mu,
                                    "epoch": model_epoch,
                                    "lr": model_lr,
                                    "seed": run_seed,
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
                            f"results_fold{fold_no}_knn{knn_k}_{EDGE_METRIC}_"
                            f"fw-{FW_LABEL}_K{K}_p{p}_mu{model_mu:.3g}.csv"
                        )

                        fold_row = {
                            "dataset": base,
                            "csv_path": csv_path,
                            "target": target,
                            "fold": fold_no,
                            "n_splits": N_SPLITS,
                            "train_count": len(train_idx),
                            "val_count": len(val_idx),
                            "kNN_graph": knn_k,
                            "edge_metrics": EDGE_METRIC,
                            "fw_source": FW_LABEL,
                            "K": K,
                            "p": p,
                            "mu_init": mu_init,
                            "mu_used": model_mu,
                            "RMSE_scaled": final["rmse_scaled"][0],
                            "RelRMSE_scaled_pct": final["rel_rmse_pct"][0],
                            "RMSE_orig": final["rmse_orig"][0],
                            "RelRMSE_orig_pct": final["rel_rmse_orig_pct"][0],
                            "R2": final["r2"][0],
                            "best_epoch": model_epoch,
                            "best_lr": model_lr,
                            "history_csv": history_path,
                            "ckpt_path": ckpt_path,
                        }
                        pd.DataFrame([fold_row]).to_csv(
                            results_csv_path, index=False
                        )

                        master_row = dict(fold_row)
                        master_row.update({
                            "results_root": RESULTS_ROOT,
                            "results_csv": results_csv_path,
                        })
                        append_master_row(master_row)
                        combination_fold_rows.append(fold_row)

                        if overall_bar is not None:
                            overall_bar.update(1)
                            overall_bar.set_postfix(
                                dataset=base,
                                target=target,
                                fold=f"{fold_no}/{N_SPLITS}"
                            )

                    # Mean and standard deviation across the five folds
                    fold_df = pd.DataFrame(combination_fold_rows)
                    summary_row = {
                        "dataset": base,
                        "csv_path": csv_path,
                        "target": target,
                        "n_splits": N_SPLITS,
                        "kNN_graph": knn_k,
                        "K": K,
                        "p": p,
                        "mu_init": mu_for(float(p), int(K)),
                        "RMSE_scaled_mean": fold_df["RMSE_scaled"].mean(),
                        "RMSE_scaled_std": fold_df["RMSE_scaled"].std(ddof=1),
                        "RelRMSE_scaled_pct_mean":
                            fold_df["RelRMSE_scaled_pct"].mean(),
                        "RelRMSE_scaled_pct_std":
                            fold_df["RelRMSE_scaled_pct"].std(ddof=1),
                        "RMSE_orig_mean": fold_df["RMSE_orig"].mean(),
                        "RMSE_orig_std": fold_df["RMSE_orig"].std(ddof=1),
                        "RelRMSE_orig_pct_mean":
                            fold_df["RelRMSE_orig_pct"].mean(),
                        "RelRMSE_orig_pct_std":
                            fold_df["RelRMSE_orig_pct"].std(ddof=1),
                        "R2_mean": fold_df["R2"].mean(),
                        "R2_std": fold_df["R2"].std(ddof=1),
                        "best_epoch_mean": fold_df["best_epoch"].mean(),
                    }
                    dataset_summary_rows.append(summary_row)

    dataset_summary_path = os.path.join(
        RESULTS_ROOT, f"{base}_5fold_summary.csv"
    )
    pd.DataFrame(dataset_summary_rows).to_csv(
        dataset_summary_path, index=False
    )
    return dataset_summary_rows

# ------------------------- Main -------------------------
def main():
    overall_start_time = time.perf_counter()

    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    csvs = [
        "/scratch/gsunka1/TDA_guoji/male.csv",
        "/scratch/gsunka1/TDA_guoji/female.csv",
        "/scratch/gsunka1/TDA_guoji/penn_data.csv"
    ]

    combinations_per_dataset = (
        len(TARGETS)
        * len(KNN_LIST)
        * sum(len(K_CHOICES_BY_P[p]) for p in HIGH_P_VALUES)
        * N_SPLITS
    )
    total_runs = len(csvs) * combinations_per_dataset

    all_summary_rows = []
    with tqdm(
        total=total_runs,
        desc="Complete 5-fold PGNN sweep",
        unit="run",
        dynamic_ncols=True
    ) as overall_bar:
        for path in csvs:
            print(f"\n=== Running {N_SPLITS}-fold PGNN sweep for: {path} ===")
            all_summary_rows.extend(
                sweep_high_p(path, overall_bar=overall_bar)
            )
            print(f"Finished: {path}")

    combined_summary_path = os.path.join(
        RESULTS_ROOT, "all_datasets_5fold_summary.csv"
    )

    combined_summary_df = pd.DataFrame(all_summary_rows)
    combined_summary_df.to_csv(
        combined_summary_path, index=False
    )

    # Select one best average K-fold configuration for each dataset and target.
    # Primary criterion: lowest mean original-scale relative RMSE.
    # Tie-breakers: lowest mean original-scale RMSE, then highest mean R2.
    best_average_kfold_df = (
        combined_summary_df.sort_values(
            by=[
                "dataset",
                "target",
                "RelRMSE_orig_pct_mean",
                "RMSE_orig_mean",
                "R2_mean",
            ],
            ascending=[True, True, True, True, False],
        )
        .groupby(["dataset", "target"], as_index=False, sort=False)
        .first()
    )

    best_average_kfold_columns = [
        "dataset",
        "csv_path",
        "target",
        "n_splits",
        "kNN_graph",
        "p",
        "K",
        "mu_init",
        "RMSE_scaled_mean",
        "RMSE_scaled_std",
        "RelRMSE_scaled_pct_mean",
        "RelRMSE_scaled_pct_std",
        "RMSE_orig_mean",
        "RMSE_orig_std",
        "RelRMSE_orig_pct_mean",
        "RelRMSE_orig_pct_std",
        "R2_mean",
        "R2_std",
        "best_epoch_mean",
    ]
    best_average_kfold_df = best_average_kfold_df[
        best_average_kfold_columns
    ]

    best_average_kfold_path = os.path.join(
        RESULTS_ROOT, "best_average_kfold_each_dataset_target.csv"
    )
    best_average_kfold_df.to_csv(
        best_average_kfold_path, index=False
    )

    # Total wall-clock time for the complete workflow:
    # all datasets, all targets, all hyperparameter combinations, and all folds.
    overall_end_time = time.perf_counter()
    total_time_seconds = overall_end_time - overall_start_time
    total_time_minutes = total_time_seconds / 60.0
    total_time_hours = total_time_seconds / 3600.0

    hours = int(total_time_seconds // 3600)
    minutes = int((total_time_seconds % 3600) // 60)
    seconds = total_time_seconds % 60
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    runtime_summary_path = os.path.join(
        RESULTS_ROOT, "overall_runtime_summary.csv"
    )

    runtime_summary = pd.DataFrame([{
        "results_root": RESULTS_ROOT,
        "device": str(DEVICE),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if DEVICE.type == "cuda"
            else "CPU"
        ),
        "datasets_completed": len(csvs),
        "targets_per_dataset": len(TARGETS),
        "k_folds": N_SPLITS,
        "total_fold_runs": total_runs,
        "total_time_seconds": total_time_seconds,
        "total_time_minutes": total_time_minutes,
        "total_time_hours": total_time_hours,
        "formatted_time_hh_mm_ss": formatted_time,
    }])
    runtime_summary.to_csv(runtime_summary_path, index=False)

    print(f"\nCombined K-fold summary saved to: {combined_summary_path}")
    print(
        "Best average K-fold runs saved to: "
        f"{best_average_kfold_path}"
    )
    print(f"Master fold-level results saved to: {MASTER_CSV}")
    print(
        f"Overall time for all tasks: {hours} hours, "
        f"{minutes} minutes, {seconds:.3f} seconds"
    )
    print(f"Overall runtime summary saved to: {runtime_summary_path}")


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    main()

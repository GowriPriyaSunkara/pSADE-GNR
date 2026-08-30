
#!/usr/bin/env python3
import os, re, math, json, copy, time
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

# Single root folder for all results
RESULTS_ROOT = "PGNN_RESULTS_L2_cc"
os.makedirs(RESULTS_ROOT, exist_ok=True)

# Which supervised target(s) to run? ,"BMD","BFP","Age"
TARGETS = ["ALM", "BMD","BFP"]

# kNN values to sweep for graph construction
KNN_LIST = [2, 5, 10, 15, 20, 25, 30, 35, 40]

# p/K sweeps
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

# Run plan: no weights, |corr|, |corr|^2 (for ALM/BMD/BFP)
RUNS = [
    # plain L2 (no correlation weights)
    {"edge_metrics": ["l2"], "fw_source": None,  "fw_power": 0.0,
     "fw_abs": False, "fw_norm": False, "fw_label": "none"},

    # ALM-based weights: |corr|, |corr|^2
    {"edge_metrics": ["l2"], "fw_source": "alm", "fw_power": 1.0,
     "fw_abs": True,  "fw_norm": True,  "fw_label": "alm_c"},
    {"edge_metrics": ["l2"], "fw_source": "alm", "fw_power": 2.0,
     "fw_abs": True,  "fw_norm": True,  "fw_label": "alm_c2"},

    # BMD-based weights
    {"edge_metrics": ["l2"], "fw_source": "bmd", "fw_power": 1.0,
     "fw_abs": True,  "fw_norm": True,  "fw_label": "bmd_c"},
    {"edge_metrics": ["l2"], "fw_source": "bmd", "fw_power": 2.0,
     "fw_abs": True,  "fw_norm": True,  "fw_label": "bmd_c2"},

    # BFP-based weights
    {"edge_metrics": ["l2"], "fw_source": "bfp", "fw_power": 1.0,
     "fw_abs": True,  "fw_norm": True,  "fw_label": "bfp_c"},
    {"edge_metrics": ["l2"], "fw_source": "bfp", "fw_power": 2.0,
     "fw_abs": True,  "fw_norm": True,  "fw_label": "bfp_c2"},
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
def _slugify(name: str) -> str:
    s = os.path.splitext(os.path.basename(name))[0]
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

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
    """
    Build kNN graph with continuous edge weights using:

      d(i,j;c) = sqrt( sum_k c_k (X_ik - X_jk)^2 ) / ||c||_beta
      w_ij     = exp( -alpha * d(i,j;c) )
    """
    n, d = X.shape

    # binary kNN for connectivity
    A = kneighbors_graph(X, k, mode='connectivity', metric='euclidean',
                         include_self=False)
    coo_mut = _mutualize(A.tocsr())
    row = coo_mut.row
    col = coo_mut.col

    if row.size == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)
        return edge_index, edge_weight

    # feature weights c_k
    if fw_used is not None:
        c = np.asarray(fw_used, dtype=float)
        if c.shape[0] != d:
            raise ValueError(f"fw_used length {c.shape[0]} != feature dim {d}")
    else:
        c = np.ones(d, dtype=float)

    # numerator sqrt(sum_k c_k (X_ik - X_jk)^2)
    diff = X[row] - X[col]           # (E, d)
    num = (diff**2) * c[None, :]     # (E, d)
    num = num.sum(axis=1)            # (E,)
    num = np.sqrt(np.maximum(num, 1e-12))

    # denominator ||c||_beta
    c_norm = beta_norm(c, beta=beta)
    d_ij = num / c_norm              # (E,)

    # alpha scale from mean distance
    mean_d = float(d_ij.mean()) if d_ij.size > 0 else 1.0
    if mean_d <= 1e-8:
        alpha = 1.0
    else:
        alpha = 1.0 / mean_d

    w_ij = np.exp(-alpha * d_ij).astype(np.float32)  # (E,)

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
    """
    Build (edge_index, edge_weight) for each named metric.
    Currently only 'l2' is supported, with continuous weights.
    """
    name_map = {
        'l2': _knn_edge_index_l2_weighted,
    }
    eis = []
    ews = []
    for nm in names:
        if nm not in name_map:
            raise ValueError(f"Unknown graph metric '{nm}'. Choose from: {list(name_map.keys())}")
        ei, ew = name_map[nm](X, k, fw_used=fw_used)
        eis.append(ei)
        ews.append(ew)
    return eis, ews

# ------------------------- Feature weighting (correlations) -------------------------
def compute_feature_weights_by_label(X_train: np.ndarray,
                                     y_train_1d: np.ndarray,
                                     power: float = 1.0,
                                     use_abs: bool = True,
                                     normalize: bool = True) -> np.ndarray:
    """
    Corr-based feature weights:
      corr_j = corr(X[:,j], y)
      w_j    = |corr_j|^power   (if use_abs)
             = (corr_j)^power   (if not use_abs)
    """
    coefs = np.array([
        np.corrcoef(X_train[:, j], y_train_1d)[0, 1] for j in range(X_train.shape[1])
    ])
    coefs = np.nan_to_num(coefs, nan=0.0)

    if use_abs:
        w = np.abs(coefs)
    else:
        w = coefs

    if power != 1.0:
        w = np.sign(w) * (np.abs(w) ** power) if not use_abs else (w ** power)

    if normalize:
        mean_w = w.mean() if w.size > 0 else 1.0
        w = w / (mean_w + 1e-12)

    return w

# ------------------------- Data loading -------------------------
def load_penn_for_target(csv_path: str,
                         k: int,
                         target: str = "alm",
                         fw_source: Optional[str] = None,
                         fw_power: float = 1.0,
                         fw_abs: bool = True,
                         fw_norm: bool = True,
                         train_size: float = TRAIN_SIZE,
                         seed: int = SEED):
    df = pd.read_csv(csv_path)
    name_map = {c.lower(): c for c in df.columns}

    target = target.lower()
    if target not in name_map:
        raise ValueError(f"CSV must contain '{target}' column (case-insensitive).")

    col_y = name_map[target]
    drop_cols = ['0', 'PPT ID', 'Site', 'Gender', 'Race']
    drop_in_df = [c for c in drop_cols if c in df.columns]

    features = df.drop(columns=drop_in_df + [col_y], errors='ignore')
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

    fw_used = None

    if fw_source is not None:
        src = fw_source.lower()
        if src not in ['alm','bmd','bfp']:
            raise ValueError("fw_source must be one of {alm,bmd,bfp}.")
        if src not in name_map:
            raise ValueError(f"fw_source '{src}' not found in CSV.")

        src_y = df[[name_map[src]]].copy()
        src_y = src_y.fillna(src_y.mean())
        _, _, src_train_raw, _ = train_test_split(
            X_df.values, src_y.values, train_size=train_size, random_state=seed
        )
        src_scaler = StandardScaler().fit(src_train_raw)
        src_train = src_scaler.transform(src_train_raw)[:, 0]

        fw = compute_feature_weights_by_label(
            Xt, src_train, power=fw_power, use_abs=fw_abs, normalize=fw_norm
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
            diff = h[row] - h[col]                         # (E, d)
            norm_diff = torch.norm(diff, dim=1).clamp(min=1e-6)  # (E,)
            plap_w = (norm_diff / norm_diff.max()).pow(self.p - 2)
            if edge_weight is not None:
                w = plap_w * edge_weight
            else:
                w = plap_w

            agg = torch.zeros_like(h)
            for d in range(h.size(1)):
                agg[:, d].index_add_(0, row, w * (h[col, d] - h[row, d]))

            deg = torch.zeros(h.size(0), device=h.device)
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
        cos = (1 + math.cos(math.cos(math.pi * (1 - t/half)))) / 2 if half > 0 else 1.0
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
        Xv_t = torch.tensor(Xv, dtype=torch.float32)
        h = model.encode(Xv_t)
        val_out = model.pconv(h, edge_indices, edge_weights)
        yp_scaled = model.out(val_out).cpu().numpy()

    if yv.ndim == 1:
        y_true_scaled = yv.reshape(-1, 1)
    else:
        y_true_scaled = yv
    if yp_scaled.ndim == 1:
        yp_scaled = yp_scaled.reshape(-1, 1)

    yp_orig   = inverse_transform_y(scaler_y, yp_scaled,   target_idx=target_idx)
    y_true_or = inverse_transform_y(scaler_y, y_true_scaled, target_idx=target_idx)

    # scaled-space metrics (unweighted)
    sq_err_scaled = (y_true_scaled - yp_scaled) ** 2
    mse_scaled    = sq_err_scaled.mean(axis=0)
    rmse_scaled   = np.sqrt(mse_scaled)

    rms_true_sc   = np.sqrt((y_true_scaled ** 2).mean(axis=0)).clip(min=1e-12)
    rel_rmse_pct  = 100.0 * rmse_scaled / rms_true_sc

    # original-scale metrics (unweighted)
    sq_err_orig    = (y_true_or - yp_orig) ** 2
    mse_orig       = sq_err_orig.mean(axis=0)
    rmse_orig      = np.sqrt(mse_orig)

    rms_true_or    = np.sqrt((y_true_or ** 2).mean(axis=0)).clip(min=1e-12)
    rel_rmse_orig_pct = 100.0 * rmse_orig / rms_true_or

    from sklearn.metrics import r2_score as _r2
    r2 = _r2(y_true_scaled, yp_scaled, multioutput='raw_values')

    return {
        "rmse_scaled": rmse_scaled,
        "rel_rmse_pct": rel_rmse_pct,
        "rmse_orig": rmse_orig,
        "rel_rmse_orig_pct": rel_rmse_orig_pct,
        "r2": r2,
        "yp_scaled": yp_scaled,
    }

# ------------------------- Sweep -------------------------
def sweep_high_p(csv_path: str):
    base = _slugify(csv_path)
    train_pct = int(round(TRAIN_SIZE * 100))

    for knn_k in KNN_LIST:
        for run in RUNS:
            edge_metrics = run["edge_metrics"]
            fw_source    = run["fw_source"]
            fw_power     = run["fw_power"]
            fw_abs       = run["fw_abs"]
            fw_norm      = run["fw_norm"]
            fw_label     = run["fw_label"]

            for target in TARGETS:
                Xt, yt, Xv, yv, scaler_x, scaler_y, fw_used, feature_columns = load_penn_for_target(
                    csv_path, k=knn_k, target=target,
                    fw_source=fw_source,
                    fw_power=fw_power,
                    fw_abs=fw_abs,
                    fw_norm=fw_norm,
                    train_size=TRAIN_SIZE,
                    seed=SEED
                )

                eis_train, ews_train = build_edge_graphs_by_names(
                    Xt, knn_k, edge_metrics, fw_used=fw_used
                )
                eis_val,   ews_val   = build_edge_graphs_by_names(
                    Xv, knn_k, edge_metrics, fw_used=fw_used
                )

                metric_tag = "+".join(edge_metrics)
                fw_tag = fw_label

                out_root = os.path.join(
                    RESULTS_ROOT,
                    f"{base}_pgnn_target-{target.upper()}_{metric_tag}_fw-{fw_tag}_{train_pct}pct_knn{knn_k}"
                )
                os.makedirs(out_root, exist_ok=True)
                task_dir = os.path.join(out_root, target.lower())
                os.makedirs(task_dir, exist_ok=True)

                # save feature-weight vector (corr or corr^2) for ALM/BMD/BFP runs
                if fw_used is not None and fw_source in ["alm","bmd","bfp"]:
                    fw_save_path = os.path.join(
                        task_dir,
                        f"feature_weights_{fw_source}_power{int(fw_power)}_knn{knn_k}.npy"
                    )
                    np.save(fw_save_path, fw_used)

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
                        mu = mu_for(float(p), int(K))

                        model = PGNNRegressor(
                            Xt.shape[1], HID, out_dim,
                            K=K, p=p, mu=mu, dropout=DROPOUT
                        )
                        opt = torch.optim.Adam(
                            model.parameters(),
                            lr=LR_MIN,
                            weight_decay=WD
                        )
                        loss_fn = nn.MSELoss(reduction='mean')

                        best_state = None
                        best_rel   = np.full(out_dim, np.inf)
                        epochs_no_improve = 0

                        T = max(EPOCHS, 50)
                        current_mu = mu

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

                            # heuristic mu backoff on exploding RelRMSE (optional)
                            if epoch % 5 == 0:
                                eval_now = evaluate(
                                    model, Xt, y_train, eis_train, ews_train,
                                    scaler_y, target_idx=0
                                )
                                if np.median(eval_now["rel_rmse_pct"]) > 200:
                                    current_mu *= MU_BACKOFF
                                    for g in opt.param_groups:
                                        g['lr'] = max(g['lr']*0.5, LR_MIN)

                            metrics = evaluate(
                                model, Xv, y_val, eis_val, ews_val,
                                scaler_y, target_idx=0
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
                                    "lr": opt.param_groups[0]['lr'],
                                }
                                epochs_no_improve = 0
                            else:
                                epochs_no_improve += 1
                                if epochs_no_improve >= PATIENCE:
                                    break

                        if best_state is not None:
                            model.load_state_dict(best_state["model"])

                        final = evaluate(
                            model, Xv, y_val, eis_val, ews_val,
                            scaler_y, target_idx=0
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

                        # --------- CSV: only the requested columns ---------
                        pd.DataFrame({
                            'target':            [target],
                            'kNN_graph':         [knn_k],
                            'edge_metrics':      [metric_tag],
                            'fw_source':         [fw_tag],
                            'K':                 [K],
                            'p':                 [p],
                            'mu_used':           [model_mu],
                            'RelRMSE_orig_pct':  [final["rel_rmse_orig_pct"][0]],
                            'R2':                [final["r2"][0]],
                            'best_epoch':        [model_epoch],
                            'best_lr':           [model_lr],
                        }).to_csv(
                            os.path.join(
                                task_dir,
                                f"results_knn{knn_k}_{metric_tag}_fw-{fw_tag}_K{K}_p{p}_mu{model_mu:.3g}.csv"
                            ),
                            index=False
                        )

# ------------------------- Main -------------------------
def main():
    csvs = [
        "/male.csv",
        "/female.csv",
        "/penn_data.csv",
    ]
    for path in csvs:
        print(f"\n=== Running PGNN sweep for: {path} ===")
        sweep_high_p(path)
        print(f"Finished: {path}")

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    main()

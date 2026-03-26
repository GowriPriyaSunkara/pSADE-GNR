
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

# =========================================================
# CONFIG
# =========================================================
# --- PGNN training ---
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

RESULTS_ROOT = "PGNN_RESULTS_VAE"
os.makedirs(RESULTS_ROOT, exist_ok=True)

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

# --- VAE model reduction (stage 1) ---
USE_VAE          = True
VAE_LATENT_DIM   = 10
VAE_BETA         = 0.0
VAE_LR           = 1e-3
VAE_WD           = 1e-4
VAE_BATCH_SIZE   = 256
VAE_MAX_EPOCHS   = 300
VAE_PATIENCE     = 30
VAE_MASK_PROB    = 0.10
VAE_GAUSS_STD    = 0.0

# Cache to avoid retraining VAE repeatedly (per csv/split/config)
VAE_CACHE = {}  # key -> (z_train, z_val)

# =========================================================
# Data wrapper (PyG-compatible if available)
# =========================================================
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

# =========================================================
# Utils
# =========================================================
def _slugify(name: str) -> str:
    s = os.path.splitext(os.path.basename(name))[0]
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def append_master_row(row: dict, master_csv_path: str = MASTER_CSV):
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(master_csv_path)
    df_row.to_csv(master_csv_path, mode="a", header=write_header, index=False)

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mu_for(p: float, K: int) -> float:
    """
    Heuristic for mu based on p and K.
    """
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

# =========================================================
# Graph builders (L2, unweighted)
# =========================================================
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

# =========================================================
# Data loading (no fw weights)
# =========================================================
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

# =========================================================
# VAE for model reduction
# =========================================================
class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims=(128, 64)):
        super().__init__()
        h1, h2 = hidden_dims

        self.enc = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.GELU(),
            nn.Linear(h1, h2),
            nn.GELU(),
        )
        self.mu     = nn.Linear(h2, latent_dim)
        self.logvar = nn.Linear(h2, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.GELU(),
            nn.Linear(h2, h1),
            nn.GELU(),
            nn.Linear(h1, input_dim),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar, z

def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl

def apply_denoising(x, mask_prob=0.0, gaussian_std=0.0):
    if mask_prob > 0:
        mask = (torch.rand_like(x) < mask_prob).float()
        x = x * (1.0 - mask)
    if gaussian_std > 0:
        x = x + gaussian_std * torch.randn_like(x)
    return x

def train_vae_and_embed(
    Xt: np.ndarray, Xv: np.ndarray,
    latent_dim: int = 10,
    beta: float = 1.0,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 300,
    patience: int = 30,
    mask_prob: float = 0.10,
    gaussian_std: float = 0.0,
    seed: int = 42,
    device: Optional[str] = None,
):
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VAE(input_dim=Xt.shape[1], latent_dim=latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xt_t = torch.tensor(Xt, dtype=torch.float32)
    Xv_t = torch.tensor(Xv, dtype=torch.float32)

    tr_loader = torch.utils.data.DataLoader(Xt_t, batch_size=batch_size, shuffle=True)
    va_loader = torch.utils.data.DataLoader(Xv_t, batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None
    patience_left = patience

    @torch.no_grad()
    def eval_val():
        model.eval()
        vals = []
        for x in va_loader:
            x = x.to(device)
            x_hat, mu, logvar, _ = model(x)
            loss, _, _ = vae_loss(x_hat, x, mu, logvar, beta=beta)
            vals.append(loss.item())
        return float(np.mean(vals)) if vals else float("inf")

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x in tr_loader:
            x = x.to(device)
            x_in = apply_denoising(x, mask_prob=mask_prob, gaussian_std=gaussian_std)
            x_hat, mu, logvar, _ = model(x_in)
            loss, _, _ = vae_loss(x_hat, x, mu, logvar, beta=beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        val_loss = eval_val()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Export embeddings using mu (deterministic)
    model.eval()
    with torch.no_grad():
        mu_tr, _ = model.encode(Xt_t.to(device))
        mu_va, _ = model.encode(Xv_t.to(device))
        z_tr = mu_tr.cpu().numpy()
        z_va = mu_va.cpu().numpy()

    return z_tr, z_va, model

# =========================================================
# Model: p-Laplacian Conv + PGNN regressor
# =========================================================
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

# =========================================================
# Train/Eval helpers
# =========================================================
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
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        h = model.encode(X_t)
        h = model.pconv(h, edge_indices, edge_weights, mu=mu)
        pred = model.out(h)

        return float(loss_fn(pred, y_t).item())

# =========================================================
# Sweep (L2 only, no weights) + VAE reduction
# =========================================================
def sweep_high_p(csv_path: str):
    base = _slugify(csv_path)
    train_pct = int(round(TRAIN_SIZE * 100))

    for knn_k in KNN_LIST:
        for target in TARGETS:
            Xt, yt, Xv, yv, scaler_x, scaler_y, feature_columns = load_penn_for_target(
                csv_path, target=target, train_size=TRAIN_SIZE, seed=SEED
            )

            # ---- VAE reduction: train on Xt, validate on Xv (same split as PGNN!) ----
            if USE_VAE:
                cache_key = (csv_path, TRAIN_SIZE, SEED, VAE_LATENT_DIM, VAE_BETA, VAE_MASK_PROB, VAE_GAUSS_STD)
                if cache_key not in VAE_CACHE:
                    zt, zv, _ = train_vae_and_embed(
                        Xt, Xv,
                        latent_dim=VAE_LATENT_DIM,
                        beta=VAE_BETA,
                        batch_size=VAE_BATCH_SIZE,
                        lr=VAE_LR,
                        weight_decay=VAE_WD,
                        max_epochs=VAE_MAX_EPOCHS,
                        patience=VAE_PATIENCE,
                        mask_prob=VAE_MASK_PROB,
                        gaussian_std=VAE_GAUSS_STD,
                        seed=SEED,
                    )
                    VAE_CACHE[cache_key] = (zt, zv)

                    # optional: persist embeddings once per dataset/split/config
                    vae_dir = os.path.join(
                        RESULTS_ROOT, "VAE_EMBEDS", base,
                        f"z{VAE_LATENT_DIM}_beta{VAE_BETA}_mask{VAE_MASK_PROB}_g{VAE_GAUSS_STD}"
                    )
                    os.makedirs(vae_dir, exist_ok=True)
                    np.save(os.path.join(vae_dir, "z_train.npy"), zt)
                    np.save(os.path.join(vae_dir, "z_val.npy"), zv)
                else:
                    zt, zv = VAE_CACHE[cache_key]

                Xt_use, Xv_use = zt, zv
            else:
                Xt_use, Xv_use = Xt, Xv

            # ---- build kNN graphs on Xt_use/Xv_use ----
            ei_train, ew_train = _knn_edge_index_l2_unweighted(Xt_use, knn_k)
            ei_val,   ew_val   = _knn_edge_index_l2_unweighted(Xv_use, knn_k)

            eis_train, ews_train = [ei_train], [ew_train]
            eis_val,   ews_val   = [ei_val],   [ew_val]

            out_root = os.path.join(
                RESULTS_ROOT,
                f"{base}_pgnn_target-{target.upper()}_{EDGE_METRIC}_fw-{FW_LABEL}_{train_pct}pct_knn{knn_k}"
                + (f"_VAEz{VAE_LATENT_DIM}" if USE_VAE else "")
            )
            os.makedirs(out_root, exist_ok=True)
            task_dir = os.path.join(out_root, target.lower())
            os.makedirs(task_dir, exist_ok=True)

            out_dim = 1
            y_train = yt[:, 0:1]
            y_val   = yv[:, 0:1]

            data_train = Data(
                x=torch.tensor(Xt_use, dtype=torch.float32),
                edge_index=None,
                y=torch.tensor(y_train, dtype=torch.float32)
            )

            for p in HIGH_P_VALUES:
                p_key = float(p)
                if p_key not in K_CHOICES_BY_P:
                    continue

                for K in K_CHOICES_BY_P[p_key]:
                    mu_init = mu_for(float(p), int(K))

                    model = PGNNRegressor(
                        Xt_use.shape[1], HID, out_dim,
                        K=K, p=p, mu=mu_init, dropout=DROPOUT
                    )
                    opt = torch.optim.Adam(model.parameters(), lr=LR_MIN, weight_decay=WD)
                    loss_fn = nn.MSELoss(reduction='mean')

                    best_state = None
                    best_rel   = np.full(out_dim, np.inf)
                    epochs_no_improve = 0

                    T = max(EPOCHS, 50)
                    current_mu = mu_init

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

                        # log losses every epoch
                        train_mse_step = float(loss.item())
                        train_mse_eval = mse_on_split(model, Xt_use, y_train, eis_train, ews_train, loss_fn, mu=current_mu)
                        val_mse        = mse_on_split(model, Xv_use, y_val,   eis_val,   ews_val,   loss_fn, mu=current_mu)

                        history_rows.append({
                            "epoch": epoch,
                            "lr": float(opt.param_groups[0]["lr"]),
                            "mu": float(current_mu),
                            "train_mse_step": train_mse_step,
                            "train_mse_eval": train_mse_eval,
                            "val_mse": val_mse,
                        })

                        # mu-backoff heuristic
                        if epoch % 5 == 0:
                            eval_now = evaluate(model, Xt_use, y_train, eis_train, ews_train, scaler_y, target_idx=0)
                            if np.median(eval_now["rel_rmse_pct"]) > 200:
                                current_mu *= MU_BACKOFF
                                for g in opt.param_groups:
                                    g['lr'] = max(g['lr']*0.5, LR_MIN)

                        # early stopping uses val RelRMSE_orig_pct
                        metrics = evaluate(model, Xv_use, y_val, eis_val, ews_val, scaler_y, target_idx=0)
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

                    # save per-epoch history
                    history_path = os.path.join(
                        task_dir,
                        f"history_knn{knn_k}_{EDGE_METRIC}_fw-{FW_LABEL}_K{K}_p{p}_mu0{mu_init:.3g}"
                        + (f"_VAEz{VAE_LATENT_DIM}.csv" if USE_VAE else ".csv")
                    )
                    pd.DataFrame(history_rows).to_csv(history_path, index=False)

                    # restore best
                    if best_state is not None:
                        model.load_state_dict(best_state["model"])

                    final = evaluate(model, Xv_use, y_val, eis_val, ews_val, scaler_y, target_idx=0)

                    model_mu = best_state["mu"] if best_state else mu_init
                    model_epoch = best_state["epoch"] if best_state else -1
                    model_lr = best_state["lr"] if best_state else LR_MIN

                    ckpt_path = os.path.join(
                        task_dir,
                        f"best_model_knn{knn_k}_{EDGE_METRIC}_fw-{FW_LABEL}_K{K}_p{p}_mu{model_mu:.3g}"
                        + (f"_VAEz{VAE_LATENT_DIM}.pt" if USE_VAE else ".pt")
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
                                "use_vae": USE_VAE,
                                "vae_latent_dim": VAE_LATENT_DIM if USE_VAE else None,
                                "vae_beta": VAE_BETA if USE_VAE else None,
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
                        f"results_knn{knn_k}_{EDGE_METRIC}_fw-{FW_LABEL}_K{K}_p{p}_mu{model_mu:.3g}"
                        + (f"_VAEz{VAE_LATENT_DIM}.csv" if USE_VAE else ".csv")
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
                        "use_vae":          [USE_VAE],
                        "vae_latent_dim":   [VAE_LATENT_DIM if USE_VAE else None],
                        "vae_beta":         [VAE_BETA if USE_VAE else None],
                    }).to_csv(results_csv_path, index=False)

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
                        "use_vae": bool(USE_VAE),
                        "vae_latent_dim": int(VAE_LATENT_DIM) if USE_VAE else None,
                        "vae_beta": float(VAE_BETA) if USE_VAE else None,
                    })

# =========================================================

# =========================================================
def main():
    set_seed(SEED)
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
    main()

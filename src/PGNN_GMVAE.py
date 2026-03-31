#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.sparse import coo_matrix, csr_matrix


# ============================================================
# CONFIG-"ALM", "BMD", "BFP", "Age"
# ============================================================

CSV_LIST = [
    "/scratch/gsunka1/TDA_guoji/male.csv",
    "/scratch/gsunka1/TDA_guoji/female.csv",
    "/scratch/gsunka1/TDA_guoji/penn_data.csv",
]

RESULTS_ROOT = "GMVAE_PGNN_Mar28_RESULTS_age2wo"
os.makedirs(RESULTS_ROOT, exist_ok=True)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_SIZE = 0.80
BATCH_SIZE = 64

ID_COLUMNS = ["row_id", "ID", "id", "subject_id", "SubjectID", "PPT ID"]
NON_FEATURE_DROP = ["Site", "Gender", "Race", "0"]
TARGET_COLUMNS = ["Age"]

# ---------------- GMVAE ----------------
N_CLUSTERS = 3
LATENT_DIM = 15
GMVAE_HIDDEN_DIMS = [128, 64]
GMVAE_DROPOUT = 0.2
GMVAE_EPOCHS = 300
GMVAE_PATIENCE = 40
GMVAE_LR = 1e-3
GMVAE_WEIGHT_DECAY = 1e-5
BETA_KL_Z_MAX = 0
BETA_KL_C_MAX = 0.01
KL_WARMUP_EPOCHS = 50
LAMBDA_REG = 5.0
LAMBDA_ENTROPY = 0.001
#lamda reg=2.0, target weight -age=1.0
#TARGET_WEIGHTS_DICT = {
  #  "ALM": 1.0,
  #  "BMD": 1.0,
 #   "BFP": 2.0,
 #   "Age": 2.0
#}

TARGET_WEIGHTS_DICT = {
     "Age": 2.0
}

GMVAE_MODEL_SELECTION = "val_reg"  # "val_reg" or "val_total"

# ---------------- PGNN ----------------
PGNN_EPOCHS = 600
PGNN_PATIENCE = 30
LR_MAX = 0.01
LR_MIN = 1e-4
WD = 1e-4
HID = 128
PGNN_DROPOUT = 0.20
ADD_SELF_LOOPS = True
INNER_ITERS = 5
MU_BACKOFF = 0.5

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

DPI = 300


# ============================================================
# SEED
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# DATA WRAPPER
# ============================================================

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


class PennDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


# ============================================================
# HELPERS
# ============================================================

def pick_existing_columns(df: pd.DataFrame, cols):
    return [c for c in cols if c in df.columns]


def build_mlp(input_dim, hidden_dims, output_dim, dropout=0.0, use_batchnorm=True):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p):
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (
        logvar_p - logvar_q
        + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8)
        - 1.0
    )
    return kl.sum(dim=-1)


def kl_anneal(epoch, warmup_epochs, max_beta):
    return min(max_beta, (epoch / float(max(warmup_epochs, 1))) * max_beta)


def _slugify(name: str) -> str:
    s = os.path.splitext(os.path.basename(name))[0]
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def make_target_weights(target_cols):
    if len(target_cols) == 0:
        return None
    weights = [TARGET_WEIGHTS_DICT.get(col, 1.0) for col in target_cols]
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def save_current_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def one_cycle_lr(t, T, lr_max=LR_MAX, lr_min=LR_MIN):
    half = T // 2
    if t <= half:
        cos = (1 + math.cos(math.pi * (1 - t / max(half, 1)))) / 2
        return lr_min + (lr_max - lr_min) * cos
    else:
        cos = (1 + math.cos(math.pi * ((t - half) / max(T - half, 1)))) / 2
        return lr_min + (lr_max - lr_min) * cos


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


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset_for_gmvae(csv_path, train_size=TRAIN_SIZE, seed=SEED):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in NON_FEATURE_DROP if c in df.columns], errors="ignore")

    numeric_df = df.select_dtypes(include=[np.number]).copy()

    existing_ids = pick_existing_columns(numeric_df, ID_COLUMNS)
    existing_targets = pick_existing_columns(numeric_df, TARGET_COLUMNS)

    if len(existing_targets) == 0:
        raise ValueError(f"No target columns found in {csv_path}. Expected some of {TARGET_COLUMNS}")

    feature_cols = [c for c in numeric_df.columns if c not in set(existing_ids + existing_targets)]
    if len(feature_cols) == 0:
        raise ValueError(f"No numeric feature columns remain in {csv_path}")

    X = numeric_df[feature_cols].copy()
    Y = numeric_df[existing_targets].copy()

    X = X.fillna(X.median())
    Y = Y.fillna(Y.median())

    train_idx, val_idx = train_test_split(
        np.arange(len(X)),
        train_size=train_size,
        random_state=seed,
        shuffle=True
    )

    X_train_raw = X.iloc[train_idx].values
    X_val_raw = X.iloc[val_idx].values
    Y_train_raw = Y.iloc[train_idx].values
    Y_val_raw = Y.iloc[val_idx].values

    scaler_x = StandardScaler().fit(X_train_raw)
    scaler_y = StandardScaler().fit(Y_train_raw)

    X_train = scaler_x.transform(X_train_raw)
    X_val = scaler_x.transform(X_val_raw)
    Y_train = scaler_y.transform(Y_train_raw)
    Y_val = scaler_y.transform(Y_val_raw)

    meta = {
        "feature_cols": feature_cols,
        "target_cols": existing_targets,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "n_total": len(X),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }
    return X_train, X_val, Y_train, Y_val, meta


# ============================================================
# GMVAE MODEL
# ============================================================

class GMVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, n_clusters, hidden_dims, dropout=0.0, n_targets=0):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.n_targets = n_targets
        hdim = hidden_dims[-1]

        self.encoder_backbone = build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims[:-1],
            output_dim=hidden_dims[-1],
            dropout=dropout,
            use_batchnorm=True
        )

        self.q_c_net = nn.Sequential(
            nn.Linear(hdim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_clusters)
        )

        self.q_z_mu = build_mlp(
            input_dim=hdim + n_clusters,
            hidden_dims=[hdim],
            output_dim=latent_dim,
            dropout=dropout,
            use_batchnorm=False
        )

        self.q_z_logvar = build_mlp(
            input_dim=hdim + n_clusters,
            hidden_dims=[hdim],
            output_dim=latent_dim,
            dropout=dropout,
            use_batchnorm=False
        )

        self.p_z_mu = nn.Parameter(torch.randn(n_clusters, latent_dim) * 0.5)
        self.p_z_logvar = nn.Parameter(torch.zeros(n_clusters, latent_dim))

        self.decoder = build_mlp(
            input_dim=latent_dim,
            hidden_dims=hidden_dims[::-1],
            output_dim=input_dim,
            dropout=dropout,
            use_batchnorm=False
        )

        if n_targets > 0:
            self.reg_head = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(32, n_targets)
            )
        else:
            self.reg_head = None

    def encode_backbone(self, x):
        return self.encoder_backbone(x)

    def infer_qc(self, h):
        logits = self.q_c_net(h)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def infer_qz_given_c(self, h, c_onehot):
        hc = torch.cat([h, c_onehot], dim=-1)
        mu = self.q_z_mu(hc)
        logvar = self.q_z_logvar(hc)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def regress(self, z):
        if self.reg_head is None:
            return None
        return self.reg_head(z)

    def forward(self, x):
        batch_size = x.size(0)
        h = self.encode_backbone(x)
        logits_c, probs_c = self.infer_qc(h)

        all_mu_q = []
        all_logvar_q = []
        all_z = []
        all_x_recon = []
        all_y_pred = []

        eye_k = torch.eye(self.n_clusters, device=x.device)

        for k in range(self.n_clusters):
            c_onehot = eye_k[k].unsqueeze(0).repeat(batch_size, 1)
            mu_q_k, logvar_q_k = self.infer_qz_given_c(h, c_onehot)
            z_k = reparameterize(mu_q_k, logvar_q_k)
            x_recon_k = self.decode(z_k)
            y_pred_k = self.regress(mu_q_k) if self.reg_head is not None else None

            all_mu_q.append(mu_q_k.unsqueeze(1))
            all_logvar_q.append(logvar_q_k.unsqueeze(1))
            all_z.append(z_k.unsqueeze(1))
            all_x_recon.append(x_recon_k.unsqueeze(1))
            if y_pred_k is not None:
                all_y_pred.append(y_pred_k.unsqueeze(1))

        mu_q = torch.cat(all_mu_q, dim=1)
        logvar_q = torch.cat(all_logvar_q, dim=1)
        z_samples = torch.cat(all_z, dim=1)
        x_recon = torch.cat(all_x_recon, dim=1)

        y_pred = None
        if self.reg_head is not None:
            y_pred = torch.cat(all_y_pred, dim=1)

        return {
            "logits_c": logits_c,
            "probs_c": probs_c,
            "mu_q": mu_q,
            "logvar_q": logvar_q,
            "z_samples": z_samples,
            "x_recon": x_recon,
            "y_pred": y_pred,
        }


# ============================================================
# GMVAE LOSS AND TRAINING
# ============================================================

def gmvae_loss(model, batch_x, batch_y=None, beta_kl_z=1.0, beta_kl_c=0.01, target_weights=None):
    out = model(batch_x)

    probs_c = out["probs_c"]
    mu_q = out["mu_q"]
    logvar_q = out["logvar_q"]
    x_recon = out["x_recon"]
    y_pred = out["y_pred"]

    B, K, _ = x_recon.shape

    x_target = batch_x.unsqueeze(1).expand_as(x_recon)
    recon_per_cluster = ((x_recon - x_target) ** 2).mean(dim=-1)
    recon_loss = (probs_c * recon_per_cluster).sum(dim=1).mean()

    mu_p = model.p_z_mu.unsqueeze(0).expand(B, -1, -1)
    logvar_p = model.p_z_logvar.unsqueeze(0).expand(B, -1, -1)

    kl_z_per_cluster = gaussian_kl(
        mu_q.reshape(B * K, -1),
        logvar_q.reshape(B * K, -1),
        mu_p.reshape(B * K, -1),
        logvar_p.reshape(B * K, -1)
    ).reshape(B, K)

    kl_z = (probs_c * kl_z_per_cluster).sum(dim=1).mean()

    log_uniform = torch.log(torch.tensor(1.0 / K, device=batch_x.device))
    kl_c = (probs_c * (torch.log(probs_c + 1e-10) - log_uniform)).sum(dim=1).mean()

    entropy_per_sample = -(probs_c * torch.log(probs_c + 1e-10)).sum(dim=1)
    entropy_loss = entropy_per_sample.mean()

    reg_loss = torch.tensor(0.0, device=batch_x.device)
    if batch_y is not None and y_pred is not None:
        y_target = batch_y.unsqueeze(1).expand_as(y_pred)
        if target_weights is None:
            reg_per_cluster = ((y_pred - y_target) ** 2).mean(dim=-1)
        else:
            tw = target_weights.view(1, 1, -1)
            reg_per_cluster = (((y_pred - y_target) ** 2) * tw).mean(dim=-1)
        reg_loss = (probs_c * reg_per_cluster).sum(dim=1).mean()

    total = (
        recon_loss
        + beta_kl_z * kl_z
        + beta_kl_c * kl_c
        + LAMBDA_REG * reg_loss
        - LAMBDA_ENTROPY * entropy_loss
    )

    return total, {
        "total": total.item(),
        "recon": recon_loss.item(),
        "kl_z": kl_z.item(),
        "kl_c": kl_c.item(),
        "reg": reg_loss.item(),
        "entropy": entropy_loss.item(),
    }


def run_gmvae_epoch(model, loader, epoch, optimizer=None, target_weights=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    beta_kl_z = kl_anneal(epoch, KL_WARMUP_EPOCHS, BETA_KL_Z_MAX)
    beta_kl_c = kl_anneal(epoch, KL_WARMUP_EPOCHS, BETA_KL_C_MAX)

    meter = {
        "total": 0.0,
        "recon": 0.0,
        "kl_z": 0.0,
        "kl_c": 0.0,
        "reg": 0.0,
        "entropy": 0.0,
        "n": 0,
    }

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            loss, parts = gmvae_loss(
                model=model,
                batch_x=x,
                batch_y=y,
                beta_kl_z=beta_kl_z,
                beta_kl_c=beta_kl_c,
                target_weights=target_weights,
            )
            if train_mode:
                loss.backward()
                optimizer.step()

        bs = x.size(0)
        for k in ["total", "recon", "kl_z", "kl_c", "reg", "entropy"]:
            meter[k] += parts[k] * bs
        meter["n"] += bs

    for k in ["total", "recon", "kl_z", "kl_c", "reg", "entropy"]:
        meter[k] /= max(meter["n"], 1)

    meter["beta_kl_z"] = beta_kl_z
    meter["beta_kl_c"] = beta_kl_c
    return meter


def train_gmvae(X_train, X_val, y_train, y_val, target_cols, out_dir):
    train_ds = PennDataset(X_train, y_train)
    val_ds = PennDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = GMVAE(
        input_dim=X_train.shape[1],
        latent_dim=LATENT_DIM,
        n_clusters=N_CLUSTERS,
        hidden_dims=GMVAE_HIDDEN_DIMS,
        dropout=GMVAE_DROPOUT,
        n_targets=y_train.shape[1],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=GMVAE_LR, weight_decay=GMVAE_WEIGHT_DECAY)
    target_weights = make_target_weights(target_cols)

    best_score = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, GMVAE_EPOCHS + 1):
        train_metrics = run_gmvae_epoch(model, train_loader, epoch, optimizer=optimizer, target_weights=target_weights)
        val_metrics = run_gmvae_epoch(model, val_loader, epoch, optimizer=None, target_weights=target_weights)

        row = {
            "epoch": epoch,
            "train_total": train_metrics["total"],
            "train_recon": train_metrics["recon"],
            "train_kl_z": train_metrics["kl_z"],
            "train_kl_c": train_metrics["kl_c"],
            "train_reg": train_metrics["reg"],
            "train_entropy": train_metrics["entropy"],
            "val_total": val_metrics["total"],
            "val_recon": val_metrics["recon"],
            "val_kl_z": val_metrics["kl_z"],
            "val_kl_c": val_metrics["kl_c"],
            "val_reg": val_metrics["reg"],
            "val_entropy": val_metrics["entropy"],
            "beta_kl_z": train_metrics["beta_kl_z"],
            "beta_kl_c": train_metrics["beta_kl_c"],
        }
        history.append(row)

        print(
            f"[GMVAE] Epoch {epoch:03d} | "
            f"train_total={train_metrics['total']:.4f} | "
            f"val_total={val_metrics['total']:.4f} | "
            f"val_reg={val_metrics['reg']:.4f}"
        )

        score = val_metrics["reg"] if GMVAE_MODEL_SELECTION == "val_reg" else val_metrics["total"]

        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= GMVAE_PATIENCE:
            print(f"[GMVAE] Early stopping at epoch {epoch}")
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(out_dir, "gmvae_best.pt"))

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(out_dir, "gmvae_training_history.csv"), index=False)

    return model, history_df


@torch.no_grad()
def extract_gmvae_latent(model, X):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    out = model(X_tensor)

    probs_c = out["probs_c"].cpu().numpy()
    mu_q = out["mu_q"].cpu().numpy()

    z_expected = np.sum(probs_c[:, :, None] * mu_q, axis=1)
    hard_cluster = np.argmax(probs_c, axis=1)
    return z_expected, probs_c, hard_cluster


# ============================================================
# GRAPH BUILDERS
# ============================================================

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
    n, _ = X.shape
    A = kneighbors_graph(X, k, mode="connectivity", metric="euclidean", include_self=False)
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


# ============================================================
# PGNN MODEL
# ============================================================

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
            max_norm = norm_diff.max().clamp(min=1e-6)
            plap_w = (norm_diff / max_norm).pow(self.p - 2)

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
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.lin2 = nn.Linear(hid_dim, hid_dim)
        self.bn2 = nn.BatchNorm1d(hid_dim)
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


# ============================================================
# EVALUATION
# ============================================================

def evaluate_single_target(model, Xv, yv, edge_indices, edge_weights, scaler_y, target_idx: int):
    """
    Original relative RMSE formula from your PGNN code:

        sq_err_orig = (y_true_orig - y_pred_orig)^2
        rmse_orig = sqrt(mean(sq_err_orig))
        rms_true_orig = sqrt(mean(y_true_orig^2))
        rel_rmse_orig_pct = 100 * rmse_orig / rms_true_orig
    """
    with torch.no_grad():
        model.eval()
        Xv_t = torch.tensor(Xv, dtype=torch.float32)
        h = model.encode(Xv_t)
        val_out = model.pconv(h, edge_indices, edge_weights)
        yp_scaled = model.out(val_out).cpu().numpy()

    y_true_scaled = yv.reshape(-1, 1) if yv.ndim == 1 else yv
    yp_scaled = yp_scaled.reshape(-1, 1) if yp_scaled.ndim == 1 else yp_scaled

    mean_t = float(scaler_y.mean_[target_idx])
    scale_t = float(scaler_y.scale_[target_idx])

    yp_orig = yp_scaled * scale_t + mean_t
    y_true_orig = y_true_scaled * scale_t + mean_t

    sq_err_scaled = (y_true_scaled - yp_scaled) ** 2
    rmse_scaled = float(np.sqrt(sq_err_scaled.mean()))
    mae_scaled = float(np.mean(np.abs(y_true_scaled - yp_scaled)))
    rms_true_sc = float(np.sqrt((y_true_scaled ** 2).mean()).clip(min=1e-12))
    rel_rmse_pct = float(100.0 * rmse_scaled / rms_true_sc)

    sq_err_orig = (y_true_orig - yp_orig) ** 2
    rmse_orig = float(np.sqrt(sq_err_orig.mean()))
    mae_orig = float(np.mean(np.abs(y_true_orig - yp_orig)))
    rms_true_orig = float(np.sqrt((y_true_orig ** 2).mean()).clip(min=1e-12))
    rel_rmse_orig_pct = float(100.0 * rmse_orig / rms_true_orig)

    r2 = float(r2_score(y_true_scaled, yp_scaled))

    return {
        "rmse_scaled": rmse_scaled,
        "mae_scaled": mae_scaled,
        "rel_rmse_pct": rel_rmse_pct,
        "rmse_orig": rmse_orig,
        "mae_orig": mae_orig,
        "rel_rmse_orig_pct": rel_rmse_orig_pct,
        "r2": r2,
        "y_true_orig": y_true_orig.flatten(),
        "y_pred_orig": yp_orig.flatten(),
    }


# ============================================================
# PGNN TRAINING PER TARGET
# ============================================================

def train_pgnn_for_target(Z_train, y_train_1d, Z_val, y_val_1d, scaler_y, target_idx, target_name, out_dir):
    rows = []
    best_global = None

    for knn_k in KNN_LIST:
        ei_train, ew_train = _knn_edge_index_l2_unweighted(Z_train, knn_k)
        ei_val, ew_val = _knn_edge_index_l2_unweighted(Z_val, knn_k)

        eis_train, ews_train = [ei_train], [ew_train]
        eis_val, ews_val = [ei_val], [ew_val]

        data_train = Data(
            x=torch.tensor(Z_train, dtype=torch.float32),
            edge_index=None,
            y=torch.tensor(y_train_1d.reshape(-1, 1), dtype=torch.float32)
        )

        for p in HIGH_P_VALUES:
            if p not in K_CHOICES_BY_P:
                continue

            for K in K_CHOICES_BY_P[p]:
                mu = mu_for(float(p), int(K))

                model = PGNNRegressor(
                    in_dim=Z_train.shape[1],
                    hid_dim=HID,
                    out_dim=1,
                    K=K,
                    p=p,
                    mu=mu,
                    dropout=PGNN_DROPOUT
                )

                opt = torch.optim.Adam(model.parameters(), lr=LR_MIN, weight_decay=WD)
                loss_fn = nn.MSELoss(reduction="mean")

                best_state = None
                best_rel = np.inf
                epochs_no_improve = 0
                current_mu = mu

                T = max(PGNN_EPOCHS, 50)

                for epoch in range(1, T + 1):
                    for g in opt.param_groups:
                        g["lr"] = one_cycle_lr(epoch, T, LR_MAX, LR_MIN)

                    model.train()
                    opt.zero_grad()

                    out = model(data_train, eis_train, ews_train, mu=current_mu)
                    loss = loss_fn(out, data_train.y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()

                    if epoch % 5 == 0:
                        eval_train = evaluate_single_target(
                            model, Z_train, y_train_1d, eis_train, ews_train, scaler_y, target_idx
                        )
                        if eval_train["rel_rmse_orig_pct"] > 200:
                            current_mu *= MU_BACKOFF
                            for g in opt.param_groups:
                                g["lr"] = max(g["lr"] * 0.5, LR_MIN)

                    metrics = evaluate_single_target(
                        model, Z_val, y_val_1d, eis_val, ews_val, scaler_y, target_idx
                    )
                    rel = metrics["rel_rmse_orig_pct"]

                    if rel < best_rel:
                        best_rel = rel
                        best_state = {
                            "model": copy.deepcopy(model.state_dict()),
                            "epoch": epoch,
                            "mu": current_mu,
                            "lr": opt.param_groups[0]["lr"],
                            "metrics": metrics,
                            "knn_k": knn_k,
                            "K": K,
                            "p": p,
                        }
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= PGNN_PATIENCE:
                            break

                if best_state is not None:
                    model.load_state_dict(best_state["model"])
                    final = best_state["metrics"]
                else:
                    final = evaluate_single_target(
                        model, Z_val, y_val_1d, eis_val, ews_val, scaler_y, target_idx
                    )

                model_mu = best_state["mu"] if best_state else mu
                model_epoch = best_state["epoch"] if best_state else -1
                model_lr = best_state["lr"] if best_state else LR_MIN

                ckpt_path = os.path.join(
                    out_dir,
                    f"best_model_target-{target_name}_knn{knn_k}_K{K}_p{p}_mu{model_mu:.3g}.pt"
                )
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": {
                            "target": target_name,
                            "target_idx": target_idx,
                            "knn_k": knn_k,
                            "K": K,
                            "p": p,
                            "mu": model_mu,
                            "best_epoch": model_epoch,
                            "best_lr": model_lr,
                            "latent_dim": Z_train.shape[1],
                        },
                        "metrics": {
                            "rmse_scaled": final["rmse_scaled"],
                            "mae_scaled": final["mae_scaled"],
                            "rel_rmse_pct": final["rel_rmse_pct"],
                            "rmse_orig": final["rmse_orig"],
                            "mae_orig": final["mae_orig"],
                            "rel_rmse_orig_pct": final["rel_rmse_orig_pct"],
                            "r2": final["r2"],
                        },
                    },
                    ckpt_path
                )

                row = {
                    "target": target_name,
                    "target_idx": target_idx,
                    "kNN_graph": knn_k,
                    "K": K,
                    "p": p,
                    "mu_used": model_mu,
                    "RMSE_scaled": final["rmse_scaled"],
                    "MAE_scaled": final["mae_scaled"],
                    "RelRMSE_scaled_pct": final["rel_rmse_pct"],
                    "RMSE_orig": final["rmse_orig"],
                    "MAE_orig": final["mae_orig"],
                    "RelRMSE_orig_pct": final["rel_rmse_orig_pct"],
                    "R2": final["r2"],
                    "best_epoch": model_epoch,
                    "best_lr": model_lr,
                }
                rows.append(row)

                if (best_global is None) or (row["RelRMSE_orig_pct"] < best_global["RelRMSE_orig_pct"]):
                    best_global = dict(row)

                print(
                    f"[PGNN][{target_name}] knn={knn_k} K={K} p={p} "
                    f"| RMSE_orig={final['rmse_orig']:.4f} "
                    f"| RelRMSE_orig={final['rel_rmse_orig_pct']:.3f}% "
                    f"| R2={final['r2']:.4f}"
                )

    results_df = pd.DataFrame(rows).sort_values(
        by=["RelRMSE_orig_pct", "RMSE_orig", "R2"],
        ascending=[True, True, False]
    )
    results_df.to_csv(os.path.join(out_dir, f"summary_{target_name}.csv"), index=False)

    if best_global is not None:
        pd.DataFrame([best_global]).to_csv(
            os.path.join(out_dir, f"best_run_{target_name}.csv"),
            index=False
        )

    return results_df


# ============================================================
# VISUALIZATION
# ============================================================

def plot_gmvae_loss(history_df, out_dir):
    epochs = history_df["epoch"].values

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_df["train_total"].values, label="train_total")
    plt.plot(epochs, history_df["val_total"].values, label="val_total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GMVAE Total Loss")
    plt.legend()
    save_current_figure(os.path.join(out_dir, "gmvae_total_loss.png"))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_df["train_reg"].values, label="train_reg")
    plt.plot(epochs, history_df["val_reg"].values, label="val_reg")
    plt.xlabel("Epoch")
    plt.ylabel("Regression Loss")
    plt.title("GMVAE Regression Loss")
    plt.legend()
    save_current_figure(os.path.join(out_dir, "gmvae_reg_loss.png"))


def plot_cluster_counts(cluster_labels, out_dir, split_name):
    vals, counts = np.unique(cluster_labels, return_counts=True)
    plt.figure(figsize=(7, 5))
    plt.bar(vals.astype(str), counts)
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.title(f"{split_name}: GMVAE Hard Cluster Counts")
    save_current_figure(os.path.join(out_dir, f"{split_name}_cluster_counts.png"))


# ============================================================
# RUN ONE DATASET
# ============================================================

def run_one_dataset(csv_path):
    dataset_name = _slugify(csv_path)
    out_dir = os.path.join(RESULTS_ROOT, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 90}")
    print(f"Running dataset: {csv_path}")
    print(f"{'=' * 90}")

    X_train, X_val, Y_train, Y_val, meta = load_dataset_for_gmvae(
        csv_path, train_size=TRAIN_SIZE, seed=SEED
    )

    print(f"Train X: {X_train.shape}, Train Y: {Y_train.shape}")
    print(f"Val   X: {X_val.shape}, Val   Y: {Y_val.shape}")
    print(f"Targets found: {meta['target_cols']}")

    pd.DataFrame([{
        "dataset": dataset_name,
        "csv_path": csv_path,
        "n_total": meta["n_total"],
        "n_train": meta["n_train"],
        "n_val": meta["n_val"],
        "n_features": len(meta["feature_cols"]),
        "targets": ",".join(meta["target_cols"]),
    }]).to_csv(os.path.join(out_dir, "dataset_info.csv"), index=False)

    # ---------------- GMVAE stage ----------------
    gmvae_dir = os.path.join(out_dir, "gmvae_stage")
    os.makedirs(gmvae_dir, exist_ok=True)

    gmvae_model, history_df = train_gmvae(
        X_train, X_val, Y_train, Y_val, meta["target_cols"], gmvae_dir
    )

    plot_gmvae_loss(history_df, gmvae_dir)

    Z_train, P_train, C_train = extract_gmvae_latent(gmvae_model, X_train)
    Z_val, P_val, C_val = extract_gmvae_latent(gmvae_model, X_val)

    np.save(os.path.join(gmvae_dir, "Z_train.npy"), Z_train)
    np.save(os.path.join(gmvae_dir, "Z_val.npy"), Z_val)
    np.save(os.path.join(gmvae_dir, "P_train.npy"), P_train)
    np.save(os.path.join(gmvae_dir, "P_val.npy"), P_val)
    np.save(os.path.join(gmvae_dir, "C_train.npy"), C_train)
    np.save(os.path.join(gmvae_dir, "C_val.npy"), C_val)

    plot_cluster_counts(C_train, gmvae_dir, "train")
    plot_cluster_counts(C_val, gmvae_dir, "val")

    latent_train_df = pd.DataFrame(Z_train, columns=[f"z_exp_{i}" for i in range(Z_train.shape[1])])
    latent_val_df = pd.DataFrame(Z_val, columns=[f"z_exp_{i}" for i in range(Z_val.shape[1])])

    latent_train_df["cluster_hard"] = C_train
    latent_val_df["cluster_hard"] = C_val

    for j in range(P_train.shape[1]):
        latent_train_df[f"cluster_prob_{j}"] = P_train[:, j]
        latent_val_df[f"cluster_prob_{j}"] = P_val[:, j]

    for i, t in enumerate(meta["target_cols"]):
        latent_train_df[f"{t}_scaled"] = Y_train[:, i]
        latent_val_df[f"{t}_scaled"] = Y_val[:, i]

    latent_train_df.to_csv(os.path.join(gmvae_dir, "latent_train.csv"), index=False)
    latent_val_df.to_csv(os.path.join(gmvae_dir, "latent_val.csv"), index=False)

    # ---------------- PGNN stage ----------------
    pgnn_dir = os.path.join(out_dir, "pgnn_stage")
    os.makedirs(pgnn_dir, exist_ok=True)

    all_target_results = []
    best_per_target_rows = []

    for target_idx, target_name in enumerate(meta["target_cols"]):
        print(f"\n--- Training PGNN for target: {target_name} ---")
        target_dir = os.path.join(pgnn_dir, target_name)
        os.makedirs(target_dir, exist_ok=True)

        res_df = train_pgnn_for_target(
            Z_train=Z_train,
            y_train_1d=Y_train[:, target_idx],
            Z_val=Z_val,
            y_val_1d=Y_val[:, target_idx],
            scaler_y=meta["scaler_y"],
            target_idx=target_idx,
            target_name=target_name,
            out_dir=target_dir
        )
        all_target_results.append(res_df)

        best_row = res_df.sort_values(
            by=["RelRMSE_orig_pct", "RMSE_orig", "R2"],
            ascending=[True, True, False]
        ).iloc[0].to_dict()

        best_row["dataset"] = dataset_name
        best_per_target_rows.append(best_row)

    master_df = pd.concat(all_target_results, axis=0, ignore_index=True)
    master_df.insert(0, "dataset", dataset_name)
    master_df.to_csv(os.path.join(out_dir, "master_summary.csv"), index=False)

    best_targets_df = pd.DataFrame(best_per_target_rows)
    best_targets_df.to_csv(os.path.join(out_dir, "best_per_target_summary.csv"), index=False)

    print("\nTop runs for dataset:")
    print(master_df.sort_values(by=["RelRMSE_orig_pct", "RMSE_orig"]).head(15))

    return {
        "dataset_name": dataset_name,
        "master_df": master_df,
        "best_targets_df": best_targets_df,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Using device: {DEVICE}")

    all_dataset_summaries = []
    all_best_per_target = []

    for csv_path in CSV_LIST:
        result = run_one_dataset(csv_path)
        all_dataset_summaries.append(result["master_df"])
        all_best_per_target.append(result["best_targets_df"])

    if len(all_dataset_summaries) > 0:
        global_df = pd.concat(all_dataset_summaries, axis=0, ignore_index=True)
        global_df.to_csv(os.path.join(RESULTS_ROOT, "all_datasets_summary.csv"), index=False)

        best_df = (
            global_df.sort_values(
                by=["dataset", "target", "RelRMSE_orig_pct", "RMSE_orig", "R2"],
                ascending=[True, True, True, True, False]
            )
            .groupby(["dataset", "target"], as_index=False)
            .first()
        )
        best_df.to_csv(os.path.join(RESULTS_ROOT, "best_runs_across_all_datasets.csv"), index=False)

    if len(all_best_per_target) > 0:
        combined_best_targets_df = pd.concat(all_best_per_target, axis=0, ignore_index=True)
        combined_best_targets_df.to_csv(
            os.path.join(RESULTS_ROOT, "combined_best_per_target_summary.csv"),
            index=False
        )

    print(f"\nDone. Results saved in: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
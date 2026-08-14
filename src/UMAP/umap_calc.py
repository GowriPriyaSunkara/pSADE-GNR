
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run 4 pipelines on three datasets:
  1) Raw scaled features -> UMAP
  2) VAE latent space -> UMAP
  3) GMVAE latent expected embedding -> UMAP
  4) Mapper on the 2D UMAP embedding (for each of the above spaces)

What this script saves for each dataset and each method:
  - train/val split indices
  - scaled feature tables
  - latent embeddings (.npy and .csv)
  - UMAP 2D coordinates (.npy and .csv)
  - PNG figures colored by each target (ALM/BMD/BFP/Age when present)
  - cluster-count plot for GMVAE
  - Mapper HTML visualization
  - Mapper graph JSON
  - Mapper nodes CSV
  - Mapper edges CSV
  - summary CSVs

Datasets expected:
  /scratch/gsunka1/TDA_guoji/male.csv
  /scratch/gsunka1/TDA_guoji/female.csv
  /scratch/gsunka1/TDA_guoji/penn_data.csv

Required packages:
  numpy, pandas, matplotlib, scikit-learn, scipy, torch, umap-learn, kmapper
"""

import os
import re
import json
import copy
import math
import random
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import umap.umap_ as umap
import kmapper as km

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================
CSV_LIST = [
    "/scratch/mlemo36/TDA_folder/Penn/pLaplacian-Graph-NN/male.csv",
    "/scratch/mlemo36/TDA_folder/Penn/pLaplacian-Graph-NN/female.csv",
    "/scratch/mlemo36/TDA_folder/Penn/pLaplacian-Graph-NN/penn_data.csv",
]

RESULTS_ROOT = "UMAP_VAE_GMVAE_MAPPER_ALL_DATASETS"
os.makedirs(RESULTS_ROOT, exist_ok=True)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SIZE = 0.80
BATCH_SIZE = 128
DPI = 300

# columns to remove from features if present
ID_COLUMNS = ["row_id", "ID", "id", "subject_id", "SubjectID", "PPT ID"]
NON_FEATURE_DROP = ["Site", "Gender", "Race", "0"]
TARGET_COLUMNS = ["ALM", "BMD", "BFP", "Age"]

# UMAP
UMAP_N_NEIGHBORS = 20
UMAP_MIN_DIST = 0.10
UMAP_METRIC = "euclidean"
UMAP_RANDOM_STATE = SEED

# VAE
VAE_LATENT_DIM = 15
VAE_HIDDEN_DIMS = [128, 64]
VAE_DROPOUT = 0.20
VAE_LR = 1e-3
VAE_WEIGHT_DECAY = 1e-4
VAE_EPOCHS = 200
VAE_PATIENCE = 25
VAE_BETA = 0.0
VAE_MASK_PROB = 0.10
VAE_GAUSS_STD = 0.0

# GMVAE
N_CLUSTERS = 3
LATENT_DIM = 15
GMVAE_HIDDEN_DIMS = [128, 64]
GMVAE_DROPOUT = 0.20
GMVAE_EPOCHS = 250
GMVAE_PATIENCE = 30
GMVAE_LR = 1e-3
GMVAE_WEIGHT_DECAY = 1e-5
BETA_KL_Z_MAX = 0.0
BETA_KL_C_MAX = 0.01
KL_WARMUP_EPOCHS = 40
LAMBDA_REG = 2.0
LAMBDA_ENTROPY = 0.001
TARGET_WEIGHTS_DICT = {
    "ALM": 1.0,
    "BMD": 1.0,
    "BFP": 2.0,
    "Age": 2.0,
}
GMVAE_MODEL_SELECTION = "val_reg"  # or "val_total"
 
# Mapper
MAPPER_N_CUBES = 5
MAPPER_OVERLAP = 0.01
MAPPER_DBSCAN_EPS = 0.3
MAPPER_DBSCAN_MIN_SAMPLES = 1

# cubes = [5, 10, 15, 20]
# overlap = [0.01, 0.1, 0.25, 0.5, 0.75]
# eps = [0.01, 0.1, 0.25, 0.5, 0.75]
# min_samp = [1, 3, 5, 10, 15]

# c = 0
# while c < len(cubes):

#     MAPPER_N_CUBES = cubes[c]

#     o = 0
#     while o < len(overlap):

#         MAPPER_OVERLAP = overlap[o]

#         e = 0
#         while e < len(eps):

#             MAPPER_DBSCAN_EPS = eps[e]

#             ms = 0
#             while ms < len(min_samp):

#                 MAPPER_DBSCAN_MIN_SAMPLES = min_samp[ms]




# =========================================================
# UTILITIES
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _slugify(name: str) -> str:
    s = os.path.splitext(os.path.basename(name))[0]
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def save_current_figure(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()


def pick_existing_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def kl_anneal(epoch: int, warmup_epochs: int, max_beta: float) -> float:
    return min(max_beta, (epoch / float(max(warmup_epochs, 1))) * max_beta)


def append_global_summary(row: dict, csv_path: str):
    df_row = pd.DataFrame([row])
    write_header = not os.path.exists(csv_path)
    df_row.to_csv(csv_path, mode="a", header=write_header, index=False)


# =========================================================
# DATA LOADING
# =========================================================
def load_dataset(csv_path: str, train_size: float = TRAIN_SIZE, seed: int = SEED):
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

    X = X.fillna(X.median(numeric_only=True))
    Y = Y.fillna(Y.median(numeric_only=True))

    train_idx, val_idx = train_test_split(
        np.arange(len(X)),
        train_size=train_size,
        random_state=seed,
        shuffle=True,
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
        "df": df,
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


def save_split_tables(out_dir: str,
                    X_train: np.ndarray,
                    X_val: np.ndarray,
                    Y_train: np.ndarray,
                    Y_val: np.ndarray,
                    meta: dict):
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame({"train_idx": meta["train_idx"]}).to_csv(
        os.path.join(out_dir, "train_indices.csv"), index=False
    )
    pd.DataFrame({"val_idx": meta["val_idx"]}).to_csv(
        os.path.join(out_dir, "val_indices.csv"), index=False
    )

    train_df = pd.DataFrame(X_train, columns=meta["feature_cols"])
    val_df = pd.DataFrame(X_val, columns=meta["feature_cols"])

    for i, t in enumerate(meta["target_cols"]):
        train_df[f"{t}_scaled"] = Y_train[:, i]
        val_df[f"{t}_scaled"] = Y_val[:, i]

    train_df.to_csv(os.path.join(out_dir, "scaled_train_table.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "scaled_val_table.csv"), index=False)


# =========================================================
# VAE
# =========================================================
class PlainTensorDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims=(128, 64), dropout=0.0):
        super().__init__()
        h1, h2 = hidden_dims

        self.enc = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(h2, latent_dim)
        self.logvar = nn.Linear(h2, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
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


def train_vae_and_embed(X_train: np.ndarray, X_val: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    model = VAE(
        input_dim=X_train.shape[1],
        latent_dim=VAE_LATENT_DIM,
        hidden_dims=tuple(VAE_HIDDEN_DIMS),
        dropout=VAE_DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=VAE_LR, weight_decay=VAE_WEIGHT_DECAY)

    train_loader = DataLoader(PlainTensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PlainTensorDataset(X_val), batch_size=BATCH_SIZE, shuffle=False)

    best_val = float("inf")
    best_state = None
    patience_left = VAE_PATIENCE
    history = []

    @torch.no_grad()
    def eval_loader(loader):
        model.eval()
        total = []
        recons = []
        kls = []
        for x in loader:
            x = x.to(DEVICE)
            x_hat, mu, logvar, _ = model(x)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=VAE_BETA)
            total.append(loss.item())
            recons.append(recon.item())
            kls.append(kl.item())
        return float(np.mean(total)), float(np.mean(recons)), float(np.mean(kls))

    for epoch in range(1, VAE_EPOCHS + 1):
        model.train()
        batch_losses = []
        batch_recons = []
        batch_kls = []

        for x in train_loader:
            x = x.to(DEVICE)
            x_in = apply_denoising(x, mask_prob=VAE_MASK_PROB, gaussian_std=VAE_GAUSS_STD)
            x_hat, mu, logvar, _ = model(x_in)
            loss, recon, kl = vae_loss(x_hat, x, mu, logvar, beta=VAE_BETA)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            batch_losses.append(loss.item())
            batch_recons.append(recon.item())
            batch_kls.append(kl.item())

        val_total, val_recon, val_kl = eval_loader(val_loader)
        train_total = float(np.mean(batch_losses))
        train_recon = float(np.mean(batch_recons))
        train_kl = float(np.mean(batch_kls))

        history.append({
            "epoch": epoch,
            "train_total": train_total,
            "train_recon": train_recon,
            "train_kl": train_kl,
            "val_total": val_total,
            "val_recon": val_recon,
            "val_kl": val_kl,
        })

        print(f"[VAE] epoch={epoch:03d} train_total={train_total:.4f} val_total={val_total:.4f}")

        if val_total < best_val - 1e-6:
            best_val = val_total
            best_state = copy.deepcopy(model.state_dict())
            patience_left = VAE_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[VAE] early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(out_dir, "vae_training_history.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(out_dir, "vae_best.pt"))

    # plt.figure(figsize=(8, 5))
    # plt.plot(history_df["epoch"], history_df["train_total"], label="train_total")
    # plt.plot(history_df["epoch"], history_df["val_total"], label="val_total")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("VAE Total Loss")
    # plt.legend()
    # save_current_figure(os.path.join(out_dir, "vae_total_loss.png"))

    @torch.no_grad()
    def encode_np(X):
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        mu, _ = model.encode(X_t)
        return mu.cpu().numpy()

    z_train = encode_np(X_train)
    z_val = encode_np(X_val)

    np.save(os.path.join(out_dir, "z_train.npy"), z_train)
    np.save(os.path.join(out_dir, "z_val.npy"), z_val)

    return z_train, z_val, history_df


# =========================================================
# GMVAE
# =========================================================
class XYDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


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


class GMVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, n_clusters, hidden_dims, dropout=0.0, n_targets=0):
        super().__init__()
        hdim = hidden_dims[-1]
        self.n_clusters = n_clusters
        self.n_targets = n_targets

        self.encoder_backbone = build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims[:-1],
            output_dim=hidden_dims[-1],
            dropout=dropout,
            use_batchnorm=True,
        )

        self.q_c_net = nn.Sequential(
            nn.Linear(hdim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_clusters),
        )

        self.q_z_mu = build_mlp(
            input_dim=hdim + n_clusters,
            hidden_dims=[hdim],
            output_dim=latent_dim,
            dropout=dropout,
            use_batchnorm=False,
        )
        self.q_z_logvar = build_mlp(
            input_dim=hdim + n_clusters,
            hidden_dims=[hdim],
            output_dim=latent_dim,
            dropout=dropout,
            use_batchnorm=False,
        )

        self.p_z_mu = nn.Parameter(torch.randn(n_clusters, latent_dim) * 0.5)
        self.p_z_logvar = nn.Parameter(torch.zeros(n_clusters, latent_dim))

        self.decoder = build_mlp(
            input_dim=latent_dim,
            hidden_dims=hidden_dims[::-1],
            output_dim=input_dim,
            dropout=dropout,
            use_batchnorm=False,
        )

        if n_targets > 0:
            self.reg_head = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(32, n_targets),
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


def make_target_weights(target_cols):
    weights = [TARGET_WEIGHTS_DICT.get(col, 1.0) for col in target_cols]
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def gmvae_loss(model, batch_x, batch_y=None, target_weights=None,
            beta_kl_z=1.0, beta_kl_c=0.01):
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
        logvar_p.reshape(B * K, -1),
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

    meter = {k: 0.0 for k in ["total", "recon", "kl_z", "kl_c", "reg", "entropy"]}
    meter["n"] = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

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


def train_gmvae_and_embed(X_train: np.ndarray, X_val: np.ndarray,
                        Y_train: np.ndarray, Y_val: np.ndarray,
                        target_cols: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    train_ds = XYDataset(X_train, Y_train)
    val_ds = XYDataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = GMVAE(
        input_dim=X_train.shape[1],
        latent_dim=LATENT_DIM,
        n_clusters=N_CLUSTERS,
        hidden_dims=GMVAE_HIDDEN_DIMS,
        dropout=GMVAE_DROPOUT,
        n_targets=Y_train.shape[1],
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
            f"[GMVAE] epoch={epoch:03d} train_total={train_metrics['total']:.4f} "
            f"val_total={val_metrics['total']:.4f} val_reg={val_metrics['reg']:.4f}"
        )

        score = val_metrics["reg"] if GMVAE_MODEL_SELECTION == "val_reg" else val_metrics["total"]

        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= GMVAE_PATIENCE:
            print(f"[GMVAE] early stopping at epoch {epoch}")
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(out_dir, "gmvae_best.pt"))

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(out_dir, "gmvae_training_history.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_total"], label="train_total")
    plt.plot(history_df["epoch"], history_df["val_total"], label="val_total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GMVAE Total Loss")
    plt.legend()
    save_current_figure(os.path.join(out_dir, "gmvae_total_loss.png"))

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_reg"], label="train_reg")
    plt.plot(history_df["epoch"], history_df["val_reg"], label="val_reg")
    plt.xlabel("Epoch")
    plt.ylabel("Regression Loss")
    plt.title("GMVAE Regression Loss")
    plt.legend()
    save_current_figure(os.path.join(out_dir, "gmvae_regression_loss.png"))

    @torch.no_grad()
    def extract_latent(X: np.ndarray):
        model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        out = model(X_t)
        probs_c = out["probs_c"].cpu().numpy()
        mu_q = out["mu_q"].cpu().numpy()
        z_expected = np.sum(probs_c[:, :, None] * mu_q, axis=1)
        hard_cluster = np.argmax(probs_c, axis=1)
        return z_expected, probs_c, hard_cluster

    z_train, p_train, c_train = extract_latent(X_train)
    z_val, p_val, c_val = extract_latent(X_val)

    np.save(os.path.join(out_dir, "Z_train.npy"), z_train)
    np.save(os.path.join(out_dir, "Z_val.npy"), z_val)
    np.save(os.path.join(out_dir, "P_train.npy"), p_train)
    np.save(os.path.join(out_dir, "P_val.npy"), p_val)
    np.save(os.path.join(out_dir, "C_train.npy"), c_train)
    np.save(os.path.join(out_dir, "C_val.npy"), c_val)

    plt.figure(figsize=(7, 5))
    vals, counts = np.unique(c_train, return_counts=True)
    plt.bar(vals.astype(str), counts)
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.title("Train GMVAE Cluster Counts")
    save_current_figure(os.path.join(out_dir, "train_cluster_counts.png"))

    plt.figure(figsize=(7, 5))
    vals, counts = np.unique(c_val, return_counts=True)
    plt.bar(vals.astype(str), counts)
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.title("Validation GMVAE Cluster Counts")
    save_current_figure(os.path.join(out_dir, "val_cluster_counts.png"))

    return z_train, z_val, p_train, p_val, c_train, c_val, history_df


# =========================================================
# UMAP + PLOTTING
# =========================================================
def fit_umap(train_X: np.ndarray, val_X: np.ndarray):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    )
    umap_train = reducer.fit_transform(train_X)
    umap_val = reducer.transform(val_X)
    return reducer, umap_train, umap_val


def save_embedding_tables(out_dir: str,
                        base_train: np.ndarray,
                        base_val: np.ndarray,
                        umap_train: np.ndarray,
                        umap_val: np.ndarray,
                        Y_train: np.ndarray,
                        Y_val: np.ndarray,
                        target_cols: List[str],
                        base_prefix: str,
                        extra_train: Optional[dict] = None,
                        extra_val: Optional[dict] = None):
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, f"{base_prefix}_train.npy"), base_train)
    np.save(os.path.join(out_dir, f"{base_prefix}_val.npy"), base_val)
    np.save(os.path.join(out_dir, f"{base_prefix}_umap_train.npy"), umap_train)
    np.save(os.path.join(out_dir, f"{base_prefix}_umap_val.npy"), umap_val)

    train_df = pd.DataFrame(umap_train, columns=["umap_1", "umap_2"])
    val_df = pd.DataFrame(umap_val, columns=["umap_1", "umap_2"])

    for j, name in enumerate(target_cols):
        train_df[f"{name}_scaled"] = Y_train[:, j]
        val_df[f"{name}_scaled"] = Y_val[:, j]

    if extra_train is not None:
        for k, v in extra_train.items():
            train_df[k] = v
    if extra_val is not None:
        for k, v in extra_val.items():
            val_df[k] = v

    train_df.to_csv(os.path.join(out_dir, f"{base_prefix}_umap_train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, f"{base_prefix}_umap_val.csv"), index=False)

    combined_df = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
    ], axis=0, ignore_index=True)
    combined_df.to_csv(os.path.join(out_dir, f"{base_prefix}_umap_all.csv"), index=False)



def plot_umap_targets(umap_train: np.ndarray,
                    umap_val: np.ndarray,
                    Y_train: np.ndarray,
                    Y_val: np.ndarray,
                    target_cols: List[str],
                    out_dir: str,
                    title_prefix: str):
    os.makedirs(out_dir, exist_ok=True)

    for j, target_name in enumerate(target_cols):
        plt.figure(figsize=(8, 6))
        vals = np.concatenate([Y_train[:, j], Y_val[:, j]])
        xy = np.vstack([umap_train, umap_val])
        plt.scatter(xy[:, 0], xy[:, 1], c=vals, s=18, alpha=0.8)
        plt.colorbar(label=f"{target_name} ")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.title(f"{title_prefix} - colored by {target_name}")
        save_current_figure(os.path.join(out_dir, f"umap_by_{target_name}.png"))

    plt.figure(figsize=(8, 6))
    plt.scatter(umap_train[:, 0], umap_train[:, 1], s=18, alpha=0.8, label="train")
    plt.scatter(umap_val[:, 0], umap_val[:, 1], s=18, alpha=0.8, label="val")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(f"{title_prefix} - train vs val")
    plt.legend()
    save_current_figure(os.path.join(out_dir, "umap_train_vs_val.png"))


# =========================================================
# MAPPER
# =========================================================
def save_mapper_outputs(lens_2d: np.ndarray,
                        X_for_clustering: np.ndarray,
                        out_dir: str,
                        mapper_name: str):
    os.makedirs(out_dir, exist_ok=True)

    mapper = km.KeplerMapper(verbose=1)

    # Try a few cluster/cover settings because Mapper can legitimately return
    # an empty graph when the cover is too fine or DBSCAN is too strict.
    candidate_params = [
        {"n_cubes": MAPPER_N_CUBES,     "overlap": MAPPER_OVERLAP, "eps": MAPPER_DBSCAN_EPS, "min_samples": MAPPER_DBSCAN_MIN_SAMPLES},
        {"n_cubes": max(8, MAPPER_N_CUBES), "overlap": 0.40,           "eps": 1.00,             "min_samples": 5},
        {"n_cubes": 10,                 "overlap": 0.45,           "eps": 1.20,             "min_samples": 4},
        {"n_cubes": 8,                  "overlap": 0.50,           "eps": 1.50,             "min_samples": 3},
    ]

    graph = None
    best_attempt = None
    attempt_rows = []

    for params in candidate_params:
        cover = km.Cover(n_cubes=params["n_cubes"], perc_overlap=params["overlap"])
        clusterer = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])

        graph_try = mapper.map(
            lens_2d,
            X=X_for_clustering,
            clusterer=clusterer,
            cover=cover,
        )

        n_nodes_try = len(graph_try.get("nodes", {}))
        n_edges_try = sum(len(v) for v in graph_try.get("links", {}).values())
        params_row = {
            "mapper_name": mapper_name,
            "n_cubes": params["n_cubes"],
            "overlap": params["overlap"],
            "eps": params["eps"],
            "min_samples": params["min_samples"],
            "n_nodes": n_nodes_try,
            "n_edges": n_edges_try,
        }
        attempt_rows.append(params_row)

        if (best_attempt is None) or (n_nodes_try > best_attempt["n_nodes"]) or (
            n_nodes_try == best_attempt["n_nodes"] and n_edges_try > best_attempt["n_edges"]
        ):
            best_attempt = params_row
            graph = graph_try

        if n_nodes_try > 0:
            break

    attempts_csv = os.path.join(out_dir, f"{mapper_name}_attempts.csv")
    pd.DataFrame(attempt_rows).to_csv(attempts_csv, index=False)

    html_path = os.path.join(out_dir, f"{mapper_name}_{MAPPER_N_CUBES}_{MAPPER_OVERLAP}_{MAPPER_DBSCAN_EPS}_{MAPPER_DBSCAN_MIN_SAMPLES}_mapper.html")
    graph_json_path = os.path.join(out_dir, f"{mapper_name}_graph.json")

    with open(graph_json_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    node_rows = []
    for node_name, members in graph.get("nodes", {}).items():
        node_rows.append({
            "node": node_name,
            "n_members": len(members),
            "members": "|".join(map(str, members)),
        })
    nodes_df = pd.DataFrame(node_rows)
    nodes_df.to_csv(os.path.join(out_dir, f"{mapper_name}_nodes.csv"), index=False)

    edge_rows = []
    for src, dst_list in graph.get("links", {}).items():
        for dst in dst_list:
            edge_rows.append({"source": src, "target": dst})
    edges_df = pd.DataFrame(edge_rows)
    edges_df.to_csv(os.path.join(out_dir, f"{mapper_name}_edges.csv"), index=False)

    html_created = False
    if len(graph.get("nodes", {})) > 0:
        mapper.visualize(graph, path_html=html_path, title=mapper_name)
        html_created = True
    else:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("<html><body><h2>Mapper graph is empty</h2><p>No nodes were created for this dataset/method with the tried parameter settings. See the attempts CSV for the tested parameters.</p></body></html>")

    summary = {
        "mapper_name": mapper_name,
        "n_nodes": len(graph.get("nodes", {})),
        "n_edges": len(edge_rows),
        "html_path": html_path,
        "graph_json_path": graph_json_path,
        "attempts_csv": attempts_csv,
        "html_created": html_created,
        "best_n_cubes": best_attempt["n_cubes"] if best_attempt else None,
        "best_overlap": best_attempt["overlap"] if best_attempt else None,
        "best_eps": best_attempt["eps"] if best_attempt else None,
        "best_min_samples": best_attempt["min_samples"] if best_attempt else None,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, f"{mapper_name}_summary.csv"), index=False)
    return summary


# =========================================================
# ONE DATASET
# =========================================================
def run_one_dataset(csv_path: str, global_summary_csv: str):
    dataset_name = _slugify(csv_path)
    dataset_dir = os.path.join(RESULTS_ROOT, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    print("\n" + "=" * 100)
    print(f"Running dataset: {csv_path}")
    print("=" * 100)

    X_train, X_val, Y_train, Y_val, meta = load_dataset(csv_path, train_size=TRAIN_SIZE, seed=SEED)

    info_row = {
        "dataset": dataset_name,
        "csv_path": csv_path,
        "n_total": meta["n_total"],
        "n_train": meta["n_train"],
        "n_val": meta["n_val"],
        "n_features": len(meta["feature_cols"]),
        "targets": ",".join(meta["target_cols"]),
        "device": DEVICE,
    }
    pd.DataFrame([info_row]).to_csv(os.path.join(dataset_dir, "dataset_info.csv"), index=False)

    split_dir = os.path.join(dataset_dir, "split_data")
    save_split_tables(split_dir, X_train, X_val, Y_train, Y_val, meta)

    method_summaries = []

    # -----------------------------------------------------
    # 1) RAW + UMAP + MAPPER
    # -----------------------------------------------------
    raw_dir = os.path.join(dataset_dir, "raw_umap_mapper")
    os.makedirs(raw_dir, exist_ok=True)

    _, raw_umap_train, raw_umap_val = fit_umap(X_train, X_val)
    save_embedding_tables(
        out_dir=raw_dir,
        base_train=X_train,
        base_val=X_val,
        umap_train=raw_umap_train,
        umap_val=raw_umap_val,
        Y_train=Y_train,
        Y_val=Y_val,
        target_cols=meta["target_cols"],
        base_prefix="raw",
    )
    plot_umap_targets(raw_umap_train, raw_umap_val, Y_train, Y_val, meta["target_cols"], raw_dir, "Raw scaled features")

    raw_mapper_summary = save_mapper_outputs(
        lens_2d=np.vstack([raw_umap_train, raw_umap_val]),
        X_for_clustering=np.vstack([X_train, X_val]),
        out_dir=os.path.join(raw_dir, "mapper"),
        mapper_name=f"{dataset_name}_raw_umap_mapper",
    )
    method_summaries.append({
        "dataset": dataset_name,
        "method": "raw",
        "space_dim": X_train.shape[1],
        **raw_mapper_summary,
    })

    append_global_summary({
        "dataset": dataset_name,
        "method": "raw",
        "n_features_in_space": X_train.shape[1],
        "n_umap_points": int(len(raw_umap_train) + len(raw_umap_val)),
        "mapper_nodes": raw_mapper_summary["n_nodes"],
        "mapper_edges": raw_mapper_summary["n_edges"],
        "result_dir": raw_dir,
    }, global_summary_csv)

    # -----------------------------------------------------
    # 2) VAE + UMAP + MAPPER
    # -----------------------------------------------------
    vae_dir = os.path.join(dataset_dir, "vae_umap_mapper")
    os.makedirs(vae_dir, exist_ok=True)

    z_train, z_val, _ = train_vae_and_embed(X_train, X_val, os.path.join(vae_dir, "vae_stage"))
    _, vae_umap_train, vae_umap_val = fit_umap(z_train, z_val)

    save_embedding_tables(
        out_dir=vae_dir,
        base_train=z_train,
        base_val=z_val,
        umap_train=vae_umap_train,
        umap_val=vae_umap_val,
        Y_train=Y_train,
        Y_val=Y_val,
        target_cols=meta["target_cols"],
        base_prefix="vae",
    )
    plot_umap_targets(vae_umap_train, vae_umap_val, Y_train, Y_val, meta["target_cols"], vae_dir, "VAE latent space")

    vae_mapper_summary = save_mapper_outputs(
        lens_2d=np.vstack([vae_umap_train, vae_umap_val]),
        X_for_clustering=np.vstack([z_train, z_val]),
        out_dir=os.path.join(vae_dir, "mapper"),
        mapper_name=f"{dataset_name}_vae_umap_mapper",
    )
    method_summaries.append({
        "dataset": dataset_name,
        "method": "vae",
        "space_dim": z_train.shape[1],
        **vae_mapper_summary,
    })

    append_global_summary({
        "dataset": dataset_name,
        "method": "vae",
        "n_features_in_space": z_train.shape[1],
        "n_umap_points": int(len(vae_umap_train) + len(vae_umap_val)),
        "mapper_nodes": vae_mapper_summary["n_nodes"],
        "mapper_edges": vae_mapper_summary["n_edges"],
        "result_dir": vae_dir,
    }, global_summary_csv)

    # -----------------------------------------------------
    # 3) GMVAE + UMAP + MAPPER
    # -----------------------------------------------------
    gmvae_dir = os.path.join(dataset_dir, "gmvae_umap_mapper")
    os.makedirs(gmvae_dir, exist_ok=True)

    Z_train, Z_val, P_train, P_val, C_train, C_val, _ = train_gmvae_and_embed(
        X_train, X_val, Y_train, Y_val, meta["target_cols"], os.path.join(gmvae_dir, "gmvae_stage")
    )
    _, gmvae_umap_train, gmvae_umap_val = fit_umap(Z_train, Z_val)

    extra_train = {"cluster_hard": C_train}
    extra_val = {"cluster_hard": C_val}
    for j in range(P_train.shape[1]):
        extra_train[f"cluster_prob_{j}"] = P_train[:, j]
        extra_val[f"cluster_prob_{j}"] = P_val[:, j]

    save_embedding_tables(
        out_dir=gmvae_dir,
        base_train=Z_train,
        base_val=Z_val,
        umap_train=gmvae_umap_train,
        umap_val=gmvae_umap_val,
        Y_train=Y_train,
        Y_val=Y_val,
        target_cols=meta["target_cols"],
        base_prefix="gmvae",
        extra_train=extra_train,
        extra_val=extra_val,
    )
    plot_umap_targets(gmvae_umap_train, gmvae_umap_val, Y_train, Y_val, meta["target_cols"], gmvae_dir, "GMVAE latent space")

    plt.figure(figsize=(8, 6))
    plt.scatter(gmvae_umap_train[:, 0], gmvae_umap_train[:, 1], c=C_train, s=18, alpha=0.8)
    plt.colorbar(label="GMVAE hard cluster")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("GMVAE latent UMAP - train clusters")
    save_current_figure(os.path.join(gmvae_dir, "umap_train_by_cluster.png"))

    plt.figure(figsize=(8, 6))
    plt.scatter(gmvae_umap_val[:, 0], gmvae_umap_val[:, 1], c=C_val, s=18, alpha=0.8)
    plt.colorbar(label="GMVAE hard cluster")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("GMVAE latent UMAP - validation clusters")
    save_current_figure(os.path.join(gmvae_dir, "umap_val_by_cluster.png"))

    gmvae_mapper_summary = save_mapper_outputs(
        lens_2d=np.vstack([gmvae_umap_train, gmvae_umap_val]),
        X_for_clustering=np.vstack([Z_train, Z_val]),
        out_dir=os.path.join(gmvae_dir, "mapper"),
        mapper_name=f"{dataset_name}_gmvae_umap_mapper",
    )
    method_summaries.append({
        "dataset": dataset_name,
        "method": "gmvae",
        "space_dim": Z_train.shape[1],
        **gmvae_mapper_summary,
    })

    append_global_summary({
        "dataset": dataset_name,
        "method": "gmvae",
        "n_features_in_space": Z_train.shape[1],
        "n_umap_points": int(len(gmvae_umap_train) + len(gmvae_umap_val)),
        "mapper_nodes": gmvae_mapper_summary["n_nodes"],
        "mapper_edges": gmvae_mapper_summary["n_edges"],
        "result_dir": gmvae_dir,
    }, global_summary_csv)

    method_df = pd.DataFrame(method_summaries)
    method_df.to_csv(os.path.join(dataset_dir, "method_summary.csv"), index=False)

    return method_df


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    global_summary_csv = os.path.join(RESULTS_ROOT, "all_datasets_method_summary.csv")
    if os.path.exists(global_summary_csv):
        os.remove(global_summary_csv)

    all_method_summaries = []

    for csv_path in CSV_LIST:
        if not os.path.exists(csv_path):
            print(f"[WARN] Missing file: {csv_path}")
            continue

        method_df = run_one_dataset(csv_path, global_summary_csv)
        all_method_summaries.append(method_df)

    if len(all_method_summaries) > 0:
        combined_df = pd.concat(all_method_summaries, axis=0, ignore_index=True)
        combined_df.to_csv(os.path.join(RESULTS_ROOT, "combined_method_summary.csv"), index=False)

    print(f"\nDone. Results saved in: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()

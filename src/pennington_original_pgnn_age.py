#!/usr/bin/env python3
"""Batch Pennington regression using Guoji Fu's original pGNNConv.

Runs male, female, and combined CSV files. Age prediction can be performed
either with or without ALM, BMD, and BFP as input features. Results are written
beneath one output directory. The implementation in src/pgnn_conv.py is
imported directly and is not copied or changed here.



   python3 run_pennington_original_pgnn_age.py \
  --csvs male.csv female.csv penn_data.csv \
  --targets Age \
  --age-feature-set without-dxa \
  --output results_age_without_dxa
  
  
  
  python3 run_pennington_original_pgnn_age.py \
  --csvs male.csv female.csv penn_data.csv \
  --targets Age \
  --age-feature-set with-dxa \
  --output results_age_with_dxa
"""

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from src.pgnn_conv import pGNNConv


TARGETS_DEFAULT = ["ALM", "BMD", "BFP"]
TARGET_CHOICES = ["Age", "ALM", "BMD", "BFP"]
P_VALUES_DEFAULT = [2.0, 3.0, 5.0, 10.0]
KNN_DEFAULT = [2, 5, 10, 15, 20, 25, 30, 35, 40]
DXA_COLUMNS = ["ALM", "BMD", "BFP"]
NON_FEATURE_COLUMNS = ["0", "PPT ID", "Site", "Gender", "Race"]


class OriginalPGNNRegressor(torch.nn.Module):
    """Two-hidden-layer regression wrapper around the original pGNNConv."""

    def __init__(self, in_features, hidden, mu, p, propagation_steps, dropout):
        super().__init__()
        self.dropout = dropout
        self.lin1 = torch.nn.Linear(in_features, hidden)
        self.bn1 = torch.nn.BatchNorm1d(hidden)
        self.lin2 = torch.nn.Linear(hidden, hidden)
        self.bn2 = torch.nn.BatchNorm1d(hidden)
        self.conv1 = pGNNConv(
            in_channels=hidden,
            out_channels=1,
            mu=mu,
            p=p,
            K=propagation_steps,
            cached=False,
        )

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.bn1(self.lin1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn2(self.lin2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv1(x, edge_index, edge_weight).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        help="Three CSV paths: male.csv female.csv and combined penn_data.csv",
    )
    parser.add_argument(
        "--targets", nargs="+", default=TARGETS_DEFAULT,
        choices=TARGET_CHOICES,
    )
    parser.add_argument(
        "--age-feature-set",
        choices=["with-dxa", "without-dxa"],
        default="without-dxa",
        help=(
            "For Age prediction, include or exclude ALM, BMD, and BFP. "
            "This option does not change non-Age tasks."
        ),
    )
    parser.add_argument("--output", required=True, help="Results directory")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--knn-list", nargs="+", type=int, default=KNN_DEFAULT,
        help="KNN graph sizes to sweep",
    )
    parser.add_argument(
        "--p-values", nargs="+", type=float, default=P_VALUES_DEFAULT,
        help="p values to sweep",
    )
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--propagation-steps", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def dataset_name(path):
    return Path(path).stem.replace(" ", "_").replace("(", "").replace(")", "")


def load_table(csv_path, target, age_feature_set):
    frame = pd.read_csv(csv_path)
    required = set(DXA_COLUMNS + ["Age"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")
    if frame.isna().any().any():
        cols = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"{csv_path} contains missing values in: {cols}")

    # The prediction target must never appear among the predictors.
    excluded = set(NON_FEATURE_COLUMNS + [target])

    if target == "Age":
        if age_feature_set == "without-dxa":
            excluded.update(DXA_COLUMNS)
        elif age_feature_set != "with-dxa":
            raise ValueError(f"Unknown Age feature set: {age_feature_set}")
    else:
        # For ALM/BMD/BFP tasks, all DXA outcomes are excluded while Age is
        # retained as an input, matching the original supplied program.
        excluded.update(DXA_COLUMNS)
    feature_names = [c for c in frame.columns if c not in excluded]
    nonnumeric = [c for c in feature_names if not pd.api.types.is_numeric_dtype(frame[c])]
    if nonnumeric:
        raise TypeError(f"Nonnumeric predictors remain in {csv_path}: {nonnumeric}")

    x = frame[feature_names].to_numpy(dtype=np.float64)
    y = frame[target].to_numpy(dtype=np.float64)
    return frame, x, y, feature_names


def make_graph(x, neighbors):
    n = x.shape[0]
    if neighbors < 1:
        raise ValueError("Every KNN value must be at least 1")
    # Validation folds can contain fewer than 41 participants. Match the
    # supplied sweep code by capping k at n-1 for each fold graph.
    effective_neighbors = min(neighbors, n - 1)

    adjacency = kneighbors_graph(
        x,
        n_neighbors=effective_neighbors,
        mode="distance",
        metric="euclidean",
        include_self=False,
    )
    # Symmetric union: an edge is retained if either endpoint selected the other.
    adjacency = adjacency.maximum(adjacency.T).tocoo()
    distances = adjacency.data.astype(np.float64)
    positive = distances[distances > 0]
    scale = float(positive.mean()) if positive.size else 1.0
    weights = np.exp(-distances / max(scale, 1e-12)).astype(np.float32)
    edge_index = torch.tensor(
        np.vstack([adjacency.row, adjacency.col]), dtype=torch.long
    )
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def prepare_fold(x_raw, y_raw, train_idx, val_idx, test_idx, neighbors, device):
    x_scaler = StandardScaler().fit(x_raw[train_idx])
    y_scaler = StandardScaler().fit(y_raw[train_idx].reshape(-1, 1))

    x_train = x_scaler.transform(x_raw[train_idx])
    x_val = x_scaler.transform(x_raw[val_idx])
    x_test = x_scaler.transform(x_raw[test_idx])
    y_train = y_scaler.transform(y_raw[train_idx].reshape(-1, 1)).ravel()
    y_val = y_scaler.transform(y_raw[val_idx].reshape(-1, 1)).ravel()
    y_test = y_raw[test_idx]

    train_edges, train_weights = make_graph(x_train, neighbors)
    val_edges, val_weights = make_graph(x_val, neighbors)
    test_edges, test_weights = make_graph(x_test, neighbors)

    return {
        "x_train": torch.tensor(x_train, dtype=torch.float32, device=device),
        "y_train": torch.tensor(y_train, dtype=torch.float32, device=device),
        "train_edges": train_edges.to(device),
        "train_weights": train_weights.to(device),
        "x_val": torch.tensor(x_val, dtype=torch.float32, device=device),
        "y_val": torch.tensor(y_val, dtype=torch.float32, device=device),
        "val_edges": val_edges.to(device),
        "val_weights": val_weights.to(device),
        "x_test": torch.tensor(x_test, dtype=torch.float32, device=device),
        "test_edges": test_edges.to(device),
        "test_weights": test_weights.to(device),
        "y_test": y_test,
        "y_scaler": y_scaler,
    }


@torch.no_grad()
def validation_loss(model, fold):
    model.eval()
    prediction = model(fold["x_val"], fold["val_edges"], fold["val_weights"])
    return F.mse_loss(prediction, fold["y_val"]).item()


def fit_fold(args, fold, fold_seed, p_value):
    seed_all(fold_seed)
    model = OriginalPGNNRegressor(
        in_features=fold["x_train"].shape[1],
        hidden=args.hidden,
        mu=args.mu,
        p=p_value,
        propagation_steps=args.propagation_steps,
        dropout=args.dropout,
    ).to(fold["x_train"].device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_loss = math.inf
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        prediction = model(
            fold["x_train"], fold["train_edges"], fold["train_weights"]
        )
        loss = F.mse_loss(prediction, fold["y_train"])
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch}; check p={p_value} and mu={args.mu}"
            )
        loss.backward()
        optimizer.step()

        current = validation_loss(model, fold)
        if current < best_loss:
            best_loss = current
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model")
    model.load_state_dict(best_state)
    return model, best_epoch


@torch.no_grad()
def predict_original_units(model, fold):
    model.eval()
    scaled = model(fold["x_test"], fold["test_edges"], fold["test_weights"])
    return fold["y_scaler"].inverse_transform(
        scaled.cpu().numpy().reshape(-1, 1)
    ).ravel()


def calculate_metrics(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # Same definition used by the supplied program: RMSE divided by RMS(y).
    denominator = float(np.sqrt(np.mean(y_true ** 2)))
    relative_rmse = 100.0 * rmse / max(denominator, 1e-12)
    return rmse, relative_rmse


def run_one_dataset_target(args, csv_path, target, output, device):
    frame, x_raw, y_raw, feature_names = load_table(
        csv_path, target, args.age_feature_set
    )
    name = dataset_name(csv_path)
    feature_set = args.age_feature_set if target == "Age" else "standard"
    task_dir = output / name / target / feature_set
    task_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    prediction_frames = []
    for p_value in args.p_values:
        for neighbors in args.knn_list:
            splitter = KFold(
                n_splits=args.folds, shuffle=True, random_state=args.seed
            )
            config_dir = task_dir / f"p_{p_value:g}" / f"knn_{neighbors}"
            config_dir.mkdir(parents=True, exist_ok=True)

            for fold_number, (train_idx, test_idx) in enumerate(
                splitter.split(x_raw), start=1
            ):
                start = time.perf_counter()
                fit_idx, val_idx = train_test_split(
                    train_idx,
                    test_size=0.15,
                    random_state=args.seed + fold_number,
                    shuffle=True,
                )
                fold = prepare_fold(
                    x_raw, y_raw, fit_idx, val_idx, test_idx, neighbors, device
                )
                model, best_epoch = fit_fold(
                    args, fold, args.seed + fold_number, p_value
                )
                predicted = predict_original_units(model, fold)
                actual = y_raw[test_idx]
                rmse, relative_rmse = calculate_metrics(actual, predicted)
                elapsed = time.perf_counter() - start

                rows.append({
                    "dataset": name,
                    "csv_path": str(Path(csv_path).resolve()),
                    "target": target,
                    "feature_set": feature_set,
                    "p": p_value,
                    "knn": neighbors,
                    "fold": fold_number,
                    "train_n": len(fit_idx),
                    "validation_n": len(val_idx),
                    "test_n": len(test_idx),
                    "mu": args.mu,
                    "propagation_steps": args.propagation_steps,
                    "best_epoch": best_epoch,
                    "RMSE": rmse,
                    "Relative_RMSE_pct": relative_rmse,
                    "elapsed_seconds": elapsed,
                })
                prediction_frames.append(pd.DataFrame({
                    "dataset": name,
                    "target": target,
                    "feature_set": feature_set,
                    "p": p_value,
                    "knn": neighbors,
                    "fold": fold_number,
                    "row_index": test_idx,
                    "PPT_ID": frame.iloc[test_idx]["PPT ID"].astype(str).to_numpy(),
                    "actual": actual,
                    "predicted": predicted,
                }))
                torch.save(
                    model.state_dict(),
                    config_dir / f"fold_{fold_number}_model.pt",
                )
                print(
                    f"{name:16s} {target} p={p_value:g} knn={neighbors} "
                    f"fold {fold_number}/{args.folds}: RMSE={rmse:.6f}, "
                    f"RelRMSE={relative_rmse:.4f}%"
                )
                del model, fold
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(task_dir / "fold_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        task_dir / "predictions.csv", index=False
    )
    with open(task_dir / "features.json", "w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2)
    return rows


def summarize(fold_frame):
    return (
        fold_frame.groupby(
            ["dataset", "csv_path", "target", "feature_set", "p", "knn"],
            as_index=False,
        )
        .agg(
            folds_completed=("fold", "nunique"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            Relative_RMSE_pct_mean=("Relative_RMSE_pct", "mean"),
            Relative_RMSE_pct_std=("Relative_RMSE_pct", "std"),
        )
    )


def save_best_rmse(summary, output):
    """Save the lowest five-fold mean RMSE for every dataset and target."""
    ordered = summary.sort_values(
        [
            "dataset",
            "target",
            "feature_set",
            "RMSE_mean",
            "Relative_RMSE_pct_mean",
            "p",
            "knn",
        ],
        ascending=True,
    )
    best = (
        ordered.groupby(
            ["dataset", "target", "feature_set"], as_index=False, sort=False
        )
        .head(1)
        .reset_index(drop=True)
    )
    best.to_csv(output / "best_rmse_all_datasets.csv", index=False)

    for name, dataset_best in best.groupby("dataset", sort=False):
        dataset_dir = output / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_best.to_csv(dataset_dir / "best_rmse.csv", index=False)

    return best


def main():
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if len(args.csvs) != 3:
        raise ValueError(f"Expected exactly 3 CSV paths, received {len(args.csvs)}")

    device = choose_device(args.device)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Results folder: {output}")

    all_rows = []
    for csv_path in args.csvs:
        for target in args.targets:
            all_rows.extend(
                run_one_dataset_target(args, csv_path, target, output, device)
            )

    fold_frame = pd.DataFrame(all_rows)
    fold_frame.to_csv(output / "all_fold_metrics.csv", index=False)
    summary = summarize(fold_frame)
    summary.to_csv(output / "summary_mean_std.csv", index=False)
    best = save_best_rmse(summary, output)

    print("\nCompleted all dataset-target combinations.")
    print(summary.to_string(index=False))
    print("\nBest configuration by lowest five-fold mean RMSE:")
    print(best.to_string(index=False))
    print(f"\nAll fold results: {output / 'all_fold_metrics.csv'}")
    print(f"Mean/std summary: {output / 'summary_mean_std.csv'}")
    print(f"Best RMSE results: {output / 'best_rmse_all_datasets.csv'}")


if __name__ == "__main__":
    main()

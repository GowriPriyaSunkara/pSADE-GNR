
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter


# ============================================================
# SETTINGS
# ============================================================

RESULTS_ROOT = "PGNN_ALL_RESULTS_May12"

TARGETS_TO_CHECK = ["ALM", "BMD", "BFP", "Age"]

FW_SOURCE_TO_CHECK = "none"
# Options:
# none, alm_c, alm_c2, bmd_c, bmd_c2,
# bfp_c, bfp_c2, age_c, age_c2

METRIC_TO_MINIMIZE = "RMSE_orig"
# Options:
# RMSE_orig or RelRMSE_orig_pct

MAIN_OUTPUT_FOLDER = (
    f"COMBINED_LOSS_LANDSCAPE_AND_HEATMAPS_ALL_TARGETS_"
    f"{FW_SOURCE_TO_CHECK}_{METRIC_TO_MINIMIZE}"
)

os.makedirs(MAIN_OUTPUT_FOLDER, exist_ok=True)

DPI = 350

FIGSIZE_LANDSCAPE = (24, 15)
FIGSIZE_HEATMAP = (18, 10)
FIGSIZE_LINE = (16, 8)

GRID_SIZE = 900
RBF_SMOOTH = 0.25
GAUSSIAN_SIGMA = 2.8


# ============================================================
# READ ALL RESULT CSV FILES ONCE
# ============================================================

csv_files = glob.glob(
    os.path.join(RESULTS_ROOT, "**", "results_*.csv"),
    recursive=True
)

if len(csv_files) == 0:
    raise FileNotFoundError(
        f"No results_*.csv files found inside {RESULTS_ROOT}. "
        "Check RESULTS_ROOT path."
    )

all_results = []

for file in csv_files:
    try:
        temp = pd.read_csv(file)
        temp["source_file"] = file
        all_results.append(temp)
    except Exception as e:
        print(f"Skipping {file}: {e}")

results = pd.concat(all_results, ignore_index=True)

print("\nAvailable columns:")
print(results.columns.tolist())

print("\nAvailable targets:")
print(results["target"].unique())

print("\nAvailable feature-weight sources:")
print(results["fw_source"].unique())


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_p_label(pval):
    pval = float(pval)

    if pval >= 1000:
        return f"{pval:.0e}"
    elif pval == int(pval):
        return str(int(pval))
    else:
        return str(pval)


def prepare_axis_labels(df):
    unique_p = sorted(df["p"].astype(float).unique())
    unique_logp = np.log10(unique_p)
    p_labels = [format_p_label(pval) for pval in unique_p]

    unique_knn = sorted(df["kNN_graph"].astype(float).unique())

    return unique_p, unique_logp, p_labels, unique_knn


def interpolate_surface(x, y, z, target_name, landscape_name):
    x_start = 0.0
    y_start = 0.0

    x_end = float(np.max(x))
    y_end = float(np.max(y))

    x_grid = np.linspace(x_start, x_end, GRID_SIZE)
    y_grid = np.linspace(y_start, y_end, GRID_SIZE)

    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    try:
        rbf = Rbf(
            x,
            y,
            z,
            function="multiquadric",
            smooth=RBF_SMOOTH
        )

        Z_grid = rbf(X_grid, Y_grid)
        Z_grid = gaussian_filter(Z_grid, sigma=GAUSSIAN_SIGMA)

        z_low = np.percentile(z, 1)
        z_high = np.percentile(z, 99)

        Z_grid = np.clip(Z_grid, z_low, z_high)

        return X_grid, Y_grid, Z_grid, x_end, y_end

    except Exception as e:
        print(
            f"RBF interpolation failed for target={target_name}, "
            f"landscape={landscape_name}: {e}"
        )
        return None, None, None, None, None


def plot_loss_landscape(
    df_for_landscape,
    overall_best,
    target_name,
    output_folder,
    landscape_type_name,
    landscape_title_extra,
    point_label
):
    """
    Creates one smooth contour loss landscape.

    df_for_landscape can be:
    1. Full df with all K values
    2. best_for_heatmap with only best K per (p, kNN_graph)
    """

    x = np.log10(df_for_landscape["p"].astype(float).values)
    y = df_for_landscape["kNN_graph"].astype(float).values
    z = df_for_landscape[METRIC_TO_MINIMIZE].astype(float).values

    x_best = np.log10(float(overall_best["p"]))
    y_best = float(overall_best["kNN_graph"])

    X_grid, Y_grid, Z_grid, x_end, y_end = interpolate_surface(
        x=x,
        y=y,
        z=z,
        target_name=target_name,
        landscape_name=landscape_type_name
    )

    if X_grid is None:
        return

    unique_p, unique_logp, p_labels, unique_knn = prepare_axis_labels(df_for_landscape)

    plt.figure(figsize=FIGSIZE_LANDSCAPE, facecolor="white")

    filled = plt.contourf(
        X_grid,
        Y_grid,
        Z_grid,
        levels=50
    )

    contour_lines = plt.contour(
        X_grid,
        Y_grid,
        Z_grid,
        levels=14,
        colors="black",
        linewidths=0.8
    )

    plt.clabel(
        contour_lines,
        inline=True,
        fontsize=10
    )

    cbar = plt.colorbar(filled, pad=0.025)
    cbar.set_label(METRIC_TO_MINIMIZE, fontsize=17, fontweight="bold")
    cbar.ax.tick_params(labelsize=14)

    # Tested / selected configurations
    plt.scatter(
        x,
        y,
        s=70,
        edgecolor="black",
        linewidth=1.1,
        label=point_label
    )

    # One star only: overall best
    plt.scatter(
        x_best,
        y_best,
        s=650,
        marker="*",
        edgecolor="black",
        linewidth=1.6,
        label="Overall best configuration"
    )

    # Best point label
    best_label = (
        f"Best\n"
        f"p={overall_best['p']}\n"
        f"kNN={int(overall_best['kNN_graph'])}\n"
        f"K={int(overall_best['K'])}\n"
        f"{METRIC_TO_MINIMIZE}={overall_best[METRIC_TO_MINIMIZE]:.4f}"
    )

    plt.text(
        x_best + 0.10,
        y_best,
        best_label,
        fontsize=14,
        fontweight="bold",
        va="center",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="black")
    )

    plt.xlim(0, x_end)
    plt.ylim(0, y_end)

    # x-axis ticks show actual p values
    x_ticks = [0] + list(unique_logp)
    x_tick_labels = ["0"] + p_labels

    plt.xticks(
        x_ticks,
        x_tick_labels,
        fontsize=15,
        fontweight="bold"
    )

    # y-axis ticks show kNN values
    y_ticks = [0] + unique_knn
    y_ticks = sorted(set(y_ticks))

    plt.yticks(
        y_ticks,
        [str(int(v)) for v in y_ticks],
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel("p", fontsize=20, fontweight="bold")
    plt.ylabel("kNN graph neighbors", fontsize=20, fontweight="bold")

    plt.title(
        f"PGNN Loss Landscape for {target_name}\n"
        f"{landscape_title_extra}",
        fontsize=25,
        fontweight="bold",
        pad=22
    )

    plt.legend(
        fontsize=15,
        loc="upper right",
        frameon=True
    )

    plt.grid(alpha=0.25)

    loss_landscape_path = os.path.join(
        output_folder,
        f"{landscape_type_name}_loss_landscape_"
        f"p_on_x_knn_on_y_{target_name}_"
        f"{FW_SOURCE_TO_CHECK}_{METRIC_TO_MINIMIZE}.png"
    )

    plt.savefig(
        loss_landscape_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(f"Saved {landscape_type_name} loss landscape to:")
    print(loss_landscape_path)


def plot_metric_heatmap(
    best_for_heatmap,
    overall_best,
    target_name,
    output_folder
):
    heatmap_table = best_for_heatmap.pivot_table(
        index="kNN_graph",
        columns="p",
        values=METRIC_TO_MINIMIZE,
        aggfunc="min"
    )

    heatmap_table = heatmap_table.sort_index(axis=0)
    heatmap_table = heatmap_table.reindex(sorted(heatmap_table.columns), axis=1)

    heatmap_csv = os.path.join(
        output_folder,
        f"heatmap_table_{target_name}_{FW_SOURCE_TO_CHECK}_{METRIC_TO_MINIMIZE}.csv"
    )

    heatmap_table.to_csv(heatmap_csv)

    plt.figure(figsize=FIGSIZE_HEATMAP, facecolor="white")

    im = plt.imshow(
        heatmap_table.values,
        aspect="auto",
        origin="lower"
    )

    cbar = plt.colorbar(im, pad=0.02)
    cbar.set_label(METRIC_TO_MINIMIZE, fontsize=15, fontweight="bold")
    cbar.ax.tick_params(labelsize=12)

    heatmap_p_values = list(heatmap_table.columns.astype(float))
    heatmap_p_labels = [format_p_label(pval) for pval in heatmap_p_values]

    plt.xticks(
        ticks=np.arange(len(heatmap_p_values)),
        labels=heatmap_p_labels,
        fontsize=12,
        fontweight="bold",
        rotation=45
    )

    heatmap_knn_values = list(heatmap_table.index.astype(float))

    plt.yticks(
        ticks=np.arange(len(heatmap_knn_values)),
        labels=[str(int(v)) for v in heatmap_knn_values],
        fontsize=12,
        fontweight="bold"
    )

    # Add numbers inside heatmap cells
    for i in range(heatmap_table.shape[0]):
        for j in range(heatmap_table.shape[1]):
            value = heatmap_table.values[i, j]

            if not np.isnan(value):
                plt.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold"
                )

    # Mark overall best cell with one star
    best_p = float(overall_best["p"])
    best_knn = float(overall_best["kNN_graph"])

    if best_p in heatmap_p_values and best_knn in heatmap_knn_values:
        best_j = heatmap_p_values.index(best_p)
        best_i = heatmap_knn_values.index(best_knn)

        plt.scatter(
            best_j,
            best_i,
            s=500,
            marker="*",
            edgecolor="black",
            linewidth=1.5,
            label="Overall best"
        )

    plt.xlabel("p", fontsize=16, fontweight="bold")
    plt.ylabel("kNN graph neighbors", fontsize=16, fontweight="bold")

    plt.title(
        f"Heatmap of Best {METRIC_TO_MINIMIZE} for {target_name}\n"
        f"Best K selected within each (p, kNN) cell",
        fontsize=19,
        fontweight="bold",
        pad=18
    )

    plt.legend(
        fontsize=12,
        loc="upper right",
        frameon=True
    )

    heatmap_path = os.path.join(
        output_folder,
        f"heatmap_best_{METRIC_TO_MINIMIZE}_"
        f"p_vs_knn_{target_name}_{FW_SOURCE_TO_CHECK}.png"
    )

    plt.savefig(
        heatmap_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print("Saved metric heatmap to:")
    print(heatmap_path)
    print("Saved metric heatmap table to:")
    print(heatmap_csv)


def plot_r2_heatmap(
    best_for_heatmap,
    overall_best,
    target_name,
    output_folder
):
    r2_table = best_for_heatmap.pivot_table(
        index="kNN_graph",
        columns="p",
        values="R2",
        aggfunc="first"
    )

    r2_table = r2_table.sort_index(axis=0)
    r2_table = r2_table.reindex(sorted(r2_table.columns), axis=1)

    r2_heatmap_csv = os.path.join(
        output_folder,
        f"heatmap_table_R2_{target_name}_{FW_SOURCE_TO_CHECK}.csv"
    )

    r2_table.to_csv(r2_heatmap_csv)

    plt.figure(figsize=FIGSIZE_HEATMAP, facecolor="white")

    im2 = plt.imshow(
        r2_table.values,
        aspect="auto",
        origin="lower"
    )

    cbar2 = plt.colorbar(im2, pad=0.02)
    cbar2.set_label("R2", fontsize=15, fontweight="bold")
    cbar2.ax.tick_params(labelsize=12)

    heatmap_p_values = list(r2_table.columns.astype(float))
    heatmap_p_labels = [format_p_label(pval) for pval in heatmap_p_values]

    plt.xticks(
        ticks=np.arange(len(heatmap_p_values)),
        labels=heatmap_p_labels,
        fontsize=12,
        fontweight="bold",
        rotation=45
    )

    heatmap_knn_values = list(r2_table.index.astype(float))

    plt.yticks(
        ticks=np.arange(len(heatmap_knn_values)),
        labels=[str(int(v)) for v in heatmap_knn_values],
        fontsize=12,
        fontweight="bold"
    )

    # Add numbers inside R2 heatmap cells
    for i in range(r2_table.shape[0]):
        for j in range(r2_table.shape[1]):
            value = r2_table.values[i, j]

            if not np.isnan(value):
                plt.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold"
                )

    # Star marks same overall best RMSE/metric cell
    best_p = float(overall_best["p"])
    best_knn = float(overall_best["kNN_graph"])

    if best_p in heatmap_p_values and best_knn in heatmap_knn_values:
        best_j = heatmap_p_values.index(best_p)
        best_i = heatmap_knn_values.index(best_knn)

        plt.scatter(
            best_j,
            best_i,
            s=500,
            marker="*",
            edgecolor="black",
            linewidth=1.5,
            label=f"Overall best {METRIC_TO_MINIMIZE} cell"
        )

    plt.xlabel("p", fontsize=16, fontweight="bold")
    plt.ylabel("kNN graph neighbors", fontsize=16, fontweight="bold")

    plt.title(
        f"Heatmap of R2 for {target_name}\n"
        f"R2 from the best-{METRIC_TO_MINIMIZE} K within each (p, kNN) cell",
        fontsize=19,
        fontweight="bold",
        pad=18
    )

    plt.legend(
        fontsize=12,
        loc="upper right",
        frameon=True
    )

    r2_heatmap_path = os.path.join(
        output_folder,
        f"heatmap_R2_p_vs_knn_{target_name}_{FW_SOURCE_TO_CHECK}.png"
    )

    plt.savefig(
        r2_heatmap_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print("Saved R2 heatmap to:")
    print(r2_heatmap_path)
    print("Saved R2 heatmap table to:")
    print(r2_heatmap_csv)


def plot_best_per_p_line(
    best_per_p,
    target_name,
    output_folder
):
    unique_p = sorted(best_per_p["p"].astype(float).unique())
    unique_logp = np.log10(unique_p)
    p_labels = [format_p_label(pval) for pval in unique_p]

    plt.figure(figsize=FIGSIZE_LINE, facecolor="white")

    plt.plot(
        np.log10(best_per_p["p"].astype(float)),
        best_per_p[METRIC_TO_MINIMIZE],
        marker="o",
        linewidth=3
    )

    for _, row in best_per_p.iterrows():
        plt.text(
            np.log10(float(row["p"])),
            row[METRIC_TO_MINIMIZE],
            f"kNN={int(row['kNN_graph'])}, K={int(row['K'])}",
            fontsize=10,
            fontweight="bold"
        )

    plt.xticks(
        unique_logp,
        p_labels,
        fontsize=13,
        fontweight="bold"
    )

    plt.xlabel("p", fontsize=16, fontweight="bold")
    plt.ylabel(METRIC_TO_MINIMIZE, fontsize=16, fontweight="bold")

    plt.title(
        f"Best {METRIC_TO_MINIMIZE} for Each p: {target_name}",
        fontsize=19,
        fontweight="bold"
    )

    plt.grid(alpha=0.3)

    best_line_path = os.path.join(
        output_folder,
        f"best_metric_for_each_p_{target_name}_{FW_SOURCE_TO_CHECK}.png"
    )

    plt.savefig(
        best_line_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print("Saved best-per-p line plot to:")
    print(best_line_path)


# ============================================================
# MAIN FUNCTION FOR ONE TARGET
# ============================================================

def make_all_plots_for_target(target_name):
    print("\n" + "=" * 90)
    print(f"Processing target: {target_name}")
    print("=" * 90)

    output_folder = os.path.join(
        MAIN_OUTPUT_FOLDER,
        f"{target_name}_{FW_SOURCE_TO_CHECK}_{METRIC_TO_MINIMIZE}"
    )

    os.makedirs(output_folder, exist_ok=True)

    # ========================================================
    # FILTER TARGET AND FEATURE-WEIGHT SOURCE
    # ========================================================

    df = results[
        (results["target"].astype(str).str.upper() == target_name.upper()) &
        (results["fw_source"].astype(str) == FW_SOURCE_TO_CHECK)
    ].copy()

    df = df.replace([np.inf, -np.inf], np.nan)

    needed_cols = [
        "p",
        "K",
        "kNN_graph",
        METRIC_TO_MINIMIZE,
        "RMSE_orig",
        "RelRMSE_orig_pct",
        "R2"
    ]

    optional_print_cols = [
        "target",
        "fw_source",
        "p",
        "kNN_graph",
        "K",
        "mu_used",
        "RMSE_orig",
        "RelRMSE_orig_pct",
        "R2",
        "best_epoch",
        "best_lr"
    ]

    missing_cols = [col for col in needed_cols if col not in df.columns]

    if missing_cols:
        print(f"Missing required columns for {target_name}: {missing_cols}")
        print("Skipping this target.")
        return

    df = df.dropna(subset=needed_cols)

    df["p"] = df["p"].astype(float)
    df["K"] = df["K"].astype(float)
    df["kNN_graph"] = df["kNN_graph"].astype(float)
    df[METRIC_TO_MINIMIZE] = df[METRIC_TO_MINIMIZE].astype(float)
    df["RMSE_orig"] = df["RMSE_orig"].astype(float)
    df["RelRMSE_orig_pct"] = df["RelRMSE_orig_pct"].astype(float)
    df["R2"] = df["R2"].astype(float)

    if df.empty:
        print(
            f"No matching rows found for target={target_name}, "
            f"fw_source={FW_SOURCE_TO_CHECK}. Skipping."
        )
        return

    # ========================================================
    # SAVE FILTERED DATA
    # ========================================================

    filtered_csv = os.path.join(
        output_folder,
        f"filtered_results_{target_name}_{FW_SOURCE_TO_CHECK}.csv"
    )

    df.to_csv(filtered_csv, index=False)

    print("Saved filtered results to:")
    print(filtered_csv)

    # ========================================================
    # OVERALL BEST RESULT
    # One star only: best across all p, kNN_graph, and K
    # ========================================================

    overall_best = df.loc[df[METRIC_TO_MINIMIZE].idxmin()].copy()

    overall_best_csv = os.path.join(
        output_folder,
        f"overall_best_{target_name}_{FW_SOURCE_TO_CHECK}_{METRIC_TO_MINIMIZE}.csv"
    )

    pd.DataFrame([overall_best]).to_csv(overall_best_csv, index=False)

    print("\nOverall best result:")

    existing_print_cols = [
        col for col in optional_print_cols if col in overall_best.index
    ]

    print(overall_best[existing_print_cols])

    print("\nSaved overall best result to:")
    print(overall_best_csv)

    # ========================================================
    # BEST RESULT FOR EACH p
    # ========================================================

    best_per_p = (
        df.sort_values(METRIC_TO_MINIMIZE, ascending=True)
          .groupby("p", as_index=False)
          .first()
          .sort_values("p")
    )

    best_per_p_csv = os.path.join(
        output_folder,
        f"best_result_for_each_p_{target_name}_{FW_SOURCE_TO_CHECK}_{METRIC_TO_MINIMIZE}.csv"
    )

    best_per_p.to_csv(best_per_p_csv, index=False)

    print("\nSaved best result for each p to:")
    print(best_per_p_csv)

    # ========================================================
    # BEST RESULT FOR EACH p AND kNN_graph
    #
    # This selects the best K inside each p and kNN cell.
    # This is used for:
    # 1. Heatmap
    # 2. Second loss landscape
    # ========================================================

    best_for_heatmap = (
        df.sort_values(METRIC_TO_MINIMIZE, ascending=True)
          .groupby(["p", "kNN_graph"], as_index=False)
          .first()
          .sort_values(["p", "kNN_graph"])
    )

    best_for_heatmap_csv = os.path.join(
        output_folder,
        f"best_for_heatmap_each_p_knn_{target_name}_{FW_SOURCE_TO_CHECK}_{METRIC_TO_MINIMIZE}.csv"
    )

    best_for_heatmap.to_csv(best_for_heatmap_csv, index=False)

    print("\nSaved best result for each p and kNN_graph to:")
    print(best_for_heatmap_csv)

    # ========================================================
    # LOSS LANDSCAPE 1:
    # ALL CONFIGURATIONS, INCLUDING ALL K VALUES
    # ========================================================

    plot_loss_landscape(
        df_for_landscape=df,
        overall_best=overall_best,
        target_name=target_name,
        output_folder=output_folder,
        landscape_type_name="all_K_configs",
        landscape_title_extra=f"Interpolated {METRIC_TO_MINIMIZE} Surface Using All K Configurations",
        point_label="All tested PGNN configurations"
    )

    # ========================================================
    # LOSS LANDSCAPE 2:
    # BEST K ONLY FOR EACH (p, kNN_graph)
    # ========================================================

    plot_loss_landscape(
        df_for_landscape=best_for_heatmap,
        overall_best=overall_best,
        target_name=target_name,
        output_folder=output_folder,
        landscape_type_name="best_K_per_p_knn",
        landscape_title_extra=f"Interpolated {METRIC_TO_MINIMIZE} Surface Using Best K per (p, kNN)",
        point_label="Best K selected per (p, kNN)"
    )

    # ========================================================
    # HEATMAP 1:
    # BEST METRIC FOR EACH p AND kNN_graph
    # ========================================================

    plot_metric_heatmap(
        best_for_heatmap=best_for_heatmap,
        overall_best=overall_best,
        target_name=target_name,
        output_folder=output_folder
    )

    # ========================================================
    # HEATMAP 2:
    # R2 FROM BEST-METRIC ROW IN EACH p AND kNN_graph
    # ========================================================

    plot_r2_heatmap(
        best_for_heatmap=best_for_heatmap,
        overall_best=overall_best,
        target_name=target_name,
        output_folder=output_folder
    )

    # ========================================================
    # LINE PLOT:
    # BEST RESULT FOR EACH p
    # ========================================================

    plot_best_per_p_line(
        best_per_p=best_per_p,
        target_name=target_name,
        output_folder=output_folder
    )

    # ========================================================
    # SAVE SUMMARY TEXT
    # ========================================================

    summary_txt = os.path.join(
        output_folder,
        f"summary_{target_name}_{FW_SOURCE_TO_CHECK}_{METRIC_TO_MINIMIZE}.txt"
    )

    with open(summary_txt, "w") as f:
        f.write(f"Target: {target_name}\n")
        f.write(f"Feature-weight source: {FW_SOURCE_TO_CHECK}\n")
        f.write(f"Metric visualized/minimized: {METRIC_TO_MINIMIZE}\n\n")

        f.write("Outputs created:\n")
        f.write("1. Loss landscape using all K configurations.\n")
        f.write("2. Loss landscape using best K per (p, kNN_graph).\n")
        f.write("3. Heatmap of best metric per (p, kNN_graph).\n")
        f.write("4. Heatmap of R2 from the best-metric row per (p, kNN_graph).\n")
        f.write("5. Best metric for each p line plot.\n\n")

        f.write("Interpretation:\n")
        f.write("x-axis: p, displayed as actual p values.\n")
        f.write("Internally, p is interpolated on log10(p).\n")
        f.write("y-axis: kNN graph neighbors.\n")
        f.write(f"color/contour value: {METRIC_TO_MINIMIZE}.\n")
        f.write("The star marks the overall best configuration across all tested p, kNN_graph, and K.\n\n")

        f.write("For the heatmap, each (p, kNN_graph) cell contains the best value after selecting the best K for that cell.\n\n")

        f.write("Overall best result:\n")
        f.write(str(overall_best[existing_print_cols]))

    print("\nSaved summary to:")
    print(summary_txt)


# ============================================================
# RUN FOR ALL TARGETS
# ============================================================

for target_name in TARGETS_TO_CHECK:
    make_all_plots_for_target(target_name)

print("\nDone. All outputs saved inside:")
print(MAIN_OUTPUT_FOLDER)
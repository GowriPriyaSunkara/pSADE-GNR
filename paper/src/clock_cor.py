
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ============================================================
# Load dataset safely
# ============================================================

script_dir = os.path.dirname(os.path.abspath(__file__))

possible_paths = [
    os.path.join(script_dir, "penn_data.csv"),
    os.path.join(script_dir, "data", "penn_data.csv"),
    os.path.join(script_dir, "..", "data", "penn_data.csv"),
    os.path.join(script_dir, "..", "..", "data", "penn_data.csv"),
    "/scratch/gsunka1/TDA_guoji/penn_data.csv",
    "/scratch/gsunka1/TDA_guoji/data/penn_data.csv",
]

DATA_PATH = None

for path in possible_paths:
    if os.path.exists(path):
        DATA_PATH = path
        break

if DATA_PATH is None:
    raise FileNotFoundError(
        "\nCould not find penn_data.csv in any expected location.\n\n"
        "Checked these paths:\n"
        + "\n".join(possible_paths)
        + "\n\nPlease put penn_data.csv in the same folder as clock_corr.py, "
        "or manually set DATA_PATH to the correct full path."
    )

df = pd.read_csv(DATA_PATH)

print("Loaded file:", DATA_PATH)
print("Dataset shape:", df.shape)
print("Columns in dataset:")
print(df.columns.tolist())

# ============================================================
# Output folder
# ============================================================

OUTPUT_DIR = os.path.join(script_dir, "Pennington_Correlation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nAll results will be saved in:", OUTPUT_DIR)

# ============================================================
# Settings
# ============================================================

TARGETS = ["ALM", "BMD", "BFP", "Age"]

# Columns that should NOT be used as clock spokes
EXCLUDE_COLS = [
    "0",
    "PPT ID",
    "Site",
    "Age",
    "Gender",
    "Race",
    "ALM",
    "BMD",
    "BFP",
]

# Remove unnamed columns automatically
unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed")]

DROP_COLS = list(set(EXCLUDE_COLS + unnamed_cols))

print("\nColumns removed from clock spokes:")
print([col for col in DROP_COLS if col in df.columns])

# ============================================================
# Build feature matrix
# ============================================================

feature_df = df.drop(columns=DROP_COLS, errors="ignore").select_dtypes(include=[np.number])

# Remove columns that are completely empty
feature_df = feature_df.dropna(axis=1, how="all")

# Remove constant columns
constant_cols = [
    col for col in feature_df.columns
    if feature_df[col].nunique(dropna=True) <= 1
]

if constant_cols:
    print("\nConstant columns removed:")
    print(constant_cols)
    feature_df = feature_df.drop(columns=constant_cols)

# Fill missing values with column means
feature_df = feature_df.fillna(feature_df.mean())

features = feature_df.columns.tolist()

if len(features) == 0:
    raise ValueError("No numeric biomarker columns found after removing excluded columns.")

print("\nNumber of biomarker spokes used in the clock:", len(features))
print("Biomarker spokes:")
print(features)

# Save feature spokes used
spokes_path = os.path.join(OUTPUT_DIR, "feature_spokes_used.csv")
pd.DataFrame({"Feature_Spoke": features}).to_csv(spokes_path, index=False)
print("\nSaved:", spokes_path)

# ============================================================
# Normalize features
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_df.values)

# ============================================================
# Compute correlations
# ============================================================

corr_dict = {}
missing_targets = []

for target in TARGETS:
    if target not in df.columns:
        missing_targets.append(target)
        continue

    y = df[target].values
    valid_idx = ~pd.isna(y)

    X_valid = X_scaled[valid_idx, :]
    y_valid = y[valid_idx]

    correlations = []

    for j in range(X_valid.shape[1]):
        xj = X_valid[:, j]

        if np.std(xj) == 0 or np.std(y_valid) == 0:
            corr = np.nan
        else:
            corr = np.corrcoef(xj, y_valid)[0, 1]

        correlations.append(corr)

    corr_dict[target] = correlations

if missing_targets:
    print("\nWarning: These targets were not found and were skipped:")
    print(missing_targets)

corr_df = pd.DataFrame(corr_dict, index=features)
corr_df.index.name = "Feature"

abs_corr_df = corr_df.abs()

# ============================================================
# Save correlation CSV files
# ============================================================

signed_csv_path = os.path.join(OUTPUT_DIR, "correlation_clock_values_signed.csv")
absolute_csv_path = os.path.join(OUTPUT_DIR, "correlation_clock_values_absolute.csv")

corr_df.to_csv(signed_csv_path)
abs_corr_df.to_csv(absolute_csv_path)

print("\nSaved:", signed_csv_path)
print("Saved:", absolute_csv_path)

# ============================================================
# Helper function: clean display labels
# ============================================================

def clean_feature_label(label):
    """
    Cleans feature names only for graph display.
    It does NOT change the original dataframe column names.

    Main change:
        Circumference -> Circum
    """

    label = str(label)

    replacements = {
        "Circumference": "Circum",
    }

    for old, new in replacements.items():
        label = label.replace(old, new)

    return label

# ============================================================
# Helper function: make column names three-line labels
# ============================================================

def make_three_line_label(label):
    """
    Converts long column names into up to three-line labels.

    Examples:
        Height (cm)
        -> Height
           (cm)

        Abdomen Circum
        -> Abdomen
           Circum

        Upper Arm Circum Right
        -> Upper Arm
           Circum
           Right
    """

    label = str(label)
    words = label.split()

    if len(words) <= 1:
        return label

    if len(words) == 2:
        return words[0] + "\n" + words[1]

    if len(words) == 3:
        return words[0] + "\n" + words[1] + "\n" + words[2]

    # For 4 or more words, split into 3 balanced lines
    n = len(words)

    first_cut = n // 3
    second_cut = 2 * n // 3

    if first_cut == 0:
        first_cut = 1

    if second_cut <= first_cut:
        second_cut = first_cut + 1

    line1 = " ".join(words[:first_cut])
    line2 = " ".join(words[first_cut:second_cut])
    line3 = " ".join(words[second_cut:])

    return line1 + "\n" + line2 + "\n" + line3


# Clean labels only for plotting
cleaned_features = [clean_feature_label(label) for label in features]

# Wrap cleaned labels into three lines
wrapped_features = [make_three_line_label(label) for label in cleaned_features]

# Save wrapped labels also
wrapped_labels_path = os.path.join(OUTPUT_DIR, "feature_spokes_three_line_cleaned_labels.csv")

pd.DataFrame({
    "Original_Feature": features,
    "Cleaned_Display_Label": cleaned_features,
    "Three_Line_Label": wrapped_features
}).to_csv(wrapped_labels_path, index=False)

print("Saved:", wrapped_labels_path)

# ============================================================
# Helper function: add correlation values beside dots, fontweight="bold",
# ============================================================

def add_value_labels(ax, angles, values, fontsize=15, offset=0.06, fmt="{:.2f}"):
    rmin, rmax = ax.get_ylim()

    for ang, val in zip(angles[:-1], values[:-1]):
        if np.isnan(val):
            continue

        if val >= 0:
            r_text = min(val + offset, rmax - 0.02)
            va = "bottom"
        else:
            r_text = max(val - offset, rmin + 0.02)
            va = "top"

        ax.text(
            ang,
            r_text,
            fmt.format(val),
            fontsize=fontsize,            
            ha="center",
            va=va
        )

# ============================================================
# Helper function: bigger bold legend box
# ============================================================

def make_big_bold_legend(ax):
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        prop={"size": 20, "weight": "bold"},
        markerscale=2.0,
        handlelength=3.0,
        handletextpad=1.0,
        borderpad=1.2,
        labelspacing=1.0,
        columnspacing=2.0,
        frameon=True
    )

    legend.get_frame().set_linewidth(2)
    legend.get_frame().set_edgecolor("black")

    return legend

# ============================================================
# Prepare radar/clock angles
# ============================================================

N = len(features)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

# Close the circle
angles += angles[:1]

# ============================================================
# Signed correlation clock
# ============================================================

plt.figure(figsize=(20, 20))
ax = plt.subplot(111, polar=True)

for target in corr_df.columns:
    values = corr_df[target].values.tolist()
    values += values[:1]

    ax.plot(
        angles,
        values,
        linewidth=2,
        marker="o",
        markersize=4,
        label=target
    )

    ax.fill(
        angles,
        values,
        alpha=0.05
    )

    add_value_labels(
        ax,
        angles,
        values,
        fontsize=15,
        offset=0.07,
        fmt="{:.2f}"
    )

ax.set_xticks(angles[:-1])
ax.set_xticklabels(wrapped_features, fontweight="bold", fontsize=15, color="navy")

ax.set_ylim(-1, 1)

# Optional radial tick labels:
# ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
# ax.set_yticklabels(["-1.0", "-0.5", "0", "0.5", "1.0"], fontsize=15, fontweight="bold")

ax.set_title(
    "Pennington Anthropometric Correlation Clock\n"
    "Signed Correlations of ALM, BMD, BFP, and Age with Anthropometric Features",
    fontsize=25,
    fontweight="bold",
    pad=45
)

make_big_bold_legend(ax)

plt.tight_layout()

signed_png_path = os.path.join(
    OUTPUT_DIR,
    "pennington_correlation_clock_signed_labeled_circum_three_line_names_big_bold_legend.png"
)

plt.savefig(signed_png_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", signed_png_path)

# ============================================================
# Absolute correlation clock
# ============================================================

plt.figure(figsize=(20, 20))
ax = plt.subplot(111, polar=True)

for target in abs_corr_df.columns:
    values = abs_corr_df[target].values.tolist()
    values += values[:1]

    ax.plot(
        angles,
        values,
        linewidth=2,
        marker="o",
        markersize=4,
        label=target
    )

    ax.fill(
        angles,
        values,
        alpha=0.05
    )

    add_value_labels(
        ax,
        angles,
        values,
        fontsize=15,
        offset=0.04,
        fmt="{:.2f}"
    )

ax.set_xticks(angles[:-1])
ax.set_xticklabels(wrapped_features, fontsize=15, color="navy")

ax.set_ylim(0, 1)

# Optional radial tick labels:
# ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
# ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1.0"], fontsize=15, fontweight="bold"), , fontweight="bold"

ax.set_title(
    "Pennington Anthropometric Correlation Clock\n"
    "Absolute Correlation Strengths for ALM, BMD, BFP, and Age",
    fontsize=25,
    fontweight="bold",
    pad=45
)

make_big_bold_legend(ax)

plt.tight_layout()

absolute_png_path = os.path.join(
    OUTPUT_DIR,
    "pennington_correlation_clock_absolute_labeled_circum_three_line_names_big_bold_legend.png"
)

plt.savefig(absolute_png_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", absolute_png_path)

# ============================================================
# Save top correlations
# ============================================================

top_txt_path = os.path.join(OUTPUT_DIR, "top_correlations_by_target.txt")

with open(top_txt_path, "w") as f:
    f.write("Top correlations by absolute value\n")
    f.write("=" * 50 + "\n\n")

    for target in corr_df.columns:
        f.write(f"Target: {target}\n")
        f.write("-" * 30 + "\n")

        top_corr = (
            corr_df[target]
            .dropna()
            .reindex(corr_df[target].abs().sort_values(ascending=False).index)
            .head(10)
        )

        f.write(top_corr.to_string())
        f.write("\n\n")

print("Saved:", top_txt_path)

# ============================================================
# Print top correlations
# ============================================================

print("\nTop correlations by absolute value:")

for target in corr_df.columns:
    print("\nTarget:", target)

    top_corr = (
        corr_df[target]
        .dropna()
        .reindex(corr_df[target].abs().sort_values(ascending=False).index)
        .head(10)
    )

    print(top_corr)

print("\nDone. All results saved inside:")
print(OUTPUT_DIR)
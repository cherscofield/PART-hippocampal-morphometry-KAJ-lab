"""
Left Hemisphere Gaussian Curvature Analysis with FDR Correction
Hippocampal morphometry in PART with and without TDP-43 co-pathology
"""

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection

# =============================================================================
# DATA PATHS - UPDATE THESE FOR YOUR SYSTEM
# =============================================================================
CURVATURE_FILE = "path/to/your/lh_gaussian_curvature.csv"
OUTPUT_DIR = "output/"  # Directory for saving results

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
df = pd.read_csv(CURVATURE_FILE)

df = df.rename(columns={
    'lh.mid-surface.gauss-curv.csv-all': 'gauss_curvature',
    'PART=1_control=0': 'group_PART_vs_control',
    'x': 'grid_x',
    'y': 'grid_y',
    'z': 'grid_z'
})

# =============================================================================
# PART VS CONTROL ANALYSIS
# =============================================================================
group_part = df[df["group_PART_vs_control"] == 1]
group_control = df[df["group_PART_vs_control"] == 0]
all_coords = sorted(df[["grid_x", "grid_y"]].drop_duplicates().values.tolist())

p_values, t_stats, coords = [], [], []
for x, y in all_coords:
    part_vals = group_part[(group_part["grid_x"] == x) & (group_part["grid_y"] == y)]["gauss_curvature"].dropna()
    control_vals = group_control[(group_control["grid_x"] == x) & (group_control["grid_y"] == y)]["gauss_curvature"].dropna()
    if len(part_vals) > 1 and len(control_vals) > 1:
        t_stat, p_val = ttest_ind(part_vals, control_vals, equal_var=False)
        p_values.append(p_val)
        t_stats.append(t_stat)
        coords.append((x, y))

rejected, pvals_corrected = fdrcorrection(p_values, alpha=0.05)

# Create significance map
x_max = df["grid_x"].max()
y_max = df["grid_y"].max()
significance_map = np.full((y_max + 1, x_max + 1), np.nan)
for idx, (x, y) in enumerate(coords):
    if rejected[idx]:
        significance_map[y, x] = t_stats[idx]

# Plot PART vs Control (flip x for LH)
fig, ax = plt.subplots(figsize=(10, 6))
c = ax.imshow(np.fliplr(significance_map), cmap="bwr", interpolation="none", origin="lower",
              extent=[x_max, 0, 0, y_max])
ax.set_title("Significant Differences (PART vs Control) in LH Gaussian Curvature (FDR)", fontsize=10)
ax.set_xlabel("X axis (Lateral to Medial)")
ax.set_ylabel("Y axis (Posterior to Anterior)")
fig.colorbar(c, ax=ax, label="t-statistic (PART - Control)")
plt.savefig(f"{OUTPUT_DIR}lh_gauss_PART_vs_Control_FDR.png", dpi=300)
plt.close()

print(f"LH Gaussian significance map saved.")

# =============================================================================
# TDP GROUP PAIRWISE COMPARISONS
# =============================================================================
def analyze_gauss_curv_tdp_groups_lh(df, group_a, group_b, label_a, label_b, output_prefix):
    """Perform FDR-corrected t-tests between TDP groups for left hemisphere."""
    group_A = df[df["tdp_status"] == group_a]
    group_B = df[df["tdp_status"] == group_b]
    all_coords = sorted(df[["grid_x", "grid_y"]].drop_duplicates().values.tolist())

    p_values, t_stats, coords = [], [], []
    for x, y in all_coords:
        a_vals = group_A[(group_A["grid_x"] == x) & (group_A["grid_y"] == y)]["gauss_curvature"].dropna()
        b_vals = group_B[(group_B["grid_x"] == x) & (group_B["grid_y"] == y)]["gauss_curvature"].dropna()
        if len(a_vals) > 1 and len(b_vals) > 1:
            t_stat, p_val = ttest_ind(a_vals, b_vals, equal_var=False)
            p_values.append(p_val)
            t_stats.append(t_stat)
            coords.append((x, y))

    rejected, _ = fdrcorrection(p_values, alpha=0.05)
    x_max, y_max = df["grid_x"].max(), df["grid_y"].max()
    significance_map = np.full((y_max + 1, x_max + 1), np.nan)
    for idx, (x, y) in enumerate(coords):
        if rejected[idx]:
            significance_map[y, x] = t_stats[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.imshow(np.fliplr(significance_map), cmap="bwr", interpolation="none", origin="lower",
                  extent=[x_max, 0, 0, y_max])
    ax.set_title(f"Significant Differences ({label_a} vs {label_b}) in LH Gaussian Curvature", fontsize=10)
    ax.set_xlabel("X axis (Lateral to Medial)")
    ax.set_ylabel("Y axis (Posterior to Anterior)")
    fig.colorbar(c, ax=ax, label=f"t-statistic ({label_a} - {label_b})")
    plt.savefig(f"{OUTPUT_DIR}lh_gauss_{output_prefix}_FDR.png", dpi=300)
    plt.close()
    print(f"Saved: lh_gauss_{output_prefix}_FDR.png")

    # Per-subject average
    significant_coords = [coords[i] for i, sig in enumerate(rejected) if sig]
    results = []
    for mrn in df["mrn"].unique():
        subj_data = df[df["mrn"] == mrn]
        group_val = subj_data["tdp_status"].iloc[0]
        if group_val in [group_a, group_b]:
            values = []
            for (x, y) in significant_coords:
                val = subj_data[(subj_data["grid_x"] == x) & (subj_data["grid_y"] == y)]["gauss_curvature"]
                if not val.empty:
                    values.append(val.values[0])
            if values:
                results.append((mrn, group_val, np.mean(values)))

    results_df = pd.DataFrame(results, columns=["mrn", "group", "gauss_curvature"])
    group_A_vals = results_df[results_df["group"] == group_a]["gauss_curvature"]
    group_B_vals = results_df[results_df["group"] == group_b]["gauss_curvature"]

    print(f"\n--- Average Gaussian Curvature ({label_a} vs {label_b}) ---")
    if len(group_A_vals) > 0:
        print(f"{label_a} (n={len(group_A_vals)}): mean = {group_A_vals.mean():.5f}, std = {group_A_vals.std():.5f}")
    if len(group_B_vals) > 0:
        print(f"{label_b} (n={len(group_B_vals)}): mean = {group_B_vals.mean():.5f}, std = {group_B_vals.std():.5f}")

# Run TDP comparisons
analyze_gauss_curv_tdp_groups_lh(df, 0, 1, "TDP-", "TDP+", "TDPneg_vs_TDPpos")
analyze_gauss_curv_tdp_groups_lh(df, 1, 2, "TDP+", "Control", "TDPpos_vs_Control")
analyze_gauss_curv_tdp_groups_lh(df, 0, 2, "TDP-", "Control", "TDPneg_vs_Control")

# =============================================================================
# LINE PLOTS ALONG AXES
# =============================================================================
def plot_gauss_curv_line_by_axis_lh(df, axis='x', group_col='tdp_status',
                                    group_a=0, group_b=1, label_a="TDP-", label_b="TDP+",
                                    output_path="lineplot.png"):
    """Plot mean Gaussian curvature along specified axis with FDR significance markers."""
    axis_col = 'grid_x' if axis == 'x' else 'grid_y'
    axis_label = "X (Lateral to Medial)" if axis == 'x' else "Y (Posterior to Anterior)"

    group_A = df[df[group_col] == group_a]
    group_B = df[df[group_col] == group_b]
    points = sorted(df[axis_col].unique())

    group_A_means, group_B_means = [], []
    group_A_sems, group_B_sems = [], []
    p_values_line = []

    for pt in points:
        a_vals = group_A[group_A[axis_col] == pt].groupby("mrn")["gauss_curvature"].mean()
        b_vals = group_B[group_B[axis_col] == pt].groupby("mrn")["gauss_curvature"].mean()

        group_A_means.append(a_vals.mean())
        group_B_means.append(b_vals.mean())
        group_A_sems.append(a_vals.sem())
        group_B_sems.append(b_vals.sem())

        if len(a_vals) > 1 and len(b_vals) > 1:
            _, p_val = ttest_ind(a_vals, b_vals, equal_var=False)
            p_values_line.append(p_val)
        else:
            p_values_line.append(np.nan)

    # FDR correction
    valid_p = [p for p in p_values_line if not np.isnan(p)]
    if len(valid_p) > 0:
        rejected_line, _ = fdrcorrection(valid_p, alpha=0.05)
        full_rejected = []
        j = 0
        for p in p_values_line:
            if np.isnan(p):
                full_rejected.append(False)
            else:
                full_rejected.append(rejected_line[j])
                j += 1
    else:
        full_rejected = [False] * len(p_values_line)

    # Plot
    plt.figure(figsize=(10, 6))
    pts = np.array(points)

    plt.plot(pts, group_A_means, label=label_a, color='blue')
    plt.fill_between(pts, np.array(group_A_means) - np.array(group_A_sems),
                     np.array(group_A_means) + np.array(group_A_sems), color='blue', alpha=0.2)

    plt.plot(pts, group_B_means, label=label_b, color='red')
    plt.fill_between(pts, np.array(group_B_means) - np.array(group_B_sems),
                     np.array(group_B_means) + np.array(group_B_sems), color='red', alpha=0.2)

    for i, sig in enumerate(full_rejected):
        if sig:
            plt.plot(pts[i], max(group_A_means[i], group_B_means[i]), 'k*', markersize=8)

    plt.xlabel(axis_label)
    plt.ylabel("Gaussian Curvature")
    plt.title(f"{label_a} vs {label_b}: Gaussian Curvature Along {'Medial-Lateral' if axis == 'x' else 'Posterior-Anterior'} Axis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Line plot saved to: {output_path}")

# Y-axis comparisons
plot_gauss_curv_line_by_axis_lh(df, axis='y', group_col='tdp_status',
    group_a=0, group_b=1, label_a="TDP-", label_b="TDP+",
    output_path=f"{OUTPUT_DIR}lh_gauss_yaxis_TDPneg_vs_TDPpos.png")

plot_gauss_curv_line_by_axis_lh(df, axis='y', group_col='tdp_status',
    group_a=1, group_b=2, label_a="TDP+", label_b="Control",
    output_path=f"{OUTPUT_DIR}lh_gauss_yaxis_TDPpos_vs_Control.png")

plot_gauss_curv_line_by_axis_lh(df, axis='y', group_col='tdp_status',
    group_a=0, group_b=2, label_a="TDP-", label_b="Control",
    output_path=f"{OUTPUT_DIR}lh_gauss_yaxis_TDPneg_vs_Control.png")

plot_gauss_curv_line_by_axis_lh(df, axis='y', group_col='group_PART_vs_control',
    group_a=1, group_b=0, label_a="PART", label_b="Control",
    output_path=f"{OUTPUT_DIR}lh_gauss_yaxis_PART_vs_Control.png")

# X-axis comparisons
plot_gauss_curv_line_by_axis_lh(df, axis='x', group_col='tdp_status',
    group_a=0, group_b=1, label_a="TDP-", label_b="TDP+",
    output_path=f"{OUTPUT_DIR}lh_gauss_xaxis_TDPneg_vs_TDPpos.png")

plot_gauss_curv_line_by_axis_lh(df, axis='x', group_col='tdp_status',
    group_a=1, group_b=2, label_a="TDP+", label_b="Control",
    output_path=f"{OUTPUT_DIR}lh_gauss_xaxis_TDPpos_vs_Control.png")

plot_gauss_curv_line_by_axis_lh(df, axis='x', group_col='tdp_status',
    group_a=0, group_b=2, label_a="TDP-", label_b="Control",
    output_path=f"{OUTPUT_DIR}lh_gauss_xaxis_TDPneg_vs_Control.png")

plot_gauss_curv_line_by_axis_lh(df, axis='x', group_col='group_PART_vs_control',
    group_a=1, group_b=0, label_a="PART", label_b="Control",
    output_path=f"{OUTPUT_DIR}lh_gauss_xaxis_PART_vs_Control.png")

print("\nAnalysis complete!")

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend to avoid Tkinter errors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# DATA PATHS - UPDATE THESE FOR YOUR SYSTEM
# =============================================================================
# Right hemisphere thickness grid data from HIPSTA (41x21 grid)
THICKNESS_FILE = "path/to/your/rh_thickness_data.csv"

# === Load the CSV file ===
df = pd.read_csv(THICKNESS_FILE)


# === Transform to long format ===
df_long = df.melt(
    id_vars=["mrn", "axis", "PART=1_control=0", "tdp_status"], 
    value_vars=[col for col in df.columns if col.startswith("y")],
    var_name="y_point", value_name="z_value"
)

# Extract numeric x and y positions
df_long["x_index"] = df_long["axis"].str.extract(r"x(\d+)").astype(int)
df_long["y_index"] = df_long["y_point"].str.extract(r"y(\d+)").astype(int)

# === Loop through all subjects and generate plots ===
#for mrn in df["mrn"].unique():
 #   subject_data = df_long[df_long["mrn"] == mrn]

  #  if subject_data.empty:
   #     print(f"Skipping subject {mrn} — no data")
    #    continue

    # Extract group and TDP info
    #group = subject_data["PART=1_control=0"].iloc[0]
    #tdp = subject_data["tdp_status"].iloc[0]

#    group_label = "Control" if group == 0 else ("PART (TDP+)" if tdp == 1 else "PART (TDP−)")

#    # Create thickness grid
#    grid = subject_data.pivot_table(index="y_index", columns="x_index", values="z_value", aggfunc="mean")

#    fig = plt.figure(figsize=(10, 6))
#    ax = fig.add_subplot(111, projection='3d')

#    flipped_columns = grid.columns[::-1]
#    X, Y = np.meshgrid(flipped_columns, grid.index)
#    Z = grid[flipped_columns].values

#    surf = ax.plot_surface(X, Y, Z, cmap='viridis')

    # Titles and labels
#    ax.set_title(f"Subject {mrn} - {group_label} (Right Side)")
#    ax.set_xlabel("X axis (Medial to Lateral)")
#    ax.set_ylabel("Y axis (Posterior to Anterior)")
#    ax.set_zlabel("Z value")

    # DO NOT invert x-axis on right side
    # ax.invert_xaxis()

    # Group label on figure
#    ax.text2D(0.05, 0.92, f"{group_label}", transform=ax.transAxes, fontsize=12, color="black", weight="bold")

#    fig.colorbar(surf, shrink=0.5, aspect=5)
#    plt.tight_layout()

    # Save with clean name
#    clean_label = group_label.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "pos").replace("−", "neg")
#    output_path = f"path/to/your/output.png"  # UPDATE: Replace with your data path
#    plt.savefig(output_path)
#    plt.close()

#    print(f"Saved: {output_path}")

######################################################################################################
# === Group-average surface plot: PART vs Control ===

# Function to compute group average grid
def get_group_average(df_long, group_value):
    group_data = df_long[df_long["PART=1_control=0"] == group_value]
    mean_values = group_data.groupby(["y_index", "x_index"])["z_value"].mean().unstack()
    return mean_values

# Compute average grids for both groups
part_grid = get_group_average(df_long, 1)
control_grid = get_group_average(df_long, 0)

# Plot both surfaces side by side
fig = plt.figure(figsize=(14, 6))

# PART
ax1 = fig.add_subplot(121, projection='3d')
X1, Y1 = np.meshgrid(part_grid.columns, part_grid.index)
Z1 = part_grid.values
surf1 = ax1.plot_surface(X1, Y1, Z1, cmap='plasma')
ax1.set_title("PART (Group = 1)")
ax1.set_xlabel("X (Medial to Lateral)")
ax1.set_ylabel("Y (Posterior to Anterior)")
ax1.set_zlabel("Z value")

# Control
ax2 = fig.add_subplot(122, projection='3d')
X2, Y2 = np.meshgrid(control_grid.columns, control_grid.index)
Z2 = control_grid.values
surf2 = ax2.plot_surface(X2, Y2, Z2, cmap='viridis')
ax2.set_title("Control (Group = 0)")
ax2.set_xlabel("X (Medial to Lateral)")
ax2.set_ylabel("Y (Posterior to Anterior)")
ax2.set_zlabel("Z value")

plt.tight_layout()
output_path_comparison = "path/to/your/output.png"  # UPDATE: Replace with your output path
plt.savefig(output_path_comparison)

print(f"PART vs Control plot saved to {output_path_comparison}")



########################################################################################################
# === Group-average surface plot: TDP-43 Status (0, 1, 2) ===

# Function to compute average grid for any tdp_status group
def get_tdp_average(df_long, tdp_value):
    group_data = df_long[df_long["tdp_status"] == tdp_value]
    mean_values = group_data.groupby(["y_index", "x_index"])["z_value"].mean().unstack()
    return mean_values

# Compute average grids
tdp0_grid = get_tdp_average(df_long, 0)
tdp1_grid = get_tdp_average(df_long, 1)
tdp2_grid = get_tdp_average(df_long, 2)

# Create figure with 3 subplots
fig = plt.figure(figsize=(20, 10))

# TDP 0
ax1 = fig.add_subplot(131, projection='3d')
X0, Y0 = np.meshgrid(tdp0_grid.columns, tdp0_grid.index)
Z0 = tdp0_grid.values
surf0 = ax1.plot_surface(X0, Y0, Z0, cmap='Greens')
ax1.set_title("PART without TDP-43")
ax1.set_xlabel("X (Medial to Lateral)")
ax1.set_ylabel("Y (Posterior to Anterior)")
ax1.set_zlabel("Z value")

# TDP 1
ax2 = fig.add_subplot(132, projection='3d')
X1, Y1 = np.meshgrid(tdp1_grid.columns, tdp1_grid.index)
Z1 = tdp1_grid.values
surf1 = ax2.plot_surface(X1, Y1, Z1, cmap='Oranges')
ax2.set_title("PART with TDP-43")
ax2.set_xlabel("X (Medial to Lateral)")
ax2.set_ylabel("Y (Posterior to Anterior)")
ax2.set_zlabel("Z value")

# TDP 2
ax3 = fig.add_subplot(133, projection='3d')
X2, Y2 = np.meshgrid(tdp2_grid.columns, tdp2_grid.index)
Z2 = tdp2_grid.values
surf2 = ax3.plot_surface(X2, Y2, Z2, cmap='Purples')
ax3.set_title("Controls")
ax3.set_xlabel("X (Medial to Lateral)")
ax3.set_ylabel("Y (Posterior to Anterior)")
ax3.set_zlabel("Z value")

plt.tight_layout()
output_path_tdp = "path/to/your/output.png"  # UPDATE: Replace with your output path
plt.savefig(output_path_tdp)

print(f"TDP status comparison plot saved to {output_path_tdp}")



# === PART vs Control Statistical Analysis ===
# Per-point t-tests with FDR correction

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection

# === Step 1: Prepare data for t-tests ===
# Create a matrix of shape (subjects, x, y) for each group
group_part = df_long[df_long["PART=1_control=0"] == 1]
group_control = df_long[df_long["PART=1_control=0"] == 0]

# Get all unique (x, y) points
all_points = sorted(df_long[["x_index", "y_index"]].drop_duplicates().values.tolist())

# Store results
p_values = []
t_stats = []
coords = []

for x, y in all_points:
    part_vals = group_part[(group_part["x_index"] == x) & (group_part["y_index"] == y)]["z_value"].dropna()
    control_vals = group_control[(group_control["x_index"] == x) & (group_control["y_index"] == y)]["z_value"].dropna()

    # Ensure both groups have data
    if len(part_vals) > 1 and len(control_vals) > 1:
        t_stat, p_val = ttest_ind(part_vals, control_vals, equal_var=False)
        p_values.append(p_val)
        t_stats.append(t_stat)
        coords.append((x, y))

# === Step 2: Apply FDR correction ===
rejected, pvals_corrected = fdrcorrection(p_values, alpha=0.05)

# === Step 3: Create a significance mask for plotting ===
significance_map = np.full((21, max(df_long["x_index"]) + 1), np.nan)  # shape = (y, x)

for idx, (x, y) in enumerate(coords):
    if rejected[idx]:
        significance_map[y, x] = t_stats[idx]  # optionally show direction (t-statistic)

# === Step 4: Plot significant areas ===
fig, ax = plt.subplots(figsize=(10, 6))
c = ax.imshow(significance_map, cmap="bwr", interpolation="none", origin="lower",
              extent=[0, max(df_long["x_index"]), 0, 20])
ax.set_title("Significant Differences (PART vs Control) in Thickness (FDR-corrected)",fontsize=10)
ax.set_xlabel("X axis (Medial to Lateral)")
ax.set_ylabel("Y axis (Posterior to Anterior)")
fig.colorbar(c, ax=ax, label="t-statistic (PART - Control)")

# Save the plot
output_path_sigmap = "path/to/your/output.png"  # UPDATE: Replace with your output path
plt.savefig(output_path_sigmap)

print(f"Significance map saved to {output_path_sigmap}")


# === Calculate per-subject average thickness at significant vertices ===
# Get (x, y) coordinates that were significant
significant_coords = [coords[i] for i, sig in enumerate(rejected) if sig]

# === Step 2: For each subject, calculate mean thickness over significant points ===
subject_ids = df["mrn"].unique()
results = []

for mrn in subject_ids:
    subject_data = df_long[df_long["mrn"] == mrn]
    
    # Extract values at significant points
    values = []
    for (x, y) in significant_coords:
        val = subject_data[(subject_data["x_index"] == x) & (subject_data["y_index"] == y)]["z_value"]
        if not val.empty:
            values.append(val.values[0])
    
    # Store mean thickness and group label
    if values:  # Ensure we have data
        mean_thickness = np.mean(values)
        group = subject_data["PART=1_control=0"].iloc[0]
        results.append((mrn, group, mean_thickness))

# === Step 3: Convert to DataFrame for analysis ===
results_df = pd.DataFrame(results, columns=["mrn", "group", "mean_thickness"])

# Split by group
part_group = results_df[results_df["group"] == 1]["mean_thickness"]
control_group = results_df[results_df["group"] == 0]["mean_thickness"]

# Print summary stats
print("\n--- Average Thickness in Significant Regions ---")
print(f"PART group (n={len(part_group)}): mean = {part_group.mean():.4f}, std = {part_group.std():.4f}")
print(f"Control group (n={len(control_group)}): mean = {control_group.mean():.4f}, std = {control_group.std():.4f}")


# === TDP Status Group Comparisons ===
# Per-point t-tests with FDR correction for TDP-negative vs TDP-positive comparisons
def analyze_tdp_groups(df_long, group_a, group_b, label_a, label_b, output_prefix):
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import fdrcorrection

    group_A = df_long[df_long["tdp_status"] == group_a]
    group_B = df_long[df_long["tdp_status"] == group_b]

    all_points = sorted(df_long[["x_index", "y_index"]].drop_duplicates().values.tolist())

    p_values, t_stats, coords = [], [], []
    for x, y in all_points:
        a_vals = group_A[(group_A["x_index"] == x) & (group_A["y_index"] == y)]["z_value"].dropna()
        b_vals = group_B[(group_B["x_index"] == x) & (group_B["y_index"] == y)]["z_value"].dropna()
        if len(a_vals) > 1 and len(b_vals) > 1:
            t_stat, p_val = ttest_ind(a_vals, b_vals, equal_var=False)
            p_values.append(p_val)
            t_stats.append(t_stat)
            coords.append((x, y))

    # Apply FDR correction
    rejected, _ = fdrcorrection(p_values, alpha=0.05)
    
    # Create significance map
    significance_map = np.full((21, df_long["x_index"].max() + 1), np.nan)
    for idx, (x, y) in enumerate(coords):
        if rejected[idx]:
            significance_map[y, x] = t_stats[idx]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.imshow(significance_map, cmap="bwr", interpolation="none", origin="lower",
                  extent=[0, df_long["x_index"].max(), 0, 20])
    ax.set_title(f"Significant Differences ({label_a} vs {label_b}) in Thickness (FDR-corrected)",fontsize=10)
    ax.set_xlabel("X axis (Medial to Lateral)")
    ax.set_ylabel("Y axis (Posterior to Anterior)")
    fig.colorbar(c, ax=ax, label=f"t-statistic ({label_a} - {label_b})")

    output_path = f"path/to/your/output.png"  # UPDATE: Replace with your data path
    plt.savefig(output_path)
    plt.close()
    print(f"\nSignificance map saved to {output_path}")

    # Quantify per-subject thickness across significant (x, y)
    significant_coords = [coords[i] for i, sig in enumerate(rejected) if sig]

    significant_coords = [coords[i] for i, p in enumerate(p_values) if p < 0.05]

    results = []
    for mrn in df["mrn"].unique():
        subj_data = df_long[df_long["mrn"] == mrn]
        group_val = subj_data["tdp_status"].iloc[0]
        if group_val in [group_a, group_b]:
            values = []
            for (x, y) in significant_coords:
                val = subj_data[(subj_data["x_index"] == x) & (subj_data["y_index"] == y)]["z_value"]
                if not val.empty:
                    values.append(val.values[0])
            if values:
                mean_thickness = np.mean(values)
                results.append((mrn, group_val, mean_thickness))

    results_df = pd.DataFrame(results, columns=["mrn", "group", "mean_thickness"])
    group_A_vals = results_df[results_df["group"] == group_a]["mean_thickness"]
    group_B_vals = results_df[results_df["group"] == group_b]["mean_thickness"]

    print(f"\n--- Average Thickness in Significant Regions ({label_a} vs {label_b}) ---")
    if not group_A_vals.empty and not group_B_vals.empty:
        print(f"{label_a} group (n={len(group_A_vals)}): mean = {group_A_vals.mean():.4f}, std = {group_A_vals.std():.4f}")
        print(f"{label_b} group (n={len(group_B_vals)}): mean = {group_B_vals.mean():.4f}, std = {group_B_vals.std():.4f}")
    else:
        print("No significant differences found after FDR correction.")

# === Run TDP comparisons ===
analyze_tdp_groups(df_long, 0, 1, "PART without TDP", "PART with TDP", "PART without TDP_vs_PART with TDP")
analyze_tdp_groups(df_long, 1, 2, "PART with TDP", "Controls", "PART with TDP_vs_Controls")
analyze_tdp_groups(df_long, 0, 2, "PART without TDP", "Controls", "PART without TDP_vs_Controls")


##########################################################################################################
################################## line plotting #######################################################
####################################################################################################


def plot_thickness_line_by_axis_rh(df_long, axis='x', group_col='tdp_status',
                                   group_a=0, group_b=1, label_a="Group A", label_b="Group B",
                                   output_path="lineplot.png"):
    axis_col = 'x_index' if axis == 'x' else 'y_index'
    axis_label = "X (Medial to Lateral)" if axis == 'x' else "Y (Posterior to Anterior)"

    group_A = df_long[df_long[group_col] == group_a]
    group_B = df_long[df_long[group_col] == group_b]

    points = sorted(df_long[axis_col].unique())

    group_A_means, group_B_means = [], []
    group_A_sems, group_B_sems = [], []
    p_values, t_stats = [], []

    for pt in points:
        a_vals = group_A[group_A[axis_col] == pt].groupby("mrn")["z_value"].mean()
        b_vals = group_B[group_B[axis_col] == pt].groupby("mrn")["z_value"].mean()

        group_A_means.append(a_vals.mean())
        group_B_means.append(b_vals.mean())
        group_A_sems.append(a_vals.sem())
        group_B_sems.append(b_vals.sem())

        if len(a_vals) > 1 and len(b_vals) > 1:
            t_stat, p_val = ttest_ind(a_vals, b_vals, equal_var=False)
            t_stats.append(t_stat)
            p_values.append(p_val)
        else:
            t_stats.append(np.nan)
            p_values.append(np.nan)

    rejected, _ = fdrcorrection(p_values, alpha=0.05)

    # Plot
    plt.figure(figsize=(10, 6))
    pts = np.array(points)  # DO NOT reverse pts for RH

    plt.plot(pts, group_A_means, label=label_a, color='blue')
    plt.fill_between(pts, np.array(group_A_means) - np.array(group_A_sems),
                     np.array(group_A_means) + np.array(group_A_sems), color='blue', alpha=0.2)

    plt.plot(pts, group_B_means, label=label_b, color='red')
    plt.fill_between(pts, np.array(group_B_means) - np.array(group_B_sems),
                     np.array(group_B_means) + np.array(group_B_sems), color='red', alpha=0.2)

    for i, sig in enumerate(rejected):
        if sig:
            plt.plot(pts[i], max(group_A_means[i], group_B_means[i]), 'k*', markersize=8)

    plt.xlabel(axis_label)
    plt.ylabel("Thickness (Z value)")
    plt.title(f"{label_a} vs {label_b}: RH Thickness Along {'Medial-Lateral' if axis == 'x' else 'Posterior-Anterior'} Axis")
    plt.legend()
    plt.tight_layout()

    # DO NOT invert X for RH
    plt.savefig(output_path)
    plt.close()
    print(f"RH line plot saved to: {output_path}")

plot_thickness_line_by_axis_rh(df_long, axis='y', group_col='tdp_status',
    group_a=0, group_b=1, label_a="TDP-", label_b="TDP+",
    output_path="path/to/your/output.png")

plot_thickness_line_by_axis_rh(df_long, axis='y', group_col='tdp_status',
    group_a=1, group_b=2, label_a="TDP+", label_b="Control",
    output_path="path/to/your/output.png")

plot_thickness_line_by_axis_rh(df_long, axis='y', group_col='tdp_status',
    group_a=0, group_b=2, label_a="TDP-", label_b="Control",
    output_path="path/to/your/output.png")

plot_thickness_line_by_axis_rh(df_long, axis='y', group_col='PART=1_control=0',
    group_a=1, group_b=0, label_a="PART", label_b="Control",
    output_path="path/to/your/output.png")

plot_thickness_line_by_axis_rh(df_long, axis='x', group_col='tdp_status',
    group_a=0, group_b=1, label_a="TDP-", label_b="TDP+",
    output_path="path/to/your/output.png")

plot_thickness_line_by_axis_rh(df_long, axis='x', group_col='tdp_status',
    group_a=1, group_b=2, label_a="TDP+", label_b="Control",
    output_path="path/to/your/output.png")

plot_thickness_line_by_axis_rh(df_long, axis='x', group_col='tdp_status',
    group_a=0, group_b=2, label_a="TDP-", label_b="Control",
    output_path="path/to/your/output.png")

plot_thickness_line_by_axis_rh(df_long, axis='x', group_col='PART=1_control=0',
    group_a=1, group_b=0, label_a="PART", label_b="Control",
    output_path="path/to/your/output.png")







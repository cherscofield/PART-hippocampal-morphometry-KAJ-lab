import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_test
from matplotlib.colors import LinearSegmentedColormap
import pyvista as pv
import statsmodels.api as sm
np.random.seed(42)

# =============================================================================
# DATA PATHS - UPDATE THESE FOR YOUR SYSTEM
# =============================================================================
# Left hemisphere thickness grid data from HIPSTA (41x21 grid)
THICKNESS_FILE = "path/to/your/lh_thickness_data.csv"
# Left hemisphere subfield labels for plot overlay
SUBFIELD_LABELS_FILE = "path/to/your/lh_subfield_labels.csv"

# === Load the CSV file ===
df = pd.read_csv(THICKNESS_FILE)

# === Load Subfield Outline ===
outline_df = pd.read_csv(SUBFIELD_LABELS_FILE)

# === Overlay subfield boundaries ===
label_matrix = outline_df.pivot_table(index="x", columns="y", values="subfield label", aggfunc="first").to_numpy()
label_matrix = np.fliplr(label_matrix.T)
label_matrix = np.fliplr(label_matrix)  # 🔥 Flip again inside plotting

# Extract numeric x index
df['x_index'] = df['axis'].str.extract(r'x(\d+)').astype(int)

# Get unique subjects
subjects = df['mrn'].unique()

# === Organize into Groups ===
group0 = []  # Control
group1 = []  # PART

for subj in subjects:
    subj_df = df[df['mrn'] == subj].sort_values("x_index")
    matrix = subj_df.loc[:, "y0":"y20"].to_numpy()
    group_label = subj_df["PART=1_control=0"].iloc[0]
    if group_label == 0:
        group0.append(matrix)
    else:
        group1.append(matrix)

group0 = np.array(group0)
group1 = np.array(group1)

print(f"Group 0 (Control): {group0.shape}")
print(f"Group 1 (PART): {group1.shape}")

# === Run Cluster Permutation with MNE ===
X = [group0, group1]

T_obs, clusters, p_values, H0 = permutation_cluster_test(
    X,
    n_permutations=1000,     # 🔸 default was 1000 permutations
    tail=0,                  # two-sided test
    n_jobs=1                 # single-threaded
    )

# === Significance Mask ===
significance_mask = np.zeros_like(T_obs, dtype=bool)
for i, p_val in enumerate(p_values):
    if p_val < 0.05:
        significance_mask[clusters[i]] = True

# === Hedges' g calculation ===
def compute_hedges_g(g1, g2):
    mean_diff = np.nanmean(g1, axis=0) - np.nanmean(g2, axis=0)
    n1, n2 = g1.shape[0], g2.shape[0]
    pooled_sd = np.sqrt(((n1 - 1) * np.nanvar(g1, axis=0) + (n2 - 1) * np.nanvar(g2, axis=0)) / (n1 + n2 - 2))
    d = mean_diff / pooled_sd
    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return d * correction

# === Plotting Cluster Permutation with Hedges' g shown only in significant regions ===
def plot_Cluster Permutation_with_effect_size(d_map, sig_mask, title, save_path, outline_df):
    flipped_d = np.fliplr(d_map.T)
    flipped_mask = np.fliplr(sig_mask.T)
    label_matrix = outline_df.pivot_table(index="x", columns="y", values="subfield label", aggfunc="first").to_numpy()
    label_matrix = np.fliplr(label_matrix.T)

    masked_d = np.ma.masked_where(~flipped_mask, flipped_d)

    fig, ax = plt.subplots(figsize=(10, 6))
    vmin = 0
    vmax = np.nanmax(d_map)
    im = ax.imshow(masked_d, cmap='Reds', origin='lower', extent=[40, 0, 0, 20], aspect='auto',
               vmin=vmin, vmax=vmax)
    ax.contour(flipped_mask, levels=[0.5], colors='black', linewidths=0.5,
               origin='lower', extent=[40, 0, 0, 20])
    ax.contour(label_matrix, levels=np.unique(label_matrix), linewidths=0.5,
               colors='gray', origin='lower', extent=[40, 0, 0, 20])
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Lateral -> Medial", fontsize=14)
    ax.set_ylabel("Posterior -> Anterior", fontsize=14)
    ax.tick_params(axis='y', labelsize=18)
    cbar = plt.colorbar(im, ax=ax, label="Hedges' g (masked by Cluster Permutation significance)")
    cbar.ax.tick_params(labelsize=18)  
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# === Cluster Permutation Result Summary ===
def summarize_Cluster Permutation_results(d_map, sig_mask, group0, group1, label):
    n_sig = np.sum(sig_mask)
    print(f"\n--- {label} ---")
    print(f"Significant Points: {n_sig}")

    if n_sig > 0 and np.sum(~np.isnan(d_map[sig_mask])) > 0:
        mean_effect = np.nanmean(d_map[sig_mask])
        max_effect = np.nanmax(np.abs(d_map[sig_mask]))
        print(f"Mean Hedges' g in sig. area: {mean_effect:.3f}")
        print(f"Max abs(Hedges' g): {max_effect:.3f}")
    else:
        print("No significant regions to summarize (empty or NaN values).")

    print(f"Sample Sizes: Group0 = {group0.shape[0]}, Group1 = {group1.shape[0]}")

# === PART vs Control plot (Control - PART) ===
d_part_control = compute_hedges_g(group0, group1)
plot_Cluster Permutation_with_effect_size(d_part_control, significance_mask, "Cluster Permutation Significance over Hedges' g\n(Control vs PART)", "lh.Cluster Permutation_hedgesg_part_vs_control.png", outline_df)
summarize_Cluster Permutation_results(d_part_control, significance_mask, group0, group1, "Control vs PART")

# === TDP subgroups Cluster Permutation and Effect Overlay ===
def run_Cluster Permutation_comparison(df, group_a, group_b, label, save_name):
    group0, group1 = [], []

    for subj in df['mrn'].unique():
        subj_df = df[df['mrn'] == subj].sort_values("x_index")
        tdp_label = subj_df['tdp_status'].iloc[0]
        if tdp_label not in [group_a, group_b]:
            continue
        matrix = subj_df.loc[:, "y0":"y20"].to_numpy()
        if tdp_label == group_a:
            group0.append(matrix)
        elif tdp_label == group_b:
            group1.append(matrix)

    group0 = np.array(group0)
    group1 = np.array(group1)
    print(f"{label} comparison -> Group {group_a}: {group0.shape}, Group {group_b}: {group1.shape}")

    d_map = compute_hedges_g(group0, group1)
    X = [group0, group1]
    T_obs, clusters, p_values, H0 = permutation_cluster_test(
        X,
        n_permutations=1000,     # 🔸 default was 1000 permutations
        tail=0,                  # two-sided test
        n_jobs=1                 # single-threaded
        )

    sig_mask = np.zeros_like(T_obs, dtype=bool)
    for i, p in enumerate(p_values):
        if p < 0.05:
            sig_mask[clusters[i]] = True

    safe_save_name = save_name.replace("−", "-").replace("–", "-")  # Ensure ASCII-safe filenames
    effect_overlay_name = safe_save_name.replace("Cluster Permutation", "Cluster Permutation_hedgesg")
    plot_Cluster Permutation_with_effect_size(d_map, sig_mask, f"Cluster Permutation Significance over Hedges' g\n{label}", effect_overlay_name, outline_df)
    summarize_Cluster Permutation_results(d_map, sig_mask, group0, group1, label)
     # Check if sig_mask has any True values
    if np.any(sig_mask) and np.sum(~np.isnan(d_map[sig_mask])) > 0:
        max_effect = np.nanmax(np.abs(d_map[sig_mask]))
        print(f"Max effect size in significant regions for {label}: {max_effect:.3f}")
    else:
        print(f"No significant regions found for {label}.")

    return T_obs, sig_mask, f"Cluster Permutation-enhanced T-value\n({label})", save_name

T1, M1, title1, file1 = run_Cluster Permutation_comparison(df, 0, 1, "(TDP− vs TDP+)", "lh.Cluster Permutation_tdp0_vs_tdp1.png")
T2, M2, title2, file2 = run_Cluster Permutation_comparison(df, 2, 0, "(Control vs TDP−)", "lh.Cluster Permutation_tdp2_vs_tdp0.png")
T3, M3, title3, file3 = run_Cluster Permutation_comparison(df, 2, 1, "(Control vs TDP+)", "lh.Cluster Permutation_tdp2_vs_tdp1.png")

# === Save VTK ===
import pyvista as pv

mesh_path = "path/to/your/data_file.csv")  # UPDATE: Replace with your data path
base_mesh = pv.read(mesh_path)
flipped_mesh = base_mesh.copy()
x_coords = flipped_mesh.points[:, 0]
flipped_mesh.points[:, 0] = x_coords.max() - x_coords + x_coords.min()
y_coords = flipped_mesh.points[:, 1]
flipped_mesh.points[:, 1] = y_coords.max() - y_coords + y_coords.min()

masks = {
    "Cluster Permutation_PART_vs_Control": significance_mask,
    "Cluster Permutation_TDPneg_vs_TDPpos": M1,
    "Cluster Permutation_TDPneg_vs_Control": M2,
    "Cluster Permutation_TDPpos_vs_Control": M3
}

for label, mask in masks.items():
    mesh = flipped_mesh.copy()
    flat_mask = mask.flatten()
    if flat_mask.shape[0] != mesh.n_points:
        raise ValueError(f"Shape mismatch for {label}: mask has {flat_mask.shape[0]} values, but mesh has {mesh.n_points} vertices")
    mesh[label] = flat_mask.astype(int)
    output_path = label.replace("Cluster Permutation_", "lh.mid-surface_") + ".vtk"
    mesh.save(output_path)
    print(f"Saved: {output_path}")



#########################################################################
############################################################################

# === Additional Cluster Permutation: TDP− vs TDP+ restricted to Subfield 234 ===

print("\nRunning Cluster Permutation for TDP− vs TDP+ restricted to Subfield 234...")

subfield_234_mask = (label_matrix == 234)  # Already correct (21,41)

group0_sub234, group1_sub234 = [], []

for subj in df['mrn'].unique():
    subj_df = df[df['mrn'] == subj].sort_values("x_index")
    tdp_label = subj_df['tdp_status'].iloc[0]
    if tdp_label not in [0, 1]:
        continue
    matrix = subj_df.loc[:, "y0":"y20"].to_numpy()  # (41,21)
    matrix = matrix.T  # 🔥 NOW (21,41)

    masked_matrix = np.where(subfield_234_mask, matrix, np.nan)

    if tdp_label == 0:
        group0_sub234.append(masked_matrix)
    elif tdp_label == 1:
        group1_sub234.append(masked_matrix)

group0_sub234 = np.array(group0_sub234)
group1_sub234 = np.array(group1_sub234)

print(f"Subfield 234 Group0 (TDP−): {group0_sub234.shape}, Group1 (TDP+): {group1_sub234.shape}")

# Extract valid points
group0_masked = np.stack([m[subfield_234_mask] for m in group0_sub234])
group1_masked = np.stack([m[subfield_234_mask] for m in group1_sub234])

print(f"Masked Subfield 234 Group0: {group0_masked.shape}, Group1: {group1_masked.shape}")

# Cluster Permutation
X_sub234 = [group0_masked, group1_masked]
T_obs_sub234, clusters_sub234, p_values_sub234, H0_sub234 = permutation_cluster_test(
    X_sub234,
    n_permutations=1000,     # 🔸 default was 1000 permutations
    tail=0,                  # two-sided test
    n_jobs=1                 # single-threaded
    )

# Significance
sig_mask_sub234 = np.zeros_like(T_obs_sub234, dtype=bool)
for i, p in enumerate(p_values_sub234):
    if p < 0.05:
        sig_mask_sub234[clusters_sub234[i]] = True

# Rebuild maps
full_d_map_sub234 = np.full(subfield_234_mask.shape, np.nan)  # (21,41)
full_sig_mask_sub234 = np.zeros(subfield_234_mask.shape, dtype=bool)

full_d_map_sub234[subfield_234_mask] = compute_hedges_g(group0_masked, group1_masked)
full_sig_mask_sub234[subfield_234_mask] = sig_mask_sub234

# Save and plot
plot_Cluster Permutation_with_effect_size(
    full_d_map_sub234.T,  # 🔥 Flip back for plotting (41,21)
    full_sig_mask_sub234.T,
    "Cluster Permutation Significance over Hedges' g\n(TDP− vs TDP+) Subfield 234",
    "lh.Cluster Permutation_hedgesg_tdp0_vs_tdp1_subfield234.png",
    outline_df
)

summarize_Cluster Permutation_results(
    full_d_map_sub234.T,
    full_sig_mask_sub234.T,
    group0_masked,
    group1_masked,
    "(TDP− vs TDP+) Subfield 234"
)

print(f"Completed Cluster Permutation for Subfield 234!")

# === Save Subfield 234 Cluster Permutation mask into VTK ===

vtk_label = "Cluster Permutation_TDPneg_vs_TDPpos_Subfield234"

# Load and flip the base mesh again
mesh_path = "path/to/your/data_file.csv")  # UPDATE: Replace with your data path
base_mesh = pv.read(mesh_path)
flipped_mesh = base_mesh.copy()
x_coords = flipped_mesh.points[:, 0]
flipped_mesh.points[:, 0] = x_coords.max() - x_coords + x_coords.min()
y_coords = flipped_mesh.points[:, 1]
flipped_mesh.points[:, 1] = y_coords.max() - y_coords + y_coords.min()

# Prepare the flat mask
flat_mask_234 = full_sig_mask_sub234.T.flatten()  # 🔥 Note .T to match mesh
if flat_mask_234.shape[0] != flipped_mesh.n_points:
    raise ValueError(f"Shape mismatch for Subfield 234 mask: mask has {flat_mask_234.shape[0]} points, mesh has {flipped_mesh.n_points} vertices.")

# Assign and save
flipped_mesh[vtk_label] = flat_mask_234.astype(int)
output_vtk_path = f"lh.mid-surface_Cluster Permutation_subfield234.vtk"
flipped_mesh.save(output_vtk_path)
print(f"Saved VTK: {output_vtk_path}")



# === Additional Cluster Permutation: TDP− vs TDP+ restricted to Subfield 236 ===


# === Additional Cluster Permutation: TDP− vs TDP+ restricted to Subfield 236 ===

print("\nRunning Cluster Permutation for TDP− vs TDP+ restricted to Subfield 236...")

subfield_236_mask = (label_matrix == 236)  # Already correct (21,41)

group0_sub236, group1_sub236 = [], []

for subj in df['mrn'].unique():
    subj_df = df[df['mrn'] == subj].sort_values("x_index")
    tdp_label = subj_df['tdp_status'].iloc[0]
    if tdp_label not in [0, 1]:
        continue
    matrix = subj_df.loc[:, "y0":"y20"].to_numpy()  # (41,21)
    matrix = matrix.T  # 🔥 NOW (21,41)

    masked_matrix = np.where(subfield_236_mask, matrix, np.nan)

    if tdp_label == 0:
        group0_sub236.append(masked_matrix)
    elif tdp_label == 1:
        group1_sub236.append(masked_matrix)

group0_sub236 = np.array(group0_sub236)
group1_sub236 = np.array(group1_sub236)

print(f"Subfield 236 Group0 (TDP−): {group0_sub236.shape}, Group1 (TDP+): {group1_sub236.shape}")

# Extract valid points
group0_masked = np.stack([m[subfield_236_mask] for m in group0_sub236])
group1_masked = np.stack([m[subfield_236_mask] for m in group1_sub236])

print(f"Masked Subfield 236 Group0: {group0_masked.shape}, Group1: {group1_masked.shape}")

# Cluster Permutation
X_sub236 = [group0_masked, group1_masked]
T_obs_sub236, clusters_sub236, p_values_sub236, H0_sub236 = permutation_cluster_test(
    X_sub236,
    n_permutations=1000,     # 🔸 default was 1000 permutations
    tail=0,                  # two-sided test
    n_jobs=1                 # single-threaded
    )

# Significance
sig_mask_sub236 = np.zeros_like(T_obs_sub236, dtype=bool)
for i, p in enumerate(p_values_sub236):
    if p < 0.05:
        sig_mask_sub236[clusters_sub236[i]] = True

# Rebuild maps
full_d_map_sub236 = np.full(subfield_236_mask.shape, np.nan)  # (21,41)
full_sig_mask_sub236 = np.zeros(subfield_236_mask.shape, dtype=bool)

full_d_map_sub236[subfield_236_mask] = compute_hedges_g(group0_masked, group1_masked)
full_sig_mask_sub236[subfield_236_mask] = sig_mask_sub236

# Save and plot
plot_Cluster Permutation_with_effect_size(
    full_d_map_sub236.T,  # 🔥 Flip back for plotting (41,21)
    full_sig_mask_sub236.T,
    "Cluster Permutation Significance over Hedges' g\n(TDP− vs TDP+) Subfield 236",
    "lh.Cluster Permutation_hedgesg_tdp0_vs_tdp1_subfield236.png",
    outline_df
)

summarize_Cluster Permutation_results(
    full_d_map_sub236.T,
    full_sig_mask_sub236.T,
    group0_masked,
    group1_masked,
    "(TDP− vs TDP+) Subfield 236"
)

print(f"Completed Cluster Permutation for Subfield 236!")

# === Save Subfield 236 Cluster Permutation mask into VTK ===

vtk_label = "Cluster Permutation_TDPneg_vs_TDPpos_Subfield236"

# Load and flip the base mesh again
mesh_path = "path/to/your/data_file.csv")  # UPDATE: Replace with your data path
base_mesh = pv.read(mesh_path)
flipped_mesh = base_mesh.copy()
x_coords = flipped_mesh.points[:, 0]
flipped_mesh.points[:, 0] = x_coords.max() - x_coords + x_coords.min()
y_coords = flipped_mesh.points[:, 1]
flipped_mesh.points[:, 1] = y_coords.max() - y_coords + y_coords.min()

# Prepare the flat mask
flat_mask_236 = full_sig_mask_sub236.T.flatten()  # 🔥 Note .T to match mesh
if flat_mask_236.shape[0] != flipped_mesh.n_points:
    raise ValueError(f"Shape mismatch for Subfield 236 mask: mask has {flat_mask_236.shape[0]} points, mesh has {flipped_mesh.n_points} vertices.")

# Assign and save
flipped_mesh[vtk_label] = flat_mask_236.astype(int)
output_vtk_path = f"lh.mid-surface_Cluster Permutation_subfield236.vtk"
flipped_mesh.save(output_vtk_path)
print(f"Saved VTK: {output_vtk_path}")



# === Additional Cluster Permutation: TDP− vs TDP+ restricted to Subfield 238 ===

print("\nRunning Cluster Permutation for TDP− vs TDP+ restricted to Subfield 238...")

subfield_238_mask = (label_matrix == 238)  # Already correct (21,41)

group0_sub238, group1_sub238 = [], []

for subj in df['mrn'].unique():
    subj_df = df[df['mrn'] == subj].sort_values("x_index")
    tdp_label = subj_df['tdp_status'].iloc[0]
    if tdp_label not in [0, 1]:
        continue
    matrix = subj_df.loc[:, "y0":"y20"].to_numpy()  # (41,21)
    matrix = matrix.T  # 🔥 NOW (21,41)

    masked_matrix = np.where(subfield_238_mask, matrix, np.nan)

    if tdp_label == 0:
        group0_sub238.append(masked_matrix)
    elif tdp_label == 1:
        group1_sub238.append(masked_matrix)

group0_sub238 = np.array(group0_sub238)
group1_sub238 = np.array(group1_sub238)

print(f"Subfield 238 Group0 (TDP−): {group0_sub238.shape}, Group1 (TDP+): {group1_sub238.shape}")

# Extract valid points
group0_masked = np.stack([m[subfield_238_mask] for m in group0_sub238])
group1_masked = np.stack([m[subfield_238_mask] for m in group1_sub238])

print(f"Masked Subfield 238 Group0: {group0_masked.shape}, Group1: {group1_masked.shape}")

# Cluster Permutation
X_sub238 = [group0_masked, group1_masked]
T_obs_sub238, clusters_sub238, p_values_sub238, H0_sub238 = permutation_cluster_test(
    X_sub238,
    n_permutations=1000,     # 🔸 default was 1000 permutations
    tail=0,                  # two-sided test
    n_jobs=1                 # single-threaded
    )

# Significance
sig_mask_sub238 = np.zeros_like(T_obs_sub238, dtype=bool)
for i, p in enumerate(p_values_sub238):
    if p < 0.05:
        sig_mask_sub238[clusters_sub238[i]] = True

# Rebuild maps
full_d_map_sub238 = np.full(subfield_238_mask.shape, np.nan)  # (21,41)
full_sig_mask_sub238 = np.zeros(subfield_238_mask.shape, dtype=bool)

full_d_map_sub238[subfield_238_mask] = compute_hedges_g(group0_masked, group1_masked)
full_sig_mask_sub238[subfield_238_mask] = sig_mask_sub238

# Save and plot
plot_Cluster Permutation_with_effect_size(
    full_d_map_sub238.T,  # 🔥 Flip back for plotting (41,21)
    full_sig_mask_sub238.T,
    "Cluster Permutation Significance over Hedges' g\n(TDP− vs TDP+) Subfield 238",
    "lh.Cluster Permutation_hedgesg_tdp0_vs_tdp1_subfield238.png",
    outline_df
)

summarize_Cluster Permutation_results(
    full_d_map_sub238.T,
    full_sig_mask_sub238.T,
    group0_masked,
    group1_masked,
    "(TDP− vs TDP+) Subfield 238"
)

print(f"Completed Cluster Permutation for Subfield 238!")

# === Save Subfield 238 Cluster Permutation mask into VTK ===

vtk_label = "Cluster Permutation_TDPneg_vs_TDPpos_Subfield238"

# Load and flip the base mesh again
mesh_path = "path/to/your/data_file.csv")  # UPDATE: Replace with your data path
base_mesh = pv.read(mesh_path)
flipped_mesh = base_mesh.copy()
x_coords = flipped_mesh.points[:, 0]
flipped_mesh.points[:, 0] = x_coords.max() - x_coords + x_coords.min()
y_coords = flipped_mesh.points[:, 1]
flipped_mesh.points[:, 1] = y_coords.max() - y_coords + y_coords.min()

# Prepare the flat mask
flat_mask_238 = full_sig_mask_sub238.T.flatten()  # 🔥 Note .T to match mesh
if flat_mask_238.shape[0] != flipped_mesh.n_points:
    raise ValueError(f"Shape mismatch for Subfield 238 mask: mask has {flat_mask_238.shape[0]} points, mesh has {flipped_mesh.n_points} vertices.")

# Assign and save
flipped_mesh[vtk_label] = flat_mask_238.astype(int)
output_vtk_path = f"lh.mid-surface_Cluster Permutation_subfield238.vtk"
flipped_mesh.save(output_vtk_path)
print(f"Saved VTK: {output_vtk_path}")



# === Additional Cluster Permutation: TDP− vs TDP+ restricted to Subfield 240 ===

print("\nRunning Cluster Permutation for TDP− vs TDP+ restricted to Subfield 240...")

subfield_240_mask = (label_matrix == 240)  # Already correct (21,41)

group0_sub240, group1_sub240 = [], []

for subj in df['mrn'].unique():
    subj_df = df[df['mrn'] == subj].sort_values("x_index")
    tdp_label = subj_df['tdp_status'].iloc[0]
    if tdp_label not in [0, 1]:
        continue
    matrix = subj_df.loc[:, "y0":"y20"].to_numpy()  # (41,21)
    matrix = matrix.T  # 🔥 NOW (21,41)

    masked_matrix = np.where(subfield_240_mask, matrix, np.nan)

    if tdp_label == 0:
        group0_sub240.append(masked_matrix)
    elif tdp_label == 1:
        group1_sub240.append(masked_matrix)

group0_sub240 = np.array(group0_sub240)
group1_sub240 = np.array(group1_sub240)

print(f"Subfield 240 Group0 (TDP−): {group0_sub240.shape}, Group1 (TDP+): {group1_sub240.shape}")

# Extract valid points
group0_masked = np.stack([m[subfield_240_mask] for m in group0_sub240])
group1_masked = np.stack([m[subfield_240_mask] for m in group1_sub240])

print(f"Masked Subfield 240 Group0: {group0_masked.shape}, Group1: {group1_masked.shape}")

# Cluster Permutation
X_sub240 = [group0_masked, group1_masked]
T_obs_sub240, clusters_sub240, p_values_sub240, H0_sub240 = permutation_cluster_test(
    X_sub240,
    n_permutations=1000,     # 🔸 default was 1000 permutations
    tail=0,                  # two-sided test
    n_jobs=1                 # single-threaded
    )

# Significance
sig_mask_sub240 = np.zeros_like(T_obs_sub240, dtype=bool)
for i, p in enumerate(p_values_sub240):
    if p < 0.05:
        sig_mask_sub240[clusters_sub240[i]] = True

# Rebuild maps
full_d_map_sub240 = np.full(subfield_240_mask.shape, np.nan)  # (21,41)
full_sig_mask_sub240 = np.zeros(subfield_240_mask.shape, dtype=bool)

full_d_map_sub240[subfield_240_mask] = compute_hedges_g(group0_masked, group1_masked)
full_sig_mask_sub240[subfield_240_mask] = sig_mask_sub240

# Save and plot
plot_Cluster Permutation_with_effect_size(
    full_d_map_sub240.T,  # 🔥 Flip back for plotting (41,21)
    full_sig_mask_sub240.T,
    "Cluster Permutation Significance over Hedges' g\n(TDP− vs TDP+) Subfield 240",
    "lh.Cluster Permutation_hedgesg_tdp0_vs_tdp1_subfield240.png",
    outline_df
)

summarize_Cluster Permutation_results(
    full_d_map_sub240.T,
    full_sig_mask_sub240.T,
    group0_masked,
    group1_masked,
    "(TDP− vs TDP+) Subfield 240"
)

print(f"Completed Cluster Permutation for Subfield 240!")

# === Save Subfield 240 Cluster Permutation mask into VTK ===

vtk_label = "Cluster Permutation_TDPneg_vs_TDPpos_Subfield240"

# Load and flip the base mesh again
mesh_path = "path/to/your/data_file.csv")  # UPDATE: Replace with your data path
base_mesh = pv.read(mesh_path)
flipped_mesh = base_mesh.copy()
x_coords = flipped_mesh.points[:, 0]
flipped_mesh.points[:, 0] = x_coords.max() - x_coords + x_coords.min()
y_coords = flipped_mesh.points[:, 1]
flipped_mesh.points[:, 1] = y_coords.max() - y_coords + y_coords.min()

# Prepare the flat mask
flat_mask_240 = full_sig_mask_sub240.T.flatten()  # 🔥 Note .T to match mesh
if flat_mask_240.shape[0] != flipped_mesh.n_points:
    raise ValueError(f"Shape mismatch for Subfield 240 mask: mask has {flat_mask_240.shape[0]} points, mesh has {flipped_mesh.n_points} vertices.")

# Assign and save
flipped_mesh[vtk_label] = flat_mask_240.astype(int)
output_vtk_path = f"lh.mid-surface_Cluster Permutation_subfield240.vtk"
flipped_mesh.save(output_vtk_path)
print(f"Saved VTK: {output_vtk_path}")

##########################################################
######### Braak Cluster Permutation Comparisons in TDP-Negatives ########
##########################################################

print("\n===== Running Braak Stage Comparisons inside TDP-negatives (TDP0 + TDP2) =====\n")

# Subset only TDP-negative subjects
df_tdpneg = df[df['tdp_status'].isin([0,2])]

# Load and prepare the base left hemisphere mesh
mesh_path = "path/to/your/data_file.csv")  # UPDATE: Replace with your data path
base_mesh = pv.read(mesh_path)
flipped_mesh = base_mesh.copy()
x_coords = flipped_mesh.points[:, 0]
flipped_mesh.points[:, 0] = x_coords.max() - x_coords + x_coords.min()
y_coords = flipped_mesh.points[:, 1]
flipped_mesh.points[:, 1] = y_coords.max() - y_coords + y_coords.min()

def run_braak_Cluster Permutation_comparison(df_subset, braak_a, braak_b, label, save_prefix):
    group0, group1 = [], []
    for subj in df_subset['mrn'].unique():
        subj_df = df_subset[df_subset['mrn'] == subj].sort_values("x_index")
        braak_label = subj_df['braak_stage'].iloc[0]  # 🔥 Corrected to braak_stage
        if braak_label not in [braak_a, braak_b]:
            continue
        matrix = subj_df.loc[:, "y0":"y20"].to_numpy()
        if braak_label == braak_a:
            group0.append(matrix)
        elif braak_label == braak_b:
            group1.append(matrix)

    group0 = np.array(group0)
    group1 = np.array(group1)
    print(f"{label}: Group {braak_a}: {group0.shape}, Group {braak_b}: {group1.shape}")

    if len(group0) == 0 or len(group1) == 0:
        print(f"Not enough subjects for {label}. Skipping...")
        return None, None
    if group0.shape[1:] != group1.shape[1:]:
        print(f"Shape mismatch for {label}: {group0.shape} vs {group1.shape}. Skipping...")
        return None, None

    X = [group0, group1]

    control_group = group0
    disease_group = group1

    d_map = compute_hedges_g(control_group, disease_group)

    T_obs, clusters, p_values, H0 = permutation_cluster_test(
        X, 
        n_permutations=1000,     # 🔸 default was 1000 permutations
        tail=0,                  # two-sided test
        n_jobs=1                 # single-threaded
        )

    sig_mask = np.zeros_like(T_obs, dtype=bool)
    for i, p in enumerate(p_values):
        if p < 0.05:
            sig_mask[clusters[i]] = True

    return d_map, sig_mask

def save_vtk_with_mask(base_mesh, sig_mask, output_name):
    mesh_copy = base_mesh.copy()
    flat_mask = sig_mask.T.flatten()
    if flat_mask.shape[0] != mesh_copy.n_points:
        raise ValueError(f"Mismatch: mask {flat_mask.shape[0]}, mesh {mesh_copy.n_points}")
    mesh_copy[output_name] = flat_mask.astype(int)
    mesh_copy.save(f"{output_name}.vtk")
    print(f"Saved VTK: {output_name}.vtk")

# Loop through Braak stages 1–4
for braak_stage in [1,2,3,4]:
    label = f"(Braak {braak_stage} vs Control) [TDP-negative]"
    save_prefix = f"lh.Cluster Permutation_braak0_vs_{braak_stage}_tdpneg"

    d_map, sig_mask = run_braak_Cluster Permutation_comparison(df_tdpneg, 0, braak_stage, label, save_prefix)

    if d_map is None or sig_mask is None:
        continue

    plot_Cluster Permutation_with_effect_size(
        d_map,
        sig_mask,
        f"Cluster Permutation Significance over Hedges' g\n{label}",
        f"{save_prefix}.png",
        outline_df
    )

    save_vtk_with_mask(flipped_mesh, sig_mask, save_prefix)



print(df_tdpneg['braak_stage'].value_counts())
np.nanmax(np.abs(d_map))

###############################################################################################################
##################################### PART without TDP-43 Braak Stratification ############################
###############################################################################################################

print("\n===== PART without TDP-43 Stratified by Braak Stages vs Controls =====\n")

# Select only PART without TDP-43 subjects (tdp_status = 0) and Controls (tdp_status = 2)
df_part_notdp = df[df['tdp_status'].isin([0, 2])]

# Load and prepare the base left hemisphere mesh
mesh_path = "path/to/your/data_file.csv")  # UPDATE: Replace with your data path
base_mesh = pv.read(mesh_path)
flipped_mesh = base_mesh.copy()
x_coords = flipped_mesh.points[:, 0]
flipped_mesh.points[:, 0] = x_coords.max() - x_coords + x_coords.min()
y_coords = flipped_mesh.points[:, 1]
flipped_mesh.points[:, 1] = y_coords.max() - y_coords + y_coords.min()

def run_part_braak_Cluster Permutation_comparison(df_subset, braak_stages, control_braak, label, save_prefix):
    """Compare PART without TDP-43 subjects with specific Braak stages vs Controls"""
    group0, group1 = [], []
    
    for subj in df_subset['mrn'].unique():
        subj_df = df_subset[df_subset['mrn'] == subj].sort_values("x_index")
        tdp_status = subj_df['tdp_status'].iloc[0]
        braak_stage = subj_df['braak_stage'].iloc[0]
        
        matrix = subj_df.loc[:, "y0":"y20"].to_numpy()
        
        # Group 0: Controls (tdp_status = 2, braak_stage = 0)
        if tdp_status == 2 and braak_stage == control_braak:
            group0.append(matrix)
        # Group 1: PART without TDP-43 with specified Braak stages
        elif tdp_status == 0 and braak_stage in braak_stages:
            group1.append(matrix)
    
    group0 = np.array(group0)
    group1 = np.array(group1)
    
    print(f"{label}: Controls (n={group0.shape[0]}), PART no-TDP Braak {braak_stages} (n={group1.shape[0]})")
    
    if len(group0) == 0 or len(group1) == 0:
        print(f"Not enough subjects for {label}. Skipping...")
        return None, None
    if group0.shape[1:] != group1.shape[1:]:
        print(f"Shape mismatch for {label}: {group0.shape} vs {group1.shape}. Skipping...")
        return None, None
    
    X = [group0, group1]
    
    # Controls first (reference group)
    control_group = group0
    disease_group = group1
    
    d_map = compute_hedges_g(control_group, disease_group)
    
    T_obs, clusters, p_values, H0 = permutation_cluster_test(
        X,
        n_permutations=1000,
        tail=0,
        n_jobs=1
    )
    
    sig_mask = np.zeros_like(T_obs, dtype=bool)
    for i, p in enumerate(p_values):
        if p < 0.05:
            sig_mask[clusters[i]] = True
    
    return d_map, sig_mask

def save_vtk_with_mask(base_mesh, sig_mask, output_name):
    mesh_copy = base_mesh.copy()
    flat_mask = sig_mask.T.flatten()
    if flat_mask.shape[0] != mesh_copy.n_points:
        raise ValueError(f"Mismatch: mask {flat_mask.shape[0]}, mesh {mesh_copy.n_points}")
    mesh_copy[output_name] = flat_mask.astype(int)
    mesh_copy.save(f"{output_name}.vtk")
    print(f"Saved VTK: {output_name}.vtk")

# Comparison 1: PART without TDP-43 Braak I-II vs Controls
d_map_braak12, sig_mask_braak12 = run_part_braak_Cluster Permutation_comparison(
    df_part_notdp, 
    braak_stages=[1, 2], 
    control_braak=0,
    label="PART no-TDP Braak I-II vs Controls", 
    save_prefix="lh.Cluster Permutation_part_notdp_braak12_vs_controls"
)

if d_map_braak12 is not None and sig_mask_braak12 is not None:
    plot_Cluster Permutation_with_effect_size(
        d_map_braak12, 
        sig_mask_braak12, 
        f"Cluster Permutation Significance over Hedges' g\n(Controls vs PART no-TDP Braak I-II)", 
        "lh.Cluster Permutation_part_notdp_braak12_vs_controls.png",
        outline_df
    )
    save_vtk_with_mask(flipped_mesh, sig_mask_braak12, "lh.Cluster Permutation_part_notdp_braak12_vs_controls")
    summarize_Cluster Permutation_results(
        d_map_braak12, 
        sig_mask_braak12, 
        d_map_braak12, 
        sig_mask_braak12, 
        "PART no-TDP Braak I-II vs Controls"
    )
    print("Completed PART no-TDP Braak I-II vs Controls comparison")

# Comparison 2: PART without TDP-43 Braak III-IV vs Controls  
d_map_braak34, sig_mask_braak34 = run_part_braak_Cluster Permutation_comparison(
    df_part_notdp, 
    braak_stages=[3, 4], 
    control_braak=0,
    label="PART no-TDP Braak III-IV vs Controls", 
    save_prefix="lh.Cluster Permutation_part_notdp_braak34_vs_controls"
)

if d_map_braak34 is not None and sig_mask_braak34 is not None:
    plot_Cluster Permutation_with_effect_size(
        d_map_braak34, 
        sig_mask_braak34, 
        f"Cluster Permutation Significance over Hedges' g\n(Controls vs PART no-TDP Braak III-IV)", 
        "lh.Cluster Permutation_part_notdp_braak34_vs_controls.png",
        outline_df
    )
    save_vtk_with_mask(flipped_mesh, sig_mask_braak34, "lh.Cluster Permutation_part_notdp_braak34_vs_controls")
    summarize_Cluster Permutation_results(
        d_map_braak34, 
        sig_mask_braak34, 
        d_map_braak34, 
        sig_mask_braak34, 
        "PART no-TDP Braak III-IV vs Controls"
    )
    print("Completed PART no-TDP Braak III-IV vs Controls comparison")

# Print summary statistics
print(f"\nSummary of PART without TDP-43 subjects by Braak stage:")
part_notdp_only = df_part_notdp[df_part_notdp['tdp_status'] == 0]
print(part_notdp_only['braak_stage'].value_counts().sort_index())

print(f"\nControls by Braak stage:")
controls_only = df_part_notdp[df_part_notdp['tdp_status'] == 2]
print(controls_only['braak_stage'].value_counts().sort_index())

# Additional comparison: Direct comparison between PART no-TDP Braak I-II vs III-IV
print("\n===== Additional: PART no-TDP Braak I-II vs Braak III-IV =====\n")

d_map_tau_progression, sig_mask_tau_progression = run_part_braak_Cluster Permutation_comparison(
    df_part_notdp, 
    braak_stages=[3, 4], 
    control_braak=[1, 2],
    label="PART no-TDP Braak III-IV vs Braak I-II", 
    save_prefix="lh.Cluster Permutation_part_notdp_braak34_vs_braak12"
)

# Tau progression comparison function
def run_tau_progression_Cluster Permutation_comparison(df_subset, label, save_prefix):
    """Compare PART without TDP-43 Braak III-IV vs Braak I-II to investigate tau progression"""
    group0, group1 = [], []  # group0 = Braak I-II, group1 = Braak III-IV
    
    for subj in df_subset['mrn'].unique():
        subj_df = df_subset[df_subset['mrn'] == subj].sort_values("x_index")
        tdp_status = subj_df['tdp_status'].iloc[0]
        braak_stage = subj_df['braak_stage'].iloc[0]
        
        # Only include PART without TDP-43 subjects
        if tdp_status != 0:
            continue
            
        matrix = subj_df.loc[:, "y0":"y20"].to_numpy()
        
        if braak_stage in [1, 2]:
            group0.append(matrix)
        elif braak_stage in [3, 4]:
            group1.append(matrix)
    
    group0 = np.array(group0)
    group1 = np.array(group1)
    
    print(f"{label}: Braak I-II (n={group0.shape[0]}), Braak III-IV (n={group1.shape[0]})")
    
    if len(group0) == 0 or len(group1) == 0:
        print(f"Not enough subjects for {label}. Skipping...")
        return None, None
    if group0.shape[1:] != group1.shape[1:]:
        print(f"Shape mismatch for {label}: {group0.shape} vs {group1.shape}. Skipping...")
        return None, None
    
    X = [group0, group1]
    
    # Braak I-II as reference (less pathology)
    control_group = group0
    disease_group = group1
    
    d_map = compute_hedges_g(control_group, disease_group)
    
    T_obs, clusters, p_values, H0 = permutation_cluster_test(
        X,
        n_permutations=1000,
        tail=0,
        n_jobs=1
    )
    
    sig_mask = np.zeros_like(T_obs, dtype=bool)
    for i, p in enumerate(p_values):
        if p < 0.05:
            sig_mask[clusters[i]] = True
    
    return d_map, sig_mask

d_map_tau_prog, sig_mask_tau_prog = run_tau_progression_Cluster Permutation_comparison(
    df_part_notdp,
    label="PART no-TDP Tau Progression (Braak III-IV vs I-II)",
    save_prefix="lh.Cluster Permutation_part_notdp_tau_progression"
)

if d_map_tau_prog is not None and sig_mask_tau_prog is not None:
    plot_Cluster Permutation_with_effect_size(
        d_map_tau_prog, 
        sig_mask_tau_prog, 
        f"Cluster Permutation Significance over Hedges' g\n(PART no-TDP: Braak I-II vs III-IV)", 
        "lh.Cluster Permutation_part_notdp_tau_progression.png",
        outline_df
    )
    save_vtk_with_mask(flipped_mesh, sig_mask_tau_prog, "lh.Cluster Permutation_part_notdp_tau_progression")
    summarize_Cluster Permutation_results(
        d_map_tau_prog, 
        sig_mask_tau_prog, 
        d_map_tau_prog, 
        sig_mask_tau_prog, 
        "PART no-TDP Tau Progression"
    )
    print("Completed PART no-TDP tau progression comparison")











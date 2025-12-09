import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import spatio_temporal_cluster_test
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pyvista as pv
np.random.seed(42)

# =============================================================================
# DATA PATHS - UPDATE THESE FOR YOUR SYSTEM
# =============================================================================
# Left hemisphere Gaussian curvature grid data from HIPSTA
CURVATURE_FILE = "path/to/your/lh_gaussian_curvature.csv"
# Left hemisphere subfield labels for plot overlay
SUBFIELD_LABELS_FILE = "path/to/your/lh_subfield_labels.csv"

# === Load Gaussian Curvature Data ===
df = pd.read_csv(CURVATURE_FILE)
df = df.rename(columns={
    'lh.mid-surface.gauss-curv.csv-all': 'gauss_curvature',
    'PART=1_control=0': 'group_PART_vs_control',
    'x': 'grid_x',
    'y': 'grid_y',
    'z': 'grid_z'
})

# === Load Subfield Outline for Plot Overlay ===
outline_df = pd.read_csv(SUBFIELD_LABELS_FILE)
label_matrix = outline_df.pivot_table(index="x", columns="y", values="subfield label", aggfunc="first").to_numpy()
label_matrix = np.fliplr(label_matrix.T)

# === Reshape into [subjects, x, y] ===
def get_group_matrix(df, group_col, group_val):
    matrices = []
    for subj in df[df[group_col] == group_val]["mrn"].unique():
        subj_df = df[df["mrn"] == subj].sort_values(["grid_x", "grid_y"])
        matrix = subj_df.pivot(index="grid_x", columns="grid_y", values="gauss_curvature").to_numpy()
        matrices.append(matrix)
    return np.array(matrices)

# === Cluster Permutation comparison function ===
def run_Cluster Permutation_comparison(df, group_a, group_b, label, group_col, save_prefix):
    group0 = get_group_matrix(df, group_col, group_a)
    group1 = get_group_matrix(df, group_col, group_b)
    print("Missing values -> Group A:", np.isnan(group0).sum(), ", Group B:", np.isnan(group1).sum())
    print(f"{label} comparison -> Group {group_a}: {group0.shape}, Group {group_b}: {group1.shape}")

    X = [group0, group1]
    T_obs, clusters, p_values, H0 = spatio_temporal_cluster_test(
        X, n_permutations=1000,
        tail=0,
        n_jobs=1
    )

    sig_mask = np.zeros_like(T_obs, dtype=bool)
    for i, p_val in enumerate(p_values):
        if p_val < 0.05:
            sig_mask[clusters[i]] = True

    # === Compute Δ Gaussian curvature (mean difference) ===
    mean_group0 = np.nanmean(group0, axis=0)
    mean_group1 = np.nanmean(group1, axis=0)
    delta_curv = mean_group1 - mean_group0
    masked_delta = np.ma.masked_where(~sig_mask, delta_curv)
    masked_delta = np.fliplr(masked_delta.T)

    # === Custom color map with no white center ===
    base_cmap = plt.get_cmap("seismic", 256)
    cmap_array = base_cmap(np.linspace(0, 1, 256))
    cut = 20
    mid = 128
    cmap_no_white = np.vstack([cmap_array[:mid - cut], cmap_array[mid + cut:]])
    custom_cmap = LinearSegmentedColormap.from_list("seismic_no_white", cmap_no_white)

    # === Plot Δ curvature map ===
    fig, ax = plt.subplots(figsize=(10, 6))
    vmax = np.nanmax(np.abs(masked_delta))
    im = ax.imshow(masked_delta, cmap=custom_cmap, origin="lower", extent=[40, 0, 0, 20],
                   vmin=-vmax, vmax=vmax)
    ax.contour(label_matrix, levels=np.unique(label_matrix), linewidths=0.5, colors='black',
               origin='lower', extent=[40, 0, 0, 20])
    ax.set_title(f"Δ Gaussian Curvature (Mean Diff) {label}", fontsize=10)
    ax.set_xlabel("X axis (Lateral -> Medial)")
    ax.set_ylabel("Y axis (Posterior -> Anterior)")
    ax.tick_params(axis='both', labelsize=20)
    cbar = plt.colorbar(im, ax=ax, label="Δ curvature (mean)")
    cbar.ax.tick_params(labelsize=20)
    plt.tight_layout()
    delta_path = f"lh.gausscurv_diff_{save_prefix}.png"
    plt.savefig(delta_path, dpi=300)
    plt.close()
    print(f"Saved: {delta_path}")

    # === Plot T-map of significant clusters ===
    masked_T = np.ma.masked_where(~sig_mask, T_obs)
    masked_T = np.fliplr(masked_T.T)
    fig, ax = plt.subplots(figsize=(10, 6))
    vmax_T = np.nanmax(np.abs(masked_T))
    im = ax.imshow(masked_T, cmap=custom_cmap, origin="lower", extent=[40, 0, 0, 20],
                   vmin=-vmax_T, vmax=vmax_T)
    ax.contour(label_matrix, levels=np.unique(label_matrix), linewidths=0.5, colors='black',
               origin='lower', extent=[40, 0, 0, 20])
    ax.set_title(f"T-map of Significant Clusters {label}", fontsize=10)
    ax.set_xlabel("X axis (Lateral -> Medial)")
    ax.set_ylabel("Y axis (Posterior -> Anterior)")
    ax.tick_params(axis='both', labelsize=20)
    cbar = plt.colorbar(im, ax=ax, label="T-statistics")
    cbar.ax.tick_params(labelsize=20)
    plt.tight_layout()
    tmap_path = f"lh.gausscurv_tstat_{save_prefix}.png"
    plt.savefig(tmap_path, dpi=300)
    plt.close()
    print(f"Saved: {tmap_path}")

    # === Print summary stats ===
    t_values_in_mask = T_obs[sig_mask]
    if t_values_in_mask.size > 0:
        print(f"Cluster Permutation Summary for {label}:")
        print(f"Number of significant points: {t_values_in_mask.size}")
        print(f"Mean T-value: {np.mean(t_values_in_mask):.4f}")
        print(f"Max T-value: {np.max(t_values_in_mask):.4f}")
        print(f"Min T-value: {np.min(t_values_in_mask):.4f}")
    else:
        print(f"Cluster Permutation Summary for {label}: No significant points detected.")

    return sig_mask

# === Run PART vs Control ===
significance_mask = run_Cluster Permutation_comparison(df, 1, 0, "(PART vs Control)", "group_PART_vs_control", "PART_vs_Control")

# === Run TDP comparisons ===
M1 = run_Cluster Permutation_comparison(df, 0, 1, "(TDP- vs TDP+)", "tdp_status", "TDPneg_vs_TDPpos")
M2 = run_Cluster Permutation_comparison(df, 0, 2, "(TDP- vs Control)", "tdp_status", "TDPneg_vs_Control")
M3 = run_Cluster Permutation_comparison(df, 1, 2, "(TDP+ vs Control)", "tdp_status", "TDPpos_vs_Control")

# === Export VTK Mesh ===
mesh_path = "path/to/your/data_file.csv")  # UPDATE: Replace with your data path
mesh = pv.read(mesh_path)
x_coords = mesh.points[:, 0]
mesh.points[:, 0] = x_coords.max() - x_coords + x_coords.min()
y_coords = mesh.points[:, 1]
mesh.points[:, 1] = y_coords.max() - y_coords + y_coords.min()

masks = {
    "LH_GaussCurv_PARTvsControl": significance_mask,
    "LH_GaussCurv_TDPneg_vs_TDPpos": M1,
    "LH_GaussCurv_TDPneg_vs_Control": M2,
    "LH_GaussCurv_TDPpos_vs_Control": M3
}

for label, mask in masks.items():
    mesh_copy = mesh.copy()
    flat_mask = mask.flatten()
    if flat_mask.shape[0] != mesh_copy.n_points:
        raise ValueError(f"Mismatch for {label}: mask has {flat_mask.shape[0]}, mesh has {mesh_copy.n_points} points")
    mesh_copy[label] = flat_mask.astype(int)
    output_path = label.replace("LH_", "lh.mid-surface_") + ".vtk"
    mesh_copy.save(output_path)
    print(f"Saved: {output_path}")







import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_test
from matplotlib.colors import LinearSegmentedColormap
np.random.seed(42)

# =============================================================================
# DATA PATHS - UPDATE THESE FOR YOUR SYSTEM
# =============================================================================
# Right hemisphere thickness grid data from HIPSTA (41x21 grid)
THICKNESS_FILE = "path/to/your/rh_thickness_data.csv"
# Right hemisphere subfield labels for plot overlay
SUBFIELD_LABELS_FILE = "path/to/your/rh_subfield_labels.csv"

# === Load the CSV file ===
df = pd.read_csv(THICKNESS_FILE)

# === Load Subfield Outline ===
outline_df = pd.read_csv(SUBFIELD_LABELS_FILE)

# === Overlay subfield boundaries ===
label_matrix = outline_df.pivot_table(index="x", columns="y", values="subfield label", aggfunc="first").to_numpy()
label_matrix = label_matrix.T  # match cluster permutation orientation

# Extract numeric x index
df['x_index'] = df['axis'].str.extract(r'x(\d+)').astype(int)

# Get unique subjects
subjects = df['mrn'].unique()

# === Organize into Groups ===
group0 = []  # Control
group1 = []  # PART

for subj in subjects:
    subj_df = df[df['mrn'] == subj].sort_values("x_index")
    matrix = subj_df.loc[:, "y0":"y20"].to_numpy()  # shape: (41, 21)
    
    group_label = subj_df["PART=1_control=0"].iloc[0]
    if group_label == 0:
        group0.append(matrix)
    else:
        group1.append(matrix)

group0 = np.array(group0)
group1 = np.array(group1)

print(f"Group 0 (Control): {group0.shape}")
print(f"Group 1 (PART): {group1.shape}")



def compute_hedges_g(group0, group1):
    mean_diff = np.nanmean(group0, axis=0) - np.nanmean(group1, axis=0)
    n0, n1 = group0.shape[0], group1.shape[0]
    pooled_std = np.sqrt(((n0 - 1) * np.nanvar(group0, axis=0) + (n1 - 1) * np.nanvar(group1, axis=0)) / (n0 + n1 - 2))
    d = mean_diff / pooled_std
    correction = 1 - (3 / (4 * (n0 + n1) - 9))
    return d * correction


# === Run Cluster Permutation with MNE ===
X = [group0, group1]
d_map = compute_hedges_g(group0, group1)
T_obs, clusters, p_values, H0 = permutation_cluster_test(
    X,
    n_permutations=1000,     #  default was 1000 permutations
    tail=0,                  # two-sided test
    n_jobs=1                 # single-threaded
    )



# === sig_mask ===
sig_mask = np.zeros_like(T_obs, dtype=bool)
for i, p_val in enumerate(p_values):
    if p_val < 0.05:
        sig_mask[clusters[i]] = True

# === Plotting ===

# === Define custom red-blue colormap with no white midpoint ===
custom_rb = LinearSegmentedColormap.from_list(
    "pure_redblue",
    [(0, "blue"), (0.5, "purple"), (1, "red")],
    N=256
)

# === Flip & transpose for correct orientation ===
flipped_T_obs = T_obs.T
flipped_mask = sig_mask.T

# === Create masked array to isolate significant regions ===
masked_T = np.ma.masked_where(~flipped_mask, flipped_T_obs)

# === Plot ===
fig, ax = plt.subplots(figsize=(10, 6))


# Show significant Cluster Permutation-enhanced stats using custom colormap
from matplotlib.colors import ListedColormap

masked_d = np.ma.masked_where(~flipped_mask, flipped_T_obs)
vmin = 0
vmax = np.nanmax(d_map)
im = ax.imshow(masked_d, cmap='Reds', origin='lower', extent=[0, 40, 0, 20], aspect='auto',
               vmin=vmin, vmax=vmax)
ax.contour(flipped_mask, levels=[0.5], colors='black', linewidths=0.5,
           origin='lower', extent=[0,40, 0, 20])
ax.contour(label_matrix, levels=np.unique(label_matrix), linewidths=0.5,
           colors='gray', origin='lower', extent=[0,40, 0, 20])
cbar = plt.colorbar(im, ax=ax, label="Hedges' g (masked by Cluster Permutation significance)")
cbar.ax.tick_params(labelsize=18)       



# === Labels and colorbar ===
ax.set_title("Significant Differences (PART vs Control) in Thickness", fontsize=10)
ax.set_xlabel("Lateral -> Medial")
ax.set_ylabel("Posterior -> Anterior")
ax.tick_params(axis='y', labelsize=18)
#fig.colorbar(c, ax=ax, label="Cluster permutation-enhanced T-value")



plt.tight_layout()
plt.savefig("rh.cluster_perm_result_custom_rb_masked.png", dpi=300)
plt.show()






##################################################################################
# === TDP subgroups #####################################################
##################################################################################


def run_cluster_permutation_comparison(df, group_a, group_b, label, save_name):
    from mne.stats import permutation_cluster_test

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


    X = [group0, group1]


    # Force Control first
    if group_a == 2:
        control_group = group0
        disease_group = group1
    elif group_b == 2:
        control_group = group1
        disease_group = group0
    else:
    # TDP− vs TDP+ -> TDP− is control
        control_group = group0
        disease_group = group1

    d_map = compute_hedges_g(control_group, disease_group)


    T_obs, clusters, p_values, H0 = permutation_cluster_test(
        X,
        n_permutations=1000,     #  default was 1000 permutations
        tail=0,                  # two-sided test
        n_jobs=1                 # single-threaded
        )
    
    # Build significance mask
    sign_mask = np.zeros_like(T_obs, dtype=bool)
    for i, p in enumerate(p_values):
        if p < 0.05:
            sign_mask[clusters[i]] = True

    return d_map, sign_mask, f"Cluster permutation Significance over Hedges' g\n({label})", save_name


def plot_cluster_permutation_with_effect_size(T_obs, sign_mask, title, save_name):
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    import numpy as np
    global label_matrix


    # Flip and transpose for correct anatomical orientation
    flipped_T_obs = T_obs.T
    flipped_mask = sign_mask.T
    masked_T = np.ma.masked_where(~flipped_mask, flipped_T_obs)

    fig, ax = plt.subplots(figsize=(10, 6))

    masked_d = np.ma.masked_where(~flipped_mask, flipped_T_obs)
    vmin = 0
    vmax = np.nanmax(d_map)
    im = ax.imshow(masked_d, cmap='Reds', origin='lower', extent=[0, 40, 0, 20], aspect='auto',
               vmin=vmin, vmax=vmax)
    ax.contour(flipped_mask, levels=[0.5], colors='black', linewidths=0.5,
           origin='lower', extent=[0,40, 0, 20])
    ax.contour(label_matrix, levels=np.unique(label_matrix), linewidths=0.5,
           colors='gray', origin='lower', extent=[0,40, 0, 20])
    cbar = plt.colorbar(im, ax=ax, label="Hedges' g (masked by Cluster Permutation significance)")
    cbar.ax.tick_params(labelsize=18)       

    ax.set_title(f"Significant Differences {title}", fontsize=10)
    ax.set_xlabel("Lateral -> Medial")
    ax.set_ylabel("Posterior -> Anterior")
    ax.tick_params(axis='y', labelsize=18)
    #fig.colorbar(c, ax=ax, label="Cluster permutation-enhanced T-value")

    # === Overlay subfield boundaries ===
    label_matrix = outline_df.pivot_table(index="x", columns="y", values="subfield label", aggfunc="first").to_numpy()
    label_matrix = label_matrix.T  # match cluster permutation orientation


    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()

# Comparison 1: TDP− (0) vs TDP+ (1)
d_map1, M1, title1, file1 = run_cluster_permutation_comparison(df, 0, 1, "(TDP+ vs TDP−)", "rh.cluster_perm_tdp1_vs_tdp0.png")
plot_cluster_permutation_with_effect_size(d_map1, M1, title1, file1)

# Comparison 2: TDP− (0) vs Control (2)
d_map2, M2, title2, file2 = run_cluster_permutation_comparison(df, 0, 2, "(Control vs TDP−)", "rh.cluster_perm_tdp2_vs_tdp0.png")
plot_cluster_permutation_with_effect_size(d_map2, M2, title2, file2)

# Comparison 3: TDP+ (1) vs Control (2)
d_map3, M3, title3, file3 = run_cluster_permutation_comparison(df, 1, 2, "(Control vs TDP+)", "rh.cluster_perm_tdp2_vs_tdp1.png")
plot_cluster_permutation_with_effect_size(d_map3, M3, title3, file3)

####################################################################################
########################## Cluster Permutation VTK Mesh Export ###############################################
####################################################################################

import pyvista as pv
import numpy as np

# === Load the right hemisphere mid-surface mesh ===
mesh_path = "path/to/your/mesh_file.vtk"  # UPDATE: Replace with your mesh path
mesh = pv.read(mesh_path)

# === Cluster Permutation significance masks — these must be in memory ===
try:
    [M1, M2, M3, sig_mask]
except NameError:
    raise RuntimeError("Cluster permutation masks (M1, M2, M3, sig_mask) must be defined in memory.")

masks = {
    "ClusterPerm_PART_vs_Control": sig_mask,
    "ClusterPerm_TDPneg_vs_TDPpos": M1,
    "ClusterPerm_TDPneg_vs_Control": M2,
    "ClusterPerm_TDPpos_vs_Control": M3
}

# === Save each mask to VTK mesh ===
for label, mask in masks.items():
    mesh_copy = mesh.copy()

    # No need to flip or transpose — flatten directly
    flat_mask = mask.flatten()

    if flat_mask.shape[0] != mesh_copy.n_points:
        raise ValueError(f"Mismatch for {label}: mask has {flat_mask.shape[0]}, mesh has {mesh_copy.n_points} vertices")

    mesh_copy[label] = flat_mask.astype(int)
    output_path = label.replace("ClusterPerm_", "rh.mid-surface_") + ".vtk"
    mesh_copy.save(output_path)
    print(f"Saved: {output_path}")

#####################################################################################
#####################################################################################

# === Additional Cluster Permutation: TDP− vs TDP+ restricted to Subfields on Right Side ===

# === Setup for Subfield Cluster Permutation Comparisons ===

subfield_masks = {
    234: (label_matrix == 234),
    236: (label_matrix == 236),
    238: (label_matrix == 238),
    240: (label_matrix == 240),
    "236_238": np.isin(label_matrix, [236, 238])
}

# Path to your right hemisphere mid-surface mesh (VTK format)
mesh_path = "path/to/your/rh.mid-surface.vtk"
base_mesh = pv.read(mesh_path)



for key, mask in subfield_masks.items():
    print(f"\nRunning Cluster Permutation for Subfield {key}...")

    group0, group1 = [], []

    for subj in df['mrn'].unique():
        subj_df = df[df['mrn'] == subj].sort_values("x_index")
        tdp_label = subj_df['tdp_status'].iloc[0]
        if tdp_label not in [0, 1]:
            continue

        matrix = subj_df.loc[:, "y0":"y20"].to_numpy().T
        masked_matrix = np.where(mask, matrix, np.nan)

        if tdp_label == 0:
            group0.append(masked_matrix)
        elif tdp_label == 1:
            group1.append(masked_matrix)

    group0 = np.array(group0)
    group1 = np.array(group1)

    print(f"Subfield {key} Group0 (TDP−): {group0.shape}, Group1 (TDP+): {group1.shape}")

    group0_masked = np.stack([m[mask] for m in group0])
    group1_masked = np.stack([m[mask] for m in group1])

    X_sub = [group0_masked, group1_masked]

    T_obs_sub, clusters_sub, p_values_sub, H0_sub = permutation_cluster_test(
        X_sub,
        n_permutations=1000,     #  default was 1000 permutations
        tail=0,                  # two-sided test
        n_jobs=1                 # single-threaded
        )

    sig_mask_sub = np.zeros_like(T_obs_sub, dtype=bool)
    for i, p in enumerate(p_values_sub):
        if p < 0.05:
            sig_mask_sub[clusters_sub[i]] = True

    full_d_map_sub = np.full(mask.shape, np.nan)
    full_sig_mask_sub = np.zeros(mask.shape, dtype=bool)


    # Always force TDP− first (control) for subfields
    control_group = group0_masked
    disease_group = group1_masked
    full_d_map_sub[mask] = compute_hedges_g(control_group, disease_group)
    full_sig_mask_sub[mask] = sig_mask_sub

    # Save VTK
    vtk_label = f"ClusterPerm_TDPneg_vs_TDPpos_Subfield{key}"
    mesh_copy = base_mesh.copy()

    flat_mask = full_sig_mask_sub.T.flatten()
    if flat_mask.shape[0] != mesh_copy.n_points:
        raise ValueError(f"Shape mismatch for Subfield {key}: mask has {flat_mask.shape[0]}, mesh has {mesh_copy.n_points} vertices.")

    mesh_copy[vtk_label] = flat_mask.astype(int)
    output_vtk_path = f"rh.mid-surface_cluster_perm_subfield{key}.vtk"
    mesh_copy.save(output_vtk_path)
    print(f"Saved VTK: {output_vtk_path}")

    # Plotting
    flipped_T_obs = full_d_map_sub.T
    flipped_mask = full_sig_mask_sub.T

    masked_T = np.ma.masked_where(~flipped_mask, flipped_T_obs)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    masked_d = np.ma.masked_where(~flipped_mask, flipped_T_obs)
    vmin = 0
    vmax = np.nanmax(d_map)
    im = ax.imshow(masked_d, cmap='Reds', origin='lower', extent=[0, 40, 0, 20], aspect='auto',
               vmin=vmin, vmax=vmax)
    ax.contour(flipped_mask, levels=[0.5], colors='black', linewidths=0.5,
           origin='lower', extent=[0,40, 0, 20])
    ax.contour(label_matrix, levels=np.unique(label_matrix), linewidths=0.5,
           colors='gray', origin='lower', extent=[0,40, 0, 20])
    cbar = plt.colorbar(im, ax=ax, label="Hedges' g (masked by Cluster Permutation significance)")
    cbar.ax.tick_params(labelsize=18)       


    ax.set_title(f"Cluster permutation Significance over Hedges' g (TDP− vs TDP+) Subfield {key}", fontsize=10)
    ax.set_xlabel("Lateral -> Medial")
    ax.set_ylabel("Posterior -> Anterior")
    ax.tick_params(axis='y', labelsize=18)

    plt.tight_layout()
    plt.savefig(f"rh.cluster_perm_hedgesg_tdp0_vs_tdp1_subfield{key}.png", dpi=300)
    plt.show()

    print(f"Completed Cluster Permutation + Plot for Subfield {key}!")



###############################################################################################################
#############################################Braak####################################################################
#################################################################################################################


def compute_hedges_g(group0, group1):
    mean_diff = np.nanmean(group0, axis=0) - np.nanmean(group1, axis=0)
    n0, n1 = group0.shape[0], group1.shape[0]
    pooled_std = np.sqrt(((n0 - 1) * np.nanvar(group0, axis=0) + (n1 - 1) * np.nanvar(group1, axis=0)) / (n0 + n1 - 2))
    d = mean_diff / pooled_std
    correction = 1 - (3 / (4 * (n0 + n1) - 9))
    return d * correction

def run_braak_cluster_permutation_comparison(df_subset, braak_a, braak_b, label, save_name_prefix):
    group0, group1 = [], []
    for subj in df_subset['mrn'].unique():
        subj_df = df_subset[df_subset['mrn'] == subj].sort_values("x_index")
        braak_label = subj_df['braak_stage'].iloc[0]
        
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

    # === FIX: Early checks ===
    if len(group0) == 0 or len(group1) == 0:
        print(f"Not enough subjects for {label}. Skipping comparison...")
        return None, None
    if group0.shape[1:] != group1.shape[1:]:
        print(f"Shape mismatch for {label}: {group0.shape} vs {group1.shape}. Skipping comparison...")
        return None, None

    # === If passed, continue ===

    X = [group0, group1]

    control_group = group0
    disease_group = group1

    d_map = compute_hedges_g(control_group, disease_group)
    

    T_obs, clusters, p_values, H0 = permutation_cluster_test(
        X,
        n_permutations=1000,     #  default was 1000 permutations
        tail=0,                  # two-sided test
        n_jobs=1                 # single-threaded
        )

    sig_mask = np.zeros_like(T_obs, dtype=bool)
    for i, p in enumerate(p_values):
        if p < 0.05:
            sig_mask[clusters[i]] = True

    return d_map, sig_mask

def plot_cluster_permutation_with_effect_size(T_obs, sig_mask, title, save_name):
    flipped_T_obs = T_obs.T
    flipped_mask = sig_mask.T
    masked_T = np.ma.masked_where(~flipped_mask, flipped_T_obs)

    fig, ax = plt.subplots(figsize=(10, 6))
    vmin = 0
    vmax = np.nanmax(d_map)
    im = ax.imshow(masked_d, cmap='Reds', origin='lower', extent=[0, 40, 0, 20], aspect='auto',
               vmin=vmin, vmax=vmax)
    ax.contour(flipped_mask, levels=[0.5], colors='black', linewidths=0.5, origin='lower', extent=[0,40,0,20])
    ax.contour(label_matrix, levels=np.unique(label_matrix), colors='gray', linewidths=0.5, origin='lower', extent=[0,40,0,20])
    cbar = plt.colorbar(im, ax=ax, label="Hedges' g (masked by Cluster Permutation significance)")
    cbar.ax.tick_params(labelsize=18)       

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Lateral -> Medial")
    ax.set_ylabel("Posterior -> Anterior")
    ax.tick_params(axis='y', labelsize=18)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()

def save_vtk_with_mask(base_mesh, sig_mask, output_name):
    mesh_copy = base_mesh.copy()
    flat_mask = sig_mask.T.flatten()
    if flat_mask.shape[0] != mesh_copy.n_points:
        raise ValueError(f"Mismatch in shape: mask {flat_mask.shape}, mesh {mesh_copy.n_points}")
    mesh_copy[output_name] = flat_mask.astype(int)
    mesh_copy.save(f"{output_name}.vtk")
    print(f"Saved: {output_name}.vtk")

# === Run Braak comparisons ===
# Select TDP-negative cohort (TDP0 + TDP2 only)
df_tdpneg = df[df['tdp_status'].isin([0, 2])]

for braak_stage in [1,2,3,4]:
    label = f"(Braak {braak_stage} vs Control) [TDP-negative subjects only]"
    save_prefix = f"rh.cluster_perm_braak0_vs_{braak_stage}_tdpneg"

    d_map, sig_mask = run_braak_cluster_permutation_comparison(df_tdpneg, 0, braak_stage, label, save_prefix)

    if d_map is None or sig_mask is None:
        continue

    plot_cluster_permutation_with_effect_size(d_map, sig_mask, f"Cluster permutation Significance over Hedges' g\n{label}", f"{save_prefix}.png")
    save_vtk_with_mask(base_mesh, sig_mask, save_prefix)

    print(df_tdpneg['tdp_status'].value_counts())
    print(df_tdpneg['braak_stage'].value_counts())
    np.nanmax(np.abs(d_map))

###############################################################################################################
##################################### PART without TDP-43 Braak Stratification ############################
###############################################################################################################

print("\n===== PART without TDP-43 Stratified by Braak Stages vs Controls =====\n")

# Select only PART without TDP-43 subjects (tdp_status = 0) and Controls (tdp_status = 2)
df_part_notdp = df[df['tdp_status'].isin([0, 2])]

def run_part_braak_cluster_permutation_comparison(df_subset, braak_stages, control_braak, label, save_prefix):
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

# Comparison 1: PART without TDP-43 Braak I-II vs Controls
d_map_braak12, sig_mask_braak12 = run_part_braak_cluster_permutation_comparison(
    df_part_notdp, 
    braak_stages=[1, 2], 
    control_braak=0,
    label="PART no-TDP Braak I-II vs Controls", 
    save_prefix="rh.cluster_perm_part_notdp_braak12_vs_controls"
)

if d_map_braak12 is not None and sig_mask_braak12 is not None:
    plot_cluster_permutation_with_effect_size(
        d_map_braak12, 
        sig_mask_braak12, 
        f"Cluster permutation Significance over Hedges' g\n(Controls vs PART no-TDP Braak I-II)", 
        "rh.cluster_perm_part_notdp_braak12_vs_controls.png"
    )
    save_vtk_with_mask(base_mesh, sig_mask_braak12, "rh.cluster_perm_part_notdp_braak12_vs_controls")
    print("Completed PART no-TDP Braak I-II vs Controls comparison")

# Comparison 2: PART without TDP-43 Braak III-IV vs Controls  
d_map_braak34, sig_mask_braak34 = run_part_braak_cluster_permutation_comparison(
    df_part_notdp, 
    braak_stages=[3, 4], 
    control_braak=0,
    label="PART no-TDP Braak III-IV vs Controls", 
    save_prefix="rh.cluster_perm_part_notdp_braak34_vs_controls"
)

if d_map_braak34 is not None and sig_mask_braak34 is not None:
    plot_cluster_permutation_with_effect_size(
        d_map_braak34, 
        sig_mask_braak34, 
        f"Cluster permutation Significance over Hedges' g\n(Controls vs PART no-TDP Braak III-IV)", 
        "rh.cluster_perm_part_notdp_braak34_vs_controls.png"
    )
    save_vtk_with_mask(base_mesh, sig_mask_braak34, "rh.cluster_perm_part_notdp_braak34_vs_controls")
    print("Completed PART no-TDP Braak III-IV vs Controls comparison")

# Print summary statistics
print(f"\nSummary of PART without TDP-43 subjects by Braak stage:")
part_notdp_only = df_part_notdp[df_part_notdp['tdp_status'] == 0]
print(part_notdp_only['braak_stage'].value_counts().sort_index())

print(f"\nControls by Braak stage:")
controls_only = df_part_notdp[df_part_notdp['tdp_status'] == 2]
print(controls_only['braak_stage'].value_counts().sort_index())






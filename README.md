# PART-hippocampal-morphometry-KAJ-lab

Analysis scripts for hippocampal subfield thickness and curvature in Primary Age-Related Tauopathy (PART) with and without TDP-43 co-pathology.

## Citation

If you use these scripts, please cite:

> Youssef H, Gatto RG, Petersen RC, Reichard RR, Jack CR Jr, Whitwell JL, Josephs KA. Hippocampal subfield thickness and shape analysis in examining the impact of TDP-43 in primary age-related tauopathy. *Alzheimer's & Dementia*. 2025. [DOI pending]

## Overview

This repository contains Python scripts for analyzing hippocampal morphometry in autopsy-confirmed PART cases. The analysis pipeline includes:

- **Thickness analysis**: Point-wise hippocampal subfield thickness comparisons
- **Curvature analysis**: Gaussian curvature surface deformation
- **Cluster-based permutation testing**: Non-parametric inference with 5,000 permutations (MNE-Python)
- **X-axis FDR correction**: False discovery rate correction for medial-to-lateral axis analyses

## Requirements

### Prerequisites

- **FreeSurfer 7.4.1+**: For hippocampal subfield segmentation
- **HIPSTA**: For thickness/curvature extraction ([https://github.com/Deep-MI/hipsta](https://github.com/Deep-MI/hipsta))

### Python Dependencies

```bash
pip install -r requirements.txt
```

See `requirements.txt` for complete list. Key packages:
- Python 3.10+
- NumPy 2.x
- SciPy 1.x
- Pandas 2.x
- MNE-Python 1.10+ (for cluster permutation testing)
- Statsmodels 0.14+
- PyVista 0.46+ (for VTK visualization)
- Matplotlib 3.x

## Repository Structure

```
PART-hippocampal-morphometry/
├── README.md
├── requirements.txt
├── LICENSE
│
├── thickness_cluster_permutation/    # Cluster permutation analyses for thickness
│   ├── lh_thickness_cluster_permutation.py
│   └── rh_thickness_cluster_permutation.py
│
├── curvature_cluster_permutation/    # Cluster permutation analyses for curvature
│   ├── lh_gauss_cluster_permutation.py
│   └── rh_gauss_cluster_permutation.py
│
├── xaxis_medial_lateral_FDR/  # X-axis (medial-to-lateral) FDR-corrected analyses
│   ├── FDR_lh_hipp_thickness.py
│   ├── FDR_rh_hipp_thickness.py
│   ├── FDR_lt_hipp_gauss_curv.py
│   └── FDR_rt_hipp_gauss_curv.py
```

## Usage

### 1. Cluster-Based Permutation Testing

```python
# Run left hemisphere thickness analysis
python thickness_cluster_permutation/lh_thickness_cluster_permutation.py
```

Key parameters:
- `n_permutations = 5000`
- `random_seed = 42` (for reproducibility)
- `alpha = 0.05` (FWER-corrected)
- Method: MNE-Python `permutation_cluster_test`

### 2. X-Axis FDR-Corrected Analysis

```python
# Run left hemisphere thickness x-axis analysis
python xaxis_medial_lateral_FDR/FDR_lh_hipp_thickness.py
```

Analyzes thickness/curvature along the medial-to-lateral axis with FDR correction.

## Input Data Format

Scripts expect HIPSTA output in CSV format:
- Grid dimensions: 41 × 21 (861 vertices per hemisphere)
- Columns: `mrn`, `axis`, `y0`-`y20`, `PART=1_control=0`, `tdp_status`

Example file structure:
```
lh.grid-segments-z-all.csv    # Left hemisphere thickness
rh.grid-segments-z-all.csv    # Right hemisphere thickness
lh.gauss_curv.csv             # Left Gaussian curvature
rh.gauss_curv.csv             # Right Gaussian curvature
```

## Statistical Methods

- **Cluster-based permutation testing**: MNE-Python `permutation_cluster_test` with cluster-level FWER correction
- **Effect sizes**: Hedges' g with 95% confidence intervals
- **Multiple comparisons**: FWER correction via permutation distribution
- **Covariates**: ANCOVA adjusting for Thal phase, Braak stage, age, and sex

## Data Availability

Due to patient privacy considerations, individual-level MRI data cannot be publicly shared. De-identified summary data and statistical maps are available from the corresponding author upon reasonable request, pending IRB approval and data use agreement.

## Contact

For questions about the code or methods:
- **First author**: Hossam Youssef M.B.,B.Ch
- **Email**: youssef.hossam@mayo.edu
- **Institution**: Mayo Clinic, Rochester, MN
**------------------------------**
- **Corresponding Author**: Keith A. Josephs, MD, MST, MSc
- **Email**: josephs.keith@mayo.edu
- **Institution**: Mayo Clinic, Rochester, MN

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- HIPSTA algorithm: Diers et al. (2023), Fischbach et al. (2023)
- FreeSurfer hippocampal subfields: Iglesias et al. (2015)
- MNE-Python: Gramfort et al. (2013)





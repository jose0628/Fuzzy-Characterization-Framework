# Fuzzy Characterization Framework

A Python toolkit for **privacy-preserving user behavior characterization** and **soft clustering** from per-user event activity.

The repository builds user-level feature vectors from event counts, reduces dimensionality (PCA), estimates a suitable number of clusters, and applies clustering algorithms with a focus on fuzzy methods (FCM/PFCM).

## What This Project Does

- Parses serialized event-count payloads into tabular numeric columns.
- Engineers robust behavioral features (shares, entropy, concentration, role-like activity composition).
- Applies robust preprocessing for heavy-tailed activity distributions.
- Produces PCA-based representations for clustering.
- Estimates the best `k` using K-Means + FCM validity metrics.
- Runs clustering with:
  - K-Means
  - Agglomerative Hierarchical Clustering (AHC)
  - Fuzzy C-Means (FCM)
  - Possibilistic Fuzzy C-Means (PFCM)
  - Gustafson–Kessel (GK)
- Exports metrics, labels/memberships, profiles, and diagnostic plots.

## Repository Highlights

- `dataset_events_serialization.py`: Expands `event_type_counts` dict-like column into numeric columns.
- `dataset_preparation.py`: Builds engineered user behavior features and PCA outputs.
- `strong_normalization.py`: Multi-step preprocessing variants and diagnostics (Step 0..4).
- `principal_components_analysis.py`: PCA diagnostics and dominant feature loadings.
- `elbow_knee_cluster_estimation.py`: Simple elbow detection with `kneed`.
- `k_cluster_determination.py`: Joint K-Means + FCM `k` selection metrics and plots.
- `Fuzzy-C-Means_standalone.py`: FCM runs for all metric-suggested `k` values with contour outputs.
- `Fuzzy_C_Means_1-12.py`: Extended FCM analysis (k=2..12), advanced metrics and zone plots.
- `PFCM_Possibilistic_Algorithm.py`: PFCM sweep over `k` with stability/interpretability reporting.
- `algorithms_master_v1.py` / `algorithms_master_v2.py`: Unified cluster suite entry points.

## Requirements

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn scikit-fuzzy scipy kneed
```

Notes:
- `scipy` is needed for AHC dendrogram support in `algorithms_master_*`.
- Most scripts read/write CSV and PNG artifacts in-place.

## Typical Data Pipeline

### 1) Expand serialized event counts (if needed)

Use when your input has an `event_type_counts` column with JSON-like dict values.

```bash
python dataset_events_serialization.py datasets/leonardo_activity.csv
```

Output example:
- `datasets/leonardo_activity_unpacked.csv`

### 2) Generate preprocessing variants (recommended before clustering)

```bash
python strong_normalization.py
```

Default input in script:
- `datasets/dufry_events_activity_unpacked.csv`

Outputs:
- Step CSVs (`_step0_raw`, `_step1_log1p`, `_step2_log1p_winsor_1_99`, `_step3_log1p_winsor_robust_1_99`, `_step4_l1comp_robust`)
- Diagnostic plots in `FMC_output_preprocess_*`

### 3) Build engineered vectors + PCA representation

```bash
python dataset_preparation.py
```

Default input in script:
- `datasets/leonardo_unpacked.csv` (adjust if needed)

Outputs:
- `users_behavior_vectors_pca.csv` (clustering input)
- `users_engineered_features_raw.csv` (interpretable features)

### 4) Estimate optimal `k`

```bash
python k_cluster_determination.py --pca users_behavior_vectors_pca.csv --kmin 2 --kmax 10
```

Outputs:
- `optimal_k_analysis_kmeans_fcm.png`
- `optimal_k_metrics_table.csv`

Optional simple elbow:

```bash
python elbow_knee_cluster_estimation.py
```

## Clustering Workflows

### Unified cluster suite (recommended)

```bash
python algorithms_master_v2.py --algo kmeans --k 4
python algorithms_master_v2.py --algo ahc --k 4
python algorithms_master_v2.py --algo fcm --k 4
python algorithms_master_v2.py --algo pfcm --k 4
python algorithms_master_v2.py --algo gk --k 4
```

Common options:
- `--pca users_behavior_vectors_pca.csv`
- `--feat users_engineered_features_raw.csv`
- `--user-col actor_id`
- `--outdir outputs`

### FCM deep-dive scripts

- `Fuzzy-C-Means_standalone.py`: runs all unique `k` suggested by multiple metrics and saves contour/membership artifacts.
- `Fuzzy_C_Means_1-12.py`: extended metrics (including stability/FSI/interpretability) and richer visualizations for `k=2..12`.

### PFCM deep-dive script

```bash
python PFCM_Possibilistic_Algorithm.py --input datasets/leonardo_activity_unpacked.csv --outdir PFCM_output_1_12 --kmin 2 --kmax 12
```

Outputs include per-`k` memberships/typicalities, centers, scatter/heatmaps, and summary metrics.

## Output Structure (Typical)

- `outputs/`: unified suite outputs (labels, profiles, scatter/contour plots)
- `FMC_output_*`: FCM-focused experiments and diagnostics
- `PFCM_output_*`: PFCM-focused experiments and diagnostics
- `optimal_k_metrics_table.csv`, `optimal_k_analysis_kmeans_fcm.png`: `k` selection artifacts

## Practical Notes

- Many scripts have dataset paths hardcoded near the top (`CSV_PATH`, `INPUT_CSV`).
- Before running, verify the selected dataset exists and contains:
  - a user id column (`actor_id`, `user_id`, or `id` depending on script), and
  - numeric event/activity columns.
- For heavy-tailed behavioral counts, the Step 3 preprocessing (`log1p + winsorize + RobustScaler`) is frequently the most stable for clustering.

## Current Datasets in `datasets/`

The repository already includes prepared datasets for multiple contexts, including:
- `dufry_events_activity_*`
- `heathrow_uses_diary_events_*`
- `leonardo_activity_*`

This allows running the pipeline immediately by choosing one dataset family and updating script paths consistently.

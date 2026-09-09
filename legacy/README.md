# Legacy scripts

These are the exploratory, standalone scripts that preceded the packaged
framework (`src/fuzzy_characterization`). They are kept for reference and
reproducibility of earlier experiments; they are **not** used by the package,
tests or CLI, and most of them have dataset paths hard-coded near the top.
`README_legacy.md` is the original README describing them.

| Legacy script | Superseded by |
|---|---|
| `dataset_events_serialization.py` | `fcf unpack-events`, `data/loaders.py` |
| `dataset_preparation.py`, `strong_normalization.py` | `preprocessing/transforms.py`, `data/indicators.py` |
| `principal_components_analysis.py` | `preprocessing/reduction.py` |
| `elbow_knee_cluster_estimation.py`, `K_cluster_estimation.py`, `k_cluster_determination.py` | `clustering/selection.py` (`fcf select-k`) |
| `Fuzzy-C-Means_standalone.py`, `Fuzzy_C_Means_1-12*.py` | `clustering/fcm.py`, `visualization/plots.py` |
| `PFCM_Possibilistic_Algorithm.py` | `clustering/pfcm.py` |
| `Gustafson_Kessel_Algorithm.py` | `clustering/gk.py` |
| `k_means_clustering.py`, `ahc_algorithm.py` | `clustering/baselines.py` |
| `algorithms_master_v1.py`, `algorithms_master_v2.py` | `evaluation/comparison.py` (`fcf compare`) |

The legacy scripts additionally require `scikit-fuzzy` and `kneed`
(`pip install scikit-fuzzy kneed`).

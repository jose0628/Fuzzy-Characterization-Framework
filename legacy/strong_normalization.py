"""
Preprocessing pipeline for leonardo_activity_unpacked.csv that generates and SAVES
a CSV output for EACH step, so you can choose the best one.

Steps:
  0) baseline (raw numeric features, no transform)                -> _step0_raw
  1) log1p(counts)                                               -> _step1_log1p
  2) log1p + winsorize (clip per-feature quantiles)              -> _step2_log1p_winsor
  3) log1p + winsorize + RobustScaler (median/IQR scaling)       -> _step3_log1p_winsor_robust
  4) per-user L1 composition + RobustScaler                      -> _step4_l1comp_robust

Also saves PCA-2D scatter and FCM hard-label plots for each step under:
  ./FMC_output_preprocess/

Important:
- Step 3 returns *scaled* features (robust z-like units). This is typically what you want
  for clustering input. It will not be in original count units.
- Non-numeric columns are preserved in each output file.
- Numeric ID-like columns (actor_id/user_id/id) are preserved as identifiers in outputs,
  but are NOT used as clustering features.

Dependencies:
  pip install numpy pandas matplotlib scikit-learn scikit-fuzzy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA


# -----------------------------
# Config
# -----------------------------
INPUT_CSV = "datasets/dufry_events_activity_unpacked.csv"
OUT_DIR = "FMC_output_preprocess_dufry"

# ID columns: kept as identifiers, but not used as features
ID_COL_CANDIDATES = {"actor_id", "user_id", "id"}

# Outlier clipping quantiles (winsorization)
CLIP_LO, CLIP_HI = 0.01, 0.99

# FCM settings for visual diagnostics (fixed k for comparison)
K_FCM = 3
M_FUZZINESS = 2.0
ERROR = 1e-5
MAXITER = 1500
SEED = 42


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def winsorize_df(X: pd.DataFrame, q_low: float, q_high: float) -> pd.DataFrame:
    """Clip each column to [q_low, q_high] quantiles."""
    lo = X.quantile(q_low)
    hi = X.quantile(q_high)
    return X.clip(lower=lo, upper=hi, axis=1)


def l1_normalize_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    row_sums = np.sum(arr, axis=1, keepdims=True)
    return arr / (row_sums + eps)


def pca2_from_array(arr: np.ndarray, scaler) -> np.ndarray:
    Xs = scaler.fit_transform(arr)
    pca = PCA(n_components=2, random_state=SEED)
    return pca.fit_transform(Xs)


def save_scatter(X2: np.ndarray, title: str, path: str) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(X2[:, 0], X2[:, 1], s=10, alpha=0.5)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_fcm_hardlabels(X2: np.ndarray, k: int, title: str, path: str) -> None:
    data2 = X2.T  # (2, N)
    cntr, U, _, _, _, _, _ = fuzz.cluster.cmeans(
        data2, c=k, m=M_FUZZINESS, error=ERROR, maxiter=MAXITER, init=None, seed=SEED
    )
    labels = np.argmax(U, axis=0)

    plt.figure(figsize=(7, 6))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=12, alpha=0.7)
    plt.scatter(cntr[:, 0], cntr[:, 1], marker="*", s=220, edgecolor="k")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_step_csv(df_original: pd.DataFrame,
                  feature_names: list[str],
                  transformed_values: np.ndarray,
                  step_tag: str) -> str:
    """
    Create a full dataframe preserving all original columns,
    but replacing the numeric feature columns used with transformed_values.
    Save to disk as INPUT_CSV basename + step_tag + .csv
    """
    out_df = df_original.copy()
    for j, col in enumerate(feature_names):
        out_df[col] = transformed_values[:, j]

    base, ext = os.path.splitext(INPUT_CSV)
    out_path = f"{base}{step_tag}{ext}"
    out_df.to_csv(out_path, index=False)
    return out_path


# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dir(OUT_DIR)

    df = pd.read_csv(INPUT_CSV)

    # Numeric features used for transformations/clustering
    num_df = df.select_dtypes(include=[np.number]).copy()

    # Drop numeric ID-like columns from FEATURES (but keep them in the saved CSVs)
    feature_names = [c for c in num_df.columns if c.lower() not in ID_COL_CANDIDATES]
    if len(feature_names) < 2:
        raise ValueError(f"Need at least 2 numeric non-ID features. Found {len(feature_names)}.")

    feats = df[feature_names].astype(float).copy()
    X0 = feats.values  # raw numeric feature matrix

    saved_paths = []

    # -----------------------------
    # Step 0: raw
    # -----------------------------
    step0_tag = "_step0_raw"
    saved_paths.append(save_step_csv(df, feature_names, X0, step0_tag))

    X0_2d = pca2_from_array(X0, StandardScaler())
    save_scatter(X0_2d, "Step 0: raw → StandardScaler → PCA-2D",
                 os.path.join(OUT_DIR, "step0_pca2d.png"))
    save_fcm_hardlabels(X0_2d, K_FCM, f"Step 0: FCM hard labels in PCA-2D (k={K_FCM})",
                        os.path.join(OUT_DIR, "step0_fcm_labels.png"))

    # -----------------------------
    # Step 1: log1p
    # -----------------------------
    X1 = np.log1p(X0)
    step1_tag = "_step1_log1p"
    saved_paths.append(save_step_csv(df, feature_names, X1, step1_tag))

    X1_2d = pca2_from_array(X1, StandardScaler())
    save_scatter(X1_2d, "Step 1: log1p → StandardScaler → PCA-2D",
                 os.path.join(OUT_DIR, "step1_pca2d.png"))
    save_fcm_hardlabels(X1_2d, K_FCM, f"Step 1: FCM hard labels (k={K_FCM})",
                        os.path.join(OUT_DIR, "step1_fcm_labels.png"))

    # -----------------------------
    # Step 2: log1p + winsorize
    # -----------------------------
    X1_df = pd.DataFrame(X1, columns=feature_names)
    X2_df = winsorize_df(X1_df, CLIP_LO, CLIP_HI)
    X2 = X2_df.values
    step2_tag = f"_step2_log1p_winsor_{int(CLIP_LO*100)}_{int(CLIP_HI*100)}"
    saved_paths.append(save_step_csv(df, feature_names, X2, step2_tag))

    X2_2d = pca2_from_array(X2, StandardScaler())
    save_scatter(X2_2d, f"Step 2: log1p + winsorize({CLIP_LO:.2f},{CLIP_HI:.2f}) → StandardScaler → PCA-2D",
                 os.path.join(OUT_DIR, "step2_pca2d.png"))
    save_fcm_hardlabels(X2_2d, K_FCM, f"Step 2: FCM hard labels (k={K_FCM})",
                        os.path.join(OUT_DIR, "step2_fcm_labels.png"))

    # -----------------------------
    # Step 3: log1p + winsorize + RobustScaler (THIS is the one you preferred)
    # -----------------------------
    robust = RobustScaler(quantile_range=(25, 75))
    X3 = robust.fit_transform(X2)  # scaled features
    step3_tag = f"_step3_log1p_winsor_robust_{int(CLIP_LO*100)}_{int(CLIP_HI*100)}"
    saved_paths.append(save_step_csv(df, feature_names, X3, step3_tag))

    X3_2d = pca2_from_array(X3, StandardScaler())  # PCA on already-robust-scaled features
    save_scatter(X3_2d, "Step 3: log1p + winsorize → RobustScaler → PCA-2D",
                 os.path.join(OUT_DIR, "step3_pca2d.png"))
    save_fcm_hardlabels(X3_2d, K_FCM, f"Step 3: FCM hard labels (k={K_FCM})",
                        os.path.join(OUT_DIR, "step3_fcm_labels.png"))

    # -----------------------------
    # Step 4: per-user L1 composition + RobustScaler
    # -----------------------------
    X4 = l1_normalize_rows(X0)
    X4 = robust.fit_transform(X4)
    step4_tag = "_step4_l1comp_robust"
    saved_paths.append(save_step_csv(df, feature_names, X4, step4_tag))

    X4_2d = pca2_from_array(X4, StandardScaler())
    save_scatter(X4_2d, "Step 4: L1 composition → RobustScaler → PCA-2D",
                 os.path.join(OUT_DIR, "step4_pca2d.png"))
    save_fcm_hardlabels(X4_2d, K_FCM, f"Step 4: FCM hard labels (k={K_FCM})",
                        os.path.join(OUT_DIR, "step4_fcm_labels.png"))

    # -----------------------------
    # Summary
    # -----------------------------
    print("\nSaved CSVs (choose the step you prefer):")
    for p in saved_paths:
        print(f"  - {p}")

    print(f"\nSaved diagnostic plots under: {OUT_DIR}/")
    print("Tip: If Step 3 looks best, use the *_step3_log1p_winsor_robust_*.csv as clustering input.")


if __name__ == "__main__":
    main()
"""
Run Fuzzy C-Means for ALL unique k values suggested by the selection metrics, and:

- Generate membership contour plots for each k
- Save plots into ./FMC_output (create if missing)
- Save membership matrix U for each k (per user)
- Save a "feature mapping" file describing:
    * which input features were used
    * scaler parameters
    * PCA(2D) loadings + explained variance (since clustering/plots are in PCA-2D)
    * cluster centers in PCA-2D for each k

Notes:
- Contour plots require 2D. This script clusters in 2D PCA space (PC1, PC2).
- If a metric returns k < 2 (e.g., 1), it is ignored (FCM requires c>=2).

Dependencies:
  pip install numpy pandas matplotlib scikit-learn scikit-fuzzy
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# -----------------------------
# Config
# -----------------------------
CSV_PATH = "datasets/dufry_events_activity_unpacked_step3_log1p_winsor_robust_1_99.csv"
OUTPUT_DIR = "FMC_output_dufry"

ID_COL_CANDIDATES = {"actor_id", "user_id", "id"}

K_MIN, K_MAX = 2, 12
M_FUZZINESS = 2.0
ERROR = 1e-5
MAXITER = 2000
RANDOM_SEED = 42

GRID_N = 300
CONTOUR_LEVELS = [0.2, 0.4, 0.6, 0.8]

# save only top loadings for PCA mapping (full vectors also saved)
TOP_PCA_LOADINGS = 20


# -----------------------------
# Helpers: fuzzy validity indices
# -----------------------------
def partition_entropy(U: np.ndarray) -> float:
    """PE = -(1/N) sum_i sum_k u_ki log(u_ki)"""
    eps = 1e-12
    U_safe = np.clip(U, eps, 1.0)
    return float(-(U_safe * np.log(U_safe)).sum() / U.shape[1])


def xie_beni_index(X: np.ndarray, cntr: np.ndarray, U: np.ndarray, m: float) -> float:
    """
    XB = [sum_k sum_i u_ki^m * ||x_i - v_k||^2] / [N * min_{p!=q} ||v_p - v_q||^2]
    X: (N, d), cntr: (c, d), U: (c, N)
    """
    diff = cntr[:, None, :] - X[None, :, :]  # (c, N, d)
    dist2 = np.sum(diff**2, axis=2)          # (c, N)
    compactness = np.sum((U**m) * dist2)

    c = cntr.shape[0]
    sep2 = np.inf
    for i in range(c):
        for j in range(i + 1, c):
            d2 = float(np.sum((cntr[i] - cntr[j])**2))
            if d2 < sep2:
                sep2 = d2

    if sep2 <= 1e-12:
        return float("inf")
    return float(compactness / (X.shape[0] * sep2))


# -----------------------------
# Elbow / knee detection: max distance to chord
# -----------------------------
def knee_point_max_distance(k_vals: np.ndarray, y_vals: np.ndarray, minimize: bool = True) -> int:
    k = np.asarray(k_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)

    # Convert to increasing curve if minimizing
    y_use = -y if minimize else y

    # Normalize
    k_n = (k - k.min()) / (k.max() - k.min() + 1e-12)
    y_n = (y_use - y_use.min()) / (y_use.max() - y_use.min() + 1e-12)

    p1 = np.array([k_n[0], y_n[0]])
    p2 = np.array([k_n[-1], y_n[-1]])

    line = p2 - p1
    line_norm = np.linalg.norm(line) + 1e-12

    dists = []
    for i in range(len(k_n)):
        p = np.array([k_n[i], y_n[i]])
        dist = np.abs(np.cross(line, p - p1)) / line_norm
        dists.append(dist)

    idx = int(np.argmax(dists))
    return int(k_vals[idx])


# -----------------------------
# Plot + save membership contours for a given k
# -----------------------------
def save_membership_contours(
        X2: np.ndarray,
        cntr: np.ndarray,
        k: int,
        out_path: str,
        m: float,
        error: float,
        maxiter: int,
        grid_n: int = 300,
        levels=None,
):
    if levels is None:
        levels = [0.2, 0.4, 0.6, 0.8]

    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_n),
                         np.linspace(y_min, y_max, grid_n))
    grid = np.vstack([xx.ravel(), yy.ravel()])  # (2, grid_n^2)

    # Predict memberships on grid
    U_grid, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        grid, cntr, m=m, error=error, maxiter=maxiter
    )

    ncols = int(np.ceil(np.sqrt(k)))
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows), squeeze=False)
    axes = axes.ravel()

    for i in range(k):
        ax = axes[i]
        Zi = U_grid[i, :].reshape(xx.shape)

        cs = ax.contour(xx, yy, Zi, levels=levels, linewidths=1.0)
        ax.clabel(cs, inline=True, fontsize=8)

        ax.scatter(X2[:, 0], X2[:, 1], s=10, alpha=0.35, label="Data")
        ax.scatter(cntr[i, 0], cntr[i, 1], marker="*", s=180, edgecolor="k", label=f"C{i+1}")

        ax.set_title(f"Cluster {i+1} Membership")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="upper right", fontsize=8)

    for j in range(k, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Fuzzy C-Means Membership Contours (k={k})", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # Determine user identifier column (if present)
    user_col = None
    for cand in ["actor_id", "user_id", "id"]:
        if cand in df.columns:
            user_col = cand
            break

    if user_col is None:
        user_ids = pd.Series(np.arange(len(df)), name="row_index")
        user_id_label = "row_index"
    else:
        user_ids = df[user_col].copy()
        user_id_label = user_col

    # Numeric feature matrix
    num_df = df.select_dtypes(include=[np.number]).copy()

    # Drop numeric ID-like columns if present
    drop_ids = [c for c in num_df.columns if c.lower() in ID_COL_CANDIDATES]
    if drop_ids:
        num_df.drop(columns=drop_ids, inplace=True)

    if num_df.shape[1] < 2:
        raise ValueError(f"Need at least 2 numeric features. Found {num_df.shape[1]}.")

    feature_names = num_df.columns.tolist()

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(num_df.values)  # (N, d)

    # PCA to 2D (for clustering + contours)
    pca2 = PCA(n_components=2, random_state=RANDOM_SEED)
    X2 = pca2.fit_transform(X)  # (N, 2)
    data2 = X2.T  # (2, N)

    # -----------------------------
    # Sweep k for metric table
    # -----------------------------
    rows = []
    k_values = np.arange(K_MIN, K_MAX + 1)

    for c in k_values:
        cntr, U, _, _, jm, _, fpc = fuzz.cluster.cmeans(
            data2, c=c, m=M_FUZZINESS, error=ERROR, maxiter=MAXITER, init=None, seed=RANDOM_SEED
        )

        fpc_pc = float(fpc)
        pe = partition_entropy(U)
        xb = xie_beni_index(X2, cntr, U, M_FUZZINESS)
        jm_final = float(jm[-1])

        labels = np.argmax(U, axis=0)

        try:
            sil = float(silhouette_score(X2, labels))
        except Exception:
            sil = np.nan

        try:
            ch = float(calinski_harabasz_score(X2, labels))
        except Exception:
            ch = np.nan

        try:
            db = float(davies_bouldin_score(X2, labels))
        except Exception:
            db = np.nan

        km = KMeans(n_clusters=c, n_init="auto", random_state=RANDOM_SEED)
        km.fit(X2)
        km_inertia = float(km.inertia_)

        rows.append({
            "k": int(c),
            "FPC_PC": fpc_pc,
            "PE": pe,
            "XB": xb,
            "JM_inertia": jm_final,
            "Silhouette": sil,
            "CalinskiHarabasz": ch,
            "DaviesBouldin": db,
            "KMeans_inertia": km_inertia,
        })

    scores = pd.DataFrame(rows)
    scores_path = os.path.join(OUTPUT_DIR, "k_selection_metrics.csv")
    scores.to_csv(scores_path, index=False)

    # -----------------------------
    # Select k per metric
    # -----------------------------
    best_k = {}
    best_k["FPC_PC"] = int(scores.loc[scores["FPC_PC"].idxmax(), "k"])
    best_k["PE"] = int(scores.loc[scores["PE"].idxmin(), "k"])
    best_k["XB"] = int(scores.loc[scores["XB"].idxmin(), "k"])
    best_k["JM_inertia"] = int(scores.loc[scores["JM_inertia"].idxmin(), "k"])

    best_k["Silhouette"] = int(scores.loc[scores["Silhouette"].idxmax(skipna=True), "k"]) \
        if scores["Silhouette"].notna().any() else best_k["FPC_PC"]
    best_k["CalinskiHarabasz"] = int(scores.loc[scores["CalinskiHarabasz"].idxmax(skipna=True), "k"]) \
        if scores["CalinskiHarabasz"].notna().any() else best_k["FPC_PC"]
    best_k["DaviesBouldin"] = int(scores.loc[scores["DaviesBouldin"].idxmin(skipna=True), "k"]) \
        if scores["DaviesBouldin"].notna().any() else best_k["FPC_PC"]

    # Elbow knee from Jm (FCM)
    knee_jm = knee_point_max_distance(scores["k"].values, scores["JM_inertia"].values, minimize=True)
    best_k["Elbow_JM"] = int(knee_jm)

    # Save chosen k per metric
    best_k_path = os.path.join(OUTPUT_DIR, "best_k_by_metric.json")
    with open(best_k_path, "w", encoding="utf-8") as f:
        json.dump(best_k, f, indent=2)

    # -----------------------------
    # NEW RULE: run FCM for ALL unique k values suggested
    # -----------------------------
    unique_ks = sorted(set(int(v) for v in best_k.values()))
    # Filter invalid (FCM requires >=2)
    unique_ks = [k for k in unique_ks if k >= 2]

    if not unique_ks:
        raise ValueError("No valid k values (>=2) found from metric suggestions.")

    # Save the list of ks we will run
    ks_path = os.path.join(OUTPUT_DIR, "ks_to_run.txt")
    with open(ks_path, "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, unique_ks)) + "\n")

    print("\nBest k per metric:")
    for metric, k in best_k.items():
        print(f"{metric:>18}: k = {k}")

    print("\nUnique k values to run (>=2):", unique_ks)
    print(f"Outputs will be saved under: {OUTPUT_DIR}/")

    # -----------------------------
    # Save feature mapping (inputs + preprocessing + PCA mapping)
    # -----------------------------
    mapping = {
        "input_csv": CSV_PATH,
        "user_identifier_column": user_id_label,
        "n_rows": int(X.shape[0]),
        "n_features_used": int(len(feature_names)),
        "features_used": feature_names,
        "scaler": {
            "type": "StandardScaler",
            "mean_": scaler.mean_.tolist(),
            "scale_": scaler.scale_.tolist(),
        },
        "pca_2d": {
            "explained_variance_ratio_": pca2.explained_variance_ratio_.tolist(),
            "components_": pca2.components_.tolist(),  # shape (2, n_features)
            "feature_loadings_top": {}
        }
    }

    # Add top loadings per PC for readability
    for pc_i in range(2):
        loadings = pca2.components_[pc_i]
        idx = np.argsort(np.abs(loadings))[::-1][:TOP_PCA_LOADINGS]
        mapping["pca_2d"]["feature_loadings_top"][f"PC{pc_i+1}"] = [
            {"feature": feature_names[j], "loading": float(loadings[j])}
            for j in idx
        ]

    mapping_path = os.path.join(OUTPUT_DIR, "feature_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    # -----------------------------
    # Also save PCA-projected data with user ids (useful for plotting/debugging)
    # -----------------------------
    pca_data = pd.DataFrame({
        user_id_label: user_ids.values,
        "PC1": X2[:, 0],
        "PC2": X2[:, 1],
    })
    pca_data.to_csv(os.path.join(OUTPUT_DIR, "users_pca2d.csv"), index=False)

    # -----------------------------
    # For each k: run FCM, save memberships, save centers, save plots
    # -----------------------------
    for k in unique_ks:
        run_dir = os.path.join(OUTPUT_DIR, f"k_{k}")
        os.makedirs(run_dir, exist_ok=True)

        cntr, U, _, _, jm, _, fpc = fuzz.cluster.cmeans(
            data2, c=k, m=M_FUZZINESS, error=ERROR, maxiter=MAXITER, init=None, seed=RANDOM_SEED
        )

        hard_labels = np.argmax(U, axis=0)

        # Save memberships (U) per user
        mem_df = pd.DataFrame({user_id_label: user_ids.values})
        for i in range(k):
            mem_df[f"cluster_{i+1}"] = U[i, :]
        mem_df["hard_cluster"] = hard_labels + 1  # 1-based for readability
        mem_df.to_csv(os.path.join(run_dir, f"memberships_k_{k}.csv"), index=False)

        # Save centers and summary
        centers_df = pd.DataFrame(cntr, columns=["PC1_center", "PC2_center"])
        centers_df.insert(0, "cluster", np.arange(1, k + 1))
        centers_df.to_csv(os.path.join(run_dir, f"centers_k_{k}.csv"), index=False)

        summary = {
            "k": int(k),
            "m": float(M_FUZZINESS),
            "fpc": float(fpc),
            "jm_final": float(jm[-1]),
            "n_points": int(X2.shape[0]),
        }
        with open(os.path.join(run_dir, f"summary_k_{k}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Save membership contour plot
        contour_path = os.path.join(run_dir, f"membership_contours_k_{k}.png")
        save_membership_contours(
            X2=X2,
            cntr=cntr,
            k=k,
            out_path=contour_path,
            m=M_FUZZINESS,
            error=ERROR,
            maxiter=MAXITER,
            grid_n=GRID_N,
            levels=CONTOUR_LEVELS,
        )

        # Save hard-label scatter (optional but useful)
        plt.figure(figsize=(7, 6))
        plt.scatter(X2[:, 0], X2[:, 1], c=hard_labels, s=12, alpha=0.7)
        plt.scatter(cntr[:, 0], cntr[:, 1], marker="*", s=220, edgecolor="k")
        plt.title(f"Hard labels (argmax membership) in PCA-2D (k={k})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"hard_labels_k_{k}.png"), dpi=200)
        plt.close()

    # Save elbow plots (FCM Jm + KMeans inertia) in OUTPUT_DIR
    k_arr = scores["k"].values
    jm_arr = scores["JM_inertia"].values
    km_arr = scores["KMeans_inertia"].values

    plt.figure(figsize=(10, 5))
    plt.plot(k_arr, jm_arr, marker="o")
    plt.axvline(knee_jm, linestyle="--")
    plt.title("Elbow Plot (FCM objective Jm vs k)")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("FCM objective Jm (lower is better)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "elbow_fcm_jm.png"), dpi=200)
    plt.close()

    knee_km = knee_point_max_distance(k_arr, km_arr, minimize=True)
    plt.figure(figsize=(10, 5))
    plt.plot(k_arr, km_arr, marker="o")
    plt.axvline(knee_km, linestyle="--")
    plt.title("Elbow Plot (K-Means inertia vs k) [reference]")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (lower is better)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "elbow_kmeans_inertia.png"), dpi=200)
    plt.close()

    print("\nDone. Files written under FMC_output/")
    print("Key outputs:")
    print(f"  - {scores_path}")
    print(f"  - {best_k_path}")
    print(f"  - {mapping_path}")
    print(f"  - {os.path.join(OUTPUT_DIR, 'users_pca2d.csv')}")
    print(f"  - Subfolders: {', '.join([f'k_{k}' for k in unique_ks])}")


if __name__ == "__main__":
    main()
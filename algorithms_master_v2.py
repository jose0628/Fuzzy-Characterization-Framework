#!/usr/bin/env python3
"""
Clustering scripts + visualizations for:
- K-Means
- Agglomerative Hierarchical Clustering (AHC)
- Fuzzy C-Means (FCM)
- Possibilistic Fuzzy C-Means (PFCM)
- Gustafson–Kessel (GK)

INPUT FILES (expected in working directory unless you pass paths):
1) users_behavior_vectors_pca.csv  -> machine-optimized clustering representation (used for clustering + scatter/contours)
2) users_engineered_features_raw.csv -> human-readable behavioral characterization (used for cluster profiling/interpretation)

USAGE EXAMPLES
--------------
python cluster_suite.py --algo kmeans --k 4
python cluster_suite.py --algo ahc --k 4
python cluster_suite.py --algo fcm --k 4
python cluster_suite.py --algo pfcm --k 4
python cluster_suite.py --algo gk --k 4

Optional:
python cluster_suite.py --algo fcm --k 4 --pca users_behavior_vectors_pca.csv --feat users_engineered_features_raw.csv

OUTPUTS
-------
Creates ./outputs/
- scatter_<algo>_k<K>.png
- contours_<algo>_k<K>.png      (for fuzzy methods, PC1–PC2 membership contours like your example)
- memberships_<algo>_k<K>.png   (for fuzzy methods, heatmap)
- dendrogram_ahc.png            (for AHC; truncated to avoid recursion errors)
- labels_<algo>_k<K>.csv
- profiles_<algo>_k<K>.csv      (cluster means on engineered features)

NOTES
-----
- AHC dendrogram is truncated by levels to prevent RecursionError for large n.
- Contour memberships are computed in PC1–PC2 space using FCM-style membership formula over grid.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering

# For dendrogram (AHC)
try:
    from scipy.cluster.hierarchy import dendrogram, linkage
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_inputs(pca_path: str, feat_path: str, user_col: str = "actor_id"):
    pca_df = pd.read_csv(pca_path)
    feat_df = pd.read_csv(feat_path)

    # Align rows robustly if both contain user_col; else assume row-order aligned.
    if user_col in pca_df.columns and user_col in feat_df.columns:
        merged = pca_df[[user_col] + [c for c in pca_df.columns if c != user_col]].merge(
            feat_df, on=user_col, how="inner"
        )
        pca_cols = [c for c in pca_df.columns if c != user_col]
        feat_cols = [c for c in feat_df.columns if c != user_col]
        aligned_pca = merged[[user_col] + pca_cols].copy()
        aligned_feat = merged[[user_col] + feat_cols].copy()
        return aligned_pca, aligned_feat, user_col
    else:
        # No key to align; assume alignment by row index
        return pca_df.copy(), feat_df.copy(), (user_col if user_col in pca_df.columns else None)


def get_X_from_pca(pca_df: pd.DataFrame, user_col: str | None):
    if user_col is not None and user_col in pca_df.columns:
        X = pca_df.drop(columns=[user_col]).values.astype(float)
        ids = pca_df[user_col].values
    else:
        X = pca_df.values.astype(float)
        ids = np.arange(len(pca_df))
    return X, ids


def scatter_pc1_pc2(X: np.ndarray, labels: np.ndarray, title: str, outpath: str):
    plt.figure()
    x = X[:, 0] if X.shape[1] > 0 else np.zeros(len(X))
    y = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))
    plt.scatter(x, y, c=labels, s=12)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_memberships(U: np.ndarray, title: str, outpath: str, max_rows: int = 300):
    """
    Membership heatmap for fuzzy methods.
    Plots first N rows to keep the figure readable.
    """
    n = min(U.shape[0], max_rows)
    plt.figure()
    plt.imshow(U[:n, :], aspect="auto")
    plt.xlabel("Cluster")
    plt.ylabel("User (first N)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_labels(ids, labels, outpath: str, user_col: str = "actor_id"):
    pd.DataFrame({user_col: ids, "cluster": labels}).to_csv(outpath, index=False)


def cluster_profiles(feat_df: pd.DataFrame, labels: np.ndarray, outpath: str, user_col: str = "actor_id"):
    df = feat_df.copy()
    if user_col in df.columns:
        df = df.drop(columns=[user_col])
    df["cluster"] = labels
    prof = df.groupby("cluster").mean(numeric_only=True)
    prof.to_csv(outpath)


# -----------------------------
# Membership contours (like your example)
# -----------------------------
def fcm_membership_from_centers_2d(grid_points: np.ndarray, centers_2d: np.ndarray, m: float = 2.0):
    """
    Compute FCM-style memberships for 2D grid points given 2D centers.

    grid_points: (G, 2)
    centers_2d: (k, 2)
    returns Ugrid: (G, k)
    """
    G = grid_points.shape[0]
    k = centers_2d.shape[0]

    dist = np.zeros((G, k))
    for j in range(k):
        diff = grid_points - centers_2d[j]
        dist[:, j] = np.sum(diff * diff, axis=1)
    dist = np.maximum(dist, 1e-12)

    power = 1.0 / (m - 1.0)
    Ugrid = np.zeros((G, k))
    for i in range(G):
        for j in range(k):
            denom = np.sum((dist[i, j] / dist[i, :]) ** power)
            Ugrid[i, j] = 1.0 / (denom + 1e-12)

    return Ugrid


def plot_fuzzy_membership_contours(
        X_2d: np.ndarray,
        centers_2d: np.ndarray,
        m: float,
        algo_name: str,
        outpath: str,
        grid_size: int = 250,
        levels=(0.1, 0.3, 0.5, 0.7, 0.9),
):
    """
    Create per-cluster contour plots (auto grid),
    similar to the example figure.
    """
    k = centers_2d.shape[0]
    ncols = int(np.ceil(np.sqrt(k)))
    nrows = int(np.ceil(k / ncols))

    x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
    y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()

    x_pad = 0.10 * (x_max - x_min + 1e-12)
    y_pad = 0.10 * (y_max - y_min + 1e-12)

    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()])

    Ugrid = fcm_membership_from_centers_2d(grid, centers_2d, m=m)
    Ugrid = Ugrid.reshape(grid_size, grid_size, k)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows))
    axes = np.array(axes).reshape(-1)

    for j in range(k):
        ax = axes[j]
        ax.scatter(X_2d[:, 0], X_2d[:, 1], s=10, alpha=0.5, label="Data")
        ax.scatter(centers_2d[j, 0], centers_2d[j, 1], marker="*", s=180, label=f"C{j+1}")

        CS = ax.contour(XX, YY, Ugrid[:, :, j], levels=levels)
        ax.clabel(CS, inline=True, fontsize=8)

        ax.set_title(f"Cluster {j+1} Membership")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="upper right", fontsize=8)

    for t in range(k, len(axes)):
        axes[t].axis("off")

    fig.suptitle(f"{algo_name} Membership Contours ({k} Clusters)", y=0.98, fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# FCM (lightweight)
# -----------------------------
def fcm(X: np.ndarray, c: int, m: float = 2.0, max_iter: int = 300, tol: float = 1e-5, seed: int = 42):
    rng = np.random.default_rng(seed)
    n, p = X.shape
    U = rng.random((n, c))
    U = U / U.sum(axis=1, keepdims=True)

    for _ in range(max_iter):
        U_m = U ** m
        centers = (U_m.T @ X) / (U_m.T.sum(axis=1, keepdims=True) + 1e-12)

        dist = np.zeros((n, c))
        for j in range(c):
            diff = X - centers[j]
            dist[:, j] = np.sum(diff * diff, axis=1)
        dist = np.maximum(dist, 1e-12)

        U_new = np.zeros_like(U)
        power = 1.0 / (m - 1.0)
        for i in range(n):
            for j in range(c):
                denom = np.sum((dist[i, j] / dist[i, :]) ** power)
                U_new[i, j] = 1.0 / (denom + 1e-12)

        if np.linalg.norm(U_new - U) < tol:
            U = U_new
            break
        U = U_new

    return centers, U


# -----------------------------
# PFCM (simplified)
# -----------------------------
def pfcm(
        X: np.ndarray,
        c: int,
        m: float = 2.0,
        eta: float = 2.0,
        a: float = 1.0,
        b: float = 1.0,
        max_iter: int = 300,
        tol: float = 1e-5,
        seed: int = 42,
):
    rng = np.random.default_rng(seed)
    n, p = X.shape

    U = rng.random((n, c))
    U = U / U.sum(axis=1, keepdims=True)

    centers, U = fcm(X, c=c, m=m, max_iter=50, tol=1e-4, seed=seed)
    T = U.copy()

    for _ in range(max_iter):
        U_m = U ** m
        T_eta = T ** eta

        num = (a * U_m + b * T_eta).T @ X
        den = (a * U_m + b * T_eta).T.sum(axis=1, keepdims=True) + 1e-12
        centers_new = num / den

        dist = np.zeros((n, c))
        for j in range(c):
            diff = X - centers_new[j]
            dist[:, j] = np.sum(diff * diff, axis=1)
        dist = np.maximum(dist, 1e-12)

        gamma = np.zeros(c)
        for j in range(c):
            gamma[j] = (np.sum((U_m[:, j]) * dist[:, j]) / (np.sum(U_m[:, j]) + 1e-12)) + 1e-12

        power_t = 1.0 / (eta - 1.0)
        T_new = np.zeros_like(T)
        for j in range(c):
            T_new[:, j] = 1.0 / (1.0 + (dist[:, j] / gamma[j]) ** power_t)

        power_u = 1.0 / (m - 1.0)
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(c):
                denom = np.sum((dist[i, j] / dist[i, :]) ** power_u)
                U_new[i, j] = 1.0 / (denom + 1e-12)

        delta = np.linalg.norm(centers_new - centers) + np.linalg.norm(U_new - U) + np.linalg.norm(T_new - T)
        centers, U, T = centers_new, U_new, T_new
        if delta < tol:
            break

    return centers, U, T


# -----------------------------
# GK
# -----------------------------
def gk(
        X: np.ndarray,
        c: int,
        m: float = 2.0,
        max_iter: int = 300,
        tol: float = 1e-5,
        seed: int = 42,
):
    n, p = X.shape
    centers, U = fcm(X, c=c, m=m, max_iter=50, tol=1e-4, seed=seed)

    for _ in range(max_iter):
        U_m = U ** m
        centers_new = (U_m.T @ X) / (U_m.T.sum(axis=1, keepdims=True) + 1e-12)

        A = []
        for j in range(c):
            diff = X - centers_new[j]
            w = U_m[:, j][:, None]
            Fj = (w * diff).T @ diff / (np.sum(U_m[:, j]) + 1e-12)
            Fj = Fj + 1e-6 * np.eye(p)

            detF = np.linalg.det(Fj)
            if detF <= 0:
                detF = 1e-12
            Fj_inv = np.linalg.inv(Fj)
            Aj = (detF ** (-1.0 / p)) * Fj_inv
            A.append(Aj)

        dist = np.zeros((n, c))
        for j in range(c):
            diff = X - centers_new[j]
            dist[:, j] = np.einsum("ni,ij,nj->n", diff, A[j], diff)
        dist = np.maximum(dist, 1e-12)

        U_new = np.zeros_like(U)
        power = 1.0 / (m - 1.0)
        for i in range(n):
            for j in range(c):
                denom = np.sum((dist[i, j] / dist[i, :]) ** power)
                U_new[i, j] = 1.0 / (denom + 1e-12)

        delta = np.linalg.norm(centers_new - centers) + np.linalg.norm(U_new - U)
        centers, U = centers_new, U_new
        if delta < tol:
            break

    return centers, U


# -----------------------------
# AHC dendrogram (truncated to avoid recursion errors)
# -----------------------------
def plot_dendrogram_truncated(X: np.ndarray, outpath: str, max_levels: int = 30):
    if not SCIPY_OK:
        return False
    Z = linkage(X, method="ward")

    plt.figure(figsize=(10, 5))
    dendrogram(
        Z,
        no_labels=True,
        truncate_mode="level",
        p=max_levels,
        show_contracted=True,
    )
    plt.title(f"AHC dendrogram (Ward) - truncated to {max_levels} levels")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return True


# -----------------------------
# Main runner
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=["kmeans", "ahc", "fcm", "pfcm", "gk"])
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--pca", default="users_behavior_vectors_pca.csv")
    parser.add_argument("--feat", default="users_engineered_features_raw.csv")
    parser.add_argument("--user-col", default="actor_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--dendro-levels", type=int, default=30, help="AHC dendrogram truncation levels")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    pca_df, feat_df, user_col = load_inputs(args.pca, args.feat, user_col=args.user_col)
    X, ids = get_X_from_pca(pca_df, user_col)

    algo = args.algo.lower()
    k = args.k

    centers = None
    U = None

    if algo == "kmeans":
        model = KMeans(n_clusters=k, n_init="auto", random_state=args.seed)
        labels = model.fit_predict(X)

    elif algo == "ahc":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)

        dendro_path = os.path.join(args.outdir, "dendrogram_ahc.png")
        ok = plot_dendrogram_truncated(X, dendro_path, max_levels=args.dendro_levels)
        if not ok:
            print("Warning: scipy not available; dendrogram not generated.", file=sys.stderr)

    elif algo == "fcm":
        centers, U = fcm(X, c=k, m=2.0, seed=args.seed)
        labels = np.argmax(U, axis=1)

    elif algo == "pfcm":
        centers, U, T = pfcm(X, c=k, m=2.0, eta=2.0, a=1.0, b=1.0, seed=args.seed)
        labels = np.argmax((U + T) / 2.0, axis=1)

        t_path = os.path.join(args.outdir, f"typicalities_pfcm_k{k}.npy")
        np.save(t_path, T)

    elif algo == "gk":
        centers, U = gk(X, c=k, m=2.0, seed=args.seed)
        labels = np.argmax(U, axis=1)

    else:
        raise ValueError("Unknown algo")

    # --- Save labels
    labels_path = os.path.join(args.outdir, f"labels_{algo}_k{k}.csv")
    save_labels(ids, labels, labels_path, user_col=args.user_col)

    # --- Profiles on engineered features (interpretation)
    profiles_path = os.path.join(args.outdir, f"profiles_{algo}_k{k}.csv")
    cluster_profiles(feat_df, labels, profiles_path, user_col=args.user_col)

    # --- Scatter plot in PCA space
    scatter_path = os.path.join(args.outdir, f"scatter_{algo}_k{k}.png")
    scatter_pc1_pc2(X, labels, f"{algo.upper()} (k={k}) in PCA space", scatter_path)

    # --- Membership heatmap + save membership matrix
    if U is not None:
        mem_path = os.path.join(args.outdir, f"memberships_{algo}_k{k}.png")
        plot_memberships(U, f"{algo.upper()} memberships (first users; k={k})", mem_path)

        u_path = os.path.join(args.outdir, f"memberships_{algo}_k{k}.npy")
        np.save(u_path, U)

    # --- Contour membership plots (like your example) for fuzzy methods only
    if U is not None and centers is not None and X.shape[1] >= 2:
        X_2d = X[:, :2]
        centers_2d = centers[:, :2]
        contour_path = os.path.join(args.outdir, f"contours_{algo}_k{k}.png")
        plot_fuzzy_membership_contours(
            X_2d=X_2d,
            centers_2d=centers_2d,
            m=2.0,
            algo_name=algo.upper(),
            outpath=contour_path,
            grid_size=250,
            levels=(0.1, 0.3, 0.5, 0.7, 0.9),
        )

    print("Done.")
    print("Inputs:")
    print(f"  PCA vectors: {args.pca}")
    print(f"  Engineered features: {args.feat}")
    print("Outputs:")
    print(f"  {args.outdir}/")


if __name__ == "__main__":
    main()

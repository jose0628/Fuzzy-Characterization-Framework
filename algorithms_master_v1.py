#!/usr/bin/env python3
"""
Clustering scripts + visualizations for:
- K-Means
- Agglomerative Hierarchical Clustering (AHC)
- Fuzzy C-Means (FCM)
- Possibilistic Fuzzy C-Means (PFCM)
- Gustafson–Kessel (GK)

INPUT FILES (expected in working directory unless you pass paths):
1) users_behavior_vectors_pca.csv  -> machine-optimized clustering representation (used for clustering + scatter plots)
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
- memberships_<algo>_k<K>.png  (for fuzzy methods)
- dendrogram_ahc.png           (for AHC)
- labels_<algo>_k<K>.csv
- profiles_<algo>_k<K>.csv     (cluster means on engineered features)
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


def safe_divide(a, b, eps=1e-12):
    return a / (b + eps)


def load_inputs(pca_path: str, feat_path: str, user_col: str = "actor_id"):
    pca_df = pd.read_csv(pca_path)
    feat_df = pd.read_csv(feat_path)

    # Align rows robustly if both contain user_col; else assume row-order aligned.
    if user_col in pca_df.columns and user_col in feat_df.columns:
        merged = pca_df[[user_col] + [c for c in pca_df.columns if c != user_col]].merge(
            feat_df, on=user_col, how="inner", suffixes=("_pca", "_feat")
        )
        # Re-split after alignment
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
    # Do not set explicit colors; let matplotlib defaults apply.
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
# FCM (fallback implementation)
# -----------------------------
def fcm(X: np.ndarray, c: int, m: float = 2.0, max_iter: int = 300, tol: float = 1e-5, seed: int = 42):
    """
    Basic Fuzzy C-Means.
    Returns:
      centers: (c, p)
      U: (n, c) membership matrix
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    # Random membership initialization, row-normalized
    U = rng.random((n, c))
    U = U / U.sum(axis=1, keepdims=True)

    for _ in range(max_iter):
        U_m = U ** m
        centers = (U_m.T @ X) / (U_m.T.sum(axis=1, keepdims=True) + 1e-12)

        # distances: (n, c)
        dist = np.zeros((n, c))
        for j in range(c):
            diff = X - centers[j]
            dist[:, j] = np.sum(diff * diff, axis=1)
        dist = np.maximum(dist, 1e-12)

        # update U
        U_new = np.zeros_like(U)
        power = 1.0 / (m - 1.0)
        for i in range(n):
            for j in range(c):
                denom = np.sum((dist[i, j] / dist[i, :]) ** power)
                U_new[i, j] = 1.0 / (denom + 1e-12)

        # convergence
        if np.linalg.norm(U_new - U) < tol:
            U = U_new
            break
        U = U_new

    return centers, U


# -----------------------------
# PFCM
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
    """
    Simplified Possibilistic Fuzzy C-Means.
    Combines fuzzy memberships U and typicalities T.

    Returns:
      centers: (c, p)
      U: (n, c)
      T: (n, c)
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape

    # init U
    U = rng.random((n, c))
    U = U / U.sum(axis=1, keepdims=True)

    # init centers using FCM step
    centers, U = fcm(X, c=c, m=m, max_iter=50, tol=1e-4, seed=seed)

    # init T (typicalities)
    T = U.copy()

    for _ in range(max_iter):
        U_m = U ** m
        T_eta = T ** eta

        # update centers
        num = (a * U_m + b * T_eta).T @ X
        den = (a * U_m + b * T_eta).T.sum(axis=1, keepdims=True) + 1e-12
        centers_new = num / den

        # distances
        dist = np.zeros((n, c))
        for j in range(c):
            diff = X - centers_new[j]
            dist[:, j] = np.sum(diff * diff, axis=1)
        dist = np.maximum(dist, 1e-12)

        # update scale parameters gamma_j (eta_j in some formulations)
        # gamma_j ~ average within-cluster distance under U
        gamma = np.zeros(c)
        for j in range(c):
            gamma[j] = (np.sum((U_m[:, j]) * dist[:, j]) / (np.sum(U_m[:, j]) + 1e-12)) + 1e-12

        # update T (possibilistic typicalities)
        # T_ij = 1 / (1 + (d_ij / gamma_j)^(1/(eta-1)))
        T_new = np.zeros_like(T)
        power_t = 1.0 / (eta - 1.0)
        for j in range(c):
            T_new[:, j] = 1.0 / (1.0 + (dist[:, j] / gamma[j]) ** power_t)

        # update U (fuzzy memberships)
        U_new = np.zeros_like(U)
        power_u = 1.0 / (m - 1.0)
        for i in range(n):
            for j in range(c):
                denom = np.sum((dist[i, j] / dist[i, :]) ** power_u)
                U_new[i, j] = 1.0 / (denom + 1e-12)

        # convergence
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
    """
    Gustafson–Kessel fuzzy clustering.
    Adapts cluster metric using covariance.

    Returns:
      centers: (c, p)
      U: (n, c)
    """
    n, p = X.shape
    centers, U = fcm(X, c=c, m=m, max_iter=50, tol=1e-4, seed=seed)

    for _ in range(max_iter):
        U_m = U ** m

        # update centers
        centers_new = (U_m.T @ X) / (U_m.T.sum(axis=1, keepdims=True) + 1e-12)

        # compute cluster covariance matrices F_j and adaptive metrics A_j
        A = []
        for j in range(c):
            diff = X - centers_new[j]
            w = U_m[:, j][:, None]  # (n,1)
            Fj = (w * diff).T @ diff / (np.sum(U_m[:, j]) + 1e-12)

            # regularize to avoid singularity
            Fj = Fj + 1e-6 * np.eye(p)

            detF = np.linalg.det(Fj)
            if detF <= 0:
                detF = 1e-12
            Fj_inv = np.linalg.inv(Fj)

            Aj = (detF ** (-1.0 / p)) * Fj_inv
            A.append(Aj)

        # distances under adaptive norm
        dist = np.zeros((n, c))
        for j in range(c):
            diff = X - centers_new[j]
            # quadratic form
            dist[:, j] = np.einsum("ni,ij,nj->n", diff, A[j], diff)
        dist = np.maximum(dist, 1e-12)

        # update U
        U_new = np.zeros_like(U)
        power = 1.0 / (m - 1.0)
        for i in range(n):
            for j in range(c):
                denom = np.sum((dist[i, j] / dist[i, :]) ** power)
                U_new[i, j] = 1.0 / (denom + 1e-12)

        # convergence
        delta = np.linalg.norm(centers_new - centers) + np.linalg.norm(U_new - U)
        centers, U = centers_new, U_new
        if delta < tol:
            break

    return centers, U


# -----------------------------
# AHC dendrogram
# -----------------------------
def plot_dendrogram(X: np.ndarray, outpath: str, sample_n: int = 2000, seed: int = 42):
    if not SCIPY_OK:
        return False

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n > sample_n:
        idx = rng.choice(n, size=sample_n, replace=False)
        X_plot = X[idx]
    else:
        X_plot = X

    Z = linkage(X_plot, method="ward")

    plt.figure(figsize=(10, 5))
    dendrogram(Z, no_labels=True)
    plt.title(f"AHC dendrogram (Ward) on sample (n={X_plot.shape[0]})")
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
    args = parser.parse_args()

    ensure_dir(args.outdir)

    pca_df, feat_df, user_col = load_inputs(args.pca, args.feat, user_col=args.user_col)
    X, ids = get_X_from_pca(pca_df, user_col)

    algo = args.algo.lower()
    k = args.k

    if algo == "kmeans":
        model = KMeans(n_clusters=k, n_init="auto", random_state=args.seed)
        labels = model.fit_predict(X)
        U = None

    elif algo == "ahc":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X)
        U = None

        # Dendrogram (optional)
        dendro_path = os.path.join(args.outdir, "dendrogram_ahc.png")
        ok = plot_dendrogram(X, dendro_path)
        if not ok:
            print("Warning: scipy not available; dendrogram not generated.", file=sys.stderr)

    elif algo == "fcm":
        _, U = fcm(X, c=k, m=2.0, seed=args.seed)
        labels = np.argmax(U, axis=1)

    elif algo == "pfcm":
        _, U, T = pfcm(X, c=k, m=2.0, eta=2.0, a=1.0, b=1.0, seed=args.seed)
        # Assign by combined strength (common practice); alternatively argmax(U)
        labels = np.argmax((U + T) / 2.0, axis=1)

        # Save typicalities too (optional)
        t_path = os.path.join(args.outdir, f"typicalities_pfcm_k{k}.npy")
        np.save(t_path, T)

    elif algo == "gk":
        _, U = gk(X, c=k, m=2.0, seed=args.seed)
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

    # --- Membership heatmap for fuzzy methods
    if U is not None:
        mem_path = os.path.join(args.outdir, f"memberships_{algo}_k{k}.png")
        plot_memberships(U, f"{algo.upper()} memberships (first users; k={k})", mem_path)

        # Save membership matrix (optional)
        u_path = os.path.join(args.outdir, f"memberships_{algo}_k{k}.npy")
        np.save(u_path, U)

    print("Done.")
    print("Inputs:")
    print(f"  PCA vectors: {args.pca}")
    print(f"  Engineered features: {args.feat}")
    print("Outputs:")
    print(f"  {args.outdir}/")


if __name__ == "__main__":
    main()

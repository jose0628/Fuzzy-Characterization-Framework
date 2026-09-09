#!/usr/bin/env python3
"""
Determine ideal number of clusters (k) using:
- K-Means internal indices: Inertia (Elbow), Silhouette, Calinski–Harabasz, Davies–Bouldin
- Fuzzy C-Means (FCM) indices: Partition Coefficient (PC), Partition Entropy (PE), Xie–Beni (XB)

INPUT
-----
users_behavior_vectors_pca.csv  (machine-optimized clustering representation)

OUTPUT
------
optimal_k_analysis_kmeans_fcm.png
optimal_k_metrics_table.csv

USAGE
-----
python determine_optimal_k_kmeans_fcm.py --pca users_behavior_vectors_pca.csv --kmin 2 --kmax 10
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


# -----------------------------
# FCM (lightweight, consistent with your clustering suite)
# -----------------------------
def fcm(X: np.ndarray, c: int, m: float = 2.0, max_iter: int = 200, tol: float = 1e-5, seed: int = 42):
    rng = np.random.default_rng(seed)
    n, p = X.shape
    U = rng.random((n, c))
    U = U / U.sum(axis=1, keepdims=True)

    for _ in range(max_iter):
        U_m = U ** m
        centers = (U_m.T @ X) / (U_m.T.sum(axis=1, keepdims=True) + 1e-12)

        # squared distances (n,c)
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

        if np.linalg.norm(U_new - U) < tol:
            U = U_new
            break
        U = U_new

    return centers, U


def fcm_partition_coefficient(U: np.ndarray) -> float:
    """PC = (1/n) * sum_i sum_j u_ij^2  (higher is better separation)"""
    return float(np.mean(np.sum(U**2, axis=1)))


def fcm_partition_entropy(U: np.ndarray) -> float:
    """PE = -(1/n) * sum_i sum_j u_ij log(u_ij)  (lower is better separation)"""
    eps = 1e-12
    return float(-np.mean(np.sum(U * np.log(U + eps), axis=1)))


def xie_beni_index(X: np.ndarray, centers: np.ndarray, U: np.ndarray, m: float = 2.0) -> float:
    """
    Xie–Beni index:
    XB = [sum_i sum_j (u_ij^m * ||x_i - v_j||^2)] / [n * min_{j!=k} ||v_j - v_k||^2]
    lower is better.
    """
    n = X.shape[0]
    c = centers.shape[0]

    # numerator
    num = 0.0
    for j in range(c):
        diff = X - centers[j]
        d2 = np.sum(diff * diff, axis=1)
        num += np.sum((U[:, j] ** m) * d2)

    # denominator: minimum squared center distance
    min_d2 = np.inf
    for j in range(c):
        for k in range(j + 1, c):
            d2 = float(np.sum((centers[j] - centers[k]) ** 2))
            if d2 < min_d2:
                min_d2 = d2
    min_d2 = max(min_d2, 1e-12)

    return float(num / (n * min_d2))


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca", default="users_behavior_vectors_pca.csv")
    parser.add_argument("--user-col", default="actor_id")
    parser.add_argument("--kmin", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--m", type=float, default=2.0, help="FCM fuzzifier")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--outfig", default="optimal_k_analysis_kmeans_fcm.png")
    parser.add_argument("--outcsv", default="optimal_k_metrics_table.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.pca)
    if args.user_col in df.columns:
        X = df.drop(columns=[args.user_col]).values.astype(float)
    else:
        X = df.values.astype(float)

    k_range = list(range(args.kmin, args.kmax + 1))

    rows = []
    for k in k_range:
        # --- KMeans metrics
        km = KMeans(n_clusters=k, n_init="auto", random_state=args.seed)
        labels = km.fit_predict(X)

        inertia = float(km.inertia_)
        sil = float(silhouette_score(X, labels))
        ch = float(calinski_harabasz_score(X, labels))
        db = float(davies_bouldin_score(X, labels))

        # --- FCM metrics
        centers, U = fcm(X, c=k, m=args.m, max_iter=args.max_iter, tol=args.tol, seed=args.seed)
        pc = fcm_partition_coefficient(U)
        pe = fcm_partition_entropy(U)
        xb = xie_beni_index(X, centers, U, m=args.m)

        rows.append(
            {
                "k": k,
                "kmeans_inertia": inertia,
                "kmeans_silhouette": sil,
                "kmeans_calinski_harabasz": ch,
                "kmeans_davies_bouldin": db,
                "fcm_pc": pc,
                "fcm_pe": pe,
                "fcm_xb": xb,
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.outcsv, index=False)

    # -----------------------------
    # Visualization (single figure, multiple panels)
    # -----------------------------
    plt.figure(figsize=(14, 10))

    # (1) Elbow
    plt.subplot(2, 3, 1)
    plt.plot(metrics["k"], metrics["kmeans_inertia"], marker="o")
    plt.title("Elbow (K-Means Inertia)")
    plt.xlabel("k")
    plt.ylabel("Inertia")

    # (2) Silhouette (higher better)
    plt.subplot(2, 3, 2)
    plt.plot(metrics["k"], metrics["kmeans_silhouette"], marker="o")
    plt.title("Silhouette (higher is better)")
    plt.xlabel("k")
    plt.ylabel("Silhouette")

    # (3) CH (higher better)
    plt.subplot(2, 3, 3)
    plt.plot(metrics["k"], metrics["kmeans_calinski_harabasz"], marker="o")
    plt.title("Calinski–Harabasz (higher is better)")
    plt.xlabel("k")
    plt.ylabel("CH")

    # (4) DB (lower better)
    plt.subplot(2, 3, 4)
    plt.plot(metrics["k"], metrics["kmeans_davies_bouldin"], marker="o")
    plt.title("Davies–Bouldin (lower is better)")
    plt.xlabel("k")
    plt.ylabel("DB")

    # (5) FCM PC (higher better)
    plt.subplot(2, 3, 5)
    plt.plot(metrics["k"], metrics["fcm_pc"], marker="o")
    plt.title("FCM Partition Coefficient (higher is better)")
    plt.xlabel("k")
    plt.ylabel("PC")

    # (6) FCM PE + XB (both lower better) — plot both with legend
    plt.subplot(2, 3, 6)
    plt.plot(metrics["k"], metrics["fcm_pe"], marker="o", label="PE (lower better)")
    plt.plot(metrics["k"], metrics["fcm_xb"], marker="o", label="XB (lower better)")
    plt.title("FCM Separation Indices")
    plt.xlabel("k")
    plt.legend()

    plt.tight_layout()
    plt.savefig(args.outfig, dpi=200)
    plt.show()

    # -----------------------------
    # Suggested k values (simple heuristics)
    # -----------------------------
    best_sil_k = int(metrics.loc[metrics["kmeans_silhouette"].idxmax(), "k"])
    best_ch_k = int(metrics.loc[metrics["kmeans_calinski_harabasz"].idxmax(), "k"])
    best_db_k = int(metrics.loc[metrics["kmeans_davies_bouldin"].idxmin(), "k"])
    best_pc_k = int(metrics.loc[metrics["fcm_pc"].idxmax(), "k"])
    best_pe_k = int(metrics.loc[metrics["fcm_pe"].idxmin(), "k"])
    best_xb_k = int(metrics.loc[metrics["fcm_xb"].idxmin(), "k"])

    print("Suggested k by metric:")
    print("  K-Means Silhouette (max):", best_sil_k)
    print("  K-Means Calinski–Harabasz (max):", best_ch_k)
    print("  K-Means Davies–Bouldin (min):", best_db_k)
    print("  FCM Partition Coefficient (max):", best_pc_k)
    print("  FCM Partition Entropy (min):", best_pe_k)
    print("  FCM Xie–Beni (min):", best_xb_k)

    print("\nSaved:")
    print(" ", args.outfig)
    print(" ", args.outcsv)


if __name__ == "__main__":
    main()

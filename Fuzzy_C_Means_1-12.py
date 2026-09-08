#!/usr/bin/env python3
"""
FULL SCRIPT (FCM k=2..12) + visualizations + extended metrics per k.

Changes included:
- Color-blind-safe contour colors (Okabe-Ito inspired palette)
- Marker-based membership visualization instead of color-coded users:
    u >= 0.8        -> '+'   (very strong, unfilled, low visual footprint)
    0.6 <= u < 0.8 -> 'x'
    0.4 <= u < 0.6 -> 'o'
    0.2 <= u < 0.4 -> '^'
    u < 0.2        -> '.'
- Tracked user is a red dot
- Cluster center remains a yellow star

Also computes and saves:
- Silhouette
- FSI
- Cluster Stability (ARI across restarts)
- Interpretability Score
- FPC, PE, XB, JM_inertia, Calinski–Harabasz, Davies–Bouldin, KMeans inertia

Additional summary outputs added:
- Global summary CSV for k=2..12:
    summary_cluster_metrics_2_12.csv
- Global summary JSON for k=2..12:
    summary_cluster_metrics_2_12.json
- Per-k summary CSV in each k-folder:
    summary_k_<k>.csv
- Per-k summary JSON in each k-folder:
    summary_k_<k>.json

Outputs:
- Membership contour plots
- Combined region heatmap
- Continuous engagement heatmap
- 10x10 zone-grid engagement plots per cluster + tracked user red point
- CSV/JSON metrics and memberships

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
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.metrics.pairwise import pairwise_distances


# -----------------------------
# Config
# -----------------------------
CSV_PATH = "datasets/leonardo_activity_unpacked_step3_log1p_winsor_robust_1_99.csv"
OUTPUT_DIR = "FMC_output_1_12_details_leonardo"

ID_COL_CANDIDATES = {"actor_id", "user_id", "id"}

K_RUN_MIN, K_RUN_MAX = 2, 12
K_METRICS_MIN, K_METRICS_MAX = 2, 12

M_FUZZINESS = 2.0
ERROR = 1e-5
MAXITER = 2000
RANDOM_SEED = 42

GRID_N = 300
CONTOUR_LEVELS = [0.2, 0.4, 0.6, 0.8]
TOP_PCA_LOADINGS = 20

PLOT_N_POINTS = 500
PLOT_RANDOM_SAMPLE = True

# Marker map instead of color map
MEM_MARKER_MAP = {
    ">=0.8": "+",   # very strong, unfilled, small footprint
    ">=0.6": "x",
    ">=0.4": "o",
    ">=0.2": "^",
    "<0.2":  ".",
}

# Common point styling
MEM_POINT_COLOR = "black"
TRACKED_USER_COLOR = "red"
CLUSTER_CENTER_FACE = "#FFA000"
CLUSTER_CENTER_EDGE = "k"

# Color-blind-safe contour colors (Okabe-Ito inspired)
CONTOUR_COLOR_CYCLE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#999999",  # gray
]

GRID_BINS = 10
ZONE_THRESH_DISCONNECT = 0.4
ZONE_THRESH_ALIGN = 0.6
ZONE_COLORS = {"weak": "#D9D9D9", "disconnect": "#F2C14E", "align": "#76C893"}

FSI_SAMPLE_MAX = 2000
STABILITY_RUNS = 8
STABILITY_SEED_OFFSET = 1000


# -----------------------------
# Helpers: fuzzy validity indices
# -----------------------------
def partition_entropy(U: np.ndarray) -> float:
    eps = 1e-12
    U_safe = np.clip(U, eps, 1.0)
    return float(-(U_safe * np.log(U_safe)).sum() / U.shape[1])


def xie_beni_index(X: np.ndarray, cntr: np.ndarray, U: np.ndarray, m: float) -> float:
    diff = cntr[:, None, :] - X[None, :, :]
    dist2 = np.sum(diff**2, axis=2)
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


def fuzzy_silhouette_index(
        X: np.ndarray,
        U: np.ndarray,
        m: float,
        max_points: int = 2000,
        random_state: int = 42
) -> float:
    N = X.shape[0]
    rng = np.random.default_rng(random_state)
    if N > max_points:
        idx = rng.choice(N, size=max_points, replace=False)
        Xs = X[idx]
        Us = U[:, idx]
    else:
        Xs = X
        Us = U

    c, Ns = Us.shape
    D = pairwise_distances(Xs)
    W = np.power(np.clip(Us, 1e-12, 1.0), m)

    labels = np.argmax(Us, axis=0)

    denom = np.sum(W, axis=1)
    denom = np.where(denom <= 1e-12, 1e-12, denom)

    dist_to_cluster = (W @ D.T) / denom[:, None]

    a = dist_to_cluster[labels, np.arange(Ns)]

    b = np.full(Ns, np.inf)
    for r in range(c):
        mask = labels != r
        b[mask] = np.minimum(b[mask], dist_to_cluster[r, mask])

    s = (b - a) / (np.maximum(a, b) + 1e-12)
    return float(np.nanmean(s))


def cluster_stability_ari(
        data2: np.ndarray,
        c: int,
        m: float,
        error: float,
        maxiter: int,
        base_seed: int,
        n_runs: int = 8,
) -> float:
    labels_list = []
    for r in range(n_runs):
        seed = base_seed + r
        _, U, _, _, _, _, _ = fuzz.cluster.cmeans(
            data2, c=c, m=m, error=error, maxiter=maxiter, init=None, seed=seed
        )
        labels_list.append(np.argmax(U, axis=0))

    if len(labels_list) < 2:
        return float("nan")

    aris = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            aris.append(adjusted_rand_score(labels_list[i], labels_list[j]))

    return float(np.mean(aris)) if aris else float("nan")


def interpretability_score(U: np.ndarray, xb: float) -> float:
    crispness = float(np.mean(np.max(U, axis=0)))
    sep_proxy = 1.0 / (1.0 + float(xb)) if np.isfinite(xb) and xb >= 0 else 0.0
    return float(0.5 * crispness + 0.5 * sep_proxy)


def build_compact_summary_row(metrics_row: dict) -> dict:
    return {
        "k": int(metrics_row["k"]),
        "Silhouette": float(metrics_row["Silhouette"]),
        "FSI": float(metrics_row["FSI"]),
        "ClusterStability_ARI": float(metrics_row["ClusterStability_ARI"]),
        "InterpretabilityScore": float(metrics_row["InterpretabilityScore"]),
    }


# -----------------------------
# Plot helpers
# -----------------------------
def membership_to_marker_groups(u: np.ndarray):
    return {
        ">=0.8": u >= 0.8,
        ">=0.6": (u >= 0.6) & (u < 0.8),
        ">=0.4": (u >= 0.4) & (u < 0.6),
        ">=0.2": (u >= 0.2) & (u < 0.4),
        "<0.2":  u < 0.2,
    }


def add_membership_legend(ax):
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='+', color='k', label='u ≥ 0.8 (very strong)',
               linestyle='None', markersize=9),
        Line2D([0], [0], marker='x', color='k', label='0.6 ≤ u < 0.8 (strong)',
               linestyle='None', markersize=9),
        Line2D([0], [0], marker='o', color='k', label='0.4 ≤ u < 0.6 (moderate)',
               linestyle='None', markersize=7, markerfacecolor='none'),
        Line2D([0], [0], marker='^', color='k', label='0.2 ≤ u < 0.4 (weak)',
               linestyle='None', markersize=7, markerfacecolor='none'),
        Line2D([0], [0], marker='.', color='k', label='u < 0.2 (very weak)',
               linestyle='None', markersize=8),
        Line2D([0], [0], marker='o', color='red', label='Tracked user',
               linestyle='None', markersize=8),
        Line2D([0], [0], marker='*', color='k', label='Cluster center',
               markerfacecolor=CLUSTER_CENTER_FACE, linestyle='None', markersize=11),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)


def scatter_membership_markers(ax, X2_plot: np.ndarray, u_cluster: np.ndarray):
    groups = membership_to_marker_groups(u_cluster)

    mask = groups["<0.2"]
    if np.any(mask):
        ax.scatter(
            X2_plot[mask, 0], X2_plot[mask, 1],
            marker=MEM_MARKER_MAP["<0.2"], c=MEM_POINT_COLOR, s=18, alpha=0.35
        )

    mask = groups[">=0.2"]
    if np.any(mask):
        ax.scatter(
            X2_plot[mask, 0], X2_plot[mask, 1],
            marker=MEM_MARKER_MAP[">=0.2"], c=MEM_POINT_COLOR, s=24, alpha=0.55,
            facecolors='none'
        )

    mask = groups[">=0.4"]
    if np.any(mask):
        ax.scatter(
            X2_plot[mask, 0], X2_plot[mask, 1],
            marker=MEM_MARKER_MAP[">=0.4"], c=MEM_POINT_COLOR, s=28, alpha=0.75,
            facecolors='none'
        )

    mask = groups[">=0.6"]
    if np.any(mask):
        ax.scatter(
            X2_plot[mask, 0], X2_plot[mask, 1],
            marker=MEM_MARKER_MAP[">=0.6"], c=MEM_POINT_COLOR, s=34, alpha=0.9
        )

    mask = groups[">=0.8"]
    if np.any(mask):
        ax.scatter(
            X2_plot[mask, 0], X2_plot[mask, 1],
            marker=MEM_MARKER_MAP[">=0.8"], c=MEM_POINT_COLOR, s=36, alpha=1.0,
            linewidths=1.0
        )


def save_membership_contours(
        X2_all: np.ndarray,
        X2_plot: np.ndarray,
        cntr: np.ndarray,
        U_plot: np.ndarray,
        tracked_xy: tuple[float, float],
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

    x_min, x_max = X2_all[:, 0].min() - 0.5, X2_all[:, 0].max() + 0.5
    y_min, y_max = X2_all[:, 1].min() - 0.5, X2_all[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_n),
                         np.linspace(y_min, y_max, grid_n))
    grid = np.vstack([xx.ravel(), yy.ravel()])

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

        contour_color = CONTOUR_COLOR_CYCLE[i % len(CONTOUR_COLOR_CYCLE)]
        cs = ax.contour(
            xx, yy, Zi,
            levels=levels,
            linewidths=1.2,
            colors=[contour_color]
        )
        ax.clabel(cs, inline=True, fontsize=8)

        scatter_membership_markers(ax, X2_plot, U_plot[i, :])

        ax.scatter(
            cntr[i, 0], cntr[i, 1],
            marker="*", s=180, edgecolor=CLUSTER_CENTER_EDGE,
            facecolor=CLUSTER_CENTER_FACE, zorder=5
        )

        ax.scatter(
            tracked_xy[0], tracked_xy[1],
            marker="o", s=70, color=TRACKED_USER_COLOR,
            edgecolor="black", linewidths=1.2, zorder=6
        )

        ax.set_title(f"Cluster {i+1} Membership")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        add_membership_legend(ax)

    for j in range(k, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Fuzzy C-Means Membership Contours (k={k})", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_combined_region_heatmap(
        X2_all: np.ndarray,
        cntr: np.ndarray,
        k: int,
        out_path: str,
        m: float,
        error: float,
        maxiter: int,
        grid_n: int = 250,
):
    x_min, x_max = X2_all[:, 0].min() - 0.5, X2_all[:, 0].max() + 0.5
    y_min, y_max = X2_all[:, 1].min() - 0.5, X2_all[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_n),
                         np.linspace(y_min, y_max, grid_n))
    grid = np.vstack([xx.ravel(), yy.ravel()])

    U_grid, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        grid, cntr, m=m, error=error, maxiter=maxiter
    )

    U_grid_2d = U_grid.reshape(k, grid_n, grid_n)
    dom = np.argmax(U_grid_2d, axis=0)
    strength = np.max(U_grid_2d, axis=0)

    base_colors = np.array([
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
        [0, 0, 0],
        [200, 200, 200],
    ], dtype=float) / 255.0
    colors = base_colors[:k]

    rgb = colors[dom]
    s_min = 1.0 / k
    alpha = 0.25 + 0.75 * (strength - s_min) / (1.0 - s_min + 1e-12)
    rgb_mod = rgb * alpha[..., None] + (1 - alpha[..., None]) * 1.0

    plt.figure(figsize=(9, 7))
    plt.imshow(rgb_mod, origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="auto")
    plt.scatter(X2_all[:, 0], X2_all[:, 1], s=3, alpha=0.12, c="k")
    plt.scatter(cntr[:, 0], cntr[:, 1], marker="*", s=220, edgecolor="k", c=CLUSTER_CENTER_FACE)
    plt.title(f"Cluster Regions Heatmap (k={k})\nHue=dominant cluster, intensity=max membership")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_engagement_heatmap_from_membership(
        X2_all: np.ndarray,
        cntr: np.ndarray,
        k: int,
        out_path: str,
        m: float,
        error: float,
        maxiter: int,
        grid_n: int = 250,
):
    x_min, x_max = X2_all[:, 0].min() - 0.5, X2_all[:, 0].max() + 0.5
    y_min, y_max = X2_all[:, 1].min() - 0.5, X2_all[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_n),
                         np.linspace(y_min, y_max, grid_n))
    grid = np.vstack([xx.ravel(), yy.ravel()])

    U_grid, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        grid, cntr, m=m, error=error, maxiter=maxiter
    )

    E_k = cntr[:, 0].reshape(k, 1)
    E = (U_grid * E_k).sum(axis=0)
    E_img = E.reshape(grid_n, grid_n)

    plt.figure(figsize=(9, 7))
    plt.imshow(E_img, origin="lower", extent=[x_min, x_max, y_min, y_max], aspect="auto")
    plt.colorbar(label="Engagement proxy (membership-weighted centroid PC1)")
    plt.scatter(X2_all[:, 0], X2_all[:, 1], s=3, alpha=0.12, c="k")
    plt.scatter(cntr[:, 0], cntr[:, 1], marker="*", s=220, edgecolor="k", c=CLUSTER_CENTER_FACE)
    plt.title(f"Engagement Heatmap (k={k})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_cluster_zone_grid_heatmaps(
        X2_all: np.ndarray,
        U: np.ndarray,
        tracked_xy: tuple[float, float],
        k: int,
        out_dir: str,
        bins: int = 10,
        thr_disconnect: float = 0.4,
        thr_align: float = 0.6,
        annotate: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    x = X2_all[:, 0]
    y = X2_all[:, 1]

    x_edges = np.linspace(x.min(), x.max(), bins + 1)
    y_edges = np.linspace(y.min(), y.max(), bins + 1)

    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, bins - 1)
    y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, bins - 1)

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([ZONE_COLORS["weak"], ZONE_COLORS["disconnect"], ZONE_COLORS["align"]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    for ci in range(k):
        count = np.zeros((bins, bins), dtype=int)
        sum_u = np.zeros((bins, bins), dtype=float)
        sum_pc1_u = np.zeros((bins, bins), dtype=float)

        u_ci = U[ci, :]

        for idx in range(X2_all.shape[0]):
            xb = x_bin[idx]
            yb = y_bin[idx]
            count[yb, xb] += 1
            sum_u[yb, xb] += u_ci[idx]
            sum_pc1_u[yb, xb] += x[idx] * u_ci[idx]

        mean_u = np.full((bins, bins), np.nan, dtype=float)
        wmean_pc1 = np.full((bins, bins), np.nan, dtype=float)

        mask = count > 0
        mean_u[mask] = sum_u[mask] / count[mask]

        wmask = sum_u > 1e-12
        wmean_pc1[wmask] = sum_pc1_u[wmask] / sum_u[wmask]

        zone = np.full((bins, bins), np.nan, dtype=float)
        zone[(mask) & (mean_u < thr_disconnect)] = 0
        zone[(mask) & (mean_u >= thr_disconnect) & (mean_u < thr_align)] = 1
        zone[(mask) & (mean_u >= thr_align)] = 2

        plt.figure(figsize=(9, 7))
        plt.pcolormesh(x_edges, y_edges, zone, cmap=cmap, norm=norm, shading="auto")

        for xe in x_edges:
            plt.axvline(xe, linewidth=0.4, color="white", alpha=0.6)
        for ye in y_edges:
            plt.axhline(ye, linewidth=0.4, color="white", alpha=0.6)

        plt.scatter(
            tracked_xy[0], tracked_xy[1],
            color=TRACKED_USER_COLOR, s=120,
            edgecolor="black", linewidth=1.5, zorder=6
        )

        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title(f"Cluster {ci+1} – Engagement Zone Grid (10×10)\nColor=membership zone; numbers=weighted PC1 (optional)")

        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(color=ZONE_COLORS["weak"], label=f"Weak/very weak (mean u < {thr_disconnect})"),
            mpatches.Patch(color=ZONE_COLORS["disconnect"], label=f"Zone of disconnect ({thr_disconnect} ≤ mean u < {thr_align})"),
            mpatches.Patch(color=ZONE_COLORS["align"], label=f"Zone of alignment (mean u ≥ {thr_align})"),
            mpatches.Patch(color=TRACKED_USER_COLOR, label="Tracked user"),
        ]
        plt.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.95)

        if annotate:
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            for r in range(bins):
                for c in range(bins):
                    if np.isfinite(zone[r, c]) and np.isfinite(wmean_pc1[r, c]):
                        plt.text(
                            x_centers[c], y_centers[r], f"{wmean_pc1[r, c]:.1f}",
                            ha="center", va="center", fontsize=7, color="black"
                        )

        out_path = os.path.join(out_dir, f"cluster_{ci+1}_zone_grid_10x10.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

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

    num_df = df.select_dtypes(include=[np.number]).copy()
    drop_ids = [c for c in num_df.columns if c.lower() in ID_COL_CANDIDATES]
    if drop_ids:
        num_df.drop(columns=drop_ids, inplace=True)
    if num_df.shape[1] < 2:
        raise ValueError(f"Need at least 2 numeric features. Found {num_df.shape[1]}.")

    feature_names = num_df.columns.tolist()

    scaler = StandardScaler()
    X = scaler.fit_transform(num_df.values)

    pca2 = PCA(n_components=2, random_state=RANDOM_SEED)
    X2 = pca2.fit_transform(X)
    data2 = X2.T

    n = X2.shape[0]
    if PLOT_N_POINTS is None or PLOT_N_POINTS >= n:
        plot_idx = np.arange(n)
    else:
        rng = np.random.default_rng(RANDOM_SEED)
        plot_idx = rng.choice(n, size=PLOT_N_POINTS, replace=False) if PLOT_RANDOM_SAMPLE else np.arange(PLOT_N_POINTS)
    X2_plot = X2[plot_idx]

    rng = np.random.default_rng(RANDOM_SEED)
    tracked_pos_in_plot = int(rng.integers(0, len(plot_idx)))
    tracked_global_idx = int(plot_idx[tracked_pos_in_plot])
    tracked_xy = (float(X2[tracked_global_idx, 0]), float(X2[tracked_global_idx, 1]))
    tracked_user_id = user_ids.iloc[tracked_global_idx] if hasattr(user_ids, "iloc") else user_ids[tracked_global_idx]

    with open(os.path.join(OUTPUT_DIR, "tracked_user.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "tracked_user_id_label": user_id_label,
                "tracked_user_id": str(tracked_user_id),
                "tracked_global_row_index": tracked_global_idx,
                "tracked_pca2d": {"PC1": tracked_xy[0], "PC2": tracked_xy[1]},
                "n_points_plotted": int(len(plot_idx)),
            },
            f,
            indent=2,
        )

    mapping = {
        "input_csv": CSV_PATH,
        "user_identifier_column": user_id_label,
        "n_rows": int(X.shape[0]),
        "n_features_used": int(len(feature_names)),
        "features_used": feature_names,
        "scaler": {"type": "StandardScaler", "mean_": scaler.mean_.tolist(), "scale_": scaler.scale_.tolist()},
        "pca_2d": {"explained_variance_ratio_": pca2.explained_variance_ratio_.tolist(),
                   "components_": pca2.components_.tolist(),
                   "feature_loadings_top": {}},
    }
    for pc_i in range(2):
        loadings = pca2.components_[pc_i]
        idx = np.argsort(np.abs(loadings))[::-1][:TOP_PCA_LOADINGS]
        mapping["pca_2d"]["feature_loadings_top"][f"PC{pc_i+1}"] = [
            {"feature": feature_names[j], "loading": float(loadings[j])} for j in idx
        ]
    with open(os.path.join(OUTPUT_DIR, "feature_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    pd.DataFrame({user_id_label: user_ids.values, "PC1": X2[:, 0], "PC2": X2[:, 1]}).to_csv(
        os.path.join(OUTPUT_DIR, "users_pca2d.csv"), index=False
    )

    metric_rows = []
    compact_summary_rows = []
    metrics_json_dir = os.path.join(OUTPUT_DIR, "metrics_json")
    os.makedirs(metrics_json_dir, exist_ok=True)

    for c in range(K_METRICS_MIN, K_METRICS_MAX + 1):
        cntr, U, _, _, jm, _, fpc = fuzz.cluster.cmeans(
            data2, c=c, m=M_FUZZINESS, error=ERROR, maxiter=MAXITER, init=None, seed=RANDOM_SEED
        )
        labels = np.argmax(U, axis=0)

        pe = partition_entropy(U)
        xb = xie_beni_index(X2, cntr, U, M_FUZZINESS)
        jm_final = float(jm[-1])

        try:
            sil = float(silhouette_score(X2, labels))
        except Exception:
            sil = float("nan")

        try:
            ch = float(calinski_harabasz_score(X2, labels))
        except Exception:
            ch = float("nan")

        try:
            db = float(davies_bouldin_score(X2, labels))
        except Exception:
            db = float("nan")

        km_inertia = float(KMeans(n_clusters=c, n_init="auto", random_state=RANDOM_SEED).fit(X2).inertia_)

        fsi = fuzzy_silhouette_index(X2, U, m=M_FUZZINESS, max_points=FSI_SAMPLE_MAX, random_state=RANDOM_SEED)
        stability = cluster_stability_ari(
            data2=data2,
            c=c,
            m=M_FUZZINESS,
            error=ERROR,
            maxiter=MAXITER,
            base_seed=RANDOM_SEED + STABILITY_SEED_OFFSET,
            n_runs=STABILITY_RUNS,
        )
        interp = interpretability_score(U, xb)

        row = {
            "k": int(c),
            "FPC_PC": float(fpc),
            "PE": float(pe),
            "XB": float(xb),
            "JM_inertia": float(jm_final),
            "Silhouette": float(sil),
            "FSI": float(fsi),
            "ClusterStability_ARI": float(stability),
            "InterpretabilityScore": float(interp),
            "CalinskiHarabasz": float(ch),
            "DaviesBouldin": float(db),
            "KMeans_inertia": float(km_inertia),
            "FSI_sample_max": int(FSI_SAMPLE_MAX),
            "Stability_runs": int(STABILITY_RUNS),
        }

        metric_rows.append(row)
        compact_summary_rows.append(build_compact_summary_row(row))

        with open(os.path.join(metrics_json_dir, f"metrics_k_{c}.json"), "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "k_selection_metrics_extended.csv"), index=False)

    summary_df = pd.DataFrame(compact_summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_cluster_metrics_2_12.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "summary_cluster_metrics_2_12.json"), "w", encoding="utf-8") as f:
        json.dump(compact_summary_rows, f, indent=2)

    ks_to_run = list(range(K_RUN_MIN, K_RUN_MAX + 1))
    with open(os.path.join(OUTPUT_DIR, "ks_to_run.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, ks_to_run)) + "\n")

    print(
        f"Running FCM for k={K_RUN_MIN}..{K_RUN_MAX}. "
        f"Plotting only n={len(X2_plot)} points. "
        f"Tracked user ({user_id_label})={tracked_user_id}. "
        f"Output: {OUTPUT_DIR}/"
    )

    metrics_by_k = {int(r["k"]): r for r in metric_rows}
    compact_summary_by_k = {int(r["k"]): r for r in compact_summary_rows}

    for k in ks_to_run:
        run_dir = os.path.join(OUTPUT_DIR, f"k_{k}")
        os.makedirs(run_dir, exist_ok=True)

        cntr, U, _, _, jm, _, fpc = fuzz.cluster.cmeans(
            data2, c=k, m=M_FUZZINESS, error=ERROR, maxiter=MAXITER, init=None, seed=RANDOM_SEED
        )
        hard_labels = np.argmax(U, axis=0)

        mem_df = pd.DataFrame({user_id_label: user_ids.values})
        for i in range(k):
            mem_df[f"cluster_{i+1}"] = U[i, :]
        mem_df["hard_cluster"] = hard_labels + 1
        mem_df.to_csv(os.path.join(run_dir, f"memberships_k_{k}.csv"), index=False)

        centers_df = pd.DataFrame(cntr, columns=["PC1_center", "PC2_center"])
        centers_df.insert(0, "cluster", np.arange(1, k + 1))
        centers_df.to_csv(os.path.join(run_dir, f"centers_k_{k}.csv"), index=False)

        with open(os.path.join(run_dir, f"metrics_k_{k}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_by_k[k], f, indent=2)

        summary = {
            "k": int(k),
            "m": float(M_FUZZINESS),
            "fpc": float(fpc),
            "jm_final": float(jm[-1]),
            "n_points_total": int(X2.shape[0]),
            "n_points_plotted": int(len(X2_plot)),
            "tracked_user": {
                "id_label": user_id_label,
                "id_value": str(tracked_user_id),
                "global_row_index": tracked_global_idx,
                "pca2d": {"PC1": tracked_xy[0], "PC2": tracked_xy[1]},
            },
            "compact_metrics_summary": compact_summary_by_k[k],
        }
        with open(os.path.join(run_dir, f"summary_k_{k}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        pd.DataFrame([compact_summary_by_k[k]]).to_csv(
            os.path.join(run_dir, f"summary_k_{k}.csv"), index=False
        )

        U_plot = U[:, plot_idx]
        save_membership_contours(
            X2_all=X2,
            X2_plot=X2_plot,
            cntr=cntr,
            U_plot=U_plot,
            tracked_xy=tracked_xy,
            k=k,
            out_path=os.path.join(run_dir, f"membership_contours_k_{k}.png"),
            m=M_FUZZINESS,
            error=ERROR,
            maxiter=MAXITER,
            grid_n=GRID_N,
            levels=CONTOUR_LEVELS,
        )

        plt.figure(figsize=(7, 6))
        plt.scatter(X2_plot[:, 0], X2_plot[:, 1], c=hard_labels[plot_idx], s=14, alpha=0.75)
        plt.scatter(
            cntr[:, 0], cntr[:, 1],
            marker="*", s=220, edgecolor=CLUSTER_CENTER_EDGE, facecolor=CLUSTER_CENTER_FACE
        )
        plt.scatter(
            tracked_xy[0], tracked_xy[1],
            marker="o", s=90, color=TRACKED_USER_COLOR, edgecolor="black", linewidths=1.2
        )
        plt.title(f"Hard labels (argmax membership) in PCA-2D (k={k}) [n={len(X2_plot)} plotted]")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"hard_labels_k_{k}.png"), dpi=200)
        plt.close()

        save_combined_region_heatmap(
            X2_all=X2,
            cntr=cntr,
            k=k,
            out_path=os.path.join(run_dir, f"combined_regions_heatmap_k_{k}.png"),
            m=M_FUZZINESS,
            error=ERROR,
            maxiter=MAXITER,
            grid_n=250,
        )

        save_engagement_heatmap_from_membership(
            X2_all=X2,
            cntr=cntr,
            k=k,
            out_path=os.path.join(run_dir, f"engagement_heatmap_k_{k}.png"),
            m=M_FUZZINESS,
            error=ERROR,
            maxiter=MAXITER,
            grid_n=250,
        )

        zone_dir = os.path.join(run_dir, "zone_grids_10x10")
        save_cluster_zone_grid_heatmaps(
            X2_all=X2,
            U=U,
            tracked_xy=tracked_xy,
            k=k,
            out_dir=zone_dir,
            bins=GRID_BINS,
            thr_disconnect=ZONE_THRESH_DISCONNECT,
            thr_align=ZONE_THRESH_ALIGN,
            annotate=True,
        )

    print("Done.")
    print(f"Tracked user info: {os.path.join(OUTPUT_DIR, 'tracked_user.json')}")
    print(f"Metrics (CSV):      {os.path.join(OUTPUT_DIR, 'k_selection_metrics_extended.csv')}")
    print(f"Summary (CSV):      {os.path.join(OUTPUT_DIR, 'summary_cluster_metrics_2_12.csv')}")
    print(f"Summary (JSON):     {os.path.join(OUTPUT_DIR, 'summary_cluster_metrics_2_12.json')}")
    print(f"Metrics (JSON dir): {os.path.join(OUTPUT_DIR, 'metrics_json')}")
    print(f"Mapping:            {os.path.join(OUTPUT_DIR, 'feature_mapping.json')}")
    print(f"PCA2D:              {os.path.join(OUTPUT_DIR, 'users_pca2d.csv')}")
    print("Generated subfolders:", ", ".join([f"k_{k}" for k in ks_to_run]))


if __name__ == "__main__":
    main()
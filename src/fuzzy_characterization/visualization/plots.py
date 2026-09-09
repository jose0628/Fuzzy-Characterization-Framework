"""Matplotlib figures. Every function saves to ``path`` and returns it."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..clustering.fcm import fcm_predict  # noqa: E402
from ..fuzzification.membership import MembershipFunction  # noqa: E402

# Colour-blind safe palette (Okabe-Ito)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999"]


def _save(fig, path: str | Path, dpi: int = 160) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_membership_functions(engine, path: str | Path, dpi: int = 160) -> Path:
    """One panel per fuzzy feature with its linguistic terms."""
    feats = engine.features
    n = len(feats)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), squeeze=False)
    for ax, f in zip(axes.ravel(), feats):
        if f.kind == "hour_profile":
            xs = np.linspace(0, 24, 481)
            xlabel = "hour of day"
        elif f.normalisation == "none":
            supports = [mf.support() for mf in f.terms.values()]
            lo = max(min(s[0] for s in supports), -1e6)
            hi = min(max(s[1] for s in supports), 1e6)
            xs = np.linspace(lo, hi, 401)
            xlabel = f"{f.indicator} (original units)"
        else:
            xs = np.linspace(0, 1, 401)
            xlabel = f"{f.indicator} (normalised, {f.normalisation})"
        for i, (term, mf) in enumerate(f.terms.items()):
            ax.plot(xs, mf(xs), color=PALETTE[i % len(PALETTE)], lw=1.8, label=term)
        if f.alpha_cut:
            ax.axhline(float(f.alpha_cut.get("level", 0.5)), ls="--", color="grey", lw=1, label=f"alpha={f.alpha_cut.get('level', 0.5)}")
        ax.set_title(f"{f.name} ({f.family})", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("mu(x)", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Fuzzy membership functions of the fuzzification engine", fontsize=12)
    return _save(fig, path, dpi)


def plot_shift_patterns(patterns: Dict[str, MembershipFunction], path: str | Path, dpi: int = 160) -> Path:
    """Working-hour membership functions for every shift pattern (Figure "shift variants")."""
    names = list(patterns)
    ncols = 2
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 2.8 * nrows), squeeze=False)
    xs = np.linspace(0, 24, 481)
    for ax, name in zip(axes.ravel(), names):
        ax.plot(xs, patterns[name](xs), color=PALETTE[0], lw=2)
        ax.set_title(f"{name}: active hours", fontsize=10)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_ylim(0, 1.2)
        ax.set_xlabel("hour of day", fontsize=8)
        ax.set_ylabel("mu(x)", fontsize=8)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    return _save(fig, path, dpi)


def plot_alpha_cuts(mf: MembershipFunction, levels: Sequence[float], path: str | Path, title: str = "", dpi: int = 160) -> Path:
    lo, hi = mf.support()
    xs = np.linspace(lo, hi, 601)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, mf(xs), color="black", lw=2, label=mf.label or "membership")
    for i, a in enumerate(sorted(levels)):
        iv = mf.alpha_cut_interval(a)
        ax.axhline(a, ls="--", color=PALETTE[i % len(PALETTE)], lw=1)
        if iv:
            ax.hlines(-0.08 * (i + 1), iv[0], iv[1], color=PALETTE[i % len(PALETTE)], lw=3)
            ax.text(iv[1], -0.08 * (i + 1), f"  A_{a}=[{iv[0]:.2f}, {iv[1]:.2f}]", va="center", fontsize=8)
    ax.set_ylim(-0.08 * (len(levels) + 1), 1.15)
    ax.set_xlabel("x")
    ax.set_ylabel("mu(x)")
    ax.set_title(title or "Alpha-cuts")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, path, dpi)


def plot_explained_variance(evr: np.ndarray, path: str | Path, limit: int = 20, dpi: int = 160) -> Path:
    """Bars = individual explained variance, line = cumulative (Figure "PCA explained variance")."""
    v = np.asarray(evr)[:limit] * 100
    cum = np.cumsum(v)
    ind = np.arange(1, len(v) + 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(ind, v, alpha=0.7, color=PALETTE[0], label="Individual explained variance (%)")
    ax.plot(ind, cum, marker="o", color=PALETTE[1], label="Cumulative explained variance (%)")
    for i, val in enumerate(v):
        ax.annotate(f"{val:.1f}%", (ind[i], val), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(ind)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Explained variance of the first {len(v)} principal components")
    ax.legend()
    ax.grid(alpha=0.25)
    return _save(fig, path, dpi)


def plot_k_selection(metrics: pd.DataFrame, suggestions: Dict[str, object], path: str | Path, dpi: int = 160) -> Path:
    """2x2 grid: PC, CE, DBI and the elbow of the FCM objective (Figure "cluster validity")."""
    panels = [
        ("partition_coefficient", "Partition Coefficient (PC)", suggestions.get("partition_coefficient_max"), "max"),
        ("classification_entropy", "Classification Entropy (CE)", suggestions.get("classification_entropy_min"), "min"),
        ("davies_bouldin", "Davies-Bouldin Index (DBI)", suggestions.get("davies_bouldin_min"), "min"),
        ("fcm_objective", "Elbow method (FCM objective J_m)", suggestions.get("elbow_fcm_objective"), "elbow"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title, best, how) in zip(axes.ravel(), panels):
        ax.plot(metrics["k"], metrics[col], marker="o", color=PALETTE[0])
        if best is not None:
            ax.axvline(best, ls=":", color="red", lw=2)
            title = f"{title} - best k = {best} ({how})"
        else:
            title = f"{title} - no clear {how}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("number of clusters k")
        ax.set_xticks(metrics["k"])
        ax.grid(alpha=0.3)
    rec = suggestions.get("recommended_k")
    fig.suptitle(f"Cluster validity analysis (recommended k = {rec})", fontsize=12)
    return _save(fig, path, dpi)


def _marker_groups(u: np.ndarray):
    return [
        (u >= 0.8, "+", 36, 1.0, "u >= 0.8"),
        ((u >= 0.6) & (u < 0.8), "x", 30, 0.9, "0.6 <= u < 0.8"),
        ((u >= 0.4) & (u < 0.6), "o", 26, 0.7, "0.4 <= u < 0.6"),
        ((u >= 0.2) & (u < 0.4), "^", 22, 0.55, "0.2 <= u < 0.4"),
        (u < 0.2, ".", 16, 0.3, "u < 0.2"),
    ]


def plot_membership_contours(
    X2: np.ndarray,
    U: np.ndarray,
    centers2: np.ndarray,
    m: float,
    path: str | Path,
    names: Optional[Sequence[str]] = None,
    max_points: int = 2000,
    seed: int = 42,
    grid_n: int = 200,
    levels: Sequence[float] = (0.2, 0.4, 0.6, 0.8),
    dpi: int = 160,
) -> Path:
    """Per-cluster membership contours in the PC1-PC2 plane (Figure "four clusters").

    Contours are FCM memberships computed on a grid from the projected centres;
    points are drawn with a marker encoding their membership in the cluster.
    """
    k = centers2.shape[0]
    n = len(X2)
    idx = np.arange(n) if n <= max_points else np.random.default_rng(seed).choice(n, max_points, replace=False)
    pad_x = 0.1 * (X2[:, 0].max() - X2[:, 0].min() + 1e-9)
    pad_y = 0.1 * (X2[:, 1].max() - X2[:, 1].min() + 1e-9)
    xs = np.linspace(X2[:, 0].min() - pad_x, X2[:, 0].max() + pad_x, grid_n)
    ys = np.linspace(X2[:, 1].min() - pad_y, X2[:, 1].max() + pad_y, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    Ugrid = fcm_predict(grid, centers2, m).reshape(grid_n, grid_n, k)
    ncols = int(np.ceil(np.sqrt(k)))
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.6 * nrows), squeeze=False)
    for j, ax in enumerate(axes.ravel()[:k]):
        cs = ax.contour(XX, YY, Ugrid[:, :, j], levels=list(levels), colors=[PALETTE[j % len(PALETTE)]], linewidths=1.2)
        ax.clabel(cs, inline=True, fontsize=7)
        u = U[idx, j]
        for mask, marker, size, alpha, label in _marker_groups(u):
            if mask.any():
                ax.scatter(X2[idx][mask, 0], X2[idx][mask, 1], marker=marker, s=size, alpha=alpha, c="black",
                           label=label, linewidths=0.8)
        ax.scatter(centers2[j, 0], centers2[j, 1], marker="*", s=220, c="#FFA000", edgecolor="black", zorder=5, label="centre")
        title = f"C{j + 1}" + (f": {names[j]}" if names is not None else "")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PC1 (general activity intensity)")
        ax.set_ylabel("PC2 (consumption vs contribution)")
        ax.legend(fontsize=6, loc="best")
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[k:]:
        ax.axis("off")
    fig.suptitle(f"Fuzzy cluster memberships in the PCA plane (k={k})", fontsize=12)
    return _save(fig, path, dpi)


def plot_cluster_scatter(X2: np.ndarray, labels: np.ndarray, centers2: Optional[np.ndarray], path: str | Path,
                         names: Optional[Sequence[str]] = None, title: str = "", max_points: int = 2000, seed: int = 42,
                         dpi: int = 160) -> Path:
    n = len(X2)
    idx = np.arange(n) if n <= max_points else np.random.default_rng(seed).choice(n, max_points, replace=False)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for j in sorted(np.unique(labels)):
        mask = labels[idx] == j
        lab = f"C{j + 1}" + (f": {names[j]}" if names is not None else "")
        ax.scatter(X2[idx][mask, 0], X2[idx][mask, 1], s=14, alpha=0.6, color=PALETTE[j % len(PALETTE)], label=lab)
    if centers2 is not None:
        ax.scatter(centers2[:, 0], centers2[:, 1], marker="*", s=240, c="#FFA000", edgecolor="black", zorder=5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title or "Clusters in the PCA plane")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    return _save(fig, path, dpi)


def plot_membership_heatmap(U: np.ndarray, path: str | Path, names: Optional[Sequence[str]] = None, max_rows: int = 400, dpi: int = 160) -> Path:
    order = np.lexsort((-U.max(axis=1), np.argmax(U, axis=1)))
    Us = U[order][:max_rows]
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(Us, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(U.shape[1]))
    ax.set_xticklabels([f"C{j + 1}" + (f"\n{names[j]}" if names is not None else "") for j in range(U.shape[1])], fontsize=7)
    ax.set_ylabel(f"users (first {len(Us)}, sorted by cluster and membership)")
    fig.colorbar(im, ax=ax, label="membership degree")
    ax.set_title("Membership matrix")
    return _save(fig, path, dpi)


def plot_cluster_profiles(averages: pd.DataFrame, path: str | Path, names: Optional[Sequence[str]] = None, dpi: int = 160) -> Path:
    """Heatmap of z-scored feature averages per cluster."""
    cols = [c for c in averages.columns if c != "description"]
    M = averages[cols].to_numpy(dtype=float)
    mu, sd = M.mean(axis=1, keepdims=True), M.std(axis=1, keepdims=True)
    Z = (M - mu) / np.where(sd > 1e-12, sd, 1.0)
    fig, ax = plt.subplots(figsize=(1.6 * len(cols) + 4, 0.45 * len(averages) + 2))
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c + (f"\n{names[i]}" if names is not None else "") for i, c in enumerate(cols)], fontsize=8)
    ax.set_yticks(range(len(averages)))
    ax.set_yticklabels(averages["description"] if "description" in averages.columns else averages.index, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            txt = f"{val * 100:.1f}%" if "share" in str(averages.index[i]) else f"{val:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, label="z-score across clusters")
    ax.set_title("Average behavioural feature values per cluster")
    return _save(fig, path, dpi)


def plot_method_comparison(table: pd.DataFrame, path: str | Path, dpi: int = 160) -> Path:
    metrics = [("silhouette_or_fsi", "Silhouette / FSI"), ("cluster_stability", "Cluster stability (ARI)"),
               ("interpretability_score", "Interpretability score")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, title) in zip(axes, metrics):
        colors = [PALETTE[0] if r == "fuzzy" else PALETTE[7] for r in table["representation"]]
        ax.bar(table["method"].str.upper(), table[col].astype(float), color=colors)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.05 if col != "silhouette_or_fsi" else max(1.0, float(table[col].max()) * 1.1))
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle("Comparative evaluation (blue = fuzzy on anonymised representation, grey = baselines on full features)", fontsize=10)
    return _save(fig, path, dpi)

#!/usr/bin/env python3
"""
Gustafson-Kessel clustering over K=2..12 with summary metrics and 2D visualizations.

Usage:
    python gk_clustering_2_to_12.py \
        --input /path/to/input.csv \
        --output GK_output_1_12

Default behavior:
- Reads a CSV file.
- Uses all numeric columns except identifier-like columns for clustering.
- Tries K = 2..12.
- Creates one subfolder per K under the output directory.
- Saves clustering assignments, memberships, centroids, covariances, metrics, and a 2D PCA visualization.
- Produces a global CSV summary across all K.

Notes:
- The script assumes the input has already been privacy-preserving transformed.
- Negative values are allowed.
- The first non-numeric/id column is preserved in outputs as an identifier when available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


EPS = 1e-10
MAX_POINTS = 200


@dataclass
class GKResult:
    k: int
    centers: np.ndarray            # (k, p)
    memberships: np.ndarray        # (n, k)
    covariances: np.ndarray        # (k, p, p)
    objective: float
    n_iter: int
    converged: bool
    hard_labels: np.ndarray        # (n,)
    distances: np.ndarray          # (n, k)
    metrics: Dict[str, float]


class GustafsonKessel:
    """Gustafson-Kessel fuzzy clustering with adaptive cluster covariances."""

    def __init__(
            self,
            n_clusters: int,
            m: float = 2.0,
            max_iter: int = 300,
            tol: float = 1e-5,
            random_state: Optional[int] = None,
            covariance_reg: float = 1e-6,
    ) -> None:
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2")
        if m <= 1:
            raise ValueError("fuzzifier m must be > 1")
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.covariance_reg = covariance_reg

    def _init_memberships(self, n_samples: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        u = rng.random((n_samples, self.n_clusters))
        u /= np.clip(u.sum(axis=1, keepdims=True), EPS, None)
        return u

    def fit(self, X: np.ndarray) -> GKResult:
        n, p = X.shape
        u = self._init_memberships(n)
        prev_u = u.copy()
        A_inv = np.array([np.eye(p) for _ in range(self.n_clusters)])
        covs = np.array([np.eye(p) for _ in range(self.n_clusters)])

        converged = False
        objective = np.inf

        for it in range(1, self.max_iter + 1):
            um = np.power(np.clip(u, EPS, None), self.m)
            denom = np.clip(um.sum(axis=0), EPS, None)  # (k,)
            centers = (um.T @ X) / denom[:, None]

            # Update cluster covariances and induced metric matrices.
            covs = np.zeros((self.n_clusters, p, p), dtype=float)
            A_inv = np.zeros((self.n_clusters, p, p), dtype=float)
            for j in range(self.n_clusters):
                diff = X - centers[j]
                Sj = np.einsum("n,ni,nj->ij", um[:, j], diff, diff) / denom[j]
                Sj += self.covariance_reg * np.eye(p)
                det_sj = max(float(np.linalg.det(Sj)), EPS)
                # GK metric: A_j = rho_j * det(S_j)^(1/p) * inv(S_j), here rho_j = 1.
                # We directly work with A_j for distances.
                A_j = (det_sj ** (1.0 / p)) * np.linalg.pinv(Sj)
                # Regularize again for numerical stability.
                A_j += self.covariance_reg * np.eye(p)
                covs[j] = Sj
                A_inv[j] = A_j

            # Distances under cluster-specific metrics.
            d = np.zeros((n, self.n_clusters), dtype=float)
            for j in range(self.n_clusters):
                diff = X - centers[j]
                d[:, j] = np.einsum("ni,ij,nj->n", diff, A_inv[j], diff)
            d = np.clip(d, EPS, None)

            # Exact handling for points sitting on a center.
            zero_mask = d <= EPS * 10
            u_new = np.zeros_like(u)
            rows_with_zero = zero_mask.any(axis=1)
            if np.any(rows_with_zero):
                for i in np.where(rows_with_zero)[0]:
                    z = zero_mask[i]
                    u_new[i, z] = 1.0 / z.sum()

            rows_no_zero = ~rows_with_zero
            if np.any(rows_no_zero):
                exponent = 1.0 / (self.m - 1.0)
                ratio = d[rows_no_zero, :, None] / d[rows_no_zero, None, :]
                u_new[rows_no_zero] = 1.0 / np.sum(np.power(ratio, exponent), axis=2)

            u_new /= np.clip(u_new.sum(axis=1, keepdims=True), EPS, None)

            objective_new = float(np.sum(np.power(u_new, self.m) * d))
            max_change = float(np.max(np.abs(u_new - prev_u)))
            u = u_new
            prev_u = u_new.copy()
            objective = objective_new

            if max_change < self.tol:
                converged = True
                break

        hard_labels = np.argmax(u, axis=1)
        return GKResult(
            k=self.n_clusters,
            centers=centers,
            memberships=u,
            covariances=covs,
            objective=objective,
            n_iter=it,
            converged=converged,
            hard_labels=hard_labels,
            distances=d,
            metrics={},
        )


def detect_id_column(df: pd.DataFrame) -> Optional[str]:
    preferred = [
        "actor_id", "user_id", "id", "identifier", "uuid", "guid", "member_id",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for key in preferred:
        if key in lower_map:
            return lower_map[key]
    # Fall back: choose first non-numeric column with high uniqueness.
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            nunique = df[col].nunique(dropna=True)
            if nunique >= 0.5 * len(df):
                return col
    return None


def choose_feature_columns(df: pd.DataFrame, id_col: Optional[str]) -> List[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    drop_like = {id_col} if id_col else set()
    for col in list(numeric_cols):
        cl = col.lower()
        if cl in {"unnamed: 0", "index"}:
            drop_like.add(col)
    feature_cols = [c for c in numeric_cols if c not in drop_like]
    if len(feature_cols) < 2:
        raise ValueError("Need at least two numeric feature columns for clustering.")
    return feature_cols


def fuzzy_silhouette_index(X: np.ndarray, U: np.ndarray, labels: np.ndarray) -> float:
    """
    Fuzzy silhouette approximation.

    Uses crisp silhouettes from argmax labels and weights each sample by how much
    its top membership exceeds the second-largest membership. This is a practical,
    transparent FSI proxy for comparing fuzzy solutions.
    """
    k = U.shape[1]
    if k < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    s = silhouette_score(X, labels, metric="euclidean", sample_size=min(len(X), 20000), random_state=42)
    # Sample-wise silhouettes for weighting would be ideal, so compute directly.
    from sklearn.metrics import silhouette_samples
    ss = silhouette_samples(X, labels, metric="euclidean")
    sorted_u = np.sort(U, axis=1)
    top1 = sorted_u[:, -1]
    top2 = sorted_u[:, -2]
    weights = np.clip(top1 - top2, 0.0, 1.0)
    if np.all(weights < EPS):
        return float(np.mean(ss))
    return float(np.sum(weights * ss) / np.sum(weights))


def stability_from_restarts(label_runs: List[np.ndarray]) -> float:
    if len(label_runs) < 2:
        return float("nan")
    scores = []
    for i in range(len(label_runs)):
        for j in range(i + 1, len(label_runs)):
            scores.append(adjusted_rand_score(label_runs[i], label_runs[j]))
    return float(np.mean(scores)) if scores else float("nan")


def interpretability_score(centers: np.ndarray, memberships: np.ndarray) -> float:
    """
    Heuristic [0,1] interpretability score combining three aspects:
    1) centroid separation,
    2) centroid dominance clarity,
    3) membership decisiveness.
    """
    k, p = centers.shape
    if k < 2:
        return float("nan")

    pairwise = []
    for i in range(k):
        for j in range(i + 1, k):
            pairwise.append(np.linalg.norm(centers[i] - centers[j]))
    sep = float(np.mean(pairwise))
    sep_norm = sep / (sep + 1.0)

    dominance = []
    for row in centers:
        abs_row = np.abs(row)
        order = np.sort(abs_row)
        top1 = order[-1]
        top2 = order[-2] if len(order) > 1 else 0.0
        dominance.append((top1 - top2) / (top1 + EPS))
    dom = float(np.mean(dominance))

    sorted_u = np.sort(memberships, axis=1)
    decisiveness = float(np.mean(sorted_u[:, -1] - sorted_u[:, -2]))

    score = 0.40 * sep_norm + 0.35 * dom + 0.25 * decisiveness
    return float(np.clip(score, 0.0, 1.0))


def project_centers_to_pca(centers: np.ndarray, pca: PCA, scaler: StandardScaler) -> np.ndarray:
    return pca.transform(centers)


def covariance_to_pca_space(cov: np.ndarray, pca_components: np.ndarray) -> np.ndarray:
    return pca_components @ cov @ pca_components.T


def draw_cov_ellipse(ax, mean: np.ndarray, cov2: np.ndarray, n_std: float = 2.0, **kwargs) -> None:
    cov2 = np.asarray(cov2, dtype=float)
    if cov2.shape != (2, 2):
        return
    vals, vecs = np.linalg.eigh(cov2)
    vals = np.clip(vals, EPS, None)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle, fill=False, **kwargs)
    ax.add_patch(ell)


# def save_visualization(
#         X: np.ndarray,
#         result: GKResult,
#         feature_cols: List[str],
#         out_png: Path,
#         title: str,
# ) -> None:
#     pca = PCA(n_components=2, random_state=42)
#     X2 = pca.fit_transform(X)
#     centers2 = project_centers_to_pca(result.centers, pca, scaler=None)  # scaler already applied to X
#
#     # Membership confidence for visual emphasis.
#     max_u = np.max(result.memberships, axis=1)
#     sizes = 10 + 30 * max_u
#     labels = result.hard_labels
#
#     plt.figure(figsize=(10, 8))
#     ax = plt.gca()
#     scatter = ax.scatter(X2[:, 0], X2[:, 1], c=labels, s=sizes, alpha=0.45)
#     ax.scatter(centers2[:, 0], centers2[:, 1], marker='*', s=350, edgecolor='black', linewidths=1.0)
#
#     # Draw ellipses from covariances projected into PCA space.
#     W = pca.components_  # (2, p)
#     for j in range(result.k):
#         cov2 = covariance_to_pca_space(result.covariances[j], W)
#         draw_cov_ellipse(ax, centers2[j], cov2, n_std=2.0, linewidth=2)
#         ax.text(centers2[j, 0], centers2[j, 1], f"C{j+1}", fontsize=11, ha='center', va='center')
#
#     ax.set_title(title)
#     ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
#     ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
#     ax.grid(True, alpha=0.25)
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=220, bbox_inches="tight")
#     plt.close()

def save_visualization(
        X: np.ndarray,
        result: GKResult,
        feature_cols: List[str],
        out_png: Path,
        title: str,
) -> None:

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)

    centers2 = project_centers_to_pca(result.centers, pca, scaler=None)

    labels = result.hard_labels
    max_u = np.max(result.memberships, axis=1)

    # ---- SAMPLE MAX 200 POINTS ----
    n = X2.shape[0]
    if n > MAX_POINTS:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, MAX_POINTS, replace=False)

        X2_plot = X2[idx]
        labels_plot = labels[idx]
        max_u_plot = max_u[idx]
    else:
        X2_plot = X2
        labels_plot = labels
        max_u_plot = max_u
    # --------------------------------

    sizes = 10 + 30 * max_u_plot

    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    ax.scatter(
        X2_plot[:, 0],
        X2_plot[:, 1],
        c=labels_plot,
        s=sizes,
        alpha=0.45
    )

    ax.scatter(
        centers2[:, 0],
        centers2[:, 1],
        marker='*',
        s=350,
        edgecolor='black',
        linewidths=1.0
    )

    W = pca.components_

    for j in range(result.k):
        cov2 = covariance_to_pca_space(result.covariances[j], W)
        draw_cov_ellipse(ax, centers2[j], cov2, n_std=2.0, linewidth=2)
        ax.text(
            centers2[j, 0],
            centers2[j, 1],
            f"C{j+1}",
            fontsize=11,
            ha='center',
            va='center'
        )

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def save_cluster_profiles(result: GKResult, feature_cols: List[str], out_csv: Path) -> pd.DataFrame:
    profiles = pd.DataFrame(result.centers, columns=feature_cols)
    profiles.insert(0, "cluster", [f"C{i+1}" for i in range(result.k)])
    profiles.to_csv(out_csv, index=False)
    return profiles


def run_single_k(
        X: np.ndarray,
        original_df: pd.DataFrame,
        id_col: Optional[str],
        feature_cols: List[str],
        k: int,
        k_dir: Path,
        n_restarts: int,
        fuzzifier: float,
        max_iter: int,
        tol: float,
) -> GKResult:
    k_dir.mkdir(parents=True, exist_ok=True)

    best: Optional[GKResult] = None
    label_runs: List[np.ndarray] = []
    run_metrics: List[Dict[str, float]] = []

    for seed in range(n_restarts):
        model = GustafsonKessel(
            n_clusters=k,
            m=fuzzifier,
            max_iter=max_iter,
            tol=tol,
            random_state=seed,
        )
        res = model.fit(X)
        label_runs.append(res.hard_labels)

        # Crisp silhouette + fuzzy silhouette proxy.
        if len(np.unique(res.hard_labels)) > 1:
            sil = float(silhouette_score(
                X,
                res.hard_labels,
                metric="euclidean",
                sample_size=min(len(X), 20000),
                random_state=42,
            ))
            fsi = fuzzy_silhouette_index(X, res.memberships, res.hard_labels)
        else:
            sil, fsi = float("nan"), float("nan")

        interp = interpretability_score(res.centers, res.memberships)
        metrics = {
            "k": k,
            "seed": seed,
            "objective": res.objective,
            "iterations": res.n_iter,
            "converged": float(res.converged),
            "silhouette": sil,
            "fsi": fsi,
            "interpretability_score": interp,
        }
        run_metrics.append(metrics)

        res.metrics = metrics.copy()
        if best is None or res.objective < best.objective:
            best = res

    assert best is not None
    stability = stability_from_restarts(label_runs)
    best.metrics["cluster_stability"] = stability

    # Persist run-wise metrics.
    pd.DataFrame(run_metrics).to_csv(k_dir / "run_metrics.csv", index=False)

    # Persist memberships and assignments.
    assignments = pd.DataFrame({
        (id_col if id_col else "row_id"): original_df[id_col].values if id_col else np.arange(len(original_df)),
        "hard_cluster": best.hard_labels + 1,
        "max_membership": np.max(best.memberships, axis=1),
    })
    membership_df = pd.DataFrame(best.memberships, columns=[f"C{i+1}_membership" for i in range(k)])
    assignments = pd.concat([assignments, membership_df], axis=1)
    assignments.to_csv(k_dir / "cluster_assignments_memberships.csv", index=False)

    # Persist centers and covariance matrices.
    save_cluster_profiles(best, feature_cols, k_dir / "cluster_centers.csv")
    cov_dir = k_dir / "covariances"
    cov_dir.mkdir(exist_ok=True)
    for j in range(k):
        pd.DataFrame(best.covariances[j], index=feature_cols, columns=feature_cols).to_csv(
            cov_dir / f"cluster_{j+1}_covariance.csv"
        )

    # Metrics JSON for the chosen best run.
    metric_payload = {
        "k": k,
        "best_objective": best.objective,
        "iterations": best.n_iter,
        "converged": bool(best.converged),
        "silhouette": best.metrics.get("silhouette"),
        "fsi": best.metrics.get("fsi"),
        "cluster_stability": stability,
        "interpretability_score": best.metrics.get("interpretability_score"),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
    with open(k_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metric_payload, f, indent=2)

    # Visualization.
    save_visualization(
        X=X,
        result=best,
        feature_cols=feature_cols,
        out_png=k_dir / f"gk_k_{k}_pca_visualization.png",
        title=f"Gustafson-Kessel clustering (K={k}) in PCA space",
    )

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gustafson-Kessel clustering for K=2..12.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", default="GK_output_1_12", help="Output directory.")
    parser.add_argument("--k-min", type=int, default=2, help="Minimum K.")
    parser.add_argument("--k-max", type=int, default=12, help="Maximum K.")
    parser.add_argument("--fuzzifier", type=float, default=2.0, help="Fuzzifier m (>1).")
    parser.add_argument("--max-iter", type=int, default=300, help="Maximum iterations per run.")
    parser.add_argument("--tol", type=float, default=1e-5, help="Convergence tolerance.")
    parser.add_argument("--restarts", type=int, default=5, help="Number of random restarts for stability.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    id_col = detect_id_column(df)
    feature_cols = choose_feature_columns(df, id_col=id_col)
    X_raw = df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    X_raw = X_raw.fillna(X_raw.median(numeric_only=True))

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw.values)

    # Save metadata and preprocessing info.
    metadata = {
        "input_file": str(input_path),
        "id_column": id_col,
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "k_range": [args.k_min, args.k_max],
        "fuzzifier": args.fuzzifier,
        "restarts": args.restarts,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "scaling": "StandardScaler on numeric features",
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    pd.DataFrame({
        "feature": feature_cols,
        "mean_before_scaling": X_raw.mean(axis=0).values,
        "std_before_scaling": X_raw.std(axis=0, ddof=0).values,
    }).to_csv(output_dir / "feature_summary.csv", index=False)

    summary_rows = []
    for k in range(args.k_min, args.k_max + 1):
        k_dir = output_dir / f"k_{k}"
        best = run_single_k(
            X=X,
            original_df=df,
            id_col=id_col,
            feature_cols=feature_cols,
            k=k,
            k_dir=k_dir,
            n_restarts=args.restarts,
            fuzzifier=args.fuzzifier,
            max_iter=args.max_iter,
            tol=args.tol,
        )
        summary_rows.append({
            "k": k,
            "best_objective": best.objective,
            "iterations": best.n_iter,
            "converged": bool(best.converged),
            "silhouette": best.metrics.get("silhouette"),
            "fsi": best.metrics.get("fsi"),
            "cluster_stability": best.metrics.get("cluster_stability"),
            "interpretability_score": best.metrics.get("interpretability_score"),
            "output_folder": str(k_dir),
            "visualization_file": str(k_dir / f"gk_k_{k}_pca_visualization.png"),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "GK_summary_metrics_k_2_12.csv", index=False)

    # Convenience ranking outputs.
    if not summary.empty:
        rank_cols = ["fsi", "cluster_stability", "interpretability_score"]
        rank_df = summary.copy()
        for col in rank_cols:
            rank_df[f"rank_{col}"] = rank_df[col].rank(ascending=False, method="min")
        rank_df["mean_rank"] = rank_df[[f"rank_{c}" for c in rank_cols]].mean(axis=1)
        rank_df.sort_values(["mean_rank", "k"]).to_csv(output_dir / "GK_summary_ranked.csv", index=False)

    print(f"Done. Results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
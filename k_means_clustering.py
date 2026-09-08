from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


RANDOM_STATE = 42
MAX_PLOT_POINTS = 200
N_STABILITY_RUNS = 8


@dataclass
class KMeansRunResult:
    k: int
    labels: np.ndarray
    centers_scaled: np.ndarray
    inertia: float
    silhouette: float
    stability: float
    interpretability: float
    cluster_sizes: List[int]
    feature_importance: dict


def load_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    df = pd.read_csv(csv_path)
    if "actor_id" not in df.columns:
        raise ValueError("Expected an 'actor_id' column in the input dataset.")

    feature_cols = [c for c in df.columns if c != "actor_id"]
    X_df = df[feature_cols].copy()

    # Keep only numeric columns and coerce safely.
    for col in feature_cols:
        X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

    # Replace inf / nan conservatively.
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    return df, X_df, feature_cols


def bounded_separation(centers_scaled: np.ndarray) -> float:
    if len(centers_scaled) < 2:
        return 0.0
    dists = []
    for i in range(len(centers_scaled)):
        for j in range(i + 1, len(centers_scaled)):
            dists.append(np.linalg.norm(centers_scaled[i] - centers_scaled[j]))
    mean_dist = float(np.mean(dists)) if dists else 0.0
    # Monotone map to [0, 1)
    return mean_dist / (mean_dist + np.sqrt(centers_scaled.shape[1]))


def size_balance_score(cluster_sizes: List[int]) -> float:
    sizes = np.array(cluster_sizes, dtype=float)
    props = sizes / sizes.sum()
    k = len(props)
    worst_std = np.sqrt((k - 1) / (k ** 2))
    std = float(np.std(props))
    score = 1.0 - min(std / (worst_std + 1e-12), 1.0)
    return float(max(0.0, min(1.0, score)))


def interpretability_score(
        centers_scaled: np.ndarray,
        feature_cols: List[str],
        cluster_sizes: List[int],
) -> Tuple[float, dict]:
    abs_centers = np.abs(centers_scaled)

    # Dominance clarity: how much of each cluster meaning is explained by top features.
    top3_share = []
    top_features = {}
    for idx, row in enumerate(abs_centers):
        total = float(np.sum(row)) + 1e-12
        order = np.argsort(row)[::-1]
        top_idx = order[:5]
        top3 = order[:3]
        top3_share.append(float(np.sum(row[top3]) / total))
        top_features[f"cluster_{idx}"] = [
            {"feature": feature_cols[i], "abs_weight": float(row[i])}
            for i in top_idx
        ]

    prominence = float(np.mean(top3_share))
    separation = bounded_separation(centers_scaled)
    balance = size_balance_score(cluster_sizes)

    # With centroid-based interpretation, prominence and separation matter most.
    score = 0.6 * prominence + 0.35 * separation + 0.05 * balance
    score = float(max(0.0, min(1.0, score)))
    return score, top_features


def compute_cluster_sizes(labels: np.ndarray, k: int) -> List[int]:
    return [int(np.sum(labels == i)) for i in range(k)]


def compute_stability(
        X_scaled: np.ndarray,
        k: int,
        random_state: int,
        n_runs: int = N_STABILITY_RUNS,
) -> float:
    label_runs = []
    for seed in range(random_state, random_state + n_runs):
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        label_runs.append(km.fit_predict(X_scaled))

    scores = []
    for i in range(len(label_runs)):
        for j in range(i + 1, len(label_runs)):
            ari = adjusted_rand_score(label_runs[i], label_runs[j])
            scores.append(float(ari))
    return float(np.mean(scores)) if scores else 0.0


def project_centers_to_pca(centers_scaled: np.ndarray, pca: PCA) -> np.ndarray:
    return pca.transform(centers_scaled)


def sample_for_plot(
        X2: np.ndarray,
        labels: np.ndarray,
        max_points: int,
        random_state: int,
) -> np.ndarray:
    n = X2.shape[0]
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def save_visualization(
        X_scaled: np.ndarray,
        labels: np.ndarray,
        centers_scaled: np.ndarray,
        feature_cols: List[str],
        out_png: Path,
        title: str,
        random_state: int = RANDOM_STATE,
) -> None:
    pca = PCA(n_components=2, random_state=random_state)
    X2 = pca.fit_transform(X_scaled)
    centers2 = project_centers_to_pca(centers_scaled, pca)

    idx = sample_for_plot(
        X2, labels, max_points=MAX_PLOT_POINTS, random_state=random_state
    )
    X2_plot = X2[idx]
    labels_plot = labels[idx]

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.scatter(X2_plot[:, 0], X2_plot[:, 1], c=labels_plot, alpha=0.6, s=35)
    ax.scatter(
        centers2[:, 0],
        centers2[:, 1],
        marker="*",
        s=350,
        edgecolor="black",
        linewidth=1.0,
    )

    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    explained = pca.explained_variance_ratio_.sum() * 100
    ax.text(
        0.02,
        0.02,
        f"Explained variance (2D PCA): {explained:.2f}%\nPlotted data points: {len(idx)} / {len(labels)}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def save_cluster_profiles(
        original_df: pd.DataFrame,
        feature_cols: List[str],
        labels: np.ndarray,
        out_csv: Path,
) -> None:
    tmp = original_df[feature_cols].copy()
    tmp["cluster"] = labels
    profile = tmp.groupby("cluster")[feature_cols].mean().round(6)
    profile.to_csv(out_csv)


def save_assignments(raw_df: pd.DataFrame, labels: np.ndarray, out_csv: Path) -> None:
    out = pd.DataFrame({"actor_id": raw_df["actor_id"], "cluster": labels})
    out.to_csv(out_csv, index=False)


def run_single_k(
        X_scaled: np.ndarray,
        X_original: pd.DataFrame,
        raw_df: pd.DataFrame,
        feature_cols: List[str],
        k: int,
        out_dir: Path,
) -> KMeansRunResult:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(X_scaled)
    centers_scaled = km.cluster_centers_

    sample_size = min(1000, len(labels))
    silhouette = float(
        silhouette_score(
            X_scaled,
            labels,
            sample_size=sample_size,
            random_state=RANDOM_STATE,
        )
    )
    stability = compute_stability(X_scaled, k=k, random_state=RANDOM_STATE)
    cluster_sizes = compute_cluster_sizes(labels, k)
    interpretability, top_features = interpretability_score(
        centers_scaled, feature_cols, cluster_sizes
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    save_visualization(
        X_scaled=X_scaled,
        labels=labels,
        centers_scaled=centers_scaled,
        feature_cols=feature_cols,
        out_png=out_dir / f"k_{k}_pca_visualization.png",
        title=f"K-Means clustering (k={k}) on standardized activity features",
    )
    save_cluster_profiles(
        X_original, feature_cols, labels, out_dir / f"k_{k}_cluster_profiles.csv"
    )
    save_assignments(raw_df, labels, out_dir / f"k_{k}_assignments.csv")

    metadata = {
        "k": k,
        "silhouette": silhouette,
        "stability": stability,
        "interpretability": interpretability,
        "cluster_sizes": cluster_sizes,
        "top_features_by_cluster": top_features,
        "inertia": float(km.inertia_),
    }
    (out_dir / f"k_{k}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return KMeansRunResult(
        k=k,
        labels=labels,
        centers_scaled=centers_scaled,
        inertia=float(km.inertia_),
        silhouette=silhouette,
        stability=stability,
        interpretability=interpretability,
        cluster_sizes=cluster_sizes,
        feature_importance=top_features,
    )


def build_summary(results: List[KMeansRunResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        rows.append(
            {
                "k": res.k,
                "silhouette_fsi": res.silhouette,
                "cluster_stability": res.stability,
                "interpretability_score": res.interpretability,
                "inertia": res.inertia,
                "min_cluster_size": int(min(res.cluster_sizes)),
                "max_cluster_size": int(max(res.cluster_sizes)),
            }
        )

    summary = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    metric_cols = [
        "silhouette_fsi",
        "cluster_stability",
        "interpretability_score",
    ]
    for col in metric_cols:
        cmin = summary[col].min()
        cmax = summary[col].max()
        if np.isclose(cmax, cmin):
            summary[f"{col}_norm"] = 1.0
        else:
            summary[f"{col}_norm"] = (summary[col] - cmin) / (cmax - cmin)

    summary["overall_rank_score"] = (
            0.40 * summary["silhouette_fsi_norm"]
            + 0.35 * summary["cluster_stability_norm"]
            + 0.25 * summary["interpretability_score_norm"]
    )
    summary = summary.sort_values(
        ["overall_rank_score", "silhouette_fsi", "cluster_stability"],
        ascending=False,
    ).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)

    ordered_cols = [
        "rank",
        "k",
        "overall_rank_score",
        "silhouette_fsi",
        "cluster_stability",
        "interpretability_score",
        "inertia",
        "min_cluster_size",
        "max_cluster_size",
        "silhouette_fsi_norm",
        "cluster_stability_norm",
        "interpretability_score_norm",
    ]
    return summary[ordered_cols]


def main() -> None:
    input_csv = Path("datasets/leonardo_activity_unpacked_step3_log1p_winsor_robust_1_99.csv")
    output_root = Path("k_means_1_12")
    output_root.mkdir(parents=True, exist_ok=True)

    raw_df, X_df, feature_cols = load_dataset(input_csv)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    results = []
    for k in range(2, 13):
        k_dir = output_root / f"k_{k}"
        res = run_single_k(
            X_scaled=X_scaled,
            X_original=X_df,
            raw_df=raw_df,
            feature_cols=feature_cols,
            k=k,
            out_dir=k_dir,
        )
        results.append(res)

    summary = build_summary(results)
    summary.to_csv(output_root / "kmeans_summary_ranked.csv", index=False)

    best_row = summary.iloc[0].to_dict()
    report = {
        "input_csv": str(input_csv),
        "n_rows": int(len(raw_df)),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "best_k_by_rank": int(best_row["k"]),
        "best_rank_score": float(best_row["overall_rank_score"]),
        "notes": {
            "silhouette_fsi": "For k-means this is the standard silhouette score; the column name is kept as silhouette_fsi for cross-method comparability.",
            "cluster_stability": "Mean pairwise Adjusted Rand Index across repeated runs with different random seeds.",
            "interpretability_score": "Composite proxy based on centroid feature dominance and centroid separation, scaled to [0,1].",
            "visualization": f"Each plot uses PCA to 2D and limits displayed datapoints to at most {MAX_PLOT_POINTS}."
        }
    }
    (output_root / "run_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("[OK] Generated outputs in", output_root)
    print(summary.head(11).to_string(index=False))


if __name__ == "__main__":
    main()
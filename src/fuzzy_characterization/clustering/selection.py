"""Determination of the number of clusters (Section "Optimal number of clusters").

For ``k`` in a range the following validity indices are computed on FCM
solutions: Partition Coefficient (PC), Classification Entropy (CE),
Davies-Bouldin Index (DBI) on the hard labels, the elbow of the objective
``J_m`` (and of the K-Means inertia), plus Xie-Beni, silhouette and
Calinski-Harabasz for completeness.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from ..config import ClusteringConfig
from ..evaluation.validity import classification_entropy, partition_coefficient, xie_beni
from .fcm import fcm_multi_restart


def detect_elbow(ks: Iterable[int], values: Iterable[float]) -> Optional[int]:
    """Knee of a decreasing convex curve (Kneedle-style maximum distance to the chord).

    Uses the ``kneed`` package when available and falls back to a self-contained
    implementation otherwise.
    """
    ks = list(ks)
    vals = np.asarray(list(values), dtype=float)
    if len(ks) < 3 or not np.all(np.isfinite(vals)):
        return None
    try:  # pragma: no cover - optional dependency
        from kneed import KneeLocator

        knee = KneeLocator(ks, vals, curve="convex", direction="decreasing").knee
        if knee is not None:
            return int(knee)
    except Exception:
        pass
    x = np.asarray(ks, dtype=float)
    xn = (x - x[0]) / max(x[-1] - x[0], 1e-12)
    yn = (vals - vals.min()) / max(vals.max() - vals.min(), 1e-12)
    # distance to the chord between first and last point
    chord = yn[0] + (yn[-1] - yn[0]) * xn
    dist = chord - yn
    idx = int(np.argmax(dist))
    return int(ks[idx]) if dist[idx] > 0 else None


def evaluate_k_range(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    cfg: Optional[ClusteringConfig] = None,
    n_restarts: int = 5,
    max_silhouette_samples: int = 5000,
) -> pd.DataFrame:
    """Validity indices for every ``k``; one row per ``k``."""
    cfg = cfg or ClusteringConfig()
    X = np.asarray(X, dtype=float)
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(cfg.seed)
    sil_idx = rng.choice(len(X), size=min(len(X), max_silhouette_samples), replace=False)
    for k in range(k_min, k_max + 1):
        best, _ = fcm_multi_restart(X, k, seeds=range(cfg.seed, cfg.seed + n_restarts), m=cfg.m,
                                    max_iter=cfg.max_iter, tol=cfg.tol)
        U, V, labels = best.memberships, best.centers, best.labels
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed).fit(X)
        row = {
            "k": k,
            "partition_coefficient": partition_coefficient(U),
            "classification_entropy": classification_entropy(U),
            "xie_beni": xie_beni(X, V, U, cfg.m),
            "fcm_objective": best.objective,
            "kmeans_inertia": float(km.inertia_),
        }
        if len(np.unique(labels)) > 1:
            row["davies_bouldin"] = float(davies_bouldin_score(X, labels))
            row["silhouette"] = float(silhouette_score(X[sil_idx], labels[sil_idx])) if len(np.unique(labels[sil_idx])) > 1 else float("nan")
            row["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
        else:
            row.update({"davies_bouldin": float("nan"), "silhouette": float("nan"), "calinski_harabasz": float("nan")})
        rows.append(row)
    return pd.DataFrame(rows)


def suggest_k(metrics: pd.DataFrame) -> Dict[str, Any]:
    """Best ``k`` per criterion and a recommendation.

    PC and CE favour small ``k`` by construction, so they are reported but do
    not vote; the recommendation is the mode of DBI, the elbow of ``J_m`` and
    Xie-Beni (ties broken towards DBI).
    """
    ks = metrics["k"].tolist()
    out: Dict[str, Any] = {
        "partition_coefficient_max": int(metrics.loc[metrics["partition_coefficient"].idxmax(), "k"]),
        "classification_entropy_min": int(metrics.loc[metrics["classification_entropy"].idxmin(), "k"]),
        "davies_bouldin_min": int(metrics.loc[metrics["davies_bouldin"].idxmin(), "k"]),
        "xie_beni_min": int(metrics.loc[metrics["xie_beni"].idxmin(), "k"]),
        "elbow_fcm_objective": detect_elbow(ks, metrics["fcm_objective"]),
        "elbow_kmeans_inertia": detect_elbow(ks, metrics["kmeans_inertia"]),
    }
    if "silhouette" in metrics.columns and metrics["silhouette"].notna().any():
        out["silhouette_max"] = int(metrics.loc[metrics["silhouette"].idxmax(), "k"])
    voters = ["davies_bouldin_min", "xie_beni_min", "elbow_fcm_objective", "elbow_kmeans_inertia", "silhouette_max"]
    votes = [out[v] for v in voters if out.get(v) is not None]
    counts = pd.Series(votes).value_counts()
    top = counts[counts == counts.max()].index.tolist()
    out["recommended_k"] = int(out["davies_bouldin_min"] if out["davies_bouldin_min"] in top else min(top))
    out["votes"] = {v: out.get(v) for v in voters}
    out["rationale"] = (
        "PC and CE favour small k by construction and are reported only; the recommendation is the majority "
        "vote of the Davies-Bouldin minimum, the Xie-Beni minimum, the elbows of the FCM objective and of the "
        "K-Means inertia, and the silhouette maximum (ties resolved towards DBI, then the smaller k)."
    )
    return out

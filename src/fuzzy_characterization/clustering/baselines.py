"""Non-fuzzy baselines: K-Means and agglomerative hierarchical clustering (Ward)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

from .base import ClusterResult, one_hot


def kmeans(X: np.ndarray, k: int, seed: Optional[int] = None, n_init: int = 10, max_iter: int = 300) -> ClusterResult:
    X = np.asarray(X, dtype=float)
    model = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=seed)
    labels = model.fit_predict(X)
    return ClusterResult(
        method="kmeans", k=k, centers=model.cluster_centers_, memberships=one_hot(labels, k), labels=labels,
        objective=float(model.inertia_), n_iter=int(model.n_iter_), converged=True, seed=seed,
    )


def ahc(X: np.ndarray, k: int, linkage: str = "ward", seed: Optional[int] = None) -> ClusterResult:
    X = np.asarray(X, dtype=float)
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X)
    centers = np.vstack([X[labels == j].mean(axis=0) if np.any(labels == j) else X.mean(axis=0) for j in range(k)])
    within = float(sum(np.sum((X[labels == j] - centers[j]) ** 2) for j in range(k)))
    return ClusterResult(
        method="ahc", k=k, centers=centers, memberships=one_hot(labels, k), labels=labels,
        objective=within, n_iter=0, converged=True, seed=seed, extras={"linkage": linkage},
    )

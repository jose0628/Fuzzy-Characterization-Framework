"""Cluster validity indices for fuzzy and crisp partitions."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score

EPS = 1e-12


def partition_coefficient(U: np.ndarray) -> float:
    """PC = (1/n) sum_i sum_j u_ij^2 (higher = crisper partition)."""
    return float(np.mean(np.sum(U ** 2, axis=1)))


def classification_entropy(U: np.ndarray) -> float:
    """CE = -(1/n) sum_i sum_j u_ij log u_ij (lower = more confident)."""
    return float(-np.mean(np.sum(U * np.log(np.clip(U, EPS, 1.0)), axis=1)))


def normalised_entropy(U: np.ndarray) -> float:
    """CE divided by log(k), in [0, 1]."""
    k = U.shape[1]
    return classification_entropy(U) / np.log(k) if k > 1 else 0.0


def xie_beni(X: np.ndarray, centers: np.ndarray, U: np.ndarray, m: float = 2.0) -> float:
    """XB = sum_ij u_ij^m ||x_i - v_j||^2 / (n min_{p != q} ||v_p - v_q||^2) (lower is better)."""
    diff = X[:, None, :] - centers[None, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    num = float(np.sum((U ** m) * d2))
    k = centers.shape[0]
    sep = np.inf
    for i in range(k):
        for j in range(i + 1, k):
            sep = min(sep, float(np.sum((centers[i] - centers[j]) ** 2)))
    return num / (X.shape[0] * max(sep, EPS))


def _subsample(n: int, max_samples: int, seed: int) -> np.ndarray:
    if n <= max_samples:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, size=max_samples, replace=False)


def crisp_silhouette(X: np.ndarray, labels: np.ndarray, max_samples: int = 5000, seed: int = 42) -> float:
    """Standard silhouette (Eq. 5) on a subsample for large n."""
    idx = _subsample(len(X), max_samples, seed)
    if len(np.unique(labels[idx])) < 2:
        return float("nan")
    return float(silhouette_score(X[idx], labels[idx]))


def fuzzy_silhouette_index(
    X: np.ndarray,
    U: np.ndarray,
    alpha: float = 1.0,
    max_samples: int = 5000,
    seed: int = 42,
) -> float:
    """Fuzzy Silhouette (Campello & Hruschka, 2006).

    ``FS = sum_i (u_pi - u_qi)^alpha s_i / sum_i (u_pi - u_qi)^alpha`` where
    ``s_i`` is the crisp silhouette of point ``i`` under its highest-membership
    label and ``u_p``, ``u_q`` are its largest and second-largest memberships.
    Ambiguous points therefore contribute less to the index.
    """
    idx = _subsample(len(X), max_samples, seed)
    Xs, Us = X[idx], U[idx]
    labels = np.argmax(Us, axis=1)
    if len(np.unique(labels)) < 2:
        return float("nan")
    s = silhouette_samples(Xs, labels)
    sorted_u = np.sort(Us, axis=1)
    w = (sorted_u[:, -1] - sorted_u[:, -2]) ** alpha
    if w.sum() <= EPS:
        return float(np.mean(s))
    return float(np.sum(w * s) / np.sum(w))


def separation_score(X: np.ndarray, U: np.ndarray, is_fuzzy: bool, alpha: float = 1.0, max_samples: int = 5000, seed: int = 42) -> float:
    """FSI for fuzzy partitions, silhouette for crisp ones (Table "segmentation quality")."""
    if is_fuzzy:
        return fuzzy_silhouette_index(X, U, alpha=alpha, max_samples=max_samples, seed=seed)
    return crisp_silhouette(X, np.argmax(U, axis=1), max_samples=max_samples, seed=seed)

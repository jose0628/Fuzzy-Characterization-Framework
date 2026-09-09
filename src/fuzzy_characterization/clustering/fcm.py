"""Fuzzy C-Means (Bezdek, 1981), Eq. 7 of the design chapter.

Minimises ``J_m = sum_i sum_j u_ij^m ||x_i - c_j||^2`` by alternating the
membership and centre updates until ``|J_m(t) - J_m(t-1)| < tol``.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

from .base import ClusterResult, squared_distances

EPS = 1e-12


def fcm_memberships(D2: np.ndarray, m: float) -> np.ndarray:
    """Membership update ``u_ij = 1 / sum_l (d_ij / d_il)^(2/(m-1))`` from squared distances."""
    n, k = D2.shape
    power = 1.0 / (m - 1.0)
    zero = D2 <= EPS
    U = np.zeros((n, k))
    rows_zero = zero.any(axis=1)
    if rows_zero.any():
        z = zero[rows_zero].astype(float)
        U[rows_zero] = z / z.sum(axis=1, keepdims=True)
    ok = ~rows_zero
    if ok.any():
        inv = (1.0 / D2[ok]) ** power
        U[ok] = inv / inv.sum(axis=1, keepdims=True)
    return U


def _init_centers(X: np.ndarray, k: int, rng: np.random.Generator, init: str) -> np.ndarray:
    n = X.shape[0]
    if init == "kmeans++":
        centers = [X[rng.integers(n)]]
        for _ in range(1, k):
            d2 = np.min(squared_distances(X, np.vstack(centers)), axis=1)
            probs = d2 / max(d2.sum(), EPS)
            centers.append(X[rng.choice(n, p=probs)])
        return np.vstack(centers)
    # random membership initialisation (classical FCM)
    U = rng.random((n, k))
    U /= U.sum(axis=1, keepdims=True)
    return _centers_from_U(X, U, 2.0)


def _centers_from_U(X: np.ndarray, U: np.ndarray, m: float) -> np.ndarray:
    Um = U ** m
    return (Um.T @ X) / np.maximum(Um.sum(axis=0)[:, None], EPS)


def fcm(
    X: np.ndarray,
    k: int,
    m: float = 2.0,
    max_iter: int = 300,
    tol: float = 1e-5,
    seed: Optional[int] = None,
    init: str = "random",
) -> ClusterResult:
    """Run FCM on ``X`` (n, d). Returns a :class:`ClusterResult` with the
    objective history in ``extras['objective_history']``."""
    X = np.asarray(X, dtype=float)
    if k < 2:
        raise ValueError("k must be >= 2")
    if m <= 1:
        raise ValueError("fuzziness m must be > 1")
    rng = np.random.default_rng(seed)
    V = _init_centers(X, k, rng, init)
    history: List[float] = []
    U = fcm_memberships(squared_distances(X, V), m)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        V = _centers_from_U(X, U, m)
        D2 = squared_distances(X, V)
        U = fcm_memberships(D2, m)
        J = float(np.sum((U ** m) * D2))
        history.append(J)
        if len(history) > 1 and abs(history[-2] - J) < tol:
            converged = True
            break
    labels = np.argmax(U, axis=1)
    return ClusterResult(
        method="fcm", k=k, centers=V, memberships=U, labels=labels,
        objective=history[-1] if history else float("nan"), n_iter=it, converged=converged, seed=seed,
        extras={"m": m, "objective_history": history},
    )


def fcm_predict(X: np.ndarray, centers: np.ndarray, m: float = 2.0) -> np.ndarray:
    """Memberships of new points (or a grid) given fitted centres."""
    return fcm_memberships(squared_distances(np.asarray(X, dtype=float), centers), m)


def fcm_multi_restart(
    X: np.ndarray,
    k: int,
    seeds: Iterable[int],
    m: float = 2.0,
    max_iter: int = 300,
    tol: float = 1e-5,
    init: str = "random",
) -> Tuple[ClusterResult, List[ClusterResult]]:
    """Run FCM once per seed; return the best (lowest objective) and all runs."""
    runs = [fcm(X, k, m=m, max_iter=max_iter, tol=tol, seed=int(s), init=init) for s in seeds]
    best = min(runs, key=lambda r: r.objective)
    return best, runs

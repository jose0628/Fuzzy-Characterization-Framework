"""Gustafson-Kessel clustering (Gustafson & Kessel, 1979).

An FCM variant with a cluster-specific Mahalanobis metric
``A_j = det(F_j)^(1/d) F_j^{-1}``, where ``F_j`` is the fuzzy covariance of
cluster ``j``; this lets clusters take ellipsoidal shapes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import ClusterResult
from .fcm import fcm, fcm_memberships

EPS = 1e-12


def gustafson_kessel(
    X: np.ndarray,
    k: int,
    m: float = 2.0,
    max_iter: int = 300,
    tol: float = 1e-5,
    seed: Optional[int] = None,
    covariance_reg: float = 1e-4,
) -> ClusterResult:
    """Run GK; ``extras['covariances']`` holds the fuzzy covariance matrices (k, d, d)."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    init = fcm(X, k, m=m, max_iter=max(50, max_iter // 4), tol=tol, seed=seed)
    U = init.memberships.copy()
    V = init.centers.copy()
    covs = np.array([np.eye(d) for _ in range(k)])
    prev_obj = None
    converged = False
    it = 0
    eye = np.eye(d)
    for it in range(1, max_iter + 1):
        Um = U ** m
        denom = np.maximum(Um.sum(axis=0), EPS)
        V = (Um.T @ X) / denom[:, None]
        D2 = np.zeros((n, k))
        for j in range(k):
            diff = X - V[j]
            F = (Um[:, j][:, None] * diff).T @ diff / denom[j]
            F = F + covariance_reg * np.trace(F) / d * eye + EPS * eye
            det = max(float(np.linalg.det(F)), EPS)
            A = (det ** (1.0 / d)) * np.linalg.pinv(F)
            covs[j] = F
            D2[:, j] = np.einsum("ni,ij,nj->n", diff, A, diff)
        D2 = np.maximum(D2, EPS)
        U_new = fcm_memberships(D2, m)
        obj = float(np.sum((U_new ** m) * D2))
        change = float(np.max(np.abs(U_new - U)))
        U = U_new
        if change < tol or (prev_obj is not None and abs(prev_obj - obj) < tol):
            prev_obj = obj
            converged = True
            break
        prev_obj = obj
    labels = np.argmax(U, axis=1)
    return ClusterResult(
        method="gk", k=k, centers=V, memberships=U, labels=labels,
        objective=float(prev_obj) if prev_obj is not None else float("nan"), n_iter=it, converged=converged,
        seed=seed, extras={"covariances": covs, "m": m},
    )

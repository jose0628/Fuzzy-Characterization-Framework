"""Possibilistic Fuzzy C-Means (Pal, Pal, Keller & Bezdek, 2005).

PFCM combines the relative memberships ``u_ij`` of FCM with typicalities
``t_ij`` that do not sum to one, which makes the prototypes robust to noise
and overlapping behaviours:

``J = sum_i sum_j (a u_ij^m + b t_ij^eta) d_ij^2 + sum_j gamma_j sum_i (1 - t_ij)^eta``
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import ClusterResult, squared_distances
from .fcm import fcm, fcm_memberships

EPS = 1e-12


def _typicalities(D2: np.ndarray, gamma: np.ndarray, eta: float, b: float) -> np.ndarray:
    exponent = 1.0 / (eta - 1.0)
    frac = (b * D2) / np.maximum(gamma[None, :], EPS)
    return np.clip(1.0 / (1.0 + frac ** exponent), 0.0, 1.0)


def pfcm(
    X: np.ndarray,
    k: int,
    m: float = 2.0,
    eta: float = 2.0,
    a: float = 1.0,
    b: float = 1.0,
    max_iter: int = 300,
    tol: float = 1e-5,
    seed: Optional[int] = None,
    K: float = 1.0,
) -> ClusterResult:
    """Run PFCM; ``extras`` holds ``typicalities`` (n, k) and ``gamma`` (k,)."""
    X = np.asarray(X, dtype=float)
    if eta <= 1 or m <= 1:
        raise ValueError("m and eta must be > 1")
    # Initialise from an FCM solution (standard practice)
    init = fcm(X, k, m=m, max_iter=max(50, max_iter // 4), tol=tol, seed=seed)
    V, U = init.centers.copy(), init.memberships.copy()
    D2 = squared_distances(X, V)
    Um = U ** m
    gamma = K * (Um * D2).sum(axis=0) / np.maximum(Um.sum(axis=0), EPS)
    gamma = np.maximum(gamma, EPS)
    T = _typicalities(D2, gamma, eta, b)

    prev = None
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        W = a * U ** m + b * T ** eta
        V_new = (W.T @ X) / np.maximum(W.sum(axis=0)[:, None], EPS)
        D2 = squared_distances(X, V_new)
        U = fcm_memberships(D2, m)
        T = _typicalities(D2, gamma, eta, b)
        J = float(np.sum((a * U ** m + b * T ** eta) * D2) + np.sum(gamma * ((1 - T) ** eta).sum(axis=0)))
        shift = float(np.max(np.linalg.norm(V_new - V, axis=1)))
        V = V_new
        if prev is not None and (shift < tol or abs(prev - J) / (abs(prev) + EPS) < tol):
            converged = True
            prev = J
            break
        prev = J
    labels = np.argmax(U, axis=1)
    return ClusterResult(
        method="pfcm", k=k, centers=V, memberships=U, labels=labels,
        objective=float(prev) if prev is not None else float("nan"), n_iter=it, converged=converged, seed=seed,
        extras={"typicalities": T, "gamma": gamma, "m": m, "eta": eta, "a": a, "b": b},
    )

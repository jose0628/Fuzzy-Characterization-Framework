"""Cluster stability across repeated runs (Section "Convergence and Stability").

For methods with random initialisation, stability is the mean pairwise
Adjusted Rand Index of the hard assignments across runs with different
seeds, complemented for fuzzy methods by the mean agreement of the
membership matrices after aligning cluster labels.  For deterministic
methods (AHC) the perturbation is a random 80 % subsample.
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from ..clustering.base import ClusterResult


def align_memberships(U_ref: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Permute the columns of ``U`` to best match ``U_ref`` (Hungarian on overlap)."""
    overlap = U_ref.T @ U  # (k, k)
    rows, cols = linear_sum_assignment(-overlap)
    perm = np.empty(U.shape[1], dtype=int)
    perm[rows] = cols
    return U[:, perm]


def membership_agreement(U_a: np.ndarray, U_b: np.ndarray) -> float:
    """1 - mean L1 distance between aligned membership rows (in [0, 1])."""
    U_b = align_memberships(U_a, U_b)
    return float(1.0 - 0.5 * np.mean(np.abs(U_a - U_b).sum(axis=1)))


def stability_from_runs(runs: List[ClusterResult]) -> Dict[str, float]:
    if len(runs) < 2:
        return {"ari_mean": float("nan"), "ari_min": float("nan"), "membership_agreement": float("nan"), "n_runs": len(runs)}
    aris, agreements = [], []
    for a, b in combinations(runs, 2):
        aris.append(adjusted_rand_score(a.labels, b.labels))
        agreements.append(membership_agreement(a.memberships, b.memberships))
    return {
        "ari_mean": float(np.mean(aris)),
        "ari_min": float(np.min(aris)),
        "membership_agreement": float(np.mean(agreements)),
        "n_runs": len(runs),
    }


def cluster_stability(
    runner: Callable[[np.ndarray, int], ClusterResult],
    X: np.ndarray,
    n_runs: int = 10,
    seed: int = 1,
    subsample: Optional[float] = None,
    runs: Optional[List[ClusterResult]] = None,
) -> Dict[str, float]:
    """Stability of ``runner(X, seed)`` over ``n_runs`` restarts or subsamples.

    ``runner`` takes the data and a seed.  With ``subsample`` set (e.g. 0.8),
    each run is fitted on a random subsample and the ARI is computed on the
    overlap of every pair.  Pre-computed ``runs`` (same data) can be reused.
    """
    if runs is not None and subsample is None:
        return stability_from_runs(runs)
    rng = np.random.default_rng(seed)
    n = len(X)
    if subsample is None:
        results = [runner(X, int(seed + r)) for r in range(n_runs)]
        return stability_from_runs(results)
    size = max(int(round(n * subsample)), 2)
    fitted = []
    for r in range(n_runs):
        idx = np.sort(rng.choice(n, size=size, replace=False))
        fitted.append((idx, runner(X[idx], int(seed + r))))
    aris, agreements = [], []
    for (ia, ra), (ib, rb) in combinations(fitted, 2):
        common, pa, pb = np.intersect1d(ia, ib, return_indices=True)
        if len(common) < 2:
            continue
        aris.append(adjusted_rand_score(ra.labels[pa], rb.labels[pb]))
        agreements.append(membership_agreement(ra.memberships[pa], rb.memberships[pb]))
    return {
        "ari_mean": float(np.mean(aris)) if aris else float("nan"),
        "ari_min": float(np.min(aris)) if aris else float("nan"),
        "membership_agreement": float(np.mean(agreements)) if agreements else float("nan"),
        "n_runs": n_runs,
    }

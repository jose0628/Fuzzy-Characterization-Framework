"""Common result container for every clustering method."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

FUZZY_METHODS = ("fcm", "pfcm", "gk")
CRISP_METHODS = ("kmeans", "ahc")
ALL_METHODS = FUZZY_METHODS + CRISP_METHODS

METHOD_NAMES = {
    "fcm": "Fuzzy C-Means (FCM)",
    "pfcm": "Possibilistic Fuzzy C-Means (PFCM)",
    "gk": "Gustafson-Kessel (GK)",
    "kmeans": "K-Means",
    "ahc": "Agglomerative Hierarchical (AHC, Ward)",
}


@dataclass
class ClusterResult:
    method: str
    k: int
    centers: np.ndarray            # (k, d)
    memberships: np.ndarray        # (n, k); one-hot for crisp methods
    labels: np.ndarray             # (n,)
    objective: float = float("nan")
    n_iter: int = 0
    converged: bool = True
    seed: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_fuzzy(self) -> bool:
        return self.method in FUZZY_METHODS

    @property
    def n(self) -> int:
        return int(self.memberships.shape[0])

    def cluster_sizes(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=self.k)

    def membership_mass(self) -> np.ndarray:
        """Sum of memberships per cluster (fuzzy cardinality)."""
        return self.memberships.sum(axis=0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "k": int(self.k),
            "objective": float(self.objective),
            "n_iter": int(self.n_iter),
            "converged": bool(self.converged),
            "seed": self.seed,
            "cluster_sizes": [int(v) for v in self.cluster_sizes()],
            "membership_mass": [float(v) for v in self.membership_mass()],
        }


def one_hot(labels: np.ndarray, k: int) -> np.ndarray:
    U = np.zeros((len(labels), k), dtype=float)
    U[np.arange(len(labels)), labels] = 1.0
    return U


def squared_distances(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``D2[i, j] = ||x_i - v_j||^2`` computed without an (n, k, d) temporary."""
    xx = np.einsum("ij,ij->i", X, X)[:, None]
    vv = np.einsum("ij,ij->i", V, V)[None, :]
    return np.maximum(xx + vv - 2.0 * X @ V.T, 0.0)

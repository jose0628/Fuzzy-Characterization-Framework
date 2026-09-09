"""Principal component analysis on the fuzzy attribute vector.

PCA serves two roles in the framework: the retained components (about 20,
around 90 % of cumulative explained variance in the retail case) form the
clustering input, and the first two components give the visualisation plane
used to read the clusters.  The two-dimensional projection never validates
anything on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAReduction:
    pca: PCA
    scaler: Optional[StandardScaler]
    feature_names: List[str]
    scores: pd.DataFrame            # users x PC1..PCk
    n_components: int
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        v = X[self.feature_names].to_numpy(dtype=float)
        if self.scaler is not None:
            v = self.scaler.transform(v)
        return self.pca.transform(v)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        v = self.pca.inverse_transform(Z)
        if self.scaler is not None:
            v = self.scaler.inverse_transform(v)
        return v

    def loadings(self, top_n: int = 10) -> pd.DataFrame:
        rows = []
        for i, comp in enumerate(self.pca.components_):
            idx = np.argsort(np.abs(comp))[::-1][:top_n]
            for j in idx:
                rows.append({"component": f"PC{i + 1}", "feature": self.feature_names[j], "loading": float(comp[j])})
        return pd.DataFrame(rows)

    def variance_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "component": [f"PC{i + 1}" for i in range(self.n_components)],
                "explained_variance_ratio": self.explained_variance_ratio[: self.n_components],
                "cumulative_variance": self.cumulative_variance[: self.n_components],
            }
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "n_components": int(self.n_components),
            "cumulative_variance": float(self.cumulative_variance[self.n_components - 1]),
            "explained_variance_ratio": [float(v) for v in self.explained_variance_ratio[: self.n_components]],
        }


def fit_pca(
    X: pd.DataFrame,
    n_components: Any = 0.90,
    max_components: int = 20,
    standardise: bool = False,
    random_state: int = 42,
) -> PCAReduction:
    """Fit PCA keeping ``n_components`` (an int, or a variance ratio capped by ``max_components``)."""
    names = list(X.columns)
    v = X.to_numpy(dtype=float)
    scaler = None
    if standardise:
        scaler = StandardScaler()
        v = scaler.fit_transform(v)
    full = PCA(svd_solver="full", random_state=random_state).fit(v)
    evr = full.explained_variance_ratio_
    cum = np.cumsum(evr)
    if isinstance(n_components, float) and 0 < n_components <= 1.0:
        k = int(np.searchsorted(cum, n_components) + 1)
    else:
        k = int(n_components)
    k = int(max(1, min(k, max_components, len(evr))))
    pca = PCA(n_components=k, svd_solver="full", random_state=random_state).fit(v)
    scores = pd.DataFrame(pca.transform(v), index=X.index, columns=[f"PC{i + 1}" for i in range(k)])
    return PCAReduction(
        pca=pca, scaler=scaler, feature_names=names, scores=scores, n_components=k,
        explained_variance_ratio=evr, cumulative_variance=cum,
    )

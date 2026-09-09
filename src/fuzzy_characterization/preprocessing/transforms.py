"""Transformations for heavy-tailed activity data.

The fuzzy attribute vector already lies in [0, 1]; the transformations here
serve two purposes:

* an optional light normalisation of the membership variables to equalise
  their influence in the distance computation (``FeatureScaler``);
* the *full-resolution* feature space handed to the non-fuzzy baselines
  (``full_resolution_features``): ``log1p`` -> winsorising -> robust scaling,
  the most stable variant found in the exploratory scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def log1p_transform(X: pd.DataFrame) -> pd.DataFrame:
    return np.log1p(X.clip(lower=0))


def winsorize(X: pd.DataFrame, q_low: float = 0.01, q_high: float = 0.99) -> pd.DataFrame:
    lo, hi = X.quantile(q_low), X.quantile(q_high)
    return X.clip(lower=lo, upper=hi, axis=1)


def robust_scale(X: pd.DataFrame) -> Tuple[pd.DataFrame, RobustScaler]:
    sc = RobustScaler(quantile_range=(25.0, 75.0))
    return pd.DataFrame(sc.fit_transform(X), index=X.index, columns=X.columns), sc


def standard_scale(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    sc = StandardScaler()
    return pd.DataFrame(sc.fit_transform(X), index=X.index, columns=X.columns), sc


def minmax_scale(X: pd.DataFrame) -> Tuple[pd.DataFrame, MinMaxScaler]:
    sc = MinMaxScaler()
    return pd.DataFrame(sc.fit_transform(X), index=X.index, columns=X.columns), sc


def l1_composition(X: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    """Per-row composition (each row sums to one)."""
    s = X.sum(axis=1)
    return X.div(s.replace(0, np.nan) + eps, axis=0).fillna(0.0)


@dataclass
class FeatureScaler:
    """Optional scaling of the fuzzy attribute vector before clustering.

    ``method``:
      * ``none``     -- keep memberships as they are (default in the thesis);
      * ``standard`` -- z-scores (equal variance per attribute);
      * ``unit_var`` -- divide by the standard deviation only (keeps the zero);
      * ``minmax``   -- rescale each column to [0, 1] on the population.
    """

    method: str = "none"
    stats: Dict[str, np.ndarray] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame) -> "FeatureScaler":
        v = X.to_numpy(dtype=float)
        self.stats = {
            "mean": v.mean(axis=0),
            "std": np.where(v.std(axis=0) > 1e-12, v.std(axis=0), 1.0),
            "min": v.min(axis=0),
            "max": v.max(axis=0),
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.method == "none":
            return X.copy()
        v = X.to_numpy(dtype=float)
        if self.method == "standard":
            out = (v - self.stats["mean"]) / self.stats["std"]
        elif self.method == "unit_var":
            out = v / self.stats["std"]
        elif self.method == "minmax":
            rng = np.where(self.stats["max"] - self.stats["min"] > 1e-12, self.stats["max"] - self.stats["min"], 1.0)
            out = (v - self.stats["min"]) / rng
        else:
            raise ValueError(f"unknown scaling method '{self.method}'")
        return pd.DataFrame(out, index=X.index, columns=X.columns)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def full_resolution_features(
    indicators: pd.DataFrame,
    columns: Optional[List[str]] = None,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> pd.DataFrame:
    """Non-anonymised behavioural feature space for the baseline methods.

    ``log1p`` stabilises the long tails, winsorising removes extreme outliers
    and the robust scaler (median / IQR) puts every indicator on a comparable
    scale without being dominated by heavy tails.
    """
    cols = columns or [c for c in indicators.columns if not c.startswith(("hour_share_", "dow_share_"))]
    X = indicators[cols].astype(float)
    X = winsorize(log1p_transform(X), q_low, q_high)
    scaled, _ = robust_scale(X)
    return scaled.replace([np.inf, -np.inf], 0.0).fillna(0.0)

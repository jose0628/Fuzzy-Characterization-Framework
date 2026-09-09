"""Alpha-cuts (Section "Alpha-Cuts" of the design chapter).

An alpha-cut ``A_alpha = {x | mu_A(x) >= alpha}`` turns a fuzzy set into the
crisp set of observations whose membership reaches at least ``alpha``.  Cuts
are nested (``A_0.8 ⊂ A_0.5 ⊂ A_0.2``) and are reported as *segments*, never
as lists of individuals.  A segment smaller than a minimum group size is
flagged so that it is not reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def alpha_cut(memberships: np.ndarray | pd.Series, alpha: float, strong: bool = False) -> np.ndarray:
    """Boolean mask of the alpha-cut. ``strong=True`` uses ``>`` instead of ``>=``."""
    mu = np.asarray(memberships, dtype=float)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    return (mu > alpha) if strong else (mu >= alpha)


@dataclass
class AlphaCutSegment:
    feature: str
    alpha: float
    size: int
    share: float
    reportable: bool
    min_group_size: int

    def to_dict(self) -> Dict[str, float | str | bool | int]:
        return self.__dict__.copy()


def nested_alpha_cuts(
    memberships: np.ndarray | pd.Series,
    levels: Iterable[float] = (0.2, 0.5, 0.8),
) -> Dict[float, np.ndarray]:
    """Return the masks for every level; masks are nested by construction."""
    levels = sorted(float(a) for a in levels)
    return {a: alpha_cut(memberships, a) for a in levels}


def alpha_cut_report(
    fuzzy_vectors: pd.DataFrame,
    levels: Iterable[float] = (0.2, 0.5, 0.8),
    min_group_size: int = 20,
    features: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Segment sizes for every fuzzy attribute at every cut level.

    The report deliberately contains only *counts* and *shares*; it is the
    privacy-preserving description of a behaviour as a group.
    """
    cols = list(features) if features is not None else list(fuzzy_vectors.columns)
    n = len(fuzzy_vectors)
    rows: List[Dict] = []
    for col in cols:
        mu = fuzzy_vectors[col].to_numpy(dtype=float)
        for a, mask in nested_alpha_cuts(mu, levels).items():
            size = int(mask.sum())
            rows.append(
                AlphaCutSegment(
                    feature=col,
                    alpha=a,
                    size=size,
                    share=float(size / n) if n else 0.0,
                    reportable=bool(size == 0 or size >= min_group_size),
                    min_group_size=int(min_group_size),
                ).to_dict()
            )
    return pd.DataFrame(rows)


def segment_stability(fuzzy_vectors: pd.DataFrame, low: float = 0.2, high: float = 0.8) -> pd.Series:
    """Ratio ``|A_high| / |A_low|`` per attribute.

    Values near 1 mean the behaviour is robust to the cut level; values near 0
    mean the segment is defined mostly by observations at the edge of the
    membership function and should be treated as weakly defined.
    """
    out = {}
    for col in fuzzy_vectors.columns:
        mu = fuzzy_vectors[col].to_numpy(dtype=float)
        n_low = alpha_cut(mu, low).sum()
        n_high = alpha_cut(mu, high).sum()
        out[col] = float(n_high / n_low) if n_low else float("nan")
    return pd.Series(out, name="segment_stability")

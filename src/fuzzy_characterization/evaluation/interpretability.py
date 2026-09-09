"""Interpretability score (Section "Interpretability score").

The score combines three components:

1. **feature dominance clarity** -- whether clusters show distinct, coherent
   patterns across the behavioural dimensions (clear dominant attributes and
   profiles that differ from one another);
2. **membership transparency** -- whether cluster affiliation can be explained
   through fuzzy membership degrees rather than binary labels (0 for crisp
   partitions, ``1 - CE / log k`` for fuzzy ones);
3. **semantic consistency** -- expert ratings of the behavioural segments; it
   is optional and is left out of the average (and flagged) when no ratings
   are provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .validity import normalised_entropy

EPS = 1e-12


@dataclass
class InterpretabilityBreakdown:
    feature_dominance: float
    membership_transparency: float
    semantic_consistency: Optional[float]
    score: float
    weights: Dict[str, float] = field(default_factory=dict)
    semantic_assessed: bool = False

    def to_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


def cluster_profiles_z(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Membership-weighted cluster means expressed as z-scores of the features."""
    mu = X.mean(axis=0)
    sd = np.where(X.std(axis=0) > EPS, X.std(axis=0), 1.0)
    W = U / np.maximum(U.sum(axis=0, keepdims=True), EPS)
    centers = W.T @ X
    return (centers - mu) / sd


def feature_dominance_clarity(profiles_z: np.ndarray, top_n: int = 3) -> float:
    """Mean of (a) how concentrated each cluster profile is on its ``top_n``
    dominant attributes (share of the |z| mass they carry, rescaled so that a
    flat profile scores 0) and (b) the distinctness of the cluster profiles
    from one another (1 - cosine similarity)."""
    k, d = profiles_z.shape
    top_n = min(top_n, d)
    baseline = top_n / d
    conc = []
    for row in np.abs(profiles_z):
        total = row.sum()
        if total <= EPS:
            conc.append(0.0)
            continue
        share = np.sort(row)[::-1][:top_n].sum() / total
        conc.append((share - baseline) / max(1.0 - baseline, EPS))
    dominance = float(np.clip(np.mean(conc), 0.0, 1.0))
    if k < 2:
        return dominance
    sims = []
    norms = np.linalg.norm(profiles_z, axis=1) + EPS
    for i in range(k):
        for j in range(i + 1, k):
            sims.append(float(profiles_z[i] @ profiles_z[j] / (norms[i] * norms[j])))
    distinctness = float(np.clip((1.0 - np.mean(sims)) / 2.0, 0.0, 1.0))
    return float(np.clip(0.5 * dominance + 0.5 * distinctness, 0.0, 1.0))


def membership_transparency(U: np.ndarray, is_fuzzy: bool) -> float:
    """0 for crisp partitions; otherwise how informative the degrees are."""
    if not is_fuzzy:
        return 0.0
    return float(np.clip(1.0 - normalised_entropy(U), 0.0, 1.0))


def semantic_consistency(expert_ratings: Optional[pd.DataFrame]) -> Optional[float]:
    """Mean expert rating in [0, 1] (columns: ``cluster``, ``rating``)."""
    if expert_ratings is None or len(expert_ratings) == 0 or "rating" not in expert_ratings.columns:
        return None
    r = expert_ratings["rating"].astype(float)
    if r.max() > 1.0:  # allow 1..5 Likert scales
        r = (r - 1.0) / 4.0
    return float(np.clip(r.mean(), 0.0, 1.0))


def interpretability_score(
    X: np.ndarray,
    U: np.ndarray,
    is_fuzzy: bool,
    expert_ratings: Optional[pd.DataFrame] = None,
    weights: Optional[Dict[str, float]] = None,
) -> InterpretabilityBreakdown:
    w = {"feature_dominance": 1.0, "membership_transparency": 1.0, "semantic_consistency": 1.0}
    if weights:
        w.update(weights)
    dom = feature_dominance_clarity(cluster_profiles_z(X, U))
    trans = membership_transparency(U, is_fuzzy)
    sem = semantic_consistency(expert_ratings)
    parts = {"feature_dominance": dom, "membership_transparency": trans}
    if sem is not None:
        parts["semantic_consistency"] = sem
    total_w = sum(w[p] for p in parts)
    score = sum(w[p] * v for p, v in parts.items()) / max(total_w, EPS)
    return InterpretabilityBreakdown(
        feature_dominance=dom, membership_transparency=trans, semantic_consistency=sem,
        score=float(score), weights={p: w[p] for p in parts}, semantic_assessed=sem is not None,
    )

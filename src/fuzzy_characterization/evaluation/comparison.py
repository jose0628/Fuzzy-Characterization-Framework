"""Comparative evaluation (Section "Comparative Evaluation", Table "segmentation quality").

Fuzzy methods operate on the anonymised fuzzy representation; the non-fuzzy
baselines receive the full-resolution feature space.  ``k`` is fixed for
every method and all are scored on separation (FSI or silhouette),
stability, interpretability and privacy compliance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..clustering.base import ClusterResult, FUZZY_METHODS, METHOD_NAMES
from ..clustering.runner import run_method, run_with_restarts
from ..config import ClusteringConfig, EvaluationConfig
from .interpretability import interpretability_score
from .privacy import privacy_assessment
from .stability import cluster_stability
from .validity import classification_entropy, partition_coefficient, separation_score, xie_beni


@dataclass
class MethodEvaluation:
    method: str
    result: ClusterResult
    representation: str
    separation: float
    stability: Dict[str, float]
    interpretability: Dict[str, Any]
    privacy: Dict[str, Any]
    extra_indices: Dict[str, float] = field(default_factory=dict)

    def row(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "method_name": METHOD_NAMES.get(self.method, self.method),
            "representation": self.representation,
            "silhouette_or_fsi": self.separation,
            "separation_metric": "FSI" if self.result.is_fuzzy else "silhouette",
            "cluster_stability": self.stability.get("ari_mean"),
            "membership_agreement": self.stability.get("membership_agreement"),
            "interpretability_score": self.interpretability.get("score"),
            "privacy_compliance": self.privacy.get("level"),
            "k": self.result.k,
            **{f"idx_{k}": v for k, v in self.extra_indices.items()},
        }


def evaluate_result(
    method: str,
    X: np.ndarray,
    result: ClusterResult,
    runs: Optional[List[ClusterResult]],
    representation: str,
    clustering_cfg: ClusteringConfig,
    eval_cfg: EvaluationConfig,
    compliance: Optional[Dict[str, Any]],
    min_group_size: int,
    expert_ratings: Optional[pd.DataFrame] = None,
    X_profile: Optional[np.ndarray] = None,
) -> MethodEvaluation:
    """Score one clustering result.

    ``X`` is the space the method clustered; ``X_profile`` is the space in which
    cluster profiles are interpreted (fuzzy attributes for fuzzy methods, the
    behavioural features for the baselines) and defaults to ``X``.
    """
    if X_profile is None:
        X_profile = X
    sep = separation_score(X, result.memberships, result.is_fuzzy, alpha=eval_cfg.fsi_alpha,
                           max_samples=eval_cfg.max_silhouette_samples, seed=clustering_cfg.seed)
    if method == "ahc":
        stab = cluster_stability(lambda Xs, s: run_method("ahc", Xs, result.k, clustering_cfg, seed=s), X,
                                 n_runs=eval_cfg.stability_runs, seed=clustering_cfg.seed, subsample=0.8)
    elif runs is not None and len(runs) >= 2:
        stab = cluster_stability(lambda Xs, s: run_method(method, Xs, result.k, clustering_cfg, seed=s), X,
                                 runs=runs[: max(eval_cfg.stability_runs, 2)])
    else:
        stab = cluster_stability(lambda Xs, s: run_method(method, Xs, result.k, clustering_cfg, seed=s), X,
                                 n_runs=eval_cfg.stability_runs, seed=clustering_cfg.seed)
    interp = interpretability_score(X_profile, result.memberships, result.is_fuzzy, expert_ratings,
                                    eval_cfg.interpretability_weights)
    priv = privacy_assessment(representation, result.cluster_sizes(), min_group_size, compliance)
    extra = {}
    if result.is_fuzzy:
        extra = {
            "partition_coefficient": partition_coefficient(result.memberships),
            "classification_entropy": classification_entropy(result.memberships),
            "xie_beni": xie_beni(X, result.centers, result.memberships, clustering_cfg.m),
        }
    return MethodEvaluation(method, result, representation, sep, stab, interp.to_dict(), priv, extra)


def compare_methods(
    X_fuzzy: np.ndarray,
    X_full: np.ndarray,
    k: int,
    clustering_cfg: ClusteringConfig,
    eval_cfg: EvaluationConfig,
    methods: Optional[Iterable[str]] = None,
    compliance: Optional[Dict[str, Any]] = None,
    min_group_size: int = 20,
    expert_ratings: Optional[pd.DataFrame] = None,
    precomputed: Optional[Dict[str, tuple]] = None,
    X_fuzzy_profile: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, Dict[str, MethodEvaluation]]:
    """Run every method with ``k`` clusters and return the comparison table.

    ``X_fuzzy`` is the (possibly PCA-reduced) anonymised space the fuzzy methods
    cluster; ``X_fuzzy_profile`` is the original fuzzy attribute vector used to
    interpret their clusters (defaults to ``X_fuzzy``).  ``precomputed`` may map
    a method name to ``(best, runs)`` to reuse results.
    """
    methods = list(methods or clustering_cfg.compare_methods)
    evaluations: Dict[str, MethodEvaluation] = {}
    for method in methods:
        is_fuzzy = method in FUZZY_METHODS
        X = X_fuzzy if is_fuzzy else X_full
        representation = "fuzzy" if is_fuzzy else "full"
        if precomputed and method in precomputed:
            best, runs = precomputed[method]
        else:
            best, runs = run_with_restarts(method, X, k, clustering_cfg)
        profile_space = (X_fuzzy_profile if X_fuzzy_profile is not None else X_fuzzy) if is_fuzzy else X_full
        evaluations[method] = evaluate_result(
            method, X, best, runs, representation, clustering_cfg, eval_cfg, compliance, min_group_size,
            expert_ratings, X_profile=profile_space,
        )
    table = pd.DataFrame([e.row() for e in evaluations.values()])
    return table, evaluations

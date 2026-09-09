"""Dispatch clustering methods by name and handle restarts."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

from ..config import ClusteringConfig
from .base import ClusterResult
from .baselines import ahc, kmeans
from .fcm import fcm
from .gk import gustafson_kessel
from .pfcm import pfcm


def run_method(method: str, X: np.ndarray, k: int, cfg: Optional[ClusteringConfig] = None, seed: Optional[int] = None) -> ClusterResult:
    cfg = cfg or ClusteringConfig()
    method = method.lower()
    if method == "fcm":
        return fcm(X, k, m=cfg.m, max_iter=cfg.max_iter, tol=cfg.tol, seed=seed)
    if method == "pfcm":
        return pfcm(X, k, m=cfg.m, eta=cfg.eta, a=cfg.a, b=cfg.b, max_iter=cfg.max_iter, tol=cfg.tol, seed=seed,
                    K=cfg.pfcm_gamma_scale)
    if method == "gk":
        return gustafson_kessel(X, k, m=cfg.m, max_iter=cfg.max_iter, tol=cfg.tol, seed=seed)
    if method == "kmeans":
        return kmeans(X, k, seed=seed, max_iter=cfg.max_iter)
    if method == "ahc":
        return ahc(X, k, linkage=cfg.linkage, seed=seed)
    raise ValueError(f"Unknown clustering method '{method}'.")


def run_with_restarts(
    method: str,
    X: np.ndarray,
    k: int,
    cfg: Optional[ClusteringConfig] = None,
    seeds: Optional[Iterable[int]] = None,
) -> Tuple[ClusterResult, List[ClusterResult]]:
    """Run ``method`` for every seed (seeds 1..n_restarts by default); the run
    with the lowest objective is returned first, followed by all runs."""
    cfg = cfg or ClusteringConfig()
    if method.lower() == "ahc":
        res = run_method(method, X, k, cfg)
        return res, [res]
    seeds = list(seeds) if seeds is not None else list(range(cfg.seed, cfg.seed + cfg.n_restarts))
    runs = [run_method(method, X, k, cfg, seed=int(s)) for s in seeds]
    best = min(runs, key=lambda r: (r.objective if np.isfinite(r.objective) else np.inf))
    return best, runs

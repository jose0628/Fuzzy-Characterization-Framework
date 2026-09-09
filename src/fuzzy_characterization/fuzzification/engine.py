"""The fuzzification engine (Section "The Fuzzification Engine").

The engine converts per-user behavioural indicators into a *fuzzy attribute
vector* ``mu_1 ... mu_n`` with ``mu_i in [0, 1]``.  Each configured fuzzy
feature names an indicator, a membership-function family and a set of
linguistic terms (``low``, ``medium``, ``high``, ``working``, ...).  Indicators
are normalised before fuzzification so that membership degrees are comparable
across features.

Two kinds of features are supported:

* ``scalar``       -- one value per user (posts per month, likes per day, ...);
* ``hour_profile`` -- the user's distribution of events over the 24 hours of the
  day; the membership in *working hours* is the share-weighted mean of the
  per-hour membership, which equals the mean membership over all events.
  Shift-specific windows (night, short, long, split) can be assigned per user.

The transformation is the privacy-preserving step of the framework: the raw
indicator values are replaced by degrees of membership and never leave the
engine.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .alpha_cuts import alpha_cut, alpha_cut_report
from .membership import (
    Complement,
    MembershipFunction,
    Trapezoidal,
    build_membership_function,
    linguistic_partition,
)

HOUR_COLUMNS = [f"hour_share_{h:02d}" for h in range(24)]


# --------------------------------------------------------------------------- #
# Normalisers
# --------------------------------------------------------------------------- #
@dataclass
class Normaliser:
    """Population-level normalisation to [0, 1] fitted once and reused."""

    method: str = "log_minmax"
    lo: float = 0.0
    hi: float = 1.0
    sorted_values: Optional[np.ndarray] = None
    clip_quantiles: Optional[tuple[float, float]] = (0.0, 0.99)

    def fit(self, x: np.ndarray) -> "Normaliser":
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if self.method == "none":
            self.lo, self.hi = 0.0, 1.0
            return self
        if self.method == "quantile":
            self.sorted_values = np.sort(x)
            return self
        v = np.log1p(np.clip(x, 0, None)) if self.method == "log_minmax" else x
        if self.clip_quantiles is not None and len(v) > 0:
            qlo, qhi = self.clip_quantiles
            self.lo = float(np.quantile(v, qlo))
            self.hi = float(np.quantile(v, qhi))
        elif len(v) > 0:
            self.lo, self.hi = float(v.min()), float(v.max())
        if self.hi <= self.lo:
            self.hi = self.lo + 1e-9
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.method == "none":
            return x
        if self.method == "quantile":
            if self.sorted_values is None or len(self.sorted_values) == 0:
                return np.zeros_like(x)
            ranks = np.searchsorted(self.sorted_values, x, side="right")
            return ranks / float(len(self.sorted_values))
        v = np.log1p(np.clip(x, 0, None)) if self.method == "log_minmax" else x
        return np.clip((v - self.lo) / (self.hi - self.lo), 0.0, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"method": self.method, "lo": self.lo, "hi": self.hi}


# --------------------------------------------------------------------------- #
# Feature specification
# --------------------------------------------------------------------------- #
@dataclass
class FuzzyFeature:
    name: str
    indicator: str
    family: str = "trapezoidal"
    kind: str = "scalar"  # scalar | hour_profile
    terms: Dict[str, MembershipFunction] = field(default_factory=dict)
    normalisation: str = "log_minmax"
    alpha_cut: Optional[Dict[str, Any]] = None
    rationale: str = ""
    category: str = ""
    normaliser: Optional[Normaliser] = None

    @classmethod
    def from_spec(cls, spec: Dict[str, Any], default_normalisation: str = "log_minmax") -> "FuzzyFeature":
        spec = copy.deepcopy(spec)
        name = spec["name"]
        family = str(spec.get("family", "trapezoidal")).lower()
        kind = str(spec.get("kind", "scalar"))
        terms_spec = spec.get("terms")
        terms: Dict[str, MembershipFunction] = {}
        if kind == "hour_profile":
            window = spec.get("window", {"a": 6, "b": 8, "c": 16, "d": 18})
            work = _build_hour_mf(window, label="working")
            terms = {"working": work, "non_working": Complement(inner=work, label="non_working")}
        elif isinstance(terms_spec, dict):
            for term, mf_spec in terms_spec.items():
                mf_spec = dict(mf_spec)
                mf_spec.setdefault("family", family)
                terms[term] = build_membership_function(mf_spec, label=term)
        elif isinstance(terms_spec, list):
            terms = linguistic_partition(family, terms_spec)
        else:
            terms = linguistic_partition(family, ["low", "medium", "high"])
        return cls(
            name=name,
            indicator=spec.get("indicator", name),
            family=family,
            kind=kind,
            terms=terms,
            normalisation=spec.get("normalisation", default_normalisation if kind == "scalar" else "none"),
            alpha_cut=spec.get("alpha_cut"),
            rationale=spec.get("rationale", ""),
            category=spec.get("category", ""),
        )

    @property
    def columns(self) -> List[str]:
        return [f"{self.name}__{term}" for term in self.terms]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "indicator": self.indicator,
            "family": self.family,
            "kind": self.kind,
            "category": self.category,
            "normalisation": self.normalisation,
            "normaliser": self.normaliser.to_dict() if self.normaliser else None,
            "alpha_cut": self.alpha_cut,
            "rationale": self.rationale,
            "terms": {t: mf.to_dict() for t, mf in self.terms.items()},
        }


def _build_hour_mf(window: Dict[str, Any] | List[float], label: str = "working") -> MembershipFunction:
    """Build the working-hour membership function from a window specification.

    Accepts ``{"a":6,"b":8,"c":16,"d":18}``, ``[6, 8, 16, 18]``,
    ``{"family": "wraparound_trapezoidal", ...}`` or ``{"windows": [[...], [...]]}``.
    """
    if isinstance(window, (list, tuple)):
        a, b, c, d = map(float, window)
        return Trapezoidal(a=a, b=b, c=c, d=d, label=label)
    window = dict(window)
    if "windows" in window:
        window.setdefault("family", "split_shift")
        return build_membership_function(window, label=label)
    if "family" in window:
        return build_membership_function(window, label=label)
    a, b, c, d = (float(window[k]) for k in ("a", "b", "c", "d"))
    if a <= b <= c <= d:
        return Trapezoidal(a=a, b=b, c=c, d=d, label=label)
    # Window crossing midnight, e.g. (20, 22, 4, 6)
    return build_membership_function(
        {"family": "wraparound_trapezoidal", "a": a, "b": b, "c": c, "d": d}, label=label
    )


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class FuzzificationResult:
    fuzzy_vectors: pd.DataFrame          # users x fuzzy attributes, values in [0, 1]
    normalised_indicators: pd.DataFrame  # users x indicators, normalised to [0, 1]
    alpha_cut_segments: pd.DataFrame     # segment sizes per attribute and alpha level
    feature_specs: List[Dict[str, Any]]
    shift_pattern: Optional[pd.Series] = None

    @property
    def n_features(self) -> int:
        return self.fuzzy_vectors.shape[1]


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class FuzzificationEngine:
    """Maps behavioural indicators to fuzzy attribute vectors."""

    def __init__(
        self,
        feature_specs: Iterable[Dict[str, Any]],
        default_normalisation: str = "log_minmax",
        alpha_levels: Iterable[float] = (0.2, 0.5, 0.8),
        shift_patterns: Optional[Dict[str, Any]] = None,
        min_group_size: int = 20,
    ) -> None:
        self.features: List[FuzzyFeature] = [
            FuzzyFeature.from_spec(s, default_normalisation) for s in feature_specs
        ]
        if not self.features:
            raise ValueError("The fuzzification engine needs at least one fuzzy feature.")
        names = [f.name for f in self.features]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate fuzzy feature names: {names}")
        self.alpha_levels = [float(a) for a in alpha_levels]
        self.shift_patterns: Dict[str, MembershipFunction] = {
            name: _build_hour_mf(spec, label=name) for name, spec in (shift_patterns or {}).items()
        }
        self.min_group_size = int(min_group_size)
        self._fitted = False

    # ------------------------------------------------------------------ #
    @property
    def columns(self) -> List[str]:
        return [c for f in self.features for c in f.columns]

    def feature_of(self, column: str) -> FuzzyFeature:
        name = column.split("__", 1)[0]
        for f in self.features:
            if f.name == name:
                return f
        raise KeyError(column)

    def fit(self, indicators: pd.DataFrame) -> "FuzzificationEngine":
        """Fit the population normalisers on the indicator table."""
        for f in self.features:
            if f.kind == "scalar":
                if f.indicator not in indicators.columns:
                    raise KeyError(
                        f"Fuzzy feature '{f.name}' needs indicator '{f.indicator}', "
                        f"available: {list(indicators.columns)}"
                    )
                f.normaliser = Normaliser(f.normalisation).fit(indicators[f.indicator].to_numpy())
            else:
                missing = [c for c in HOUR_COLUMNS if c not in indicators.columns]
                if missing:
                    raise KeyError(f"hour_profile feature '{f.name}' needs columns {HOUR_COLUMNS[:3]}...")
        self._fitted = True
        return self

    def transform(
        self,
        indicators: pd.DataFrame,
        shift_pattern: Optional[pd.Series] = None,
    ) -> FuzzificationResult:
        """Fuzzify an indicator table.

        ``shift_pattern`` optionally assigns a named shift window to each user
        (index aligned with ``indicators``); those users get the corresponding
        working-hour membership function instead of the default one.
        """
        if not self._fitted:
            self.fit(indicators)
        n = len(indicators)
        fuzzy: Dict[str, np.ndarray] = {}
        normalised: Dict[str, np.ndarray] = {}
        for f in self.features:
            if f.kind == "scalar":
                x = indicators[f.indicator].to_numpy(dtype=float)
                xn = f.normaliser.transform(x) if f.normaliser else x
                normalised[f.indicator] = xn
                for term, mf in f.terms.items():
                    fuzzy[f"{f.name}__{term}"] = np.clip(mf(xn), 0.0, 1.0)
            elif f.kind == "hour_profile":
                shares = indicators[HOUR_COLUMNS].to_numpy(dtype=float)
                hours = np.arange(24, dtype=float) + 0.5  # bin centres
                mu_default = f.terms["working"](hours)
                mu_matrix = np.tile(mu_default, (n, 1))
                if shift_pattern is not None and len(self.shift_patterns):
                    sp = shift_pattern.reindex(indicators.index)
                    for pattern, mf in self.shift_patterns.items():
                        mask = (sp == pattern).to_numpy()
                        if mask.any():
                            mu_matrix[mask] = mf(hours)
                totals = shares.sum(axis=1, keepdims=True)
                w = np.divide(shares, totals, out=np.zeros_like(shares), where=totals > 0)
                working = np.einsum("nh,nh->n", w, mu_matrix)
                fuzzy[f"{f.name}__working"] = np.clip(working, 0.0, 1.0)
                fuzzy[f"{f.name}__non_working"] = np.clip(1.0 - working, 0.0, 1.0)
            else:
                raise ValueError(f"unknown feature kind '{f.kind}'")

            # Alpha-cut filtering: memberships below the level are set to 0 for
            # the selected terms so that only characteristic behaviour remains.
            if f.alpha_cut:
                level = float(f.alpha_cut.get("level", 0.5))
                mode = f.alpha_cut.get("mode", "filter")
                for term in f.alpha_cut.get("terms", list(f.terms)):
                    col = f"{f.name}__{term}"
                    if col in fuzzy and mode == "filter":
                        mu = fuzzy[col]
                        fuzzy[col] = np.where(alpha_cut(mu, level), mu, 0.0)

        fuzzy_df = pd.DataFrame(fuzzy, index=indicators.index)
        norm_df = pd.DataFrame(normalised, index=indicators.index)
        segments = alpha_cut_report(fuzzy_df, self.alpha_levels, self.min_group_size)
        return FuzzificationResult(
            fuzzy_vectors=fuzzy_df,
            normalised_indicators=norm_df,
            alpha_cut_segments=segments,
            feature_specs=[f.to_dict() for f in self.features],
            shift_pattern=shift_pattern,
        )

    def fit_transform(self, indicators: pd.DataFrame, shift_pattern: Optional[pd.Series] = None) -> FuzzificationResult:
        return self.fit(indicators).transform(indicators, shift_pattern)

    # ------------------------------------------------------------------ #
    def describe(self) -> pd.DataFrame:
        """Human-readable table mirroring Table "membership functions"."""
        rows = []
        for f in self.features:
            for term, mf in f.terms.items():
                rows.append(
                    {
                        "category": f.category or f.name,
                        "feature": f.name,
                        "indicator": f.indicator,
                        "term": term,
                        "family": mf.family,
                        "params": mf.params(),
                        "normalisation": f.normalisation,
                        "alpha_cut": (f.alpha_cut or {}).get("level"),
                        "rationale": f.rationale,
                    }
                )
        return pd.DataFrame(rows)

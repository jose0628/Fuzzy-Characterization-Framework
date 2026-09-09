"""Compliance checks reported by the evaluation layer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ComplianceReport:
    identifiers_excluded: List[str] = field(default_factory=list)
    content_columns_excluded: List[str] = field(default_factory=list)
    pseudonymised: bool = False
    consent: Dict[str, Any] = field(default_factory=dict)
    min_group_size: int = 20
    demographic_min_group: Optional[int] = None
    demographic_groups_below_threshold: int = 0
    feature_space_columns: List[str] = field(default_factory=list)
    feature_space_clean: bool = True
    violations: List[str] = field(default_factory=list)

    @property
    def level(self) -> str:
        return "High" if self.feature_space_clean and not self.violations else "Low"

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["level"] = self.level
        return d


FORBIDDEN_TOKENS = frozenset(
    {"name", "fullname", "full_name", "username", "email", "phone", "text", "message", "payload", "body", "content",
     "user_id", "actor_id", "airport", "terminal", "age", "country"}
)


def check_feature_space(columns: Iterable[str], allowed_exceptions: Iterable[str] = ()) -> List[str]:
    """Return feature columns whose name is (or contains as a whole word) an
    identifier, a raw demographic attribute or a content field.

    Names are split on ``__`` (feature/term separator) and ``_``; a column is
    flagged when any of its tokens, or the full feature name, is forbidden.
    Fuzzy attributes derived from an attribute (e.g. ``age__young``) are allowed
    because only a degree of membership in a broad band reaches the vector.
    """
    allowed = set(allowed_exceptions)
    bad = []
    for c in columns:
        if c in allowed:
            continue
        cl = str(c).lower()
        feature = cl.split("__", 1)[0]
        tokens = set(re.split(r"[^a-z0-9]+", cl)) - {""}
        if cl in FORBIDDEN_TOKENS or ("__" not in cl and (feature in FORBIDDEN_TOKENS or tokens & FORBIDDEN_TOKENS)):
            bad.append(c)
    return bad


def demographic_group_sizes(demographics: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    cols = [c for c in columns if c in demographics.columns]
    if not cols:
        return pd.Series(dtype=int)
    return demographics.groupby(cols, dropna=False).size()


def assess_compliance(
    identifiers_excluded: List[str],
    content_columns_excluded: List[str],
    pseudonymised: bool,
    consent: Dict[str, Any],
    feature_columns: Iterable[str],
    demographics: Optional[pd.DataFrame],
    min_group_size: int,
    demographic_columns: Iterable[str] = ("age_band", "region", "employee_category"),
) -> ComplianceReport:
    rep = ComplianceReport(
        identifiers_excluded=list(identifiers_excluded),
        content_columns_excluded=list(content_columns_excluded),
        pseudonymised=pseudonymised,
        consent=dict(consent),
        min_group_size=int(min_group_size),
        feature_space_columns=list(feature_columns),
    )
    bad = check_feature_space(rep.feature_space_columns)
    if bad:
        rep.feature_space_clean = False
        rep.violations.append(f"feature space contains identifier/content-like columns: {bad}")
    if consent.get("consent_available") and consent.get("consent_coverage") is not None:
        if consent["consent_coverage"] < 1.0 and consent.get("users_after", 0) == consent.get("users_before", 0):
            rep.violations.append("users without consent were not excluded")
    if demographics is not None and len(demographics):
        sizes = demographic_group_sizes(demographics, demographic_columns)
        if len(sizes):
            rep.demographic_min_group = int(sizes.min())
            rep.demographic_groups_below_threshold = int((sizes < min_group_size).sum())
    return rep


def small_groups(sizes: pd.Series | np.ndarray, min_group_size: int) -> List[int]:
    """Indices of groups (clusters / segments) that are too small to report."""
    s = np.asarray(sizes)
    return [int(i) for i in np.where(s < min_group_size)[0]]

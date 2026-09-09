"""Pseudonymisation, identifier removal, consent filtering and generalisation.

These steps run *before* any behavioural analysis: direct identifiers are
dropped, user keys are replaced by salted hashes, users without consent are
excluded and demographic attributes are generalised into broad bins.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from ..config import ComplianceConfig
from ..data.schema import Datasets


class Pseudonymiser:
    """Deterministic, salted pseudonyms for user identifiers."""

    def __init__(self, salt: str = "fcf", length: int = 12) -> None:
        self.salt = salt
        self.length = length
        self._map: dict = {}

    def pseudonym(self, value: object) -> str:
        key = str(value)
        if key not in self._map:
            digest = hashlib.sha256(f"{self.salt}:{key}".encode("utf-8")).hexdigest()
            self._map[key] = "p" + digest[: self.length]
        return self._map[key]

    def apply(self, df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        df = df.copy()
        for col in columns:
            if col in df.columns:
                df[col] = df[col].map(lambda v: self.pseudonym(v) if pd.notna(v) else v)
        return df

    def apply_datasets(self, data: Datasets) -> Datasets:
        uid = data.user_id_col
        cols = {uid, "author_id"}
        new = Datasets(
            users=self.apply(data.users, cols),
            events=self.apply(data.events, cols),
            posts=self.apply(data.posts, cols),
            comments=self.apply(data.comments, cols),
            reactions=self.apply(data.reactions, cols),
            streams=data.streams.copy(),
            stream_permissions=self.apply(data.stream_permissions, cols),
            group_memberships=self.apply(data.group_memberships, cols),
            consent=self.apply(data.consent, cols) if data.consent is not None else None,
            user_id_col=uid,
            metadata=dict(data.metadata, pseudonymised=True),
        )
        return new


def drop_identifiers(df: pd.DataFrame, identifiers: Iterable[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Drop direct identifier columns; return the frame and the dropped names."""
    present = [c for c in identifiers if c in df.columns]
    return df.drop(columns=present), present


def apply_consent(data: Datasets, require_consent: bool = True) -> Tuple[Datasets, dict]:
    """Restrict every table to users with recorded consent."""
    uid = data.user_id_col
    n_users = data.users[uid].nunique()
    if data.consent is None or not require_consent:
        return data, {"consent_available": data.consent is not None, "users_before": n_users,
                      "users_after": n_users, "consent_coverage": None}
    consented = set(data.consent.loc[data.consent["consent"].astype(int) == 1, uid])

    def keep(df: pd.DataFrame, col: str) -> pd.DataFrame:
        return df[df[col].isin(consented)].reset_index(drop=True) if col in df.columns else df

    new = Datasets(
        users=keep(data.users, uid),
        events=keep(data.events, uid),
        posts=keep(data.posts, "author_id"),
        comments=keep(data.comments, "author_id"),
        reactions=keep(data.reactions, uid),
        streams=data.streams,
        stream_permissions=keep(data.stream_permissions, uid),
        group_memberships=keep(data.group_memberships, uid),
        consent=data.consent,
        user_id_col=uid,
        metadata=dict(data.metadata),
    )
    n_after = new.users[uid].nunique()
    return new, {
        "consent_available": True,
        "users_before": int(n_users),
        "users_after": int(n_after),
        "consent_coverage": float(n_after / n_users) if n_users else 0.0,
    }


def generalise_demographics(users: pd.DataFrame, cfg: ComplianceConfig) -> pd.DataFrame:
    """Compliant demographic bins: age band, region, employee category, shift pattern.

    Exact age, airport and terminal are removed; country is generalised to a
    region.  The output keeps the user id column plus broad categorical bins.
    """
    df = users.copy()
    out = pd.DataFrame(index=df.index)
    if "user_id" in df.columns:
        out["user_id"] = df["user_id"]
    if "age" in df.columns:
        bins = list(cfg.age_bins)
        labels = [f"{bins[i]}-{bins[i + 1] - 1}" for i in range(len(bins) - 1)]
        out["age_band"] = pd.cut(df["age"], bins=bins, labels=labels, right=False).astype(str)
        # Band midpoint: the only numeric trace of age that reaches the fuzzifier.
        mid = pd.cut(df["age"], bins=bins, right=False).apply(
            lambda iv: (iv.left + iv.right) / 2.0 if pd.notna(iv) else float("nan")
        )
        out["age_band_midpoint"] = mid.astype(float)
    if "country" in df.columns:
        out["region"] = df["country"].map(_region_of).fillna("other")
    for col in ("division", "department", "employee_category", "shift_pattern", "preferred_language"):
        if col in df.columns:
            out[col] = df[col]
    return out


_EU = {"CH", "DE", "FR", "IT", "ES", "GB", "UK", "NL", "BE", "AT", "PT", "SE", "DK", "NO", "FI", "PL", "GR", "IE"}
_NA = {"US", "CA", "MX"}


def _region_of(country: object) -> str:
    c = str(country).upper()
    if c in _EU:
        return "europe"
    if c in _NA:
        return "north_america"
    return "other"

"""Loading the source tables and the legacy ``event_type_counts`` parser."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ..config import DataConfig
from .schema import Datasets


def _read_csv(path: Path, parse_dates: Optional[list] = None) -> pd.DataFrame:
    """Read ``path`` or its gzip-compressed sibling ``path.gz``."""
    candidates = [path, path.with_name(path.name + ".gz")]
    for cand in candidates:
        if cand.exists():
            return pd.read_csv(cand, parse_dates=parse_dates)
    raise FileNotFoundError(f"None of {[str(c) for c in candidates]} exists.")


def load_datasets(cfg: DataConfig, root: Optional[str | Path] = None) -> Datasets:
    """Load every table referenced in the data configuration."""
    base = Path(root) if root is not None else Path(cfg.root)
    users = _read_csv(base / cfg.users_file)
    events = _read_csv(base / cfg.events_file, parse_dates=["timestamp"])
    posts = _read_csv(base / cfg.posts_file, parse_dates=["created_at"])
    comments = _read_csv(base / cfg.comments_file, parse_dates=["created_at"])
    reactions = _read_csv(base / cfg.reactions_file, parse_dates=["created_at"])
    streams = _read_csv(base / cfg.streams_file)
    perms = _read_csv(base / cfg.stream_permissions_file)
    groups = _read_csv(base / cfg.group_memberships_file)
    consent = None
    if cfg.consent_file:
        try:
            consent = _read_csv(base / cfg.consent_file)
        except FileNotFoundError:
            consent = None
    uid = cfg.user_id_col
    for name, df in (("users", users), ("events", events)):
        if uid not in df.columns:
            raise KeyError(f"Table '{name}' has no user id column '{uid}'.")
    period_start = pd.Timestamp(cfg.period_start) if cfg.period_start else events["timestamp"].min()
    period_end = pd.Timestamp(cfg.period_end) if cfg.period_end else events["timestamp"].max()
    return Datasets(
        users=users, events=events, posts=posts, comments=comments, reactions=reactions,
        streams=streams, stream_permissions=perms, group_memberships=groups, consent=consent,
        user_id_col=uid,
        metadata={"root": str(base), "period_start": str(period_start), "period_end": str(period_end)},
    )


# --------------------------------------------------------------------------- #
# Legacy support: expand a serialised ``event_type_counts`` column
# --------------------------------------------------------------------------- #
def parse_counts_cell(x: Any) -> Dict[str, Any]:
    """Parse a dict-like cell (JSON, ClickHouse-style ``{a:1,b:2}`` or Python literal)."""
    if x is None:
        return {}
    s = str(x).strip()
    if s == "" or s.lower() in {"null", "none", "nan"}:
        return {}
    for candidate in (s, re.sub(r"([{,]\s*)([A-Za-z0-9_\-\.]+)\s*:", r'\1"\2":', s)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def unpack_event_type_counts(df: pd.DataFrame, column: str = "event_type_counts") -> pd.DataFrame:
    """Expand a serialised per-user event-count column into numeric columns."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Found columns: {list(df.columns)}")
    parsed = df[column].apply(parse_counts_cell)
    expanded = pd.json_normalize(parsed)
    expanded = expanded.apply(pd.to_numeric, errors="coerce").fillna(0).astype("int64")
    expanded.index = df.index
    return pd.concat([df.drop(columns=[column]), expanded], axis=1)

"""Stream permissions (Algorithm "Checking user permissions for streams").

A stream that does not appear in the permissions table has no access
restriction, so every user can reach it.  For restricted streams the user's
permission row decides.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def stream_permission_matrix(
    permissions: pd.DataFrame,
    user_col: str = "user_id",
    stream_col: str = "stream_id",
    value_col: str = "can_access",
) -> pd.DataFrame:
    """Users x restricted-streams matrix with 1 for access and 0 otherwise."""
    if permissions.empty:
        return pd.DataFrame()
    return permissions.pivot_table(index=user_col, columns=stream_col, values=value_col, aggfunc="max", fill_value=0)


def accessible_streams(
    user: object,
    streams: Iterable[object],
    permission_matrix: pd.DataFrame,
) -> List[object]:
    """Verified list of the streams (out of ``streams``) that ``user`` can access."""
    out: List[object] = []
    for s in streams:
        if not permission_matrix.empty and s in permission_matrix.columns:
            if user in permission_matrix.index and permission_matrix.at[user, s] == 1:
                out.append(s)
        else:
            out.append(s)  # no restriction recorded for this stream
    return out


def accessible_stream_counts(
    users: Iterable[object],
    streams: pd.DataFrame,
    permission_matrix: pd.DataFrame,
    stream_col: str = "stream_id",
    type_col: Optional[str] = "stream_type",
) -> pd.DataFrame:
    """Per user: number of accessible streams and the share that are broadcast streams."""
    all_streams = list(streams[stream_col])
    stype = dict(zip(streams[stream_col], streams[type_col])) if type_col and type_col in streams.columns else {}
    rows = []
    for u in users:
        acc = accessible_streams(u, all_streams, permission_matrix)
        n_broadcast = sum(1 for s in acc if stype.get(s) == "broadcast")
        rows.append(
            {
                "user_id": u,
                "accessible_streams": len(acc),
                "broadcast_stream_share": (n_broadcast / len(acc)) if acc else 0.0,
            }
        )
    return pd.DataFrame(rows).set_index("user_id")

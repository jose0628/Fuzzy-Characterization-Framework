"""Assemble the heuristic attribute block of the characterized user vector."""

from __future__ import annotations

import pandas as pd

from ..data.schema import Datasets
from .engagement import flag_engaging_posts
from .hierarchy import hierarchy_score, role_category
from .permissions import accessible_stream_counts, stream_permission_matrix


def build_heuristic_attributes(data: Datasets, user_ids: pd.Index) -> pd.DataFrame:
    """Per-user heuristic attributes: hierarchy score, role category, group
    memberships, accessible streams, broadcast share and post interactions."""
    uid = data.user_id_col
    users = data.users.set_index(uid).reindex(user_ids)
    out = pd.DataFrame(index=pd.Index(user_ids, name=uid))
    out["hierarchy_score"] = hierarchy_score(users.reset_index(), "job_title").to_numpy()
    out["role_category"] = role_category(out["hierarchy_score"]).to_numpy()
    groups = data.group_memberships.groupby(uid).size() if len(data.group_memberships) else pd.Series(dtype=int)
    out["group_memberships"] = groups.reindex(user_ids).fillna(0).astype(int).to_numpy()

    perm = stream_permission_matrix(data.stream_permissions, user_col=uid)
    acc = accessible_stream_counts(user_ids, data.streams, perm)
    out["accessible_streams"] = acc["accessible_streams"].reindex(user_ids).fillna(0).to_numpy()
    out["broadcast_stream_share"] = acc["broadcast_stream_share"].reindex(user_ids).fillna(0.0).to_numpy()

    posts = flag_engaging_posts(data.posts) if len(data.posts) else data.posts
    authored = posts.groupby("author_id").size() if len(posts) else pd.Series(dtype=int)
    engaging = posts.groupby("author_id")["engaging"].sum() if len(posts) else pd.Series(dtype=int)
    out["posts_authored"] = authored.reindex(user_ids).fillna(0).astype(int).to_numpy()
    out["engaging_posts_authored"] = engaging.reindex(user_ids).fillna(0).astype(int).to_numpy()
    interactions = (
        (data.comments.groupby("author_id").size() if len(data.comments) else pd.Series(dtype=int))
        .add(data.reactions.groupby(uid).size() if len(data.reactions) else pd.Series(dtype=int), fill_value=0)
        .add(authored, fill_value=0)
    )
    out["post_interactions"] = interactions.reindex(user_ids).fillna(0).astype(int).to_numpy()
    return out

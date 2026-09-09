"""Post engagement and similarity (Algorithms "engaging post", "post similarity").

These heuristics operate on post *metadata* (comment, like and mention
counts, presence of a video) and never on content.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def flag_engaging_posts(posts: pd.DataFrame, stream_col: str = "stream_id") -> pd.DataFrame:
    """Add an ``engaging`` column (1/0) by comparing each post with its stream's means.

    A post is engaging when its comment count exceeds the stream mean and it
    either has a video or its like count also exceeds the stream mean.
    """
    df = posts.copy()
    df["engaging"] = 0
    for stream, grp in df.groupby(stream_col):
        c_mean = round(grp["comment_count"].mean())
        l_mean = round(grp["like_count"].mean())
        has_video = grp["has_video"].astype(bool) if "has_video" in grp.columns else pd.Series(False, index=grp.index)
        cond = (grp["comment_count"] > c_mean) & (has_video | (grp["like_count"] > l_mean))
        df.loc[grp.index[cond.to_numpy()], "engaging"] = 1
    return df


def post_similarity_matrix(posts: pd.DataFrame, post_col: str = "post_id") -> pd.DataFrame:
    """Cosine item-item similarity on post metadata features."""
    cols = [c for c in ("comment_count", "like_count", "comment_like_count", "mention_count", "has_video", "engaging")
            if c in posts.columns]
    if "engaging" not in posts.columns:
        posts = flag_engaging_posts(posts)
        cols.append("engaging")
    X = posts[cols].astype(float).to_numpy()
    sim = cosine_similarity(X) if len(X) else np.zeros((0, 0))
    ids = posts[post_col].tolist()
    return pd.DataFrame(sim, index=ids, columns=ids)


def post_activity_frame(
    posts: pd.DataFrame,
    comments: pd.DataFrame,
    reactions: pd.DataFrame,
    comment_likes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Users x posts interaction counts (Algorithm "user post activity data frame").

    Counts posts created, comments made, post likes and (optionally) comment
    likes per user and post.
    """
    parts: List[pd.DataFrame] = []
    if len(posts):
        parts.append(posts[["author_id", "post_id"]].rename(columns={"author_id": "user_id"}))
    if len(comments):
        parts.append(comments[["author_id", "post_id"]].rename(columns={"author_id": "user_id"}))
    if len(reactions):
        parts.append(reactions[["user_id", "post_id"]])
    if comment_likes is not None and len(comment_likes):
        parts.append(comment_likes[["user_id", "post_id"]])
    if not parts:
        return pd.DataFrame()
    long = pd.concat(parts, ignore_index=True)
    return long.pivot_table(index="user_id", columns="post_id", aggfunc="size", fill_value=0)

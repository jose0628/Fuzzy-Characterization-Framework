"""Column names and dataset container shared by the whole framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

import pandas as pd


class EventType(str, Enum):
    """Interaction event types recorded by the platform API (metadata only)."""

    POST_CREATED = "post.created"
    COMMENT_CREATED = "comment.created"
    REACTION_CREATED = "reaction.created"
    CHAT_MESSAGE_SENT = "chat.message.sent"
    LOGIN = "login"
    STREAM_VIEWED = "stream.viewed"
    TOPIC_VIEWED = "topic.viewed"
    POST_READ = "post.read"
    SEARCH_PERFORMED = "search.performed"


EVENT_TYPES = [e.value for e in EventType]

# Event types that count as "topic and stream access" (navigation).
NAVIGATION_EVENTS = {EventType.STREAM_VIEWED.value, EventType.TOPIC_VIEWED.value}

# Columns of each source table (documented in data/README.md)
USER_COLUMNS = [
    "user_id", "full_name", "username", "job_title", "department", "division",
    "country", "airport", "terminal", "employee_category", "shift_pattern",
    "age", "preferred_language",
]
EVENT_COLUMNS = [
    "event_id", "user_id", "timestamp", "event_type", "normalised_path",
    "method", "client", "object_type", "object_id",
]
POST_COLUMNS = [
    "post_id", "author_id", "stream_id", "created_at", "has_video", "mention_count",
    "comment_count", "like_count", "comment_like_count",
]
COMMENT_COLUMNS = ["comment_id", "post_id", "author_id", "created_at", "like_count"]
REACTION_COLUMNS = ["reaction_id", "post_id", "user_id", "created_at", "reaction_type"]
STREAM_COLUMNS = ["stream_id", "stream_name", "stream_type", "restricted"]
STREAM_PERMISSION_COLUMNS = ["user_id", "stream_id", "can_access"]
GROUP_MEMBERSHIP_COLUMNS = ["user_id", "group_id", "group_name"]
CONSENT_COLUMNS = ["user_id", "consent", "jurisdiction"]


@dataclass
class Datasets:
    """All source tables of a case, keyed by the user identifier column."""

    users: pd.DataFrame
    events: pd.DataFrame
    posts: pd.DataFrame
    comments: pd.DataFrame
    reactions: pd.DataFrame
    streams: pd.DataFrame
    stream_permissions: pd.DataFrame
    group_memberships: pd.DataFrame
    consent: Optional[pd.DataFrame] = None
    user_id_col: str = "user_id"
    metadata: Dict[str, object] = field(default_factory=dict)

    def tables(self) -> Dict[str, pd.DataFrame]:
        out = {
            "users": self.users,
            "events": self.events,
            "posts": self.posts,
            "comments": self.comments,
            "reactions": self.reactions,
            "streams": self.streams,
            "stream_permissions": self.stream_permissions,
            "group_memberships": self.group_memberships,
        }
        if self.consent is not None:
            out["consent"] = self.consent
        return out

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"table": k, "rows": len(v), "columns": len(v.columns)} for k, v in self.tables().items()]
        )

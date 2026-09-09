"""Behavioural indicators (Table "behavioural indicators" of the implementation chapter).

All indicators are computed **per user over the observation period** from
interaction metadata only: the fact that an event occurred, when it occurred
and against which kind of object.  Message, comment and chat text and the
identity of communication partners are never read.

Two families are produced:

* *interaction intensity* -- how much a user acts (posting, commenting,
  reactions, chat, logins by channel, topic/stream access);
* *temporal structure*    -- when a user acts (working vs non-working hours,
  daily and weekly rhythm, per-hour and per-weekday activity profiles).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import IndicatorConfig
from .schema import NAVIGATION_EVENTS, Datasets, EventType

HOUR_COLUMNS = [f"hour_share_{h:02d}" for h in range(24)]
DOW_COLUMNS = [f"dow_share_{d}" for d in range(7)]

INDICATOR_DESCRIPTIONS: Dict[str, str] = {
    "posts_per_month": "Posts created by the user (count per month)",
    "comments_per_month": "Comments written by the user (count per month)",
    "reactions_per_month": "Likes / reactions given to posts or comments (count per month)",
    "chat_events_per_month": "Chat messages sent, from interaction metadata only (count per month)",
    "logins_per_month": "Login events (count per month)",
    "logins_web_per_month": "Login events from the web client (count per month)",
    "logins_mobile_per_month": "Login events from the mobile client (count per month)",
    "mobile_login_share": "Share of logins from the mobile client",
    "topic_accesses_per_month": "Visits to topics and streams, by normalised API path (count per month)",
    "post_reads_per_month": "Posts opened / read (count per month)",
    "searches_per_month": "Search actions (count per month)",
    "total_events_per_month": "All platform events (count per month)",
    "active_days_share": "Share of days in the period with at least one event",
    "working_hours_share": "Share of events inside the crisp working-hour window",
    "non_working_hours_share": "Share of events outside the crisp working-hour window",
    "weekend_share": "Share of events on Saturday or Sunday",
    "weekday_entropy": "Normalised entropy of activity across weekdays (1 = evenly spread)",
    "comment_working_hours_share": "Share of the user's comments written in working hours",
    "authoring_share": "Share of events that create content (posts, comments)",
    "reactive_share": "Share of events that react to content (reactions)",
    "consumption_share": "Share of events that consume content (reads, views, searches)",
}


def _per_month(count: pd.Series, months: float) -> pd.Series:
    return count.astype(float) / max(months, 1e-9)


def build_behavioural_indicators(
    data: Datasets,
    cfg: Optional[IndicatorConfig] = None,
    user_ids: Optional[pd.Index] = None,
) -> pd.DataFrame:
    """Aggregate the event log into one indicator row per user.

    ``user_ids`` restricts (and orders) the output; users without events get
    zero activity.
    """
    cfg = cfg or IndicatorConfig()
    uid = data.user_id_col
    ev = data.events.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])
    start = pd.Timestamp(data.metadata.get("period_start", ev["timestamp"].min()))
    end = pd.Timestamp(data.metadata.get("period_end", ev["timestamp"].max()))
    ev = ev[(ev["timestamp"] >= start) & (ev["timestamp"] <= end)]
    n_days = max((end - start).days + 1, 1)
    months = n_days / cfg.days_per_month

    if user_ids is None:
        user_ids = pd.Index(data.users[uid].unique(), name=uid)
    else:
        user_ids = pd.Index(user_ids, name=uid)
    ev = ev[ev[uid].isin(user_ids)]

    ev["hour"] = ev["timestamp"].dt.hour + ev["timestamp"].dt.minute / 60.0
    ev["dow"] = ev["timestamp"].dt.dayofweek
    ev["date"] = ev["timestamp"].dt.date

    # Crisp working-hour window (plateau of the trapezoid, b..c)
    b, c = float(cfg.working_hours["b"]), float(cfg.working_hours["c"])
    ev["in_working_hours"] = ((ev["hour"] >= b) & (ev["hour"] < c)).astype(float)

    g = ev.groupby(uid)
    counts = ev.pivot_table(index=uid, columns="event_type", values="event_id", aggfunc="count", fill_value=0)
    counts = counts.reindex(user_ids, fill_value=0)

    def col(event_type: str) -> pd.Series:
        return counts[event_type] if event_type in counts.columns else pd.Series(0, index=user_ids)

    ind = pd.DataFrame(index=user_ids)
    ind["posts_per_month"] = _per_month(col(EventType.POST_CREATED.value), months)
    ind["comments_per_month"] = _per_month(col(EventType.COMMENT_CREATED.value), months)
    ind["reactions_per_month"] = _per_month(col(EventType.REACTION_CREATED.value), months)
    ind["chat_events_per_month"] = _per_month(col(EventType.CHAT_MESSAGE_SENT.value), months)
    logins = ev[ev["event_type"] == EventType.LOGIN.value]
    login_by_client = logins.pivot_table(index=uid, columns="client", values="event_id", aggfunc="count", fill_value=0)
    login_by_client = login_by_client.reindex(user_ids, fill_value=0)
    web = login_by_client["web"] if "web" in login_by_client.columns else pd.Series(0, index=user_ids)
    mobile = login_by_client["mobile"] if "mobile" in login_by_client.columns else pd.Series(0, index=user_ids)
    ind["logins_per_month"] = _per_month(col(EventType.LOGIN.value), months)
    ind["logins_web_per_month"] = _per_month(web, months)
    ind["logins_mobile_per_month"] = _per_month(mobile, months)
    total_logins = (web + mobile).astype(float)
    ind["mobile_login_share"] = np.divide(mobile, total_logins, out=np.zeros(len(user_ids)), where=total_logins > 0)
    nav = sum(col(e) for e in NAVIGATION_EVENTS)
    ind["topic_accesses_per_month"] = _per_month(nav, months)
    ind["post_reads_per_month"] = _per_month(col(EventType.POST_READ.value), months)
    ind["searches_per_month"] = _per_month(col(EventType.SEARCH_PERFORMED.value), months)
    total = counts.sum(axis=1).astype(float)
    ind["total_events_per_month"] = _per_month(total, months)

    # Temporal structure
    active_days = g["date"].nunique().reindex(user_ids, fill_value=0)
    ind["active_days_share"] = active_days / float(n_days)
    wh = g["in_working_hours"].mean().reindex(user_ids).fillna(0.0)
    ind["working_hours_share"] = wh
    ind["non_working_hours_share"] = 1.0 - wh
    weekend = g["dow"].apply(lambda s: float((s >= 5).mean())).reindex(user_ids).fillna(0.0)
    ind["weekend_share"] = weekend

    dow_counts = ev.pivot_table(index=uid, columns="dow", values="event_id", aggfunc="count", fill_value=0)
    dow_counts = dow_counts.reindex(index=user_ids, columns=range(7), fill_value=0).astype(float)
    dow_tot = dow_counts.sum(axis=1).replace(0, np.nan)
    dow_share = dow_counts.div(dow_tot, axis=0).fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(dow_share * np.log(dow_share.where(dow_share > 0, 1.0))).sum(axis=1) / np.log(7)
    ind["weekday_entropy"] = ent.fillna(0.0)
    for d in range(7):
        ind[f"dow_share_{d}"] = dow_share[d]

    hour_bins = np.clip(ev["hour"].astype(int), 0, 23)
    hour_counts = pd.crosstab(ev[uid], hour_bins).reindex(index=user_ids, columns=range(24), fill_value=0).astype(float)
    hour_tot = hour_counts.sum(axis=1).replace(0, np.nan)
    hour_share = hour_counts.div(hour_tot, axis=0).fillna(0.0)
    for h in range(24):
        ind[f"hour_share_{h:02d}"] = hour_share[h]

    comments = ev[ev["event_type"] == EventType.COMMENT_CREATED.value]
    ind["comment_working_hours_share"] = (
        comments.groupby(uid)["in_working_hours"].mean().reindex(user_ids).fillna(0.0)
    )

    # Composition of activity (authoring / reactive / consumption)
    authoring = col(EventType.POST_CREATED.value) + col(EventType.COMMENT_CREATED.value)
    reactive = col(EventType.REACTION_CREATED.value)
    consumption = nav + col(EventType.POST_READ.value) + col(EventType.SEARCH_PERFORMED.value)
    safe_total = total.replace(0, np.nan)
    ind["authoring_share"] = (authoring / safe_total).fillna(0.0)
    ind["reactive_share"] = (reactive / safe_total).fillna(0.0)
    ind["consumption_share"] = (consumption / safe_total).fillna(0.0)

    ind = ind.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    ind.attrs["observation_days"] = n_days
    ind.attrs["observation_months"] = months
    return ind


def indicator_table(ind: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics of the scalar indicators (for reports)."""
    scalar = [c for c in ind.columns if not c.startswith(("hour_share_", "dow_share_"))]
    desc = ind[scalar].describe().T
    desc["description"] = [INDICATOR_DESCRIPTIONS.get(c, "") for c in desc.index]
    return desc

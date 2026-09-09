"""Synthetic, privacy-safe demo data shaped like the retail case.

The generator draws users from four latent behavioural archetypes whose
monthly activity rates mirror Table "Average behavioural feature values per
cluster" of the implementation chapter (high communicators, reactive
consumers, operational browsers, low-engagement users).  Every table of the
real case is reproduced *without any content*: posts, comments and chats are
metadata rows only.  Names and usernames are fabricated so that the
compliance layer has something to exclude.

The data are coherent across tables: posts/comments/reactions are derived
from the event log, aggregates on posts match the comment and reaction
tables, and stream permissions constrain where users can post.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .schema import EventType


@dataclass
class Archetype:
    name: str
    share: float
    posts: float
    comments: float
    reactions: float
    chat: float
    topic_access: float
    post_reads: float
    searches: float
    logins: float
    working_hours_share: float
    active_day_prob: float
    manager_prob: float = 0.05
    mobile_shift: float = 0.0      # added to the mobile-login share of the category
    weekend_factor: float = 1.0    # probability that a shift-based user is active on a weekend day
    notes: str = ""


# Monthly rates from the thesis cluster table, complemented with plausible
# navigation, reading, search and login volumes.
ARCHETYPES: Dict[str, Archetype] = {
    "high_communicators": Archetype(
        "High Communicators", 0.10, 12.4, 18.9, 64.7, 310.5, 145.3, 120.0, 20.0, 40.0, 0.582, 0.85,
        manager_prob=0.35, mobile_shift=-0.30, weekend_factor=0.7,
        notes="initiate content, discuss, chat; active in and out of working hours",
    ),
    "reactive_consumers": Archetype(
        "Reactive Consumers", 0.30, 1.1, 3.4, 42.3, 52.8, 89.7, 100.0, 8.0, 25.0, 0.637, 0.65,
        manager_prob=0.05, mobile_shift=0.0, weekend_factor=0.85,
        notes="react to posts, occasionally comment, rarely publish",
    ),
    "operational_browsers": Archetype(
        "Operational Browsers", 0.35, 0.6, 1.2, 15.6, 21.4, 210.4, 150.0, 40.0, 30.0, 0.814, 0.70,
        manager_prob=0.05, mobile_shift=0.10, weekend_factor=1.0,
        notes="consult operational content during working hours",
    ),
    "low_engagement": Archetype(
        "Low-Engagement Users", 0.25, 0.1, 0.2, 2.8, 3.6, 28.3, 15.0, 2.0, 4.0, 0.729, 0.20,
        manager_prob=0.02, mobile_shift=-0.10, weekend_factor=0.4,
        notes="log in rarely, sporadic navigation",
    ),
}

FIRST_NAMES = ["Alex", "Sam", "Jordan", "Taylor", "Casey", "Robin", "Morgan", "Jamie", "Noor", "Luca",
               "Elena", "Mateo", "Sofia", "Yusuf", "Aiko", "Priya", "Omar", "Lena", "Ivan", "Chloe"]
LAST_NAMES = ["Muller", "Rossi", "Martin", "Garcia", "Smith", "Fischer", "Novak", "Silva", "Kaur", "Chen",
              "Weber", "Dubois", "Moreau", "Schneider", "Bianchi", "Lopez", "Brown", "Meier", "Costa", "Keller"]
DEPARTMENTS = ["Store Operations", "Customer Service", "Logistics", "Communications", "Human Resources", "Management"]
DEPARTMENT_WEIGHTS = [0.50, 0.22, 0.12, 0.05, 0.05, 0.06]
FRONTLINE_TITLES = ["Sales Associate", "Cashier", "Senior Sales Associate", "Stock Assistant", "Customer Service Agent",
                    "Beauty Advisor", "Logistics Associate", "Store Assistant"]
SUPERVISORY_TITLES = ["Shift Supervisor", "Team Lead", "Senior Customer Service Specialist", "Logistics Coordinator"]
MANAGEMENT_TITLES = ["Store Manager", "Assistant Manager", "General Manager", "Regional Manager",
                     "Communications Manager", "HR Manager", "Head of Retail Operations", "Director of Operations"]
COUNTRIES = {"CH": 0.30, "DE": 0.15, "FR": 0.12, "IT": 0.10, "ES": 0.08, "GB": 0.10, "US": 0.15}
AIRPORTS = {"CH": ["ZRH", "GVA", "BSL"], "DE": ["FRA", "MUC"], "FR": ["CDG", "NCE"], "IT": ["FCO", "MXP"],
            "ES": ["MAD", "BCN"], "GB": ["LHR", "LGW"], "US": ["JFK", "LAX", "MIA"]}
LANGUAGES = {"CH": ["de", "fr", "it"], "DE": ["de"], "FR": ["fr"], "IT": ["it"], "ES": ["es"], "GB": ["en"], "US": ["en", "es"]}
SHIFT_PATTERNS = {"day": 0.45, "night": 0.15, "short": 0.15, "long": 0.15, "split": 0.10}
# Working windows (plateau start, plateau end) per pattern for sampling timestamps.
SHIFT_WINDOWS = {"office": (8.0, 16.0), "day": (8.0, 16.0), "night": (22.0, 30.0), "short": (9.0, 13.0),
                 "long": (7.0, 19.0), "split": None}

REACTION_TYPES = ["like", "celebrate", "support", "insightful"]


@dataclass
class SyntheticOptions:
    n_users: int = 400
    days: int = 42
    start: str = "2025-03-03"
    seed: int = 7
    consent_rate_us: float = 0.7
    n_discussion_streams: int = 8
    n_broadcast_streams: int = 4
    n_interest_groups: int = 6
    archetype_shares: Optional[Dict[str, float]] = None
    rate_noise_sigma: float = 0.35
    extra: Dict[str, object] = field(default_factory=dict)


def _choice(rng: np.random.Generator, options: Dict[str, float]) -> str:
    keys = list(options)
    p = np.array([options[k] for k in keys], dtype=float)
    return str(rng.choice(keys, p=p / p.sum()))


def _sample_hours(rng: np.random.Generator, n: int, pattern: str, working_share: float) -> np.ndarray:
    """Hours of day: with probability ``working_share`` inside the shift window."""
    inside = rng.random(n) < working_share
    hours = np.empty(n)
    if pattern == "split":
        w1, w2 = (8.0, 11.5), (14.5, 18.0)
        pick = rng.random(n) < 0.5
        hours_in = np.where(pick, rng.uniform(*w1, n), rng.uniform(*w2, n))
        # outside: before 7, the pause, or after 19
        segs = np.array([(0.0, 7.0), (12.0, 14.0), (19.0, 24.0)])
        seg = rng.choice(3, size=n, p=np.array([7.0, 2.0, 5.0]) / 14.0)
        hours_out = rng.uniform(segs[seg, 0], segs[seg, 1])
    else:
        lo, hi = SHIFT_WINDOWS.get(pattern, (8.0, 16.0))
        hours_in = rng.uniform(lo - 0.5, hi + 0.5, n)
        gap = 24.0 - (hi - lo)
        hours_out = (hi + 0.5 + rng.uniform(0, max(gap - 1.0, 0.5), n))
    hours[inside] = hours_in[inside]
    hours[~inside] = hours_out[~inside]
    return np.mod(hours, 24.0)


def generate_synthetic_datasets(options: Optional[SyntheticOptions] = None) -> Dict[str, pd.DataFrame]:
    """Generate every table of the case. Returns a dict of data frames."""
    opt = options or SyntheticOptions()
    rng = np.random.default_rng(opt.seed)
    start = pd.Timestamp(opt.start)
    months = opt.days / 30.4375

    # ---------------------------------------------------------------- users
    shares = opt.archetype_shares or {k: v.share for k, v in ARCHETYPES.items()}
    arche_keys = list(shares)
    p = np.array([shares[k] for k in arche_keys], dtype=float)
    p /= p.sum()
    archetype_of = rng.choice(arche_keys, size=opt.n_users, p=p)

    users: List[dict] = []
    for i in range(opt.n_users):
        a = ARCHETYPES[archetype_of[i]]
        country = _choice(rng, COUNTRIES)
        dept = str(rng.choice(DEPARTMENTS, p=DEPARTMENT_WEIGHTS))
        r = rng.random()
        if dept in ("Management",) or r < a.manager_prob:
            title = str(rng.choice(MANAGEMENT_TITLES))
        elif r < a.manager_prob + 0.15:
            title = str(rng.choice(SUPERVISORY_TITLES))
        else:
            title = str(rng.choice(FRONTLINE_TITLES))
        office = dept in ("Communications", "Human Resources", "Management") or rng.random() < 0.10
        if office:
            category, pattern = "office-based", "office"
        else:
            category, pattern = "shift-based", _choice(rng, SHIFT_PATTERNS)
        first, last = str(rng.choice(FIRST_NAMES)), str(rng.choice(LAST_NAMES))
        users.append(
            {
                "user_id": f"u{i:05d}",
                "full_name": f"{first} {last}",
                "username": f"{first[0].lower()}{last.lower()}{i % 97:02d}",
                "job_title": title,
                "department": dept,
                "division": "US Retail" if country == "US" else "Europe Retail",
                "country": country,
                "airport": str(rng.choice(AIRPORTS[country])),
                "terminal": f"T{int(rng.integers(1, 4))}",
                "employee_category": category,
                "shift_pattern": pattern,
                "age": int(np.clip(rng.normal(36, 11), 18, 64)),
                "preferred_language": str(rng.choice(LANGUAGES[country])),
                "_archetype": archetype_of[i],
            }
        )
    users_df = pd.DataFrame(users)

    consent_df = pd.DataFrame(
        {
            "user_id": users_df["user_id"],
            "jurisdiction": np.where(users_df["country"] == "US", "US", "EU"),
        }
    )
    consent_df["consent"] = np.where(
        consent_df["jurisdiction"] == "EU", 1, (rng.random(opt.n_users) < opt.consent_rate_us).astype(int)
    )

    # -------------------------------------------------------------- streams
    streams: List[dict] = []
    for j in range(opt.n_broadcast_streams):
        streams.append({"stream_id": f"s{j:03d}", "stream_name": ["Company News", "Division Announcements",
                        "HR Updates", "Safety & Compliance"][j % 4], "stream_type": "broadcast", "restricted": 0})
    for j in range(opt.n_discussion_streams):
        dept = DEPARTMENTS[j % len(DEPARTMENTS)]
        streams.append({"stream_id": f"s{opt.n_broadcast_streams + j:03d}", "stream_name": f"{dept} Talk {j // len(DEPARTMENTS) + 1}",
                        "stream_type": "discussion", "restricted": 1, "_department": dept})
    streams_df = pd.DataFrame(streams)
    restricted = streams_df[streams_df["restricted"] == 1]
    perms: List[dict] = []
    access_map: Dict[str, List[str]] = {}
    broadcast_ids = list(streams_df.loc[streams_df["stream_type"] == "broadcast", "stream_id"])
    for _, u in users_df.iterrows():
        acc = list(broadcast_ids)
        for _, s in restricted.iterrows():
            can = int(s["_department"] == u["department"] or rng.random() < 0.25)
            perms.append({"user_id": u["user_id"], "stream_id": s["stream_id"], "can_access": can})
            if can:
                acc.append(s["stream_id"])
        access_map[u["user_id"]] = acc
    perms_df = pd.DataFrame(perms)
    streams_df = streams_df.drop(columns=["_department"])

    groups: List[dict] = []
    interest = [f"Interest group {g + 1}" for g in range(opt.n_interest_groups)]
    for _, u in users_df.iterrows():
        groups.append({"user_id": u["user_id"], "group_id": f"g_dep_{DEPARTMENTS.index(u['department'])}", "group_name": u["department"]})
        groups.append({"user_id": u["user_id"], "group_id": f"g_div_{0 if u['division'] == 'Europe Retail' else 1}", "group_name": u["division"]})
        for gi in rng.choice(len(interest), size=int(rng.integers(0, 3)), replace=False):
            groups.append({"user_id": u["user_id"], "group_id": f"g_int_{gi}", "group_name": interest[gi]})
    groups_df = pd.DataFrame(groups)

    # --------------------------------------------------------------- events
    day_offsets = np.arange(opt.days)
    rows: List[dict] = []
    event_specs = [
        (EventType.POST_CREATED.value, "posts", "/posts", "POST", "post"),
        (EventType.COMMENT_CREATED.value, "comments", "/posts/{post_id}/comments", "POST", "post"),
        (EventType.REACTION_CREATED.value, "reactions", "/posts/{post_id}/reactions", "POST", "post"),
        (EventType.CHAT_MESSAGE_SENT.value, "chat", "/chats/{chat_id}/messages", "POST", "chat"),
        (EventType.STREAM_VIEWED.value, "topic_access", "/streams/{stream_id}", "GET", "stream"),
        (EventType.TOPIC_VIEWED.value, "topic_access", "/topics/{topic_id}", "GET", "topic"),
        (EventType.POST_READ.value, "post_reads", "/posts/{post_id}", "GET", "post"),
        (EventType.SEARCH_PERFORMED.value, "searches", "/search", "GET", "search"),
        (EventType.LOGIN.value, "logins", "/auth/login", "POST", "session"),
    ]
    for _, u in users_df.iterrows():
        a = ARCHETYPES[u["_archetype"]]
        pattern = u["shift_pattern"]
        # Channel and weekly-rhythm habits are tied to the archetype so that the
        # temporal / channel attributes carry behavioural signal too.
        mobile_share = 0.3 if u["employee_category"] == "office-based" else 0.8
        mobile_share = float(np.clip(mobile_share + a.mobile_shift, 0.05, 0.95))
        # Active days: which days the user shows up at all.
        active = rng.random(opt.days) < a.active_day_prob
        weekday = ((start + pd.to_timedelta(day_offsets, unit="D")).dayofweek < 5)
        if u["employee_category"] == "office-based":
            active &= weekday | (rng.random(opt.days) < 0.12)
        else:
            active &= weekday | (rng.random(opt.days) < a.weekend_factor)
        active_days = day_offsets[active]
        if len(active_days) == 0:
            active_days = np.array([int(rng.integers(0, opt.days))])
        user_noise = float(np.exp(rng.normal(0.0, opt.rate_noise_sigma)))
        for etype, attr, path, method, otype in event_specs:
            rate = getattr(a, attr) * user_noise
            if etype in (EventType.STREAM_VIEWED.value, EventType.TOPIC_VIEWED.value):
                rate *= 0.5
            n = int(rng.poisson(rate * months))
            if n == 0:
                continue
            days = rng.choice(active_days, size=n)
            hours = _sample_hours(rng, n, pattern, a.working_hours_share)
            ts = start + pd.to_timedelta(days, unit="D") + pd.to_timedelta(hours * 3600, unit="s")
            client = np.where(rng.random(n) < mobile_share, "mobile", "web")
            for t, c in zip(ts, client):
                rows.append({"user_id": u["user_id"], "timestamp": t, "event_type": etype, "normalised_path": path,
                             "method": method, "client": c, "object_type": otype})
    events_df = pd.DataFrame(rows).sort_values("timestamp", kind="stable").reset_index(drop=True)
    events_df.insert(0, "event_id", np.arange(1, len(events_df) + 1))
    events_df["object_id"] = pd.array([None] * len(events_df), dtype="object")

    # ---------------------------------------------- posts / comments / reactions
    post_mask = events_df["event_type"] == EventType.POST_CREATED.value
    post_events = events_df[post_mask]
    posts = []
    for n_post, (idx, e) in enumerate(post_events.iterrows(), start=1):
        acc = access_map[e["user_id"]]
        dept = users_df.loc[users_df["user_id"] == e["user_id"], "department"].iloc[0]
        disc = [s for s in acc if s not in broadcast_ids]
        if dept in ("Communications", "Management", "Human Resources") and rng.random() < 0.6:
            stream = str(rng.choice(broadcast_ids))
        else:
            stream = str(rng.choice(disc)) if disc else str(rng.choice(broadcast_ids))
        pid = f"post{n_post:06d}"
        posts.append({"post_id": pid, "author_id": e["user_id"], "stream_id": stream, "created_at": e["timestamp"],
                      "has_video": int(rng.random() < 0.15), "mention_count": int(rng.poisson(0.4))})
        events_df.at[idx, "object_id"] = pid
    posts_df = pd.DataFrame(posts)
    post_times = posts_df["created_at"].to_numpy() if len(posts_df) else np.array([], dtype="datetime64[ns]")
    post_ids = posts_df["post_id"].to_numpy() if len(posts_df) else np.array([])

    def pick_post(ts: pd.Timestamp) -> Optional[str]:
        if len(post_ids) == 0:
            return None
        n_before = int(np.searchsorted(post_times, np.datetime64(ts)))
        if n_before == 0:
            return str(post_ids[0])
        # Prefer recent posts (recency bias) among those already published.
        lo = max(0, n_before - 60)
        return str(post_ids[int(rng.integers(lo, n_before))])

    comments, reactions = [], []
    for idx, e in events_df[events_df["event_type"] == EventType.COMMENT_CREATED.value].iterrows():
        pid = pick_post(e["timestamp"])
        if pid is None:
            continue
        cid = f"c{len(comments) + 1:07d}"
        comments.append({"comment_id": cid, "post_id": pid, "author_id": e["user_id"], "created_at": e["timestamp"],
                         "like_count": int(rng.poisson(0.8))})
        events_df.at[idx, "object_id"] = pid
    for idx, e in events_df[events_df["event_type"] == EventType.REACTION_CREATED.value].iterrows():
        pid = pick_post(e["timestamp"])
        if pid is None:
            continue
        reactions.append({"reaction_id": f"r{len(reactions) + 1:07d}", "post_id": pid, "user_id": e["user_id"],
                          "created_at": e["timestamp"], "reaction_type": str(rng.choice(REACTION_TYPES, p=[0.7, 0.1, 0.1, 0.1]))})
        events_df.at[idx, "object_id"] = pid
    comments_df = pd.DataFrame(comments, columns=["comment_id", "post_id", "author_id", "created_at", "like_count"])
    reactions_df = pd.DataFrame(reactions, columns=["reaction_id", "post_id", "user_id", "created_at", "reaction_type"])
    if len(posts_df):
        posts_df["comment_count"] = posts_df["post_id"].map(comments_df.groupby("post_id").size()).fillna(0).astype(int)
        posts_df["like_count"] = posts_df["post_id"].map(reactions_df.groupby("post_id").size()).fillna(0).astype(int)
        posts_df["comment_like_count"] = posts_df["post_id"].map(comments_df.groupby("post_id")["like_count"].sum()).fillna(0).astype(int)

    # Read / view events point to streams and posts, chat to chat ids (metadata only).
    for etype, prefix, pool in (
        (EventType.STREAM_VIEWED.value, None, None),
        (EventType.POST_READ.value, None, None),
    ):
        mask = events_df["event_type"] == etype
        n = int(mask.sum())
        if n == 0:
            continue
        if etype == EventType.STREAM_VIEWED.value:
            events_df.loc[mask, "object_id"] = [str(rng.choice(access_map[u])) for u in events_df.loc[mask, "user_id"]]
        else:
            events_df.loc[mask, "object_id"] = [pick_post(t) for t in events_df.loc[mask, "timestamp"]]
    mask = events_df["event_type"] == EventType.TOPIC_VIEWED.value
    events_df.loc[mask, "object_id"] = [f"topic{int(x):03d}" for x in rng.integers(0, 40, int(mask.sum()))]
    mask = events_df["event_type"] == EventType.CHAT_MESSAGE_SENT.value
    events_df.loc[mask, "object_id"] = [f"chat{int(x):05d}" for x in rng.integers(0, max(opt.n_users * 3, 1), int(mask.sum()))]

    ground_truth = users_df[["user_id", "_archetype"]].rename(columns={"_archetype": "archetype"})
    ground_truth["archetype_name"] = ground_truth["archetype"].map(lambda k: ARCHETYPES[k].name)
    users_df = users_df.drop(columns=["_archetype"])

    return {
        "users": users_df,
        "consent": consent_df,
        "streams": streams_df,
        "stream_permissions": perms_df,
        "group_memberships": groups_df,
        "api_events": events_df,
        "posts": posts_df,
        "comments": comments_df,
        "reactions": reactions_df,
        "ground_truth_archetypes": ground_truth,
    }


def write_synthetic_datasets(
    out_dir: str | Path,
    options: Optional[SyntheticOptions] = None,
    compress_events: bool = True,
) -> Dict[str, Path]:
    """Generate and write the tables to ``out_dir``; returns the file paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = generate_synthetic_datasets(options)
    paths: Dict[str, Path] = {}
    for name, df in tables.items():
        if name == "api_events" and compress_events:
            path = out / "api_events.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                df.to_csv(fh, index=False, date_format="%Y-%m-%dT%H:%M:%S")
        else:
            path = out / f"{name}.csv"
            df.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S")
        paths[name] = path
    return paths

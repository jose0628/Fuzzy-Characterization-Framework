"""Profiles of the behavioural clusters (Section "Results: Four Behavioural Clusters").

Profiles are *group-level* descriptions: membership-weighted averages of the
behavioural indicators (Table "average behavioural feature values per
cluster"), the dominant fuzzy attributes of every cluster and the degree of
overlap between clusters.  Individual users never appear in a profile.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

EPS = 1e-12

PROFILE_INDICATORS: Dict[str, str] = {
    "posts_per_month": "Posts created (per month)",
    "comments_per_month": "Comments written (per month)",
    "reactions_per_month": "Likes / reactions given (per month)",
    "chat_events_per_month": "Chat events (per month)",
    "topic_accesses_per_month": "Topic / stream accesses (per month)",
    "working_hours_share": "Working-hours activity (% of events)",
    "non_working_hours_share": "Non-working-hours activity (% of events)",
}

# Signature of each archetype over the profile indicators (z-scores across clusters).
ARCHETYPE_TEMPLATES: Dict[str, Dict[str, float]] = {
    "High Communicators": {"posts_per_month": 1.5, "comments_per_month": 1.5, "reactions_per_month": 1.0,
                           "chat_events_per_month": 1.5, "topic_accesses_per_month": 0.3, "working_hours_share": -1.0},
    "Reactive Consumers": {"posts_per_month": -0.3, "comments_per_month": 0.0, "reactions_per_month": 0.8,
                           "chat_events_per_month": 0.0, "topic_accesses_per_month": -0.3, "working_hours_share": -0.2},
    "Operational Browsers": {"posts_per_month": -0.5, "comments_per_month": -0.5, "reactions_per_month": -0.4,
                             "chat_events_per_month": -0.4, "topic_accesses_per_month": 1.5, "working_hours_share": 1.2},
    "Low-Engagement Users": {"posts_per_month": -0.8, "comments_per_month": -0.9, "reactions_per_month": -1.2,
                             "chat_events_per_month": -1.0, "topic_accesses_per_month": -1.3, "working_hours_share": 0.2},
}

ARCHETYPE_DESCRIPTIONS: Dict[str, str] = {
    "High Communicators": "Frequently initiate content, discuss and chat; activity spread across working and non-working hours.",
    "Reactive Consumers": "Predominantly react to posts and occasionally comment, rarely publish content.",
    "Operational Browsers": "Consume information across streams and topics, task-driven usage aligned with working hours.",
    "Low-Engagement Users": "Consistently low membership across the fuzzy variables, infrequent logins, sporadic navigation.",
}


def cluster_feature_averages(
    indicators: pd.DataFrame,
    memberships: np.ndarray,
    columns: Optional[Sequence[str]] = None,
    weighted: bool = True,
) -> pd.DataFrame:
    """Average indicator values per cluster (rows = indicators, columns = C1..Ck).

    With ``weighted=True`` the average is weighted by the membership degree,
    which respects the fuzzy overlaps; otherwise the highest-membership
    assignment is used.
    """
    cols = list(columns or [c for c in PROFILE_INDICATORS if c in indicators.columns])
    X = indicators[cols].to_numpy(dtype=float)
    k = memberships.shape[1]
    if weighted:
        W = memberships / np.maximum(memberships.sum(axis=0, keepdims=True), EPS)
        means = (W.T @ X)
    else:
        labels = np.argmax(memberships, axis=1)
        means = np.vstack([X[labels == j].mean(axis=0) if np.any(labels == j) else np.full(len(cols), np.nan) for j in range(k)])
    out = pd.DataFrame(means.T, index=cols, columns=[f"C{j + 1}" for j in range(k)])
    out.insert(0, "description", [PROFILE_INDICATORS.get(c, c) for c in cols])
    return out


def membership_distribution(memberships: np.ndarray) -> pd.DataFrame:
    """Cluster sizes by highest membership and by membership mass."""
    k = memberships.shape[1]
    labels = np.argmax(memberships, axis=1)
    n = len(labels)
    sizes = np.bincount(labels, minlength=k)
    mass = memberships.sum(axis=0)
    return pd.DataFrame(
        {
            "cluster": [f"C{j + 1}" for j in range(k)],
            "users_by_highest_membership": sizes,
            "share_by_highest_membership": sizes / max(n, 1),
            "membership_mass": mass,
            "share_by_membership_mass": mass / max(mass.sum(), EPS),
            "mean_membership_of_members": [float(memberships[labels == j, j].mean()) if sizes[j] else float("nan") for j in range(k)],
        }
    )


def label_archetypes(
    averages: pd.DataFrame,
    templates: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """Match clusters to the behavioural archetypes of the thesis.

    Each cluster's profile is z-scored across clusters and compared (cosine
    similarity) with the archetype templates; the Hungarian algorithm gives a
    one-to-one assignment.  When ``k`` exceeds the number of templates, the
    remaining clusters are labelled ``Segment i``.
    """
    templates = templates or ARCHETYPE_TEMPLATES
    cluster_cols = [c for c in averages.columns if c != "description"]
    feats = [f for f in next(iter(templates.values())) if f in averages.index]
    M = averages.loc[feats, cluster_cols].to_numpy(dtype=float)  # (features, k)
    mu, sd = M.mean(axis=1, keepdims=True), M.std(axis=1, keepdims=True)
    Z = (M - mu) / np.where(sd > EPS, sd, 1.0)
    names = list(templates)
    T = np.array([[templates[nm].get(f, 0.0) for f in feats] for nm in names])  # (templates, features)
    sims = np.zeros((len(cluster_cols), len(names)))
    for j in range(len(cluster_cols)):
        for t in range(len(names)):
            sims[j, t] = float(Z[:, j] @ T[t] / ((np.linalg.norm(Z[:, j]) + EPS) * (np.linalg.norm(T[t]) + EPS)))
    rows, cols = linear_sum_assignment(-sims)
    label = {f"C{j + 1}": f"Segment {j + 1}" for j in range(len(cluster_cols))}
    score = {f"C{j + 1}": float("nan") for j in range(len(cluster_cols))}
    for r, c in zip(rows, cols):
        label[cluster_cols[r]] = names[c]
        score[cluster_cols[r]] = float(sims[r, c])
    return pd.DataFrame(
        {
            "cluster": cluster_cols,
            "archetype": [label[c] for c in cluster_cols],
            "match_score": [score[c] for c in cluster_cols],
            "description": [ARCHETYPE_DESCRIPTIONS.get(label[c], "") for c in cluster_cols],
        }
    )


def dominant_attributes(fuzzy_vectors: pd.DataFrame, memberships: np.ndarray, top_n: int = 5) -> pd.DataFrame:
    """Top positive and negative fuzzy attributes of each cluster (z-scored centre)."""
    X = fuzzy_vectors.to_numpy(dtype=float)
    mu, sd = X.mean(axis=0), np.where(X.std(axis=0) > EPS, X.std(axis=0), 1.0)
    W = memberships / np.maximum(memberships.sum(axis=0, keepdims=True), EPS)
    centers = (W.T @ X - mu) / sd
    rows: List[dict] = []
    for j, row in enumerate(centers):
        order = np.argsort(row)
        for rank, idx in enumerate(order[::-1][:top_n], start=1):
            rows.append({"cluster": f"C{j + 1}", "direction": "high", "rank": rank,
                         "attribute": fuzzy_vectors.columns[idx], "z": float(row[idx]),
                         "mean_membership": float((W[:, j] @ X[:, idx]))})
        for rank, idx in enumerate(order[:top_n], start=1):
            rows.append({"cluster": f"C{j + 1}", "direction": "low", "rank": rank,
                         "attribute": fuzzy_vectors.columns[idx], "z": float(row[idx]),
                         "mean_membership": float((W[:, j] @ X[:, idx]))})
    return pd.DataFrame(rows)


def overlap_report(memberships: np.ndarray, mixed_threshold: float = 0.3) -> Dict[str, object]:
    """How many users show mixed behaviours and how much clusters overlap pairwise."""
    k = memberships.shape[1]
    sorted_u = np.sort(memberships, axis=1)
    second = sorted_u[:, -2] if k > 1 else np.zeros(len(memberships))
    overlap = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            overlap[i, j] = float(np.mean(np.minimum(memberships[:, i], memberships[:, j])))
    return {
        "share_users_mixed": float(np.mean(second >= mixed_threshold)),
        "mixed_threshold": mixed_threshold,
        "mean_max_membership": float(np.mean(sorted_u[:, -1])),
        "pairwise_overlap": pd.DataFrame(overlap, index=[f"C{i + 1}" for i in range(k)], columns=[f"C{i + 1}" for i in range(k)]),
    }

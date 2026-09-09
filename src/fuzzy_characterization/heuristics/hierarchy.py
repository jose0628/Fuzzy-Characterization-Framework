"""Hierarchy definition (Algorithm "Hierarchy definition").

Job titles are normalised to a *hierarchy score*: keywords describing the
position (``chief`` = 20, ``manager`` = 6, ...) and the employee category are
matched in the title and their values are added.  Descriptive terms such as
``senior`` or ``executive`` are additive, so ``senior manager`` scores 2 + 6.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd

# Values reflect the relative hierarchical level and the frequency of the keyword.
DEFAULT_POSITION_KEYWORDS: Dict[str, int] = {
    "chief": 20,
    "ceo": 20,
    "cfo": 18,
    "coo": 18,
    "president": 18,
    "vice president": 14,
    "director": 12,
    "head": 10,
    "general manager": 9,
    "regional manager": 8,
    "manager": 6,
    "assistant manager": 4,
    "supervisor": 4,
    "team lead": 4,
    "lead": 3,
    "specialist": 2,
    "coordinator": 2,
    "associate": 1,
    "assistant": 1,
    "sales": 0,
    "cashier": 0,
    "intern": 0,
}

# Modifiers conveying duration or expertise in a role (additive).
DEFAULT_CATEGORY_KEYWORDS: Dict[str, int] = {
    "executive": 6,
    "senior": 2,
    "expert": 1,
    "principal": 3,
    "junior": -1,
    "trainee": -1,
}


def _score_title(title: str, position_keywords: Dict[str, int], category_keywords: Dict[str, int]) -> int:
    t = str(title).lower()
    score = 0
    # Longest keywords first so that "general manager" is not double counted as "manager".
    consumed = t
    for key in sorted(position_keywords, key=len, reverse=True):
        if key in consumed:
            score += position_keywords[key]
            consumed = consumed.replace(key, " ")
    for key, val in category_keywords.items():
        if key in t:
            score += val
    return max(score, 0)


def hierarchy_score(
    df: pd.DataFrame,
    title_col: str = "job_title",
    category_col: Optional[str] = None,
    position_keywords: Optional[Dict[str, int]] = None,
    category_keywords: Optional[Dict[str, int]] = None,
) -> pd.Series:
    """Hierarchy score per row of ``df``.

    ``category_col`` optionally names an employee-category column whose text
    is also matched against ``category_keywords`` (as in the algorithm).
    """
    pk = position_keywords or DEFAULT_POSITION_KEYWORDS
    ck = category_keywords or DEFAULT_CATEGORY_KEYWORDS
    titles = df[title_col].fillna("")
    if category_col and category_col in df.columns:
        titles = titles + " " + df[category_col].fillna("").astype(str)
    return titles.apply(lambda t: _score_title(t, pk, ck)).astype(int).rename("hierarchy_score")


def role_category(score: pd.Series, bins: Iterable[int] = (0, 3, 8, 1000)) -> pd.Series:
    """Broad, non-identifying role bins used as compliant demographics."""
    labels = ["frontline", "supervisory", "management"]
    return pd.cut(score, bins=list(bins), labels=labels, right=False, include_lowest=True).astype(str)

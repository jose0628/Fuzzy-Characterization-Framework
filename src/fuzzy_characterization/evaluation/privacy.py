"""Privacy-compliance assessment of a segmentation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np


def privacy_assessment(
    representation: str,
    cluster_sizes: Iterable[int],
    min_group_size: int,
    compliance: Optional[Dict[str, Any]] = None,
    n_membership_features: Optional[int] = None,
    n_features: Optional[int] = None,
) -> Dict[str, Any]:
    """Summarise the privacy properties of one method's result.

    ``representation`` is ``"fuzzy"`` when the method consumed the anonymised
    membership representation and ``"full"`` when it consumed full-resolution
    behavioural features.  The level is *High* only when the representation
    is fuzzy, the compliance report has no violations and every reported
    group reaches the minimum size.
    """
    sizes = np.asarray(list(cluster_sizes), dtype=int)
    small = [int(i) for i in np.where(sizes < min_group_size)[0]]
    comp = compliance or {}
    violations = list(comp.get("violations", []))
    degree = 1.0 if representation == "fuzzy" else 0.0
    if n_membership_features is not None and n_features:
        degree = float(n_membership_features / n_features)
    level = "High" if representation == "fuzzy" and not violations and not small else "Low"
    return {
        "representation": representation,
        "degree_of_anonymisation": degree,
        "consent_coverage": comp.get("consent", {}).get("consent_coverage"),
        "identifiers_excluded": comp.get("identifiers_excluded", []),
        "pseudonymised": comp.get("pseudonymised", False),
        "min_group_size": int(min_group_size),
        "smallest_group": int(sizes.min()) if len(sizes) else None,
        "groups_below_min_size": small,
        "violations": violations,
        "level": level,
    }

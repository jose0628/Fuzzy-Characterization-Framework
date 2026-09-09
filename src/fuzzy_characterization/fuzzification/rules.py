"""Fuzzy rule-based layer (Section "Fuzzy Rule-Based Layer").

Rules encode organisational constraints and communication policies and are
written by hand, from customer feedback.  Three kinds of consequent are
supported, matching the three examples in the thesis:

* ``scale_weight``       -- *IF hierarchy score is low AND streams are
  predominantly broadcast THEN reduce the weight of authoring activity and
  increase the weight of reactive activity*;
* ``relative_indicator`` -- *IF the number of accessible streams is small THEN
  activity is assessed relative to available content*;
* ``set_shift_pattern``  -- *IF the employee category is shift-based THEN the
  active-hours membership function is defined over the shift window*.

Antecedents are evaluated with the membership degree of the user's context
attribute in a linguistic term (``low``, ``small``, ``high``...).  The firing
strength of a rule is the minimum (AND) or maximum (OR) of its antecedent
degrees; consequents are applied proportionally to that strength, so a rule
never acts as a crisp switch.  Only attributes already permitted in the data
(hierarchy score, division, employee category, stream permissions, fuzzy
attributes) may appear in a rule; no new personal data is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .membership import MembershipFunction, build_membership_function


@dataclass
class Antecedent:
    attribute: str
    term: Optional[str] = None     # linguistic term (fuzzy attribute)
    equals: Optional[Any] = None   # crisp equality (categorical attribute)
    negate: bool = False

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "Antecedent":
        return cls(
            attribute=spec["attribute"],
            term=spec.get("is"),
            equals=spec.get("equals"),
            negate=bool(spec.get("not", False)),
        )

    def describe(self) -> str:
        if self.term is not None:
            s = f"{self.attribute} is {self.term}"
        else:
            s = f"{self.attribute} == {self.equals!r}"
        return f"NOT ({s})" if self.negate else s


@dataclass
class Consequent:
    action: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "Consequent":
        spec = dict(spec)
        action = spec.pop("action")
        return cls(action=action, params=spec)

    def describe(self) -> str:
        p = self.params
        if self.action == "scale_weight":
            return f"scale weight of {p.get('features')} by {p.get('factor')}"
        if self.action == "relative_indicator":
            return f"assess {p.get('indicators')} relative to {p.get('denominator')}"
        if self.action == "set_shift_pattern":
            return f"active-hours window from {p.get('from_column') or p.get('pattern')}"
        return f"{self.action} {p}"


@dataclass
class FuzzyRule:
    name: str
    antecedents: List[Antecedent]
    consequents: List[Consequent]
    operator: str = "and"
    description: str = ""
    assumption: str = ""

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "FuzzyRule":
        ants = [Antecedent.from_spec(a) for a in spec.get("if", [])]
        cons = [Consequent.from_spec(c) for c in spec.get("then", [])]
        if not ants or not cons:
            raise ValueError(f"Rule '{spec.get('name')}' needs both 'if' and 'then' parts.")
        return cls(
            name=spec["name"],
            antecedents=ants,
            consequents=cons,
            operator=str(spec.get("operator", "and")).lower(),
            description=spec.get("description", ""),
            assumption=spec.get("assumption", ""),
        )

    def describe(self) -> str:
        joiner = " AND " if self.operator == "and" else " OR "
        return (
            f"IF {joiner.join(a.describe() for a in self.antecedents)} "
            f"THEN {'; '.join(c.describe() for c in self.consequents)}"
        )


@dataclass
class RuleEffect:
    """Everything the rule layer changes, kept explicit for transparency."""

    firing_strengths: pd.DataFrame       # users x rules, in [0, 1]
    feature_weights: pd.DataFrame        # users x fuzzy feature names, default 1.0
    indicators: pd.DataFrame             # indicators after relative adjustments
    shift_pattern: Optional[pd.Series]   # per-user shift window name (or None)
    rule_descriptions: List[Dict[str, str]]

    def column_weights(self, fuzzy_columns: Iterable[str]) -> pd.DataFrame:
        """Expand feature-level weights to fuzzy attribute columns (``feature__term``)."""
        cols = list(fuzzy_columns)
        out = pd.DataFrame(1.0, index=self.feature_weights.index, columns=cols)
        for col in cols:
            feat = col.split("__", 1)[0]
            if feat in self.feature_weights.columns:
                out[col] = self.feature_weights[feat].to_numpy()
        return out


class FuzzyRuleLayer:
    """Evaluate a set of fuzzy rules against per-user context attributes."""

    def __init__(
        self,
        rules: Iterable[Dict[str, Any]],
        context_terms: Optional[Dict[str, Dict[str, Any]]] = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.rules: List[FuzzyRule] = [FuzzyRule.from_spec(r) for r in rules]
        self.context_terms: Dict[str, Dict[str, MembershipFunction]] = {}
        for attr, terms in (context_terms or {}).items():
            if isinstance(terms, dict):
                self.context_terms[attr] = {
                    t: build_membership_function(dict(spec), label=t) for t, spec in terms.items()
                }

    # ------------------------------------------------------------------ #
    def _antecedent_degree(self, ant: Antecedent, context: pd.DataFrame) -> np.ndarray:
        if ant.attribute not in context.columns:
            raise KeyError(f"Rule antecedent needs context attribute '{ant.attribute}'.")
        values = context[ant.attribute]
        if ant.equals is not None:
            deg = (values.astype(str) == str(ant.equals)).to_numpy(dtype=float)
        else:
            terms = self.context_terms.get(ant.attribute)
            if not terms or ant.term not in terms:
                raise KeyError(
                    f"No membership function for '{ant.attribute} is {ant.term}'. "
                    f"Define it under rules.context_terms."
                )
            deg = np.clip(terms[ant.term](values.to_numpy(dtype=float)), 0.0, 1.0)
        return 1.0 - deg if ant.negate else deg

    def firing_strength(self, rule: FuzzyRule, context: pd.DataFrame) -> np.ndarray:
        degrees = np.vstack([self._antecedent_degree(a, context) for a in rule.antecedents])
        if rule.operator == "or":
            return degrees.max(axis=0)
        return degrees.min(axis=0)

    # ------------------------------------------------------------------ #
    def apply(
        self,
        indicators: pd.DataFrame,
        context: pd.DataFrame,
        feature_names: Iterable[str],
    ) -> RuleEffect:
        """Evaluate all rules.

        ``indicators`` and ``context`` must share the same index (user ids).
        ``feature_names`` are the fuzzy feature names (not the ``__term`` columns).
        """
        feature_names = list(feature_names)
        indicators = indicators.copy()
        context = context.reindex(indicators.index)
        weights = pd.DataFrame(1.0, index=indicators.index, columns=feature_names)
        strengths = pd.DataFrame(0.0, index=indicators.index, columns=[r.name for r in self.rules])
        shift_pattern: Optional[pd.Series] = None
        descriptions = [
            {"name": r.name, "rule": r.describe(), "description": r.description, "assumption": r.assumption}
            for r in self.rules
        ]
        if not self.enabled or not self.rules:
            return RuleEffect(strengths, weights, indicators, None, descriptions)

        for rule in self.rules:
            s = self.firing_strength(rule, context)
            strengths[rule.name] = s
            for cons in rule.consequents:
                p = cons.params
                if cons.action == "scale_weight":
                    factor = float(p.get("factor", 1.0))
                    for feat in p.get("features", []):
                        if feat not in weights.columns:
                            raise KeyError(f"Rule '{rule.name}' scales unknown feature '{feat}'.")
                        weights[feat] = weights[feat].to_numpy() * (1.0 + s * (factor - 1.0))
                elif cons.action == "relative_indicator":
                    denom_col = p["denominator"]
                    if denom_col not in context.columns:
                        raise KeyError(f"Rule '{rule.name}' needs context column '{denom_col}'.")
                    denom = context[denom_col].to_numpy(dtype=float)
                    ref = float(np.nanmedian(denom[denom > 0])) if np.any(denom > 0) else 1.0
                    ratio = np.divide(ref, denom, out=np.ones_like(denom), where=denom > 0)
                    for ind in p.get("indicators", []):
                        if ind not in indicators.columns:
                            raise KeyError(f"Rule '{rule.name}' adjusts unknown indicator '{ind}'.")
                        x = indicators[ind].to_numpy(dtype=float)
                        indicators[ind] = (1.0 - s) * x + s * x * ratio
                elif cons.action == "set_shift_pattern":
                    if shift_pattern is None:
                        shift_pattern = pd.Series(None, index=indicators.index, dtype=object)
                    active = s >= float(p.get("threshold", 0.5))
                    if "from_column" in p:
                        col = p["from_column"]
                        if col not in context.columns:
                            raise KeyError(f"Rule '{rule.name}' needs context column '{col}'.")
                        vals = context[col].astype(object).where(context[col].notna(), None)
                        shift_pattern[active] = vals[active]
                    else:
                        shift_pattern[active] = p.get("pattern")
                else:
                    raise ValueError(f"Unknown rule action '{cons.action}' in rule '{rule.name}'.")
        return RuleEffect(strengths, weights, indicators, shift_pattern, descriptions)

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"name": r.name, "rule": r.describe(), "description": r.description, "assumption": r.assumption}
             for r in self.rules]
        )

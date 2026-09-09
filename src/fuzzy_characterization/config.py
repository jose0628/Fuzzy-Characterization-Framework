"""Configuration objects and YAML loading for the framework.

All tunable parameters of the pipeline live in a single :class:`PipelineConfig`
so that a run is fully described by one YAML file (see ``configs/``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# --------------------------------------------------------------------------- #
# Small typed sections
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Where the source datasets live and how the observation period is defined."""

    root: str = "data/synthetic"
    users_file: str = "users.csv"
    events_file: str = "api_events.csv"
    posts_file: str = "posts.csv"
    comments_file: str = "comments.csv"
    reactions_file: str = "reactions.csv"
    streams_file: str = "streams.csv"
    stream_permissions_file: str = "stream_permissions.csv"
    group_memberships_file: str = "group_memberships.csv"
    consent_file: Optional[str] = "consent.csv"
    user_id_col: str = "user_id"
    # Observation window; ``None`` means "use the min/max event timestamp".
    period_start: Optional[str] = None
    period_end: Optional[str] = None


@dataclass
class ComplianceConfig:
    """Compliance principles applied globally (GDPR / nFADP)."""

    # Columns that are direct identifiers and are always dropped.
    direct_identifiers: List[str] = field(
        default_factory=lambda: ["full_name", "username", "email", "phone", "mobile"]
    )
    # Columns that carry message payloads and are never opened.
    content_columns: List[str] = field(
        default_factory=lambda: ["text", "message", "payload", "body", "content"]
    )
    age_bins: List[int] = field(default_factory=lambda: [18, 25, 35, 45, 55, 65, 100])
    # Any reported group (cluster, alpha-cut segment) must contain at least this
    # many users, otherwise it is flagged as too small to report.
    min_group_size: int = 20
    require_consent: bool = True
    pseudonymise: bool = True
    pseudonym_salt: str = "fcf"


@dataclass
class IndicatorConfig:
    """How behavioural indicators are aggregated."""

    working_hours: Dict[str, float] = field(
        default_factory=lambda: {"a": 6.0, "b": 8.0, "c": 16.0, "d": 18.0}
    )
    # Indicators are reported "per month" of observation.
    days_per_month: float = 30.4375


@dataclass
class FuzzificationConfig:
    """Membership-function specification (loaded from ``membership_functions``)."""

    # Normalisation applied to an indicator before fuzzification.
    default_normalisation: str = "log_minmax"  # minmax | log_minmax | quantile | none
    # Each entry: name, indicator, family, terms -> {term: params}, normalisation, alpha_cut
    features: List[Dict[str, Any]] = field(default_factory=list)
    alpha_levels: List[float] = field(default_factory=lambda: [0.2, 0.5, 0.8])
    shift_patterns: Dict[str, Any] = field(default_factory=dict)
    # Users column naming their shift window (a key of ``shift_patterns``); the
    # working-hour membership function is then defined over that window.
    shift_pattern_column: Optional[str] = None


@dataclass
class RulesConfig:
    enabled: bool = False
    rules: List[Dict[str, Any]] = field(default_factory=list)
    # Membership functions used to fuzzify the *context* attributes appearing
    # in rule antecedents (hierarchy score, accessible streams, ...).
    context_terms: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReductionConfig:
    enabled: bool = True
    # Either an integer number of components or a variance ratio in (0, 1].
    n_components: Any = 0.90
    max_components: int = 20
    standardise_before_pca: bool = False
    # Optional normalisation of the membership variables before PCA/clustering:
    # none | standard | unit_var | minmax
    scaling: str = "none"
    # The clustering space: "fuzzy" (original membership vector) or "pca".
    clustering_space: str = "pca"


@dataclass
class ClusteringConfig:
    method: str = "fcm"
    k: Optional[int] = 4
    k_min: int = 2
    k_max: int = 10
    m: float = 2.0
    eta: float = 2.0
    a: float = 1.0
    b: float = 1.0
    # PFCM: scale of the typicality reference distances gamma_j (K in Pal et al. 2005).
    # Values below 1 keep typicalities local and prevent coincident clusters.
    pfcm_gamma_scale: float = 1.0
    max_iter: int = 300
    tol: float = 1e-5
    n_restarts: int = 30
    seed: int = 1
    linkage: str = "ward"
    # Which methods are executed in the comparative evaluation.
    compare_methods: List[str] = field(
        default_factory=lambda: ["fcm", "pfcm", "gk", "kmeans", "ahc"]
    )


@dataclass
class EvaluationConfig:
    fsi_alpha: float = 1.0
    max_silhouette_samples: int = 5000
    stability_runs: int = 10
    expert_ratings_file: Optional[str] = None
    interpretability_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "feature_dominance": 1.0,
            "membership_transparency": 1.0,
            "semantic_consistency": 1.0,
        }
    )


@dataclass
class OutputConfig:
    root: str = "outputs"
    run_name: str = "retail_case"
    save_figures: bool = True
    figure_dpi: int = 160
    max_plot_points: int = 2000


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    fuzzification: FuzzificationConfig = field(default_factory=FuzzificationConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    reduction: ReductionConfig = field(default_factory=ReductionConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    # Path of the YAML the config was loaded from (for relative includes).
    source_path: Optional[str] = None

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, raw: Dict[str, Any], source_path: Optional[str] = None) -> "PipelineConfig":
        raw = copy.deepcopy(raw or {})
        cfg = cls(source_path=source_path)
        section_types = {
            "data": DataConfig,
            "compliance": ComplianceConfig,
            "indicators": IndicatorConfig,
            "fuzzification": FuzzificationConfig,
            "rules": RulesConfig,
            "reduction": ReductionConfig,
            "clustering": ClusteringConfig,
            "evaluation": EvaluationConfig,
            "output": OutputConfig,
        }
        for name, typ in section_types.items():
            section = raw.get(name, {}) or {}
            if not isinstance(section, dict):
                raise TypeError(f"Config section '{name}' must be a mapping.")
            known = {f for f in typ.__dataclass_fields__}
            unknown = set(section) - known
            if unknown:
                raise KeyError(f"Unknown keys in config section '{name}': {sorted(unknown)}")
            setattr(cfg, name, typ(**section))
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path)
        return cls.from_dict(_load_yaml_with_includes(path), source_path=str(path))

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name in (
            "data", "compliance", "indicators", "fuzzification", "rules",
            "reduction", "clustering", "evaluation", "output",
        ):
            out[name] = copy.deepcopy(getattr(self, name).__dict__)
        return out

    def resolve(self, relative: str | Path) -> Path:
        """Resolve a path relative to the project root (cwd) unless absolute."""
        p = Path(relative)
        return p if p.is_absolute() else Path.cwd() / p


def _load_yaml_with_includes(path: Path, _seen: Optional[set] = None) -> Dict[str, Any]:
    """Load a YAML file, recursively merging the files listed under ``include``.

    Included files are merged first (in order); the including file wins.
    """
    path = path.resolve()
    seen = set(_seen or ())
    if path in seen:
        raise ValueError(f"Circular include detected at {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    includes = raw.pop("include", []) or []
    merged: Dict[str, Any] = {}
    for inc in includes:
        inc_path = Path(inc) if Path(inc).is_absolute() else (path.parent / inc)
        merged = _deep_merge(merged, _load_yaml_with_includes(inc_path, seen))
    return _deep_merge(merged, raw)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: Optional[str | Path] = None) -> PipelineConfig:
    """Load a configuration file, falling back to the packaged defaults."""
    if path is None:
        return PipelineConfig()
    return PipelineConfig.from_yaml(path)

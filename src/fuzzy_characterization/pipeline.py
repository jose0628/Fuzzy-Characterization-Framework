"""End-to-end orchestration of the user characterization framework.

Stages (Figure "framework overview"):

1. **Data compliance** -- consent filtering, pseudonymisation, removal of
   direct identifiers, generalisation of demographics;
2. **Characterized user vector** -- API-event indicators + heuristic
   attributes + compliant demographics;
3. **Fuzzification engine** -- fuzzy rule layer, membership functions,
   alpha-cuts -> fuzzy attribute vector;
4. **Reduction** -- optional scaling and PCA (retained components);
5. **Clustering** -- k selection, FCM (or another method) with restarts;
6. **Performance and metrics** -- comparison of fuzzy methods and baselines,
   stability, interpretability, privacy assessment, profiles and figures.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .clustering import ClusterResult, evaluate_k_range, run_with_restarts, suggest_k
from .clustering.base import METHOD_NAMES
from .compliance import Pseudonymiser, apply_consent, assess_compliance, drop_identifiers, generalise_demographics
from .config import PipelineConfig
from .data import Datasets, build_behavioural_indicators, load_datasets
from .data.indicators import indicator_table
from .evaluation import MethodEvaluation, compare_methods
from .fuzzification import FuzzificationEngine, FuzzificationResult, FuzzyRuleLayer
from .fuzzification.rules import RuleEffect
from .heuristics import build_heuristic_attributes
from .preprocessing import FeatureScaler, PCAReduction, fit_pca, full_resolution_features
from .profiling import (
    cluster_feature_averages,
    dominant_attributes,
    label_archetypes,
    membership_distribution,
    overlap_report,
)
from .visualization import plots


@dataclass
class PipelineResult:
    config: PipelineConfig
    output_dir: Path
    compliance: Dict[str, Any]
    indicators: pd.DataFrame
    heuristics: pd.DataFrame
    demographics: pd.DataFrame
    rule_effect: RuleEffect
    fuzzification: FuzzificationResult
    clustering_input: pd.DataFrame
    pca: Optional[PCAReduction]
    k_metrics: pd.DataFrame
    k_suggestions: Dict[str, Any]
    k: int
    result: ClusterResult
    runs: List[ClusterResult]
    comparison: pd.DataFrame
    evaluations: Dict[str, MethodEvaluation]
    profiles: Dict[str, Any]
    artifacts: Dict[str, str] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def archetypes(self) -> List[str]:
        return list(self.profiles["labels"]["archetype"])

    def summary(self) -> Dict[str, Any]:
        best = self.comparison.set_index("method") if len(self.comparison) else pd.DataFrame()
        return {
            "n_users": int(len(self.indicators)),
            "n_fuzzy_attributes": int(self.fuzzification.n_features),
            "n_components": int(self.pca.n_components) if self.pca else None,
            "cumulative_variance": float(self.pca.cumulative_variance[self.pca.n_components - 1]) if self.pca else None,
            "k": int(self.k),
            "k_suggestions": self.k_suggestions,
            "method": self.result.method,
            "archetypes": dict(zip(self.profiles["labels"]["cluster"], self.profiles["labels"]["archetype"])),
            "cluster_sizes": [int(v) for v in self.result.cluster_sizes()],
            "comparison": best[["silhouette_or_fsi", "cluster_stability", "interpretability_score", "privacy_compliance"]].to_dict("index") if len(best) else {},
            "compliance_level": self.compliance.get("level"),
            "output_dir": str(self.output_dir),
        }


# --------------------------------------------------------------------------- #
def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[fcf] {msg}", flush=True)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_pipeline(
    cfg: PipelineConfig,
    data: Optional[Datasets] = None,
    k: Optional[int] = None,
    write: bool = True,
    verbose: bool = True,
) -> PipelineResult:
    """Run the framework end to end and (optionally) write all artefacts."""
    t0 = time.time()
    timings: Dict[str, float] = {}
    out_dir = Path(cfg.output.root) / cfg.output.run_name
    fig_dir = out_dir / "figures"
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 1. data
    if data is None:
        _log(verbose, f"loading datasets from {cfg.data.root}")
        data = load_datasets(cfg.data)
    uid = data.user_id_col
    timings["load"] = time.time() - t0

    # ------------------------------------------------- 2. data compliance
    t = time.time()
    data, consent_info = apply_consent(data, cfg.compliance.require_consent)
    pseudonymised = False
    if cfg.compliance.pseudonymise:
        data = Pseudonymiser(cfg.compliance.pseudonym_salt).apply_datasets(data)
        pseudonymised = True
    users, identifiers_dropped = drop_identifiers(data.users, cfg.compliance.direct_identifiers)
    content_dropped: List[str] = []
    for name, table in data.tables().items():
        present = [c for c in cfg.compliance.content_columns if c in table.columns]
        content_dropped.extend(f"{name}.{c}" for c in present)
        if present:
            table.drop(columns=present, inplace=True)
    demographics = generalise_demographics(users, cfg.compliance).set_index(uid)
    user_ids = pd.Index(users[uid].unique(), name=uid)
    _log(verbose, f"compliance: {consent_info.get('users_after', len(user_ids))} users with consent, "
                  f"identifiers dropped {identifiers_dropped}, pseudonymised={pseudonymised}")
    timings["compliance"] = time.time() - t

    # ------------------------------------- 3. characterized user vector
    t = time.time()
    indicators = build_behavioural_indicators(data, cfg.indicators, user_ids)
    heuristics = build_heuristic_attributes(data, user_ids)
    _log(verbose, f"indicators: {indicators.shape[1]} columns for {len(indicators)} users "
                  f"over {indicators.attrs.get('observation_days')} days")
    timings["indicators"] = time.time() - t

    # ---------------------------------------- 4. fuzzification engine
    t = time.time()
    fuzz_input = indicators.copy()
    for col in ("age_band_midpoint",):
        if col in demographics.columns:
            fuzz_input[col] = demographics[col].reindex(fuzz_input.index).to_numpy()
    context = pd.concat([heuristics, demographics.drop(columns=[c for c in demographics.columns if c in heuristics.columns])], axis=1)
    context = context.reindex(fuzz_input.index)
    engine = FuzzificationEngine(
        cfg.fuzzification.features,
        default_normalisation=cfg.fuzzification.default_normalisation,
        alpha_levels=cfg.fuzzification.alpha_levels,
        shift_patterns=cfg.fuzzification.shift_patterns,
        min_group_size=cfg.compliance.min_group_size,
    )
    rule_layer = FuzzyRuleLayer(cfg.rules.rules, cfg.rules.context_terms, enabled=cfg.rules.enabled)
    effect = rule_layer.apply(fuzz_input, context, [f.name for f in engine.features])
    shift_pattern: Optional[pd.Series] = None
    if cfg.fuzzification.shift_pattern_column and cfg.fuzzification.shift_pattern_column in context.columns:
        shift_pattern = context[cfg.fuzzification.shift_pattern_column].astype(object)
    if effect.shift_pattern is not None:
        override = effect.shift_pattern.dropna()
        if shift_pattern is None:
            shift_pattern = pd.Series(None, index=fuzz_input.index, dtype=object)
        shift_pattern.loc[override.index] = override.values
    fz = engine.fit_transform(effect.indicators, shift_pattern)
    weights = effect.column_weights(fz.fuzzy_vectors.columns)
    weighted = fz.fuzzy_vectors * weights
    _log(verbose, f"fuzzification: {fz.n_features} fuzzy attributes from {len(engine.features)} features; "
                  f"rules {'enabled' if cfg.rules.enabled else 'disabled'} ({len(rule_layer.rules)} defined)")
    timings["fuzzification"] = time.time() - t

    # ------------------------------------------------- 5. reduction
    t = time.time()
    scaler = FeatureScaler(cfg.reduction.scaling)
    scaled = scaler.fit_transform(weighted)
    pca: Optional[PCAReduction] = None
    if cfg.reduction.enabled:
        pca = fit_pca(scaled, cfg.reduction.n_components, cfg.reduction.max_components,
                      cfg.reduction.standardise_before_pca, random_state=cfg.clustering.seed)
        _log(verbose, f"PCA: {pca.n_components} components, cumulative variance "
                      f"{pca.cumulative_variance[pca.n_components - 1]:.3f}")
    if cfg.reduction.enabled and cfg.reduction.clustering_space == "pca":
        X_cluster = pca.scores.copy()
    else:
        X_cluster = scaled.copy()
    # 2-D plane for the figures (always PCA of the scaled fuzzy vectors)
    pca2 = pca if pca is not None else fit_pca(scaled, 2, 2, False, cfg.clustering.seed)
    X2 = pca2.scores.iloc[:, :2].to_numpy()
    timings["reduction"] = time.time() - t

    # ------------------------------------------- 6. number of clusters
    t = time.time()
    Xc = X_cluster.to_numpy(dtype=float)
    k_metrics = evaluate_k_range(Xc, cfg.clustering.k_min, cfg.clustering.k_max, cfg.clustering,
                                 n_restarts=min(5, cfg.clustering.n_restarts),
                                 max_silhouette_samples=cfg.evaluation.max_silhouette_samples)
    k_sugg = suggest_k(k_metrics)
    chosen_k = int(k or cfg.clustering.k or k_sugg["recommended_k"])
    _log(verbose, f"k selection: recommended {k_sugg['recommended_k']} (DBI {k_sugg['davies_bouldin_min']}, "
                  f"elbow {k_sugg['elbow_fcm_objective']}, XB {k_sugg['xie_beni_min']}); using k={chosen_k}")
    timings["k_selection"] = time.time() - t

    # -------------------------------------------------- 7. clustering
    t = time.time()
    best, runs = run_with_restarts(cfg.clustering.method, Xc, chosen_k, cfg.clustering)
    _log(verbose, f"{METHOD_NAMES.get(best.method, best.method)}: k={chosen_k}, {len(runs)} restarts, "
                  f"objective {best.objective:.4f}, converged in {best.n_iter} iterations")
    timings["clustering"] = time.time() - t

    # ------------------------------------------- 8. evaluation / comparison
    t = time.time()
    full_cols = [c for c in indicators.columns if not c.startswith(("hour_share_", "dow_share_"))]
    full_input = pd.concat([indicators[full_cols], heuristics.select_dtypes(include=[np.number])], axis=1)
    X_full = full_resolution_features(full_input, list(full_input.columns))
    compliance = assess_compliance(
        identifiers_dropped, content_dropped, pseudonymised, consent_info,
        list(fz.fuzzy_vectors.columns), demographics, cfg.compliance.min_group_size,
    ).to_dict()
    expert = None
    if cfg.evaluation.expert_ratings_file and Path(cfg.evaluation.expert_ratings_file).exists():
        expert = pd.read_csv(cfg.evaluation.expert_ratings_file)
    comparison, evaluations = compare_methods(
        Xc, X_full.to_numpy(dtype=float), chosen_k, cfg.clustering, cfg.evaluation,
        methods=cfg.clustering.compare_methods, compliance=compliance,
        min_group_size=cfg.compliance.min_group_size, expert_ratings=expert,
        precomputed={best.method: (best, runs)},
        X_fuzzy_profile=fz.fuzzy_vectors.to_numpy(dtype=float),
    )
    _log(verbose, "comparison:\n" + comparison[["method", "silhouette_or_fsi", "cluster_stability",
                                                   "interpretability_score", "privacy_compliance"]].round(3).to_string(index=False))
    timings["evaluation"] = time.time() - t

    # ----------------------------------------------------- 9. profiling
    t = time.time()
    U = best.memberships
    averages = cluster_feature_averages(indicators, U, weighted=False)      # by highest membership
    averages_weighted = cluster_feature_averages(indicators, U, weighted=True)  # membership-weighted
    labels_df = label_archetypes(averages)
    names = list(labels_df["archetype"])
    profiles = {
        "feature_averages": averages,
        "feature_averages_weighted": averages_weighted,
        "labels": labels_df,
        "distribution": membership_distribution(U),
        "dominant_attributes": dominant_attributes(fz.fuzzy_vectors, U),
        "overlap": overlap_report(U),
        "alpha_cut_segments": fz.alpha_cut_segments,
    }
    _log(verbose, "clusters: " + ", ".join(f"C{i + 1}={n} ({s})" for i, (n, s) in enumerate(zip(names, best.cluster_sizes()))))
    timings["profiling"] = time.time() - t

    result = PipelineResult(
        config=cfg, output_dir=out_dir, compliance=compliance, indicators=indicators, heuristics=heuristics,
        demographics=demographics, rule_effect=effect, fuzzification=fz, clustering_input=X_cluster, pca=pca,
        k_metrics=k_metrics, k_suggestions=k_sugg, k=chosen_k, result=best, runs=runs, comparison=comparison,
        evaluations=evaluations, profiles=profiles, timings=timings,
    )
    if write:
        _write_outputs(result, engine, rule_layer, effect, X2, names, fig_dir, verbose)
    timings["total"] = time.time() - t0
    _log(verbose, f"done in {timings['total']:.1f}s -> {out_dir}")
    return result


# --------------------------------------------------------------------------- #
def _write_outputs(res: PipelineResult, engine: FuzzificationEngine, rule_layer: FuzzyRuleLayer, effect: RuleEffect,
                   X2: np.ndarray, names: List[str], fig_dir: Path, verbose: bool) -> None:
    out = res.output_dir
    cfg = res.config
    art = res.artifacts
    uid = res.indicators.index.name or "user_id"

    def save_df(df: pd.DataFrame, name: str, index: bool = True) -> None:
        path = out / name
        df.to_csv(path, index=index)
        art[name] = str(path)

    def save_json(obj: Any, name: str) -> None:
        path = out / name
        with path.open("w", encoding="utf-8") as fh:
            json.dump(_to_jsonable(obj), fh, indent=2)
        art[name] = str(path)

    save_json(cfg.to_dict(), "config_used.json")
    save_json(res.compliance, "compliance_report.json")
    save_df(res.indicators, "user_indicators.csv")
    save_df(indicator_table(res.indicators), "indicator_summary.csv")
    save_df(res.heuristics, "user_heuristic_attributes.csv")
    save_df(res.demographics, "user_compliant_demographics.csv")
    save_df(engine.describe(), "membership_functions.csv", index=False)
    save_json(res.fuzzification.feature_specs, "membership_functions.json")
    save_df(rule_layer.describe(), "fuzzy_rules.csv", index=False)
    save_df(effect.firing_strengths, "rule_firing_strengths.csv")
    save_df(res.fuzzification.fuzzy_vectors, "user_fuzzy_vectors.csv")
    save_df(res.fuzzification.alpha_cut_segments, "alpha_cut_segments.csv", index=False)
    if res.pca is not None:
        save_df(res.pca.variance_table(), "pca_explained_variance.csv", index=False)
        save_df(res.pca.loadings(10), "pca_loadings.csv", index=False)
        save_df(res.pca.scores, "user_principal_components.csv")
    save_df(res.k_metrics, "k_selection_metrics.csv", index=False)
    save_json(res.k_suggestions, "k_selection_suggestions.json")

    k = res.k
    U = res.result.memberships
    mem = pd.DataFrame(U, index=res.indicators.index, columns=[f"C{j + 1}" for j in range(k)])
    mem["highest_membership_cluster"] = [f"C{j + 1}" for j in res.result.labels]
    mem["archetype"] = [names[j] for j in res.result.labels]
    save_df(mem, "user_memberships.csv")
    if "typicalities" in res.result.extras:
        T = pd.DataFrame(res.result.extras["typicalities"], index=res.indicators.index, columns=[f"C{j + 1}" for j in range(k)])
        save_df(T, "user_typicalities.csv")
    centers = pd.DataFrame(res.result.centers, index=[f"C{j + 1}" for j in range(k)], columns=list(res.clustering_input.columns))
    save_df(centers, "cluster_centers_clustering_space.csv")
    if res.pca is not None and cfg.reduction.clustering_space == "pca":
        back = res.pca.inverse_transform(res.result.centers)
        save_df(pd.DataFrame(back, index=centers.index, columns=res.pca.feature_names), "cluster_centers_fuzzy_space.csv")
    save_json({**res.result.to_dict(), "archetypes": names,
               "runs": [{"seed": r.seed, "objective": r.objective, "n_iter": r.n_iter, "converged": r.converged} for r in res.runs]},
              "clustering_result.json")

    prof = res.profiles
    save_df(prof["feature_averages"], "cluster_feature_averages.csv")
    save_df(prof["feature_averages_weighted"], "cluster_feature_averages_membership_weighted.csv")
    save_df(prof["labels"], "cluster_archetypes.csv", index=False)
    save_df(prof["distribution"], "cluster_distribution.csv", index=False)
    save_df(prof["dominant_attributes"], "cluster_dominant_attributes.csv", index=False)
    save_df(prof["overlap"]["pairwise_overlap"], "cluster_pairwise_overlap.csv")
    save_json({k2: v for k2, v in prof["overlap"].items() if k2 != "pairwise_overlap"}, "cluster_overlap_summary.json")
    save_df(res.comparison, "method_comparison.csv", index=False)
    save_json({m: {"separation": e.separation, "stability": e.stability, "interpretability": e.interpretability,
                   "privacy": e.privacy, "indices": e.extra_indices, "result": e.result.to_dict()}
               for m, e in res.evaluations.items()}, "method_evaluations.json")
    save_json(res.summary(), "summary.json")
    save_json(res.timings, "timings.json")

    if cfg.output.save_figures:
        dpi = cfg.output.figure_dpi
        art["fig_membership_functions"] = str(plots.plot_membership_functions(engine, fig_dir / "membership_functions.png", dpi))
        if engine.shift_patterns:
            art["fig_shift_patterns"] = str(plots.plot_shift_patterns(engine.shift_patterns, fig_dir / "shift_patterns.png", dpi))
        posting = next((f for f in engine.features if f.alpha_cut), None)
        if posting is not None:
            term = posting.alpha_cut.get("terms", list(posting.terms))[0]
            art["fig_alpha_cuts"] = str(plots.plot_alpha_cuts(posting.terms[term], cfg.fuzzification.alpha_levels,
                                                             fig_dir / "alpha_cuts.png", f"Alpha-cuts on '{posting.name} is {term}'", dpi))
        if res.pca is not None:
            art["fig_explained_variance"] = str(plots.plot_explained_variance(res.pca.explained_variance_ratio, fig_dir / "pca_explained_variance.png", 20, dpi))
        art["fig_k_selection"] = str(plots.plot_k_selection(res.k_metrics, res.k_suggestions, fig_dir / "k_selection.png", dpi))
        centers2 = _centers_in_plane(res, X2)
        if res.result.is_fuzzy:
            art["fig_membership_contours"] = str(plots.plot_membership_contours(
                X2, U, centers2, cfg.clustering.m, fig_dir / "membership_contours.png", names,
                cfg.output.max_plot_points, cfg.clustering.seed, dpi=dpi))
        art["fig_cluster_scatter"] = str(plots.plot_cluster_scatter(
            X2, res.result.labels, centers2, fig_dir / "cluster_scatter.png", names,
            f"{METHOD_NAMES.get(res.result.method)} (k={k}) in the PCA plane", cfg.output.max_plot_points, cfg.clustering.seed, dpi))
        art["fig_membership_heatmap"] = str(plots.plot_membership_heatmap(U, fig_dir / "membership_heatmap.png", names, dpi=dpi))
        art["fig_cluster_profiles"] = str(plots.plot_cluster_profiles(prof["feature_averages"], fig_dir / "cluster_profiles.png", names, dpi))
        art["fig_method_comparison"] = str(plots.plot_method_comparison(res.comparison, fig_dir / "method_comparison.png", dpi))

    _write_report(res, names, out / "report.md")
    art["report.md"] = str(out / "report.md")
    save_json(art, "artifacts.json")
    _log(verbose, f"wrote {len(art)} artefacts")


def _centers_in_plane(res: PipelineResult, X2: np.ndarray) -> np.ndarray:
    """Cluster centres expressed in the PC1-PC2 plane."""
    centers = res.result.centers
    if res.pca is not None and res.config.reduction.clustering_space == "pca":
        return centers[:, :2]
    if res.pca is not None:
        return res.pca.pca.transform(centers)[:, :2]
    # membership-weighted mean of the 2-D coordinates
    W = res.result.memberships / np.maximum(res.result.memberships.sum(axis=0, keepdims=True), 1e-12)
    return W.T @ X2


def _write_report(res: PipelineResult, names: List[str], path: Path) -> None:
    s = res.summary()
    comp = res.comparison.copy()
    lines = [
        f"# Fuzzy characterization report: {res.config.output.run_name}",
        "",
        f"* Users analysed: **{s['n_users']}** (consent coverage: {res.compliance.get('consent', {}).get('consent_coverage')})",
        f"* Fuzzy attributes: **{s['n_fuzzy_attributes']}**; principal components retained: **{s['n_components']}**"
        + (f" ({s['cumulative_variance']:.1%} cumulative variance)" if s["cumulative_variance"] else ""),
        f"* Number of clusters: **k = {s['k']}** (recommended {res.k_suggestions['recommended_k']}; "
        f"DBI {res.k_suggestions['davies_bouldin_min']}, elbow {res.k_suggestions['elbow_fcm_objective']}, "
        f"XB {res.k_suggestions['xie_beni_min']}, PC {res.k_suggestions['partition_coefficient_max']}, CE {res.k_suggestions['classification_entropy_min']})",
        f"* Main method: **{METHOD_NAMES.get(res.result.method, res.result.method)}**, {len(res.runs)} restarts",
        f"* Privacy compliance of the fuzzy representation: **{res.compliance.get('level')}**; identifiers excluded: "
        f"{res.compliance.get('identifiers_excluded')}; pseudonymised: {res.compliance.get('pseudonymised')}",
        "",
        "## Behavioural clusters",
        "",
        "| Cluster | Archetype | Users (highest membership) | Share | Membership mass | Mean membership |",
        "|---|---|---|---|---|---|",
    ]
    dist = res.profiles["distribution"]
    for i, row in dist.iterrows():
        lines.append(f"| {row['cluster']} | {names[i]} | {int(row['users_by_highest_membership'])} | "
                     f"{row['share_by_highest_membership']:.1%} | {row['membership_mass']:.1f} | {row['mean_membership_of_members']:.2f} |")
    lines += ["", "## Average behavioural feature values per cluster", ""]
    avg = res.profiles["feature_averages"]
    cols = [c for c in avg.columns if c != "description"]
    lines.append("| Behavioural feature | " + " | ".join(f"{c} {names[i]}" for i, c in enumerate(cols)) + " |")
    lines.append("|---|" + "---|" * len(cols))
    for idx, row in avg.iterrows():
        vals = [f"{row[c] * 100:.1f}%" if "share" in idx else f"{row[c]:.1f}" for c in cols]
        lines.append(f"| {row['description']} | " + " | ".join(vals) + " |")
    ov = res.profiles["overlap"]
    lines += ["", f"Users with a second membership of at least {ov['mixed_threshold']}: **{ov['share_users_mixed']:.1%}** "
                  f"(mean highest membership {ov['mean_max_membership']:.2f}).", "",
              "## Segmentation quality comparison", "",
              "| Method | Representation | Silhouette / FSI | Cluster stability | Interpretability | Privacy |",
              "|---|---|---|---|---|---|"]
    for _, r in comp.iterrows():
        lines.append(f"| {r['method_name']} | {r['representation']} | {r['silhouette_or_fsi']:.2f} ({r['separation_metric']}) | "
                     f"{r['cluster_stability']:.2f} | {r['interpretability_score']:.2f} | {r['privacy_compliance']} |")
    sem = any(e.interpretability.get("semantic_assessed") for e in res.evaluations.values())
    lines += ["", ("Semantic consistency was assessed from expert ratings." if sem else
                   "Semantic consistency (expert review) was not assessed; the interpretability score combines feature "
                   "dominance clarity and membership transparency only."),
              "", "## Alpha-cut segments (posting and other cut features)", ""]
    seg = res.profiles["alpha_cut_segments"]
    seg = seg[seg["feature"].str.contains("__high")]
    lines.append("| Attribute | alpha | Users | Share | Reportable |")
    lines.append("|---|---|---|---|---|")
    for _, r in seg.iterrows():
        lines.append(f"| {r['feature']} | {r['alpha']} | {int(r['size'])} | {r['share']:.1%} | {'yes' if r['reportable'] else 'no (too small)'} |")
    lines += ["", "## Artefacts", ""] + [f"* `{k}`: {v}" for k, v in sorted(res.artifacts.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

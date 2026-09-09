"""Command-line interface: ``fcf <command>`` (or ``python -m fuzzy_characterization``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _cfg(args):
    from .config import load_config

    cfg = load_config(args.config)
    if getattr(args, "out", None):
        cfg.output.root = args.out
    if getattr(args, "run_name", None):
        cfg.output.run_name = args.run_name
    if getattr(args, "data_root", None):
        cfg.data.root = args.data_root
    return cfg


def cmd_generate_data(args) -> int:
    from .data.synthetic import SyntheticOptions, write_synthetic_datasets

    opt = SyntheticOptions(n_users=args.n_users, days=args.days, seed=args.seed, start=args.start)
    paths = write_synthetic_datasets(args.out, opt, compress_events=not args.no_compress)
    for name, p in paths.items():
        print(f"{name:26s} -> {p}")
    return 0


def cmd_run(args) -> int:
    from .pipeline import run_pipeline

    cfg = _cfg(args)
    if args.method:
        cfg.clustering.method = args.method
    if args.restarts:
        cfg.clustering.n_restarts = args.restarts
    if args.rules is not None:
        cfg.rules.enabled = bool(args.rules)
    res = run_pipeline(cfg, k=args.k, write=not args.no_write, verbose=not args.quiet)
    print(json.dumps(res.summary(), indent=2, default=str))
    return 0


def cmd_select_k(args) -> int:
    from .pipeline import run_pipeline

    cfg = _cfg(args)
    cfg.clustering.k_min, cfg.clustering.k_max = args.k_min, args.k_max
    cfg.clustering.compare_methods = [cfg.clustering.method]
    cfg.output.run_name = args.run_name or "select_k"
    res = run_pipeline(cfg, write=not args.no_write, verbose=not args.quiet)
    print(res.k_metrics.round(4).to_string(index=False))
    print(json.dumps(res.k_suggestions, indent=2))
    return 0


def cmd_compare(args) -> int:
    from .pipeline import run_pipeline

    cfg = _cfg(args)
    if args.methods:
        cfg.clustering.compare_methods = [m.strip() for m in args.methods.split(",")]
    res = run_pipeline(cfg, k=args.k, write=not args.no_write, verbose=not args.quiet)
    cols = ["method_name", "representation", "silhouette_or_fsi", "separation_metric", "cluster_stability",
            "interpretability_score", "privacy_compliance"]
    print(res.comparison[cols].round(3).to_string(index=False))
    return 0


def cmd_plot_mf(args) -> int:
    from .fuzzification import FuzzificationEngine
    from .visualization import plots

    cfg = _cfg(args)
    engine = FuzzificationEngine(cfg.fuzzification.features, cfg.fuzzification.default_normalisation,
                                 cfg.fuzzification.alpha_levels, cfg.fuzzification.shift_patterns)
    out = Path(args.out or "outputs/figures")
    p1 = plots.plot_membership_functions(engine, out / "membership_functions.png")
    print(p1)
    if engine.shift_patterns:
        print(plots.plot_shift_patterns(engine.shift_patterns, out / "shift_patterns.png"))
    for f in engine.features:
        if f.alpha_cut:
            term = f.alpha_cut.get("terms", list(f.terms))[0]
            print(plots.plot_alpha_cuts(f.terms[term], cfg.fuzzification.alpha_levels, out / f"alpha_cuts_{f.name}.png",
                                        f"Alpha-cuts on '{f.name} is {term}'"))
    return 0


def cmd_describe(args) -> int:
    from .fuzzification import FuzzificationEngine, FuzzyRuleLayer

    cfg = _cfg(args)
    engine = FuzzificationEngine(cfg.fuzzification.features, cfg.fuzzification.default_normalisation,
                                 cfg.fuzzification.alpha_levels, cfg.fuzzification.shift_patterns)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print("== Membership functions ==")
    print(engine.describe().drop(columns=["rationale"]).to_string(index=False))
    print("\n== Shift patterns ==")
    for name, mf in engine.shift_patterns.items():
        print(f"  {name:8s} {mf.family} {mf.params()}")
    layer = FuzzyRuleLayer(cfg.rules.rules, cfg.rules.context_terms, enabled=cfg.rules.enabled)
    print(f"\n== Fuzzy rules (enabled={cfg.rules.enabled}) ==")
    for _, r in layer.describe().iterrows():
        print(f"  [{r['name']}] {r['rule']}")
    return 0


def cmd_unpack_events(args) -> int:
    from .data.loaders import unpack_event_type_counts

    inp = Path(args.input_csv)
    out = Path(args.output) if args.output else inp.with_name(inp.stem + "_unpacked.csv")
    df = unpack_event_type_counts(pd.read_csv(inp), args.column)
    df.to_csv(out, index=False)
    print(f"Wrote: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fcf", description="Privacy-compliant fuzzy user characterization framework")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, config_required: bool = True):
        sp.add_argument("--config", "-c", default="configs/retail_case.yaml", help="pipeline YAML configuration")
        sp.add_argument("--out", help="output root directory (overrides output.root)")
        sp.add_argument("--run-name", dest="run_name", help="run name (overrides output.run_name)")
        sp.add_argument("--data-root", dest="data_root", help="dataset directory (overrides data.root)")
        sp.add_argument("--no-write", action="store_true", help="do not write artefacts")
        sp.add_argument("--quiet", "-q", action="store_true")

    g = sub.add_parser("generate-data", help="generate the synthetic demo datasets")
    g.add_argument("--out", default="data/synthetic")
    g.add_argument("--n-users", type=int, default=400)
    g.add_argument("--days", type=int, default=42)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--start", default="2025-03-03")
    g.add_argument("--no-compress", action="store_true", help="write api_events.csv uncompressed")
    g.set_defaults(func=cmd_generate_data)

    r = sub.add_parser("run", help="run the complete pipeline")
    common(r)
    r.add_argument("--k", type=int, help="number of clusters (overrides the config / recommendation)")
    r.add_argument("--method", choices=["fcm", "pfcm", "gk", "kmeans", "ahc"], help="main clustering method")
    r.add_argument("--restarts", type=int, help="number of random restarts")
    r.add_argument("--rules", type=int, choices=[0, 1], help="enable (1) or disable (0) the fuzzy rule layer")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("select-k", help="validity indices over a range of k")
    common(s)
    s.add_argument("--k-min", type=int, default=2)
    s.add_argument("--k-max", type=int, default=10)
    s.set_defaults(func=cmd_select_k)

    c = sub.add_parser("compare", help="comparative evaluation of clustering methods")
    common(c)
    c.add_argument("--k", type=int)
    c.add_argument("--methods", help="comma-separated subset of fcm,pfcm,gk,kmeans,ahc")
    c.set_defaults(func=cmd_compare)

    m = sub.add_parser("plot-mf", help="plot the configured membership functions")
    common(m)
    m.set_defaults(func=cmd_plot_mf)

    d = sub.add_parser("describe", help="print membership functions and fuzzy rules of a config")
    common(d)
    d.set_defaults(func=cmd_describe)

    u = sub.add_parser("unpack-events", help="expand a serialised event_type_counts column (legacy datasets)")
    u.add_argument("input_csv")
    u.add_argument("--output")
    u.add_argument("--column", default="event_type_counts")
    u.set_defaults(func=cmd_unpack_events)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

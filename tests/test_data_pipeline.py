import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from fuzzy_characterization.compliance import Pseudonymiser, apply_consent, generalise_demographics
from fuzzy_characterization.compliance.checks import check_feature_space
from fuzzy_characterization.data.indicators import build_behavioural_indicators
from fuzzy_characterization.data.loaders import parse_counts_cell, unpack_event_type_counts
from fuzzy_characterization.heuristics import build_heuristic_attributes, flag_engaging_posts, hierarchy_score
from fuzzy_characterization.pipeline import run_pipeline
from fuzzy_characterization.profiling import label_archetypes


def test_sample_tables_are_coherent(sample_tables):
    t = sample_tables
    users = set(t["users"]["user_id"])
    assert set(t["api_events"]["user_id"]) <= users
    assert set(t["posts"]["author_id"]) <= users
    assert set(t["comments"]["post_id"]) <= set(t["posts"]["post_id"])
    assert set(t["reactions"]["post_id"]) <= set(t["posts"]["post_id"])
    n_posts = (t["api_events"]["event_type"] == "post.created").sum()
    assert len(t["posts"]) == n_posts
    counts = t["comments"].groupby("post_id").size()
    merged = t["posts"].set_index("post_id")["comment_count"]
    assert (merged.loc[counts.index] == counts).all()
    assert "text" not in t["posts"].columns  # no content anywhere


def test_hierarchy_score_examples():
    df = pd.DataFrame({"job_title": ["Chief Executive Officer", "Senior Manager", "manager", "Sales Associate", "General Manager"]})
    s = hierarchy_score(df)
    assert s.iloc[0] >= 20 and s.iloc[1] == 8 and s.iloc[2] == 6 and s.iloc[3] == 1
    assert s.iloc[4] == 9  # general manager counted once, not as manager too


def test_engaging_posts_flag():
    posts = pd.DataFrame({"post_id": list("abcd"), "stream_id": ["s"] * 4, "comment_count": [0, 1, 5, 6],
                          "like_count": [1, 1, 10, 0], "has_video": [0, 0, 0, 1]})
    out = flag_engaging_posts(posts)
    assert out.set_index("post_id")["engaging"].tolist() == [0, 0, 1, 1]


def test_parse_counts_cell_variants():
    assert parse_counts_cell('{"a": 1, "b": 2}') == {"a": 1, "b": 2}
    assert parse_counts_cell("{post-created:3,login:2}") == {"post-created": 3, "login": 2}
    assert parse_counts_cell("{'x': 1}") == {"x": 1}
    assert parse_counts_cell(None) == {}
    df = unpack_event_type_counts(pd.DataFrame({"actor_id": [1], "event_type_counts": ['{"a": 2}']}))
    assert df.loc[0, "a"] == 2


def test_compliance_steps(datasets):
    data, info = apply_consent(datasets, True)
    assert info["users_after"] <= info["users_before"] and 0 < info["consent_coverage"] <= 1
    ps = Pseudonymiser("salt")
    pdata = ps.apply_datasets(data)
    assert not set(pdata.users["user_id"]) & set(data.users["user_id"])
    assert ps.pseudonym("u1") == ps.pseudonym("u1")
    from fuzzy_characterization.config import ComplianceConfig
    demo = generalise_demographics(pdata.users, ComplianceConfig())
    assert "age" not in demo.columns and "age_band" in demo.columns and "airport" not in demo.columns
    assert check_feature_space(["posting__high", "age__young", "mobile_channel__mobile_oriented"]) == []
    assert check_feature_space(["full_name", "message_text", "age"]) == ["full_name", "message_text", "age"]


def test_indicators_and_heuristics(datasets):
    ids = pd.Index(datasets.users["user_id"])
    ind = build_behavioural_indicators(datasets, user_ids=ids)
    assert len(ind) == len(ids)
    hour_cols = [c for c in ind.columns if c.startswith("hour_share_")]
    assert len(hour_cols) == 24
    active = ind["total_events_per_month"] > 0
    np.testing.assert_allclose(ind.loc[active, hour_cols].sum(axis=1), 1.0)
    np.testing.assert_allclose(ind["working_hours_share"] + ind["non_working_hours_share"], 1.0)
    assert (ind["posts_per_month"] >= 0).all()
    heur = build_heuristic_attributes(datasets, ids)
    assert {"hierarchy_score", "accessible_streams", "broadcast_stream_share", "group_memberships"} <= set(heur.columns)
    assert (heur["accessible_streams"] >= datasets.streams["restricted"].eq(0).sum()).all()


def test_full_pipeline_recovers_archetypes(quick_config, datasets, sample_tables):
    res = run_pipeline(quick_config, data=datasets, k=4, write=True, verbose=False)
    assert res.k == 4 and res.result.memberships.shape[1] == 4
    assert res.fuzzification.n_features >= 20
    assert res.compliance["level"] == "High"
    assert set(res.comparison["method"]) == {"fcm", "kmeans"}
    assert (res.output_dir / "report.md").exists() and (res.output_dir / "user_memberships.csv").exists()
    # archetype labels are unique and match the sample ground truth reasonably
    labels = res.profiles["labels"]
    assert labels["archetype"].is_unique
    gt = sample_tables["ground_truth_archetypes"].copy()
    ps = Pseudonymiser(quick_config.compliance.pseudonym_salt)
    gt["pid"] = gt["user_id"].map(ps.pseudonym)
    gt = gt.set_index("pid").reindex(res.indicators.index)
    ari = adjusted_rand_score(gt["archetype"], res.result.labels)
    assert ari > 0.6, ari
    # the privacy check on the fuzzy representation holds
    fcm_row = res.comparison.set_index("method").loc["fcm"]
    assert fcm_row["privacy_compliance"] == "High" and fcm_row["separation_metric"] == "FSI"


def test_pipeline_with_rules(quick_config, datasets):
    import copy
    cfg = copy.deepcopy(quick_config)
    cfg.rules.enabled = True
    cfg.output.run_name = "quick_rules"
    res = run_pipeline(cfg, data=datasets, k=4, write=False, verbose=False)
    fs = res.rule_effect.firing_strengths
    assert fs.shape[1] == 3 and (fs.values >= 0).all() and (fs.values <= 1).all()
    assert fs["operational_rhythm"].max() == 1.0
    assert (res.rule_effect.feature_weights["posting"] <= 1.0).all()

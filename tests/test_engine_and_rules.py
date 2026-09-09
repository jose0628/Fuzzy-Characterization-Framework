import numpy as np
import pandas as pd
import pytest

from fuzzy_characterization.config import load_config
from fuzzy_characterization.fuzzification import FuzzificationEngine, FuzzyRuleLayer
from fuzzy_characterization.fuzzification.engine import HOUR_COLUMNS


def _indicators(n=50, seed=0):
    rng = np.random.default_rng(seed)
    ind = pd.DataFrame(index=pd.Index([f"u{i}" for i in range(n)], name="user_id"))
    for col in ("posts_per_month", "comments_per_month", "reactions_per_month", "chat_events_per_month",
                "topic_accesses_per_month", "logins_per_month"):
        ind[col] = rng.gamma(1.0, 5.0, n)
    ind["mobile_login_share"] = rng.random(n)
    ind["weekend_share"] = rng.random(n) * 0.5
    ind["weekday_entropy"] = rng.random(n)
    ind["age_band_midpoint"] = rng.choice([21.5, 30, 40, 50, 60], n)
    hours = rng.dirichlet(np.ones(24), n)
    for h in range(24):
        ind[f"hour_share_{h:02d}"] = hours[:, h]
    return ind


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/retail_case.yaml")


def test_engine_produces_memberships_in_unit_interval(cfg):
    ind = _indicators()
    engine = FuzzificationEngine(cfg.fuzzification.features, shift_patterns=cfg.fuzzification.shift_patterns)
    res = engine.fit_transform(ind)
    fv = res.fuzzy_vectors
    assert fv.shape[0] == len(ind) and fv.shape[1] >= 20
    assert (fv.values >= 0).all() and (fv.values <= 1).all()
    assert "posting__high" in fv.columns and "working_hours__working" in fv.columns
    # alpha-cut filter on posting__high: no memberships strictly between 0 and 0.5
    ph = fv["posting__high"].values
    assert not np.any((ph > 0) & (ph < 0.5))
    # working + non-working sum to one
    np.testing.assert_allclose(fv["working_hours__working"] + fv["working_hours__non_working"], 1.0)


def test_working_hours_membership_uses_shift_pattern(cfg):
    ind = _indicators(n=2)
    # user 0 active only at 02:00, user 1 only at 12:00
    for h in range(24):
        ind[f"hour_share_{h:02d}"] = 0.0
    ind.loc["u0", "hour_share_02"] = 1.0
    ind.loc["u1", "hour_share_12"] = 1.0
    engine = FuzzificationEngine(cfg.fuzzification.features, shift_patterns=cfg.fuzzification.shift_patterns)
    default = engine.fit_transform(ind).fuzzy_vectors["working_hours__working"]
    assert default["u0"] == 0.0 and default["u1"] == 1.0
    night = engine.transform(ind, shift_pattern=pd.Series(["night", "night"], index=ind.index)).fuzzy_vectors["working_hours__working"]
    assert night["u0"] == 1.0 and night["u1"] == 0.0


def test_rule_layer_scales_weights_and_sets_shift(cfg):
    ind = _indicators(n=4)
    context = pd.DataFrame(
        {
            "hierarchy_score": [0, 0, 20, 20],
            "broadcast_stream_share": [1.0, 0.0, 1.0, 0.0],
            "accessible_streams": [2, 20, 2, 20],
            "employee_category": ["shift-based", "office-based", "shift-based", "office-based"],
            "shift_pattern": ["night", "office", "long", "office"],
        },
        index=ind.index,
    )
    layer = FuzzyRuleLayer(cfg.rules.rules, cfg.rules.context_terms, enabled=True)
    effect = layer.apply(ind, context, ["posting", "commenting", "reactions", "topic_access"])
    w = effect.feature_weights
    # top-down rule fires fully for user 0 (low hierarchy, broadcast streams) and not for user 3
    assert w.loc["u0", "posting"] == pytest.approx(0.5, abs=0.05)
    assert w.loc["u0", "reactions"] == pytest.approx(1.5, abs=0.05)
    assert w.loc["u3", "posting"] == pytest.approx(1.0, abs=1e-6)
    # opportunity rule rescales indicators of users with few streams
    assert effect.indicators.loc["u0", "topic_accesses_per_month"] > ind.loc["u0", "topic_accesses_per_month"]
    assert effect.indicators.loc["u1", "topic_accesses_per_month"] == pytest.approx(ind.loc["u1", "topic_accesses_per_month"])
    # operational rhythm sets the shift pattern from the column for shift-based users only
    assert effect.shift_pattern["u0"] == "night" and effect.shift_pattern["u2"] == "long"
    assert pd.isna(effect.shift_pattern["u1"])
    assert set(effect.firing_strengths.columns) == {r["name"] for r in cfg.rules.rules}
    assert "IF" in layer.describe().iloc[0]["rule"]


def test_disabled_rule_layer_is_identity(cfg):
    ind = _indicators(n=3)
    context = pd.DataFrame({"hierarchy_score": [0, 1, 2]}, index=ind.index)
    layer = FuzzyRuleLayer(cfg.rules.rules, cfg.rules.context_terms, enabled=False)
    effect = layer.apply(ind, context, ["posting"])
    assert (effect.feature_weights == 1.0).all().all()
    pd.testing.assert_frame_equal(effect.indicators, ind)

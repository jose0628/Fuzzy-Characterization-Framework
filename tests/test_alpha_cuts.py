import numpy as np
import pandas as pd

from fuzzy_characterization.fuzzification.alpha_cuts import alpha_cut, alpha_cut_report, nested_alpha_cuts, segment_stability


def test_alpha_cuts_are_nested():
    mu = np.random.default_rng(0).random(500)
    cuts = nested_alpha_cuts(mu, [0.2, 0.5, 0.8])
    assert cuts[0.8].sum() <= cuts[0.5].sum() <= cuts[0.2].sum()
    assert np.all(cuts[0.8] <= cuts[0.5]) and np.all(cuts[0.5] <= cuts[0.2])


def test_alpha_cut_threshold_semantics():
    mu = np.array([0.1, 0.5, 0.9])
    assert alpha_cut(mu, 0.5).tolist() == [False, True, True]
    assert alpha_cut(mu, 0.5, strong=True).tolist() == [False, False, True]


def test_report_flags_small_groups():
    df = pd.DataFrame({"posting__high": np.r_[np.ones(5), np.zeros(95)]})
    rep = alpha_cut_report(df, [0.5], min_group_size=20)
    row = rep.iloc[0]
    assert row["size"] == 5 and not row["reportable"]
    stab = segment_stability(df)
    assert stab["posting__high"] == 1.0

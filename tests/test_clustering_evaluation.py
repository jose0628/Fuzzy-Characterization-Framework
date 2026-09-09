import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from fuzzy_characterization.clustering import ahc, detect_elbow, evaluate_k_range, fcm, gustafson_kessel, kmeans, pfcm, suggest_k
from fuzzy_characterization.clustering.fcm import fcm_predict
from fuzzy_characterization.evaluation import (
    align_memberships, classification_entropy, cluster_stability, fuzzy_silhouette_index, interpretability_score,
    partition_coefficient, privacy_assessment, xie_beni,
)
from fuzzy_characterization.evaluation.stability import membership_agreement


def blobs(n_per=60, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [6, 0], [0, 6], [6, 6]], dtype=float)
    X = np.vstack([c + rng.normal(0, 0.6, (n_per, 2)) for c in centers])
    y = np.repeat(np.arange(4), n_per)
    return X, y


def test_fcm_recovers_blobs_and_memberships_sum_to_one():
    X, y = blobs()
    res = fcm(X, 4, m=2.0, seed=1)
    assert res.converged and res.memberships.shape == (len(X), 4)
    np.testing.assert_allclose(res.memberships.sum(axis=1), 1.0, atol=1e-9)
    assert adjusted_rand_score(y, res.labels) > 0.95
    # objective decreases monotonically
    hist = res.extras["objective_history"]
    assert all(b <= a + 1e-9 for a, b in zip(hist, hist[1:]))
    # predict on the centres gives crisp memberships
    U = fcm_predict(res.centers, res.centers, 2.0)
    assert np.allclose(U.max(axis=1), 1.0)


@pytest.mark.parametrize("method", [pfcm, gustafson_kessel, kmeans, ahc])
def test_other_methods_recover_blobs(method):
    X, y = blobs()
    res = method(X, 4, seed=1) if method is not ahc else method(X, 4)
    assert adjusted_rand_score(y, res.labels) > 0.9
    assert res.memberships.shape == (len(X), 4)


def test_pfcm_typicalities_and_gk_covariances():
    X, _ = blobs()
    r = pfcm(X, 4, seed=1)
    T = r.extras["typicalities"]
    assert T.shape == (len(X), 4) and (T >= 0).all() and (T <= 1).all()
    g = gustafson_kessel(X, 4, seed=1)
    assert g.extras["covariances"].shape == (4, 2, 2)


def test_validity_indices():
    X, _ = blobs()
    res = fcm(X, 4, seed=1)
    U = res.memberships
    assert 0.25 <= partition_coefficient(U) <= 1.0
    assert classification_entropy(U) >= 0
    assert xie_beni(X, res.centers, U) < 1.0
    fsi = fuzzy_silhouette_index(X, U)
    assert 0.5 < fsi <= 1.0


def test_k_selection_recommends_four():
    X, _ = blobs(n_per=40)
    metrics = evaluate_k_range(X, 2, 6, n_restarts=2)
    sugg = suggest_k(metrics)
    assert sugg["davies_bouldin_min"] == 4
    assert sugg["recommended_k"] == 4
    assert detect_elbow([2, 3, 4, 5, 6], [100, 60, 20, 18, 17]) == 4


def test_stability_and_alignment():
    X, _ = blobs()
    stab = cluster_stability(lambda Xs, s: fcm(Xs, 4, seed=s), X, n_runs=4, seed=1)
    assert stab["ari_mean"] > 0.95 and stab["membership_agreement"] > 0.95
    U = fcm(X, 4, seed=1).memberships
    perm = U[:, [2, 0, 3, 1]]
    np.testing.assert_allclose(align_memberships(U, perm), U)
    assert membership_agreement(U, perm) == pytest.approx(1.0)
    sub = cluster_stability(lambda Xs, s: ahc(Xs, 4), X, n_runs=3, seed=1, subsample=0.8)
    assert sub["ari_mean"] > 0.9


def test_interpretability_and_privacy():
    X, _ = blobs()
    r = fcm(X, 4, seed=1)
    fz = interpretability_score(X, r.memberships, is_fuzzy=True)
    cr = interpretability_score(X, kmeans(X, 4, seed=1).memberships, is_fuzzy=False)
    assert 0 <= fz.score <= 1 and fz.membership_transparency > 0
    assert cr.membership_transparency == 0.0 and not cr.semantic_assessed
    priv = privacy_assessment("fuzzy", r.cluster_sizes(), 20, {"violations": []})
    assert priv["level"] == "High"
    assert privacy_assessment("full", r.cluster_sizes(), 20)["level"] == "Low"
    assert privacy_assessment("fuzzy", [100, 5], 20)["level"] == "Low"

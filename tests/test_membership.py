import numpy as np
import pytest

from fuzzy_characterization.fuzzification.membership import (
    Complement, Gaussian, Sigmoidal, SplitShift, Trapezoidal, Triangular, WrapAroundTrapezoidal,
    build_membership_function, gaussian, linguistic_partition, sigmoidal, trapezoidal, triangular,
)


def test_trapezoidal_working_hours_eq2():
    # mu_trap(x; 6, 8, 16, 18): 0 below 6, ramps to 1 at 8, plateau to 16, ramps to 0 at 18
    x = np.array([0, 6, 7, 8, 12, 16, 17, 18, 24])
    expected = np.array([0, 0, 0.5, 1, 1, 1, 0.5, 0, 0])
    np.testing.assert_allclose(trapezoidal(x, 6, 8, 16, 18), expected)


def test_non_working_hours_is_complement_eq3():
    work = Trapezoidal(a=6, b=8, c=16, d=18)
    non_work = Complement(inner=work)
    x = np.linspace(0, 24, 97)
    np.testing.assert_allclose(work(x) + non_work(x), 1.0)


def test_triangular_eq4_and_alpha_cut_interval():
    tri = Triangular(a=0, b=10, c=20)
    np.testing.assert_allclose(tri([0, 5, 10, 15, 20]), [0, 0.5, 1, 0.5, 0])
    lo, hi = tri.alpha_cut_interval(0.5)
    assert lo == pytest.approx(5, abs=0.05) and hi == pytest.approx(15, abs=0.05)


def test_gaussian_and_sigmoid_shapes():
    assert gaussian(30, 30, 10) == pytest.approx(1.0)
    assert gaussian(40, 30, 10) == pytest.approx(np.exp(-0.5))
    assert sigmoidal(25, 0.15, 25) == pytest.approx(0.5)
    assert sigmoidal(60, 0.15, 25) > 0.99
    assert sigmoidal(60, -0.15, 25) < 0.01


def test_wraparound_night_shift():
    night = WrapAroundTrapezoidal(a=20, b=22, c=4, d=6)
    np.testing.assert_allclose(night([0, 2, 4, 5, 6, 12, 20, 21, 22, 23.9]), [1, 1, 1, 0.5, 0, 0, 0, 0.5, 1, 1], atol=1e-9)


def test_split_shift_is_max_of_windows():
    split = SplitShift(windows=[[7, 8, 11.5, 12.5], [13.5, 14.5, 18, 19]])
    assert split(10) == 1.0 and split(16) == 1.0
    assert split(13) == 0.0 and split(3) == 0.0


def test_factory_from_spec():
    mf = build_membership_function({"family": "trapezoidal", "params": [6, 8, 16, 18]})
    assert isinstance(mf, Trapezoidal) and mf(12) == 1.0
    mf = build_membership_function({"family": "gaussian", "c": 0.5, "sigma": 0.1})
    assert isinstance(mf, Gaussian)
    mf = build_membership_function({"family": "sigmoid", "a": 10, "c": 0.5})
    assert isinstance(mf, Sigmoidal)
    with pytest.raises(ValueError):
        build_membership_function({"family": "unknown"})


def test_linguistic_partition_covers_domain():
    terms = linguistic_partition("triangular", ["low", "medium", "high"])
    x = np.linspace(0, 1, 101)
    total = sum(mf(x) for mf in terms.values())
    assert np.all(total > 0.99)  # Ruspini-like partition

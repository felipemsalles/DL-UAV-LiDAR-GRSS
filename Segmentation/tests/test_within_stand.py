"""Tests for the within-stand decomposition.

A model that only knows which stand is which must not score as if it also knew what happens inside
a stand. These tests pin that: a between-only predictor shows no within-stand skill, and the
permutation null is the within-stand one.
"""
import numpy as np
import pandas as pd
import pytest

from greenvista.eval.within_stand import (center_by_group, decompose_correlation,
                                          within_stand_permutation)


def _groups(n_groups=6, per_group=3):
    return np.repeat(np.arange(n_groups), per_group)


def test_center_by_group_removes_group_means():
    g = _groups()
    v = np.arange(len(g), dtype=float)
    c = center_by_group(v, g)
    for u in np.unique(g):
        assert c[g == u].mean() == pytest.approx(0.0, abs=1e-12)


def test_between_only_predictor_has_no_within_skill():
    """Predict each stand's mean exactly and nothing else: r_between perfect, r_within ~0."""
    rng = np.random.default_rng(0)
    g = _groups()
    stand_mean = np.array([100.0, 200, 300, 400, 500, 600])[g]
    y = stand_mean + rng.normal(0, 20, len(g))
    yhat = stand_mean.astype(float)                 # knows the stand, nothing more

    dec = decompose_correlation(y, yhat, g)
    assert dec["r_total"] > 0.9
    assert dec["r_between"] > 0.99
    assert np.isnan(dec["r_within"])                # zero variance within → undefined, not "good"


def test_within_signal_is_detected_when_present():
    rng = np.random.default_rng(1)
    g = _groups(per_group=4)
    stand_mean = np.array([100.0, 200, 300, 400, 500, 600])[g]
    local = rng.normal(0, 30, len(g))
    y = stand_mean + local
    yhat = stand_mean + 0.9 * local                 # tracks the within-stand part too

    dec = decompose_correlation(y, yhat, g)
    r, p, sd = within_stand_permutation(y, yhat, g, n_perm=2000, seed=1)
    assert dec["r_within"] > 0.8
    assert p < 0.01
    assert sd > 0


def test_within_permutation_preserves_stand_means_so_between_skill_cannot_leak():
    """A predictor that is right between stands and pure noise within must not reach significance."""
    rng = np.random.default_rng(2)
    g = _groups(per_group=4)
    stand_mean = np.array([100.0, 200, 300, 400, 500, 600])[g]
    y = stand_mean + rng.normal(0, 30, len(g))
    yhat = stand_mean + rng.normal(0, 30, len(g))   # within part is independent noise

    _, p, _ = within_stand_permutation(y, yhat, g, n_perm=2000, seed=2)
    assert p > 0.05


def test_decompose_reports_degrees_of_freedom_left_for_the_within_test():
    g = _groups(n_groups=6, per_group=3)
    y = np.arange(len(g), dtype=float)
    dec = decompose_correlation(y, y * 2 + 1, g)
    assert dec["n"] == 18
    assert dec["n_groups"] == 6
    assert dec["n_within_df"] == 12


def test_permutation_handles_a_degenerate_within_case():
    """One plot per stand leaves no within-stand variation; must return NaN, not a fake number."""
    g = np.arange(5)
    y = np.array([1.0, 2, 3, 4, 5])
    r, p, sd = within_stand_permutation(y, y, g, n_perm=100, seed=0)
    assert np.isnan(r)
    assert np.isnan(p)


def test_pandas_and_numpy_group_inputs_agree():
    g = _groups()
    rng = np.random.default_rng(3)
    y = rng.normal(300, 50, len(g))
    yhat = y + rng.normal(0, 10, len(g))
    a = decompose_correlation(y, yhat, g)
    b = decompose_correlation(y, yhat, pd.Series(g).astype("category"))
    assert a["r_within"] == pytest.approx(b["r_within"])

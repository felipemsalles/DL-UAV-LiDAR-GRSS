"""Tests for the orthogonalised "beyond age" test and the model-assisted estimator.

Both modules make claims that are easy to get subtly wrong, so each test pins a property that
would break silently otherwise: no group leakage in the two-stage CV, a null that is calibrated by
refitting rather than by permuting an output vector, and a relative efficiency that is 1 for a
useless model and only exceeds 1 when the residuals really are tighter than the raw values.
"""
import numpy as np
import pandas as pd
import pytest

from greenvista import estimation
from greenvista.area_based import models as abm
from greenvista.eval.orthogonal import base_and_residual_cv, lidar_beyond_base, shuffled_null


def _frame(n_groups=6, per_group=3, seed=0, extra_signal=0.0):
    """Synthetic plots: volume driven by `age`, plus `extra` carrying an independent component.

    With extra_signal=0 the LiDAR-like column `extra` is pure noise and must test as null.
    With extra_signal>0 it genuinely explains part of what age cannot, and must be detected.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        age = 5.0 + g * 1.5
        for _ in range(per_group):
            extra = rng.normal()
            vol = 40.0 * age + extra_signal * extra + rng.normal(scale=3.0)
            rows.append({"talhao": g, "idade_anos": age, "extra": extra,
                         "zmean": age * 2.5 + rng.normal(scale=0.2),  # near-copy of age
                         "Vol_ha": vol})
    return pd.DataFrame(rows)


BASE = lambda: abm.make_huber_allom(("idade_anos",))


def test_combined_is_exactly_base_plus_residual():
    df = _frame()
    base, comb, resid, _ = base_and_residual_cv(BASE, lambda: abm.make_ridge(("extra",)), df)
    assert np.allclose(comb, base + resid)


def test_two_stage_cv_never_sees_the_held_out_group():
    """Changing a group's volumes must not change the predictions made for that group."""
    df = _frame()
    make_r = lambda: abm.make_ridge(("extra",))
    base_a, comb_a, _, _ = base_and_residual_cv(BASE, make_r, df)

    df2 = df.copy()
    held = df2.talhao == 0
    df2.loc[held, "Vol_ha"] = df2.loc[held, "Vol_ha"] * 1.5 + 100.0
    base_b, comb_b, _, _ = base_and_residual_cv(BASE, make_r, df2)

    assert np.allclose(base_a[held.to_numpy()], base_b[held.to_numpy()])
    assert np.allclose(comb_a[held.to_numpy()], comb_b[held.to_numpy()])


def test_residual_signal_is_detected_when_it_exists():
    df = _frame(extra_signal=60.0)
    out = lidar_beyond_base(BASE, lambda: abm.make_ridge(("extra",)), df, n_rep=60, seed=1)
    assert out["resid_r"] > 0.5
    assert out["p_vs_null"] < 0.1
    assert out["combined"]["rRMSE_pct"] < out["base"]["rRMSE_pct"]


def test_a_pure_copy_of_the_base_covariate_tests_as_null():
    """`zmean` is a near-duplicate of age, so it carries nothing beyond it and must not pass."""
    df = _frame(extra_signal=0.0)
    out = lidar_beyond_base(BASE, lambda: abm.make_ridge(("zmean",)), df, n_rep=60, seed=1)
    assert out["p_vs_null"] > 0.1
    assert out["resid_R2"] < 0.5


def test_refit_null_is_wide_at_small_n_and_not_a_point_mass():
    """The whole reason `shuffled_null` refits: the null of r is broad at n=13, not ~0."""
    df = _frame()
    null = shuffled_null(BASE, lambda: abm.make_ridge(("extra",)), df, n_rep=80, seed=2)
    assert null.std(ddof=1) > 0.15
    assert np.isfinite(null).all()


def test_relative_efficiency_is_one_for_a_useless_model():
    rng = np.random.default_rng(0)
    y = rng.normal(400, 60, 40)
    eff = estimation.relative_efficiency(y, np.full_like(y, y.mean()))
    assert eff["RE"] == pytest.approx(1.0, rel=1e-6)
    assert eff["equivalent_plots"] == pytest.approx(0.0, abs=1e-6)


def test_relative_efficiency_rewards_a_model_that_tracks_the_data():
    rng = np.random.default_rng(0)
    y = rng.normal(400, 60, 40)
    good = y + rng.normal(0, 10, 40)
    eff = estimation.relative_efficiency(y, good)
    assert eff["RE"] > 3
    assert eff["se_model_assisted"] < eff["se_field_only"]
    assert eff["se_reduction_pct"] > 40


def test_model_assisted_mean_recovers_the_population_mean():
    """With an unbiased model over the population, GREG must land on the true mean."""
    rng = np.random.default_rng(3)
    pop = rng.normal(400, 60, 5000)
    sample = pop[:30]
    yhat_sample = sample + rng.normal(0, 8, 30)     # unbiased predictions on the sample
    est = estimation.model_assisted_mean(sample, yhat_sample, pop)
    assert abs(est["mean_ha"] - pop.mean()) < 4 * est["se"]
    assert est["lo"] < est["mean_ha"] < est["hi"]


def test_model_assisted_se_beats_field_only_when_the_model_helps():
    rng = np.random.default_rng(4)
    pop = rng.normal(400, 60, 2000)
    sample = pop[:25]
    est = estimation.model_assisted_mean(sample, sample + rng.normal(0, 8, 25), pop)
    field = estimation.field_only_mean(sample)
    assert est["se"] < field["se"]


def test_bootstrap_re_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(5)
    y = rng.normal(400, 60, 30)
    yhat = y + rng.normal(0, 20, 30)
    re = estimation.relative_efficiency(y, yhat)["RE"]
    lo, hi = estimation.bootstrap_relative_efficiency(y, yhat, n_boot=800, seed=5)
    assert lo < re < hi

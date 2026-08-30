"""Synthetic smoke tests for greenvista/site_index_yield.py, where the mechanistic yield curve
extrapolates outside the range empirical fits support. They exercise the plumbing on synthetic
data; the real-data numbers come from scripts/exp_site_index_extrapolation.py.
"""
import numpy as np
import pandas as pd

from greenvista.site_index_yield import (mechanistic_model, yield_curve, metrics_for_group,
                                          metrics_excluding_group, holdout_table, stand_site_index)


def _synthetic(rng, n_stands=6, per_stand=3, young_stand=4, young_age=4.8):
    """Vol_ha = exp(2.6 + 0.35 ln(Hdom) + 0.98 ln(age)) with noise — the functional form recovered by
    the real hdom_yield_pipeline run (manual_match/hdom_yield_manifest.json). One stand (talhao 4) is
    both younger and shorter than the rest, mirroring the real young talhao-4 extrapolation case.
    """
    rows = []
    for t in range(1, n_stands + 1):
        age = young_age if t == young_stand else rng.uniform(9.5, 10.6)
        hdom_lo, hdom_hi = (18, 22) if t == young_stand else (28, 37)
        for _ in range(per_stand):
            hd = rng.uniform(hdom_lo, hdom_hi)
            vol = np.exp(2.6 + 0.35 * np.log(hd) + 0.98 * np.log(age)) * rng.uniform(0.97, 1.03)
            rows.append({"talhao": t, "parcela": len(rows), "Hdom": hd, "age": age, "Vol_ha": vol})
    return pd.DataFrame(rows)


def test_yield_curve_monotonic_increasing_in_hdom():
    rng = np.random.default_rng(0)
    df = _synthetic(rng)
    m = mechanistic_model().fit(df, df.Vol_ha.to_numpy())
    grid = np.linspace(15, 40, 30)
    curve = yield_curve(m, grid, age=10.0)
    assert np.all(np.diff(curve) > 0), "Vol_ha must increase monotonically with Hdom at fixed age"


def test_yield_curve_extrapolates_without_going_negative_or_exploding():
    """Extrapolate well outside the training Hdom range (18-37 m), to a much shorter and a much taller
    stand than any observed. The log-log form guarantees Vol_ha > 0 by construction and stays
    monotonic out of range, which an unconstrained empirical fit does not: a linear or RBF fit can
    cross zero or turn non-monotonic outside its training support."""
    rng = np.random.default_rng(1)
    df = _synthetic(rng)
    m = mechanistic_model().fit(df, df.Vol_ha.to_numpy())
    extreme = np.array([5.0, 60.0])  # far below/above any training Hdom
    curve = yield_curve(m, extreme, age=10.0)
    assert np.all(curve > 0) and np.all(np.isfinite(curve))
    assert curve[0] < curve[1]


def test_metrics_for_and_excluding_group_partition_correctly():
    rng = np.random.default_rng(2)
    df = _synthetic(rng)
    y = df.Vol_ha.to_numpy()
    pred = y * rng.uniform(0.95, 1.05, len(y))  # near-perfect fake predictions
    only_t4 = metrics_for_group(df, y, pred, "talhao", 4)
    excl_t4 = metrics_excluding_group(df, y, pred, "talhao", 4)
    n_t4 = int((df.talhao == 4).sum())
    assert only_t4["n"] == n_t4
    assert excl_t4["n"] == len(df) - n_t4
    # a near-perfect predictor should look good on both splits
    assert only_t4["rRMSE_pct"] < 15 and excl_t4["rRMSE_pct"] < 15


def test_holdout_table_reports_bias_pct():
    rng = np.random.default_rng(3)
    df = _synthetic(rng)
    y = df.Vol_ha.to_numpy()
    pred = y * 1.10  # uniform +10% over-prediction
    tab = holdout_table(df, y, pred, "talhao", 4)
    assert len(tab) == int((df.talhao == 4).sum())
    assert tab.bias_pct.round(0).eq(10).all()
    assert (tab.abs_pct_err == tab.bias_pct.abs()).all()


def test_stand_site_index_projects_young_stand_up():
    """The young stand (age~4.8 < ref_age=7) has its site index projected up above its raw Hdom: a
    shorter, younger stand can still be high-productivity once age is normalized away."""
    rng = np.random.default_rng(4)
    df = _synthetic(rng)
    si = stand_site_index(df, ref_age=7.0)
    assert si.loc[4, "SI"] > si.loc[4, "Hdom_mean"]
    # a mature stand (age > ref_age) projects down
    mature = [t for t in si.index if t != 4][0]
    assert si.loc[mature, "SI"] < si.loc[mature, "Hdom_mean"]

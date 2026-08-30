"""Synthetic smoke and property tests for the mechanistic volume decomposition (no real data).

Covers: G2 monotonicity, Campos forward/inverse round-trip, count calibration, exact V recovery from
known (N, D, H), interval bracketing, a synthetic G2 refit, and area_based.validate compatibility.
"""
import numpy as np
import pandas as pd
import pytest

from greenvista import mechanistic_volume as mv


# --- G2 volume equation -------------------------------------------------------------------------
def test_g2_monotone_in_D_and_H():
    D = np.array([10.0, 20.0])
    H = np.array([15.0, 15.0])
    v = mv.g2_volume(D, H)
    assert v[1] > v[0] > 0                       # volume grows with diameter
    assert mv.g2_volume(15.0, 25.0) > mv.g2_volume(15.0, 15.0)   # and with height


def test_g2_matches_known_mean_tree_scale():
    # a ~17.6 cm / 27 m mature eucalypt is ~0.30–0.35 m³ (São Manuel inventory scale)
    v = float(mv.g2_volume(17.6, 27.0))
    assert 0.25 < v < 0.40


# --- Campos hypsometric + its inverse -----------------------------------------------------------
@pytest.mark.parametrize("D,Hdom", [(15.0, 30.0), (12.0, 22.0), (22.0, 33.0)])
def test_campos_forward_inverse_roundtrip(D, Hdom):
    H = mv.campos_height(D, Hdom)
    D_rec = mv.campos_invert_diameter(H, Hdom)
    assert np.isclose(D_rec, D, atol=1e-6)


def test_campos_inverse_is_clamped_and_finite():
    # pathological height that would blow the denominator up must stay finite + within DBH range
    D = mv.campos_invert_diameter(np.array([5.0, 60.0]), np.array([30.0, 30.0]))
    assert np.all(np.isfinite(D))
    assert np.all((D >= mv.D_MIN_CM) & (D <= mv.D_MAX_CM))


# --- FF3D count calibration ---------------------------------------------------------------------
def test_count_calibrator_ratio_recovers_factor():
    obs = np.array([500.0, 700.0, 900.0])
    truth = 1.6 * obs                              # true density is a constant multiple of observed
    cal = mv.CountCalibrator("ratio").fit(obs, truth)
    assert np.isclose(cal.k_, 1.6)
    assert np.allclose(cal.predict(obs), truth)


def test_count_calibrator_linear_and_none():
    obs = np.array([400.0, 600.0, 800.0])
    truth = 100.0 + 1.2 * obs
    lin = mv.CountCalibrator("linear").fit(obs, truth)
    assert np.allclose(lin.predict(obs), truth, atol=1e-6)
    assert np.allclose(mv.CountCalibrator("none").fit(obs, truth).predict(obs), obs)


# --- exact V recovery from known N, D, H (the integration property) -----------------------------
def _synthetic_plots(n=6, seed=0):
    rng = np.random.default_rng(seed)
    Hdom = rng.uniform(20, 33, n)
    D = rng.uniform(12, 20, n)
    Hmean = mv.campos_height(D, Hdom)             # make Hmean consistent → inversion recovers D
    ff3d = rng.uniform(500, 1100, n)
    df = pd.DataFrame({"talhao": np.repeat(np.arange(1, n // 2 + 1), 2)[:n],
                       "Hdom": Hdom, "Hmean": Hmean, "ff3d_n_ha": ff3d})
    return df, D


def test_mechanistic_recovers_known_volume():
    df, D = _synthetic_plots()
    df["field_n_ha"] = df.ff3d_n_ha              # perfect detection
    vbar = mv.g2_volume(D, df.Hmean.to_numpy())
    y = df.ff3d_n_ha.to_numpy() * vbar            # V_ha = N_ha · v̄  by construction
    m = mv.MechanisticVolume(count_correction="none", fit_bias=False).fit(df, y)
    assert np.allclose(m.predict(df), y, rtol=1e-6)
    assert np.allclose(m.dbar(df), D, atol=1e-6)  # Campos inversion recovered the true DBH


def test_ratio_calibration_scales_count():
    df, D = _synthetic_plots()
    df["field_n_ha"] = 2.0 * df.ff3d_n_ha         # true density = 2× detected (constant bias)
    vbar = mv.g2_volume(D, df.Hmean.to_numpy())
    y = 2.0 * df.ff3d_n_ha.to_numpy() * vbar
    m = mv.MechanisticVolume(count_correction="ratio", fit_bias=False).fit(df, y)
    assert np.isclose(m.cal_.k_, 2.0)
    assert np.allclose(m.predict(df), y, rtol=1e-6)


def test_bias_term_matches_total_on_train():
    df, D = _synthetic_plots()
    df["field_n_ha"] = df.ff3d_n_ha
    y = np.full(len(df), 400.0)                    # arbitrary target; bias absorbs the offset
    m = mv.MechanisticVolume(count_correction="none", fit_bias=True).fit(df, y)
    assert np.isclose(m.predict(df).sum(), y.sum(), rtol=1e-6)   # unbiased in total on train


def test_predict_interval_brackets_prediction():
    df, D = _synthetic_plots()
    df["field_n_ha"] = df.ff3d_n_ha
    vbar = mv.g2_volume(D, df.Hmean.to_numpy())
    y = df.ff3d_n_ha.to_numpy() * vbar * np.random.default_rng(1).uniform(0.9, 1.1, len(df))
    m = mv.MechanisticVolume().fit(df, y)
    lo, hi = m.predict_interval(df, alpha=0.05)
    p = m.predict(df)
    assert np.all(lo <= p) and np.all(p <= hi) and np.all(lo > 0)


# --- synthetic G2 refit -------------------------------------------------------------------------
def test_fit_g2_recovers_linear_coefficients(tmp_path):
    rng = np.random.default_rng(3)
    D = rng.uniform(6, 30, 80)
    H = rng.uniform(10, 34, 80)
    true = (-0.005, 1.4e-4, 3.2e-5, 3.6e-4)
    V = true[0] + true[1] * D ** 2 + true[2] * D ** 2 * H + true[3] * H
    csv = tmp_path / "volumes.csv"
    pd.DataFrame({"D_cm": D, "H_m": H, "V": V}).to_csv(csv, index=False)
    coef, r2, rmse, n = mv.fit_g2_from_cubagem(csv)
    assert r2 > 0.999 and n == 80
    assert np.allclose(coef, true, atol=1e-6)


# --- area_based.validate compatibility ----------------------------------------------------------
def test_estimator_runs_in_validate_harness():
    from greenvista.area_based import validate as abv
    df, D = _synthetic_plots(n=6)
    df["field_n_ha"] = df.ff3d_n_ha
    vbar = mv.g2_volume(D, df.Hmean.to_numpy())
    df["Vol_ha"] = df.ff3d_n_ha * vbar * np.random.default_rng(2).uniform(0.9, 1.1, len(df))
    y = df.Vol_ha.to_numpy()
    make = lambda: mv.make_mechanistic()
    lopo, lo, hi = abv._lopo(make, df)
    loto = abv._group_cv(make, df)
    assert lopo.shape == y.shape and np.all(np.isfinite(lopo))
    assert loto.shape == y.shape and np.all(np.isfinite(loto))
    assert np.all(lo <= hi)

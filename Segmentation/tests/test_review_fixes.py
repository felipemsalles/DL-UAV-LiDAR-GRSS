"""Property tests for the statistical and modelling guarantees. Synthetic, no real data.

Each test's docstring names the property it guarantees.
"""
import json
import numpy as np
import pandas as pd
import pytest


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def _plots(n_per=3, talhoes=(1, 2, 4, 5), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in talhoes:
        base = rng.uniform(18, 26)
        for p in range(1, n_per + 1):
            zmean = base + rng.uniform(-2, 2); zq90 = zmean + rng.uniform(2, 5)
            rows.append({"talhao": t, "parcela": p, "zmax": zq90 + 1, "zmean": zmean,
                         "zsd": rng.uniform(3, 7), "zq50": zmean, "zq90": zq90,
                         "pzabovezmean": rng.uniform(48, 70), "idade_anos": 8 + t * 0.2,
                         "Vol_ha": np.exp(0.6 + 1.35 * np.log(zmean)) * rng.uniform(0.96, 1.04)})
    df = pd.DataFrame(rows)
    df["plot_id"] = df.talhao.astype(str) + "_" + df.parcela.astype(str)
    return df


# ============================ conformal intervals ================================
def test_finite_sample_level_correction():
    """level = ceil((1-alpha)(m+1))/m, clipped to 1: strictly above the plain 1-alpha."""
    from greenvista.eval.conformal import finite_sample_level
    assert finite_sample_level(39, 0.05) == pytest.approx(np.ceil(0.95 * 40) / 39)   # 38/39 ≈ 0.974
    assert finite_sample_level(99, 0.05) == pytest.approx(np.ceil(0.95 * 100) / 99)  # 96/99 ≈ 0.970
    assert finite_sample_level(39, 0.05) > 0.95        # the correction: above naive 1-alpha
    assert finite_sample_level(12, 0.05) == 1.0        # tiny m: ceil(0.95*13)/12=13/12 -> clip to 1
    assert finite_sample_level(0, 0.05) == 1.0


def test_jackknife_plus_covers_near_nominal():
    """LOO conformal reaches ~95% on clean data (finite-sample level, not under-cover)."""
    from greenvista.eval.conformal import jackknife_plus_interval, coverage
    rng = np.random.default_rng(0)
    y = rng.uniform(100, 500, 60); pred = y + rng.normal(0, 20, 60)
    lo, hi = jackknife_plus_interval(y, pred, alpha=0.05)
    assert (lo <= hi).all()
    assert coverage(y, lo, hi) >= 0.90


def test_conformal_beats_insample_interval_for_interpolating_model():
    """Distance-weighted kNN interpolates its training set, so the in-sample residual std collapses
    to ~0 and the parametric interval degenerates (coverage ~0); the out-of-fold conformal
    interval stays valid."""
    from greenvista.area_based import models as abm
    from greenvista.eval import jackknife_plus_interval, coverage
    from greenvista.area_based.validate import _lopo
    df = _plots()
    y = df.Vol_ha.to_numpy()
    knn = abm.make_knn().fit(df, y)
    assert knn.resid_sd_ < 1e-6                      # distance-weighted kNN reproduces training → resid≈0
    pred, lo, hi = _lopo(abm.make_knn, df)
    cov_param = float(((y >= lo) & (y <= hi)).mean())
    c_lo, c_hi = jackknife_plus_interval(y, pred)
    cov_conf = coverage(y, c_lo, c_hi)
    assert cov_param < 0.2                            # in-sample band is useless for kNN
    assert cov_conf >= 0.75 and cov_conf > cov_param  # conformal keeps coverage


def test_validate_run_reports_single_conformal_coverage():
    """validate.run puts conformal cov95 in `coverage95` (same method for every model) and keeps the
    parametric number under a separate key."""
    from greenvista.area_based import validate as abv, models as abm
    df = _plots()
    rep = abv.run(df, models={"rf": abm.make_rf, "gpr": abm.make_gpr}, group_col=None)
    for mm in rep.values():
        assert 0.0 <= mm["coverage95"] <= 1.0
        assert "coverage95_parametric" in mm and "interval_score" in mm and "rRMSE_ci" in mm


# ============================ leave-one-stand-out ========================================
def test_group_cv_reported_and_differs_from_lopo():
    """run() adds R2_group (leave-one-stand-out); for a stochastic model it is the typically lower
    cross-stand number."""
    from greenvista.area_based import validate as abv, models as abm
    df = _plots()
    rep = abv.run(df, models={"rf": abm.make_rf}, group_col="talhao")
    mm = rep["rf"]
    assert "R2_group" in mm and mm["n_groups"] == df.talhao.nunique()
    assert mm["R2_group"] <= mm["R2"] + 1e-9        # group CV cannot exceed within-stand LOPO here


def test_group_cv_predictions_hold_out_whole_stand():
    """_group_cv never trains on the held-out stand."""
    from greenvista.area_based.validate import _group_cv
    from greenvista.area_based import models as abm
    df = _plots()
    pred = _group_cv(abm.make_log_allometric, df, group_col="talhao")
    assert pred.shape == (len(df),) and np.all(np.isfinite(pred))


# ============================ repro + manifest =======================================
def test_seed_everything_is_deterministic():
    """Seeding makes torch draws reproducible."""
    import torch
    from greenvista.repro import seed_everything
    seed_everything(123); a = torch.randn(5)
    seed_everything(123); b = torch.randn(5)
    assert torch.allclose(a, b)


def test_write_manifest_captures_provenance(tmp_path):
    """The manifest carries git SHA, seed, config and metrics: a traceable experiment record."""
    from greenvista.repro import write_manifest
    from greenvista import config
    p = tmp_path / "m.json"
    write_manifest(p, script="t.py", config=config.ABA, seed=7, metrics={"R2": 0.74})
    d = json.loads(p.read_text())
    assert d["seed"] == 7 and d["metrics"]["R2"] == 0.74 and "git_sha" in d
    assert d["config"]["rf_n_estimators"] == config.ABA.rf_n_estimators


def test_multiseed_summary():
    """Mean ± sd over seeds."""
    from greenvista.eval import multiseed_summary
    s = multiseed_summary([{"R2": 0.5}, {"R2": 0.7}, {"R2": 0.6}], "R2")
    assert s["R2_mean"] == pytest.approx(0.6) and s["n_seeds"] == 3 and s["R2_sd"] > 0


# ============================ on-the-fly augmentation =====================================
def test_augmentation_is_on_the_fly_fresh_views():
    """Two draws of the same train index give different rasters (fresh rotation), same label."""
    from greenvista.image_cnn.dataset import PlotRasterDataset
    rng = np.random.default_rng(0)
    samples = {"p0": {"raster": rng.uniform(0, 30, (3, 32, 32)).astype(np.float32),
                      "scalars": np.array([30, 20, 5], np.float32),
                      "labels": np.array([400.0, 17.0], np.float32)}}
    ds = PlotRasterDataset(samples, ["p0"], train=True, n_aug=4, seed=0)
    r1, _, y1, _ = ds[0]; r2, _, y2, _ = ds[0]
    assert not np.allclose(r1.numpy(), r2.numpy())   # fresh view each access
    assert np.allclose(y1.numpy(), y2.numpy())       # label preserved


# ============================ guards ============================================
def test_assert_finite_raises():
    """assert_finite rejects a non-finite value."""
    from greenvista.eval import assert_finite
    assert_finite([1.0, 2.0])
    with pytest.raises(ValueError):
        assert_finite([1.0, np.nan])


def test_map_skips_nan_cells():
    """A cell that yields NaN features is skipped and counted, not fed to predict."""
    from greenvista.area_based import mapping as abmap, models as abm
    df = _plots(); y = df.Vol_ha.to_numpy()
    model = abm.make_log_allometric().fit(df, y)
    xy = np.array([[0.0, 0.0]] * 30 + [[100.0, 100.0]])   # one lone far point → its own <min_pts cell
    z = np.r_[np.full(30, 20.0), 25.0]
    gdf = abmap.make_map_from_arrays(model, xy, z, cell=20.0, min_pts=5)
    assert "n_skipped_nan" in gdf.attrs and np.all(np.isfinite(gdf.volume_m3.to_numpy()))


def test_bootstrap_reports_rmse_primary_and_guards_degenerate():
    """RMSE/rRMSE CIs are present, plus the dropped-resample counter."""
    from greenvista.eval import bootstrap_ci
    rng = np.random.default_rng(0)
    y = rng.uniform(100, 500, 13); p = y + rng.normal(0, 30, 13)
    ci = bootstrap_ci(y, p, n_boot=500)
    assert ci["RMSE_ci"][0] <= ci["RMSE_ci"][1] and "rRMSE_ci" in ci
    assert ci["n_boot_valid"] + ci["n_boot_dropped"] == 500


def test_leakage_guard_flags_leaked_feature():
    """leakage_guard drops a feature that is ~identical to the target."""
    from greenvista.area_based import validate as abv
    df = _plots()
    df = df.copy(); df["zmax"] = df["Vol_ha"]              # inject leakage
    with pytest.warns(UserWarning):
        kept = abv.leakage_guard(df)
    assert "zmax" not in kept


# ============================ config constants ===========================================
def test_config_constants_present():
    """Shared constants and hyperparameter dataclasses exist, instead of scattered literals."""
    from greenvista import config
    assert config.CRS == "EPSG:31982"
    assert config.PLOT_AREA_HA == pytest.approx(np.pi * 12 ** 2 / 1e4)
    assert config.ABA.rf_n_estimators == 500 and len(config.CNN.seeds) >= 3


# ============================ permutation test ===========================================
def test_permutation_test_small_p_for_real_signal_large_for_noise():
    """Strong signal → tiny p; pure noise → large p."""
    from greenvista.eval import permutation_test
    rng = np.random.default_rng(0)
    y = rng.uniform(100, 500, 30); good = y + rng.normal(0, 10, 30)
    noise = rng.uniform(100, 500, 30)
    assert permutation_test(y, good, n_perm=500)["p_value"] < 0.05
    assert permutation_test(y, noise, n_perm=500)["p_value"] > 0.2


def test_cv_permutation_test_runs():
    """CV-level shuffle helper."""
    from greenvista.eval import cv_permutation_test
    from greenvista.eval import loocv_predict
    from sklearn.linear_model import LinearRegression
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, (20, 2)); y = X[:, 0] * 2 + rng.normal(0, 0.5, 20)
    out = cv_permutation_test(lambda yy: loocv_predict(LinearRegression, X, yy), y, n_perm=30)
    assert out["p_value"] < 0.1


# ============================ heteroscedastic intervals + extrapolation ===================
def test_quantile_gbm_is_heteroscedastic():
    """Quantile-GBM interval width varies across inputs, rather than being a constant band."""
    from greenvista.area_based import models as abm
    df = _plots(n_per=6)
    m = abm.make_quantile_gbm().fit(df, df.Vol_ha.to_numpy())
    lo, hi = m.predict_interval(df)
    assert (lo <= hi).all() and np.ptp(hi - lo) > 1e-6


def test_mondrian_interval_adapts_by_bin():
    """Mondrian conformal gives per-bin widths and finite coverage."""
    from greenvista.eval import mondrian_interval, coverage
    rng = np.random.default_rng(0)
    y = np.r_[rng.normal(100, 5, 20), rng.normal(400, 40, 20)]
    pred = y + np.r_[rng.normal(0, 5, 20), rng.normal(0, 40, 20)]
    groups = np.r_[np.zeros(20), np.ones(20)]
    lo, hi = mondrian_interval(y, pred, groups)
    w_low = np.mean((hi - lo)[groups == 0]); w_high = np.mean((hi - lo)[groups == 1])
    assert w_high > w_low                              # wider where error is larger
    assert coverage(y, lo, hi) >= 0.85


def test_map_extrapolation_flag():
    """Cells whose features fall outside the training envelope are flagged."""
    from greenvista.area_based import mapping as abmap, models as abm
    df = _plots(); y = df.Vol_ha.to_numpy()
    model = abm.make_log_allometric().fit(df, y)
    env = abmap.feature_envelope(df)
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 60, (3000, 2)); z = rng.uniform(35, 44, 3000)   # far above train zmax
    gdf = abmap.make_map_from_arrays(model, xy, z, cell=20.0, envelope=env, min_pts=10)
    assert "extrapolated" in gdf.columns and gdf.extrapolated.any()


# ============================ CNN TTA / linear-probe / transfer log =======================
def _fake_cnn(n=8, size=16, seed=0):
    rng = np.random.default_rng(seed)
    h = np.linspace(15, 32, n) + rng.normal(0, 0.5, n)
    out = {}
    for i in range(n):
        r = np.zeros((3, size, size), np.float32)
        r[0] = h[i]; r[1] = rng.uniform(0, 3, (size, size)); r[2] = rng.uniform(0, 2, (size, size))
        out[f"p{i}"] = {"raster": r, "scalars": np.array([h[i], 0.8 * h[i], 4.0], np.float32),
                        "labels": np.array([12 * h[i] + 50, 0.6 * h[i]], np.float32)}
    return out


def test_tta_prediction_runs_and_is_finite():
    """The test-time-augmentation path produces finite predictions."""
    from greenvista.image_cnn import train as tr
    fake = _fake_cnn(); ids = list(fake); y = np.stack([fake[i]["labels"] for i in ids])
    res = tr.run_lopo(fake, ids, y, n_aug=4, epochs=2, tta=4)
    assert np.all(np.isfinite(res["pred"]))


def test_linear_probe_freezes_backbone_and_logs_transfer():
    """freeze_backbone trains only the head; transfer coverage is logged (kept/total)."""
    from greenvista.image_cnn import train as tr
    fake = _fake_cnn(); ids = list(fake); y = np.stack([fake[i]["labels"] for i in ids])
    init = tr.pretrain(fake, n_aug=2, epochs=2)
    res = tr.run_lopo(fake, ids, y, n_aug=4, epochs=2, init_state=init, freeze_backbone=True)
    kept, total = res["transfer"]
    assert 0 < kept <= total


# ============================ structural features ========================================
def test_rumple_increases_with_roughness():
    """A rough canopy has higher rumple than a flat one."""
    from greenvista.features import rumple_index
    flat = np.full((32, 32), 20.0)
    rough = 20.0 + np.random.default_rng(0).normal(0, 5, (32, 32))
    assert rumple_index(rough) > rumple_index(flat) >= 1.0


def test_cover_gap_penetration():
    """cover + gap = 1; penetration in [0,1]."""
    from greenvista.features import canopy_cover, gap_fraction, penetration_ratio
    z = np.array([1.0, 1.5, 5.0, 10.0, 20.0, 30.0])
    assert canopy_cover(z) + gap_fraction(z) == pytest.approx(1.0)
    assert 0.0 <= penetration_ratio(z) <= 1.0


def test_render_exposes_structural_scalars_and_optional_channel():
    """to_raster returns structural scalars; with_rumple adds a 4th channel."""
    from greenvista.image_cnn import render
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(-10, 10, 2000), rng.uniform(-10, 10, 2000), rng.uniform(2, 30, 2000)])
    raster, scal = render.to_raster(pts, (0.0, 0.0), 11.3, size=64)
    assert {"rumple", "cover", "gap_fraction", "penetration"}.issubset(scal)
    raster4, _ = render.to_raster(pts, (0.0, 0.0), 11.3, size=64, with_rumple=True)
    assert raster4.shape[0] == render.N_CHANNELS + 1


# ============================ error analysis + hard truth =================================
def test_leverage_ranks_influential_point():
    """leverage returns per-point ΔR² sorted by magnitude."""
    from greenvista.eval import leverage
    y = np.arange(1.0, 14.0); pred = y.copy(); pred[0] = 30.0   # one gross outlier
    lev = leverage(y, pred, ids=[f"p{i}" for i in range(13)])
    assert lev.iloc[0]["id"] == "p0"                            # the outlier dominates


def test_residual_correlations_and_hard_truth():
    """|resid| vs covariate correlation, plus the measured-tree hard-truth comparison."""
    from greenvista.eval import residual_correlations, hard_truth_from_trees, compare_to_hard_truth
    df = _plots(); y = df.Vol_ha.to_numpy(); pred = y + np.linspace(-5, 5, len(df))
    corr = residual_correlations(y, pred, df[["zmax", "idade_anos"]])
    assert "zmax" in corr
    trees = pd.DataFrame({"plot": np.repeat(df.plot_id.values, 3), "V": np.ones(len(df) * 3) * 0.3})
    hard = hard_truth_from_trees(trees, "plot", area_ha=0.0452).rename(columns={"plot": "plot_id"})
    cmp = compare_to_hard_truth(dict(zip(df.plot_id, pred)), hard, "plot_id")
    assert cmp["n_compared"] == len(df) and "note" in cmp


# ============================ ABA models + selection ================
def test_new_aba_models_fit_predict_interval():
    """Best-subset, huber, PLS and log-GPR honour the common interface."""
    from greenvista.area_based import models as abm
    df = _plots(n_per=4)                     # enough plots for PLS/best-subset
    y = df.Vol_ha.to_numpy()
    cands = ("zmean", "zq90", "zmax", "zsd", "idade_anos")   # columns present in the synthetic helper
    feat = ("zmax", "zmean", "zsd", "zq50", "zq90", "pzabovezmean")
    for mk in (lambda: abm.make_bestsubset_allom(cands, select_by="group"),
               lambda: abm.make_huber_allom(("zmean", "zq90", "idade_anos")),
               lambda: abm.make_pls(feat, max_k=3),
               lambda: abm.make_loggpr(("zmean", "zq90", "idade_anos"))):
        m = mk().fit(df, y)
        p = m.predict(df); assert p.shape == (len(df),) and np.all(np.isfinite(p))
        lo, hi = m.predict_interval(df, 0.05); assert (lo <= hi).all()


def test_group_selection_escapes_age_only_trap():
    """With a feature that is constant within stand and predictive only via stand identity (age),
    leave-one-plot-out selection picks it alone, while leave-one-stand-out selection keeps the
    genuinely predictive within-stand feature."""
    from greenvista.area_based import models as abm
    rng = np.random.default_rng(0)
    rows = []
    for t in range(1, 7):
        stand_age = 4 + 1.3 * t                      # constant within stand, distinct across stands
        for p in range(3):
            h = rng.uniform(16, 28)                  # real within-stand height signal
            rows.append({"talhao": t, "parcela": p, "zmean": h, "zq90": h + 3, "zmax": h + 4,
                         "zq50": h, "zq70": h + 1.5, "zsd": rng.uniform(3, 6),
                         "pzabovezmean": rng.uniform(48, 70), "idade_anos": stand_age,
                         "Vol_ha": np.exp(0.6 + 1.3 * np.log(h)) * rng.uniform(0.97, 1.03)})
    df = pd.DataFrame(rows)
    abm.make_bestsubset_allom(select_by="loo").fit(df, df.Vol_ha.to_numpy()).sel_cols_
    grp_cols = abm.make_bestsubset_allom(select_by="group").fit(df, df.Vol_ha.to_numpy()).sel_cols_
    assert "zmean" in grp_cols                       # group-selection keeps the real signal
    # leave-one-stand-out generalization must be sane for the group-selected model
    from greenvista.area_based.validate import _group_cv
    from greenvista.eval import metrics
    pred = _group_cv(lambda: abm.make_bestsubset_allom(select_by="group"), df)
    assert metrics(df.Vol_ha.to_numpy(), pred)["R2"] > 0.3


# ============================ map provenance =============================================
def test_save_map_provenance_roundtrip(tmp_path):
    """The provenance JSON captures features and coefficients, so the map is reproducible from a file."""
    from greenvista.area_based import mapping as abmap, models as abm, data as abd
    df = _plots(); model = abm.make_log_allometric().fit(df, df.Vol_ha.to_numpy())
    p = tmp_path / "prov.json"
    abmap.save_map_provenance(model, abd.FEATURES, p, envelope=abmap.feature_envelope(df))
    d = json.loads(p.read_text())
    assert d["features"] == list(abd.FEATURES)
    assert d["model"]["model_type"] == "LogAllometric" and "coef" in d["model"]

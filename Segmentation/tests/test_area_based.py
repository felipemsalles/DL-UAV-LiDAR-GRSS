import numpy as np, pandas as pd, pytest
from greenvista.area_based import data as abd


def _tiny_plot_csv(tmp_path):
    # 3 fake plots with the columns load_plots needs (subset of the real schema)
    pm = pd.DataFrame({
        "talhao": [1, 2, 4], "parcela": [1, 1, 1],
        "zmax": [30.0, 25.0, 35.0], "zmean": [20.0, 16.0, 26.0], "zsd": [5.0, 4.0, 7.0],
        "zskew": [0.1, 0.2, 0.0], "zkurt": [2.0, 2.1, 1.9], "zentropy": [0.8, 0.7, 0.85],
        "zq50": [20.0, 16.0, 27.0], "zq70": [24.0, 19.0, 30.0], "zq90": [27.0, 22.0, 33.0],
        "zpcum7": [80.0, 78.0, 82.0], "pzabove2": [100.0, 100.0, 100.0],
        "pzabovezmean": [55.0, 50.0, 60.0], "Vol_ha": [300.0, 200.0, 450.0]})
    inv = pd.DataFrame({"talhao": [1, 2, 4], "parcela": [1, 1, 1], "Vol_ha": [300.0, 200.0, 450.0]})
    p1 = tmp_path / "pm.csv"; p2 = tmp_path / "inv.csv"
    pm.to_csv(p1, index=False); inv.to_csv(p2, index=False)
    return str(p1), str(p2)


def test_load_plots_returns_clean_table(tmp_path):
    pm, inv = _tiny_plot_csv(tmp_path)
    df = abd.load_plots(pm, inv)
    assert len(df) == 3
    assert "Vol_ha" in df.columns
    for f in abd.FEATURES:
        assert f in df.columns
    # degenerate constant column pzabove2 must be excluded from FEATURES
    assert "pzabove2" not in abd.FEATURES


def test_load_plots_fails_loud_on_y_mismatch(tmp_path):
    pm, inv = _tiny_plot_csv(tmp_path)
    bad = pd.read_csv(inv); bad.loc[0, "Vol_ha"] = 999.0; bad.to_csv(inv, index=False)
    with pytest.raises(AssertionError):
        abd.load_plots(pm, inv)


import os
from greenvista import config
REAL_PM = str(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv")
REAL_INV = str(config.DATA / "5-dados_campo/inventario_est.csv")
real_data = pytest.mark.skipif(not os.path.exists(REAL_PM), reason="real dataset not present")
field_data = pytest.mark.skipif(not config.VOLUMES_CSV.exists(),
                                reason="field data not present (set GREENVISTA_DATA_DIR)")

@real_data
def test_load_plots_real_13_plots():
    df = abd.load_plots(REAL_PM, REAL_INV)
    assert len(df) == 13
    assert df.Vol_ha.between(150, 600).all()


# --- models.py ---
from greenvista.area_based import models as abm


def _synth_allometric(n=40, seed=0):
    rng = np.random.default_rng(seed)
    zmean = rng.uniform(15, 27, n); zq90 = zmean + rng.uniform(2, 6, n)
    df = pd.DataFrame({"zmax": zq90 + 1, "zmean": zmean, "zsd": rng.uniform(3, 8, n),
                       "zq50": zmean, "zq90": zq90, "pzabovezmean": rng.uniform(48, 70, n)})
    vol = np.exp(0.5 + 1.3 * np.log(zmean))  # clean log-allometric truth
    return df, vol


def test_log_allometric_recovers_clean_signal():
    df, y = _synth_allometric()
    m = abm.LogAllometric(cols=("zmean",)); m.fit(df, y)
    pred = m.predict(df)
    assert np.corrcoef(pred, y)[0, 1] > 0.999


def test_all_models_fit_predict_and_interval():
    df, y = _synth_allometric()
    for make in (abm.make_log_allometric, abm.make_elasticnet, abm.make_gpr):
        m = make(); m.fit(df, y)
        p = m.predict(df); assert p.shape == (len(df),)
        lo, hi = m.predict_interval(df, alpha=0.05)
        assert (lo <= hi).all()


def test_new_ladder_models_fit_predict_interval():
    """RF, kNN, and the nested-selection PCA-OLS all honour the common interface."""
    from greenvista.area_based import data as abd
    df, y = _synth_allometric()
    df = df.copy()
    for q in range(5, 100, 5):          # PCA-OLS needs the full metric suite present
        df[f"zq{q}"] = df["zq50"] + (q - 50) * 0.05
    for i in range(1, 10):
        df[f"zpcum{i}"] = 50.0 + i
    for k in ("zskew", "zkurt", "zentropy"):
        df[k] = 1.0
    for make in (abm.make_rf, abm.make_knn, abm.make_pca_ols):
        m = make(); m.fit(df, y)
        p = m.predict(df); assert p.shape == (len(df),)
        lo, hi = m.predict_interval(df, 0.05); assert (lo <= hi).all()
    # column parametrization: age-aware variant uses a different feature set without error
    df["idade_anos"] = np.linspace(4, 11, len(df))
    m = abm.make_gpr(cols=abd.FEATURES_AGE).fit(df, y)
    assert m.predict(df.iloc[[0]]).shape == (1,)


def test_metrics_reports_bias_and_agb_converts():
    from greenvista.eval import metrics
    from greenvista.area_based import data as abd
    mm = metrics(np.array([100.0, 200, 300]), np.array([110.0, 210, 310]))  # +10 everywhere
    assert "bias" in mm and abs(mm["bias"] - 10.0) < 1e-9
    assert abs(mm["bias_pct"] - 5.0) < 1e-6                                   # 10 / 200 mean
    assert abs(abd.volume_to_agb(400.0, basic_density=0.5) - 200.0) < 1e-9


@field_data
def test_itd_stand_hybrid_detects_and_aggregates():
    """ITD→stand: H→V allometry is increasing (V∝H³), and detection+aggregation runs on a cloud."""
    from greenvista.area_based import stand_from_trees as sft
    coef = sft.fit_hv_allometry()
    assert 2.0 < coef[1] < 4.0                                  # b ≈ 3 (volume ~ height³)
    assert sft.tree_volume(30, coef) > sft.tree_volume(15, coef)
    # synthetic plot: 5 tall "trees" (point columns) inside a 12 m disc
    rng = np.random.default_rng(0)
    pts = []
    for cx, cy, h in [(-6, -6, 28), (6, -6, 24), (0, 0, 30), (-6, 6, 22), (6, 6, 26)]:
        col = np.column_stack([cx + rng.uniform(-0.6, 0.6, 200), cy + rng.uniform(-0.6, 0.6, 200),
                               rng.uniform(h - 1, h, 200)])
        pts.append(col)
    pts = np.vstack(pts)
    vol_ha, n_det = sft.plot_vol_ha(pts, (0.0, 0.0), coef)
    assert n_det >= 3 and vol_ha > 0                            # detects the peaks, positive volume


# --- validate.py ---
from greenvista.area_based import validate as abv


def test_lopo_recovers_identity_signal():
    df, y = _synth_allometric(n=13, seed=3)
    df = df.copy(); df["Vol_ha"] = y
    rep = abv.run(df, models={"log": abm.make_log_allometric})
    assert rep["log"]["R2"] > 0.95
    assert "R2_ci" in rep["log"] and rep["log"]["R2_ci"][0] <= rep["log"]["R2"]
    assert 0.0 <= rep["log"]["coverage95"] <= 1.0


# --- mapping.py ---
from greenvista.area_based import mapping as abmap


def test_stdmetrics_subset_matches_definitions():
    z = np.array([3.0, 5.0, 10.0, 20.0, 30.0])  # already within [2,45]
    m = abmap.stdmetrics_subset(z)
    assert np.isclose(m["zmax"], 30.0)
    assert np.isclose(m["zmean"], z.mean())
    assert np.isclose(m["zsd"], z.std(ddof=1))
    assert np.isclose(m["zq50"], np.quantile(z, 0.50))   # R type-7 == numpy default
    assert np.isclose(m["zq90"], np.quantile(z, 0.90))
    assert np.isclose(m["pzabovezmean"], 100.0 * np.mean(z > z.mean()))


def test_make_map_grid_shape_on_synthetic():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-9, 9, (500, 2)); z = rng.uniform(10, 30, 500)
    df, y = _synth_allometric(n=13, seed=1); df = df.copy(); df["Vol_ha"] = y
    model = abm.make_log_allometric().fit(df, y)
    gdf = abmap.make_map_from_arrays(model, xy, z, cell=20.0, zmin=2.0, zmax=45.0)
    assert len(gdf) >= 1
    assert "volume_m3" in gdf.columns and "vol_lo" in gdf.columns and "vol_hi" in gdf.columns
    assert (gdf.volume_m3 >= 0).all()


REAL_LAZ = str(config.TILE_LAZ)
REAL_PARC = str(config.DATA / "2-shapes/Parcelas/parcelas.shp")
real_cloud = pytest.mark.skipif(not os.path.exists(REAL_LAZ), reason="cloud not present")


@real_cloud
@real_data
def test_recomputed_metrics_match_csv_for_talhao1_plots():
    import laspy, geopandas as gpd
    df = abd.load_plots(REAL_PM, REAL_INV)
    parc = gpd.read_file(REAL_PARC)
    parc["t"] = parc.talhao.astype(int); parc["p"] = parc.parcela.astype(int)
    las = laspy.read(REAL_LAZ)
    xy = np.column_stack([np.asarray(las.x), np.asarray(las.y)]); z = np.asarray(las.z)
    checked = 0
    for _, row in df[df.talhao == 1].iterrows():
        g = parc[(parc.t == 1) & (parc.p == int(row.parcela))]
        if g.empty:
            continue
        c = (g.geometry.x.iloc[0], g.geometry.y.iloc[0])
        rec = abmap.plot_footprint_metrics(xy, z, c)
        # demo cloud is identical to the lidR source -> metrics match to <0.1 m
        assert abs(rec["zmean"] - row.zmean) < 0.1, (row.parcela, rec["zmean"], row.zmean)
        assert abs(rec["zmax"] - row.zmax) < 0.1, (row.parcela, rec["zmax"], row.zmax)
        checked += 1
    assert checked >= 1, "no stand-1 plot footprint found in the demo cloud"


def test_integration_known_signal_recovered_end_to_end():
    # Inject Vol = exp(0.4 + 1.4 ln(zmean)) over 13 synthetic plots; LOPO must recover it strongly.
    rng = np.random.default_rng(7)
    zmean = rng.uniform(15, 27, 13)
    df = pd.DataFrame({"zmax": zmean + 4, "zmean": zmean, "zsd": rng.uniform(3, 8, 13),
                       "zq50": zmean, "zq90": zmean + 5, "pzabovezmean": rng.uniform(48, 70, 13)})
    df["Vol_ha"] = np.exp(0.4 + 1.4 * np.log(zmean)) * rng.uniform(0.97, 1.03, 13)
    rep = abv.run(df, models={"log_allometric": abm.make_log_allometric})
    assert rep["log_allometric"]["R2"] > 0.8

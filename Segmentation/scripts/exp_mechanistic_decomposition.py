#!/usr/bin/env python3
"""Mechanistic volume decomposition  V_ha = (N_stems/area)·v̄(D̄, H̄).

Instead of regressing Vol_ha directly, build it from LiDAR-native / fixed-forestry factors:
  N_stems ← FF3D detection (manual_match/ff3d_detection.csv), calibrated under the CV fold;
  Hdom, H̄ ← LiDAR (greenvista.stand_volume.lidar_hdom, exactly as hdom_yield_pipeline.py);
  D̄ ← invert the Campos hypsometric (R²=0.82);  v̄ ← the G2 volume equation (R²=0.995).

Validated under LOPO + leave-one-stand-out (LOTO), with jackknife+ conformal coverage, a
permutation/shuffle-label control, and bootstrap rRMSE CIs, then compared head-to-head against
the empirical height+age deliverable  StandVolume(("Hdom","age")).  Question: does the
mechanistic decomposition generalize cross-stand (LOTO) better than the empirical fit, given
that each factor is physically grounded and the forestry equations extrapolate to the young
stand by construction?

Run:  PYTHONPATH=. WANDB_MODE=offline GREENVISTA_LAZ_DIR=/tmp/lazall \
        python scripts/exp_mechanistic_decomposition.py
"""
import warnings
import numpy as np, pandas as pd, laspy, geopandas as gpd
warnings.simplefilter("ignore")

from greenvista import config
from greenvista.repro import seed_everything, write_manifest
from greenvista.stand_volume import StandVolume, lidar_hdom
from greenvista.area_based import data as abd, validate as abv
from greenvista.eval import (metrics, bootstrap_ci, jackknife_plus_interval, coverage,
                             interval_score, permutation_test)
from greenvista import mechanistic_volume as mv

FF3D_CSV = config.OUT_DIR / "ff3d_detection.csv"
HDOM_CACHE = config.OUT_DIR / "mechanistic_hdom_cache.csv"


# ------------------------------------------------------------------------------------------------
# Build the 13-plot mechanistic table (plot truth + FF3D counts + LiDAR Hdom/H̄)
# ------------------------------------------------------------------------------------------------
def lidar_heights(df):
    """Add LiDAR Hdom (top 50% cell-maxima) and H̄ (all cell-maxima) at each plot centre — the same
    lidar_hdom used by hdom_yield_pipeline.py.  Cached so reruns skip the cloud reads."""
    if HDOM_CACHE.exists():
        c = pd.read_csv(HDOM_CACHE)
        m = df.merge(c, on=["talhao", "parcela"], how="left")
        if m[["Hdom", "Hmean"]].notna().all().all():
            return m
    parc = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for col in ("talhao", "parcela"):
        parc[col] = pd.to_numeric(parc[col], errors="coerce").astype("Int64")
    parc["cx"], parc["cy"] = parc.geometry.x, parc.geometry.y
    df = df.merge(parc[["talhao", "parcela", "cx", "cy"]], on=["talhao", "parcela"])
    # Iterate df in its natural order (stands are contiguous) so results align to the rows, exactly
    # like hdom_yield_pipeline.build_table.  Sorting here and assigning back misaligns the rows.
    cloud, hdom, hmean = {}, [], []
    for _, r in df.iterrows():
        t = int(r.talhao)
        if t not in cloud:
            cloud.clear()   # keep one cloud in memory; stands are contiguous in df order
            las = laspy.read(str(config.LAZ_DIR / f"SaoManuelTotal_{t:03d}.laz"))
            cloud[t] = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
        xyz = cloud[t]
        hdom.append(lidar_hdom(xyz, (r.cx, r.cy), top_frac=0.5))   # dominant height
        hmean.append(lidar_hdom(xyz, (r.cx, r.cy), top_frac=1.0))  # mean canopy-top height
    df = df.assign(Hdom=hdom, Hmean=hmean)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    df[["talhao", "parcela", "Hdom", "Hmean"]].to_csv(HDOM_CACHE, index=False)
    return df


def build_table():
    df = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                        config.DATA / "5-dados_campo/inventario_est.csv")
    df["age"] = df[abd.AGE]
    ff = pd.read_csv(FF3D_CSV)
    ff = ff.rename(columns={"ff3d_stems_ha": "ff3d_n_ha", "field_stems_ha": "field_n_ha"})
    df = df.merge(ff[["plot_id", "ff3d_n_ha", "field_n_ha", "ff3d_in_plot", "field_n_arv",
                      "det_ratio"]], on="plot_id", how="inner")
    # field mean height + field Hdom (to expose the perfect-input ceiling / isolate the LiDAR-height error)
    inv = (pd.read_csv(config.DATA / "5-dados_campo/inventario_est.csv")[["talhao", "parcela", "H_est", "D_cm"]]
           .rename(columns={"H_est": "H_field", "D_cm": "D_field"}))
    hd = (pd.read_csv(config.DATA / "5-dados_campo/inv_euc.csv")
          .groupby(["talhao", "parcela"]).H_dom.first().reset_index().rename(columns={"H_dom": "Hdom_field"}))
    df = df.merge(inv, on=["talhao", "parcela"]).merge(hd, on=["talhao", "parcela"])
    df = lidar_heights(df)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------------------------
def evaluate(name, make, df, y, n_perm=5000):
    """LOPO + LOTO metrics, conformal cov95, permutation p, bootstrap rRMSE CI for one model."""
    lopo, lo, hi = abv._lopo(make, df)
    loto = abv._group_cv(make, df)
    ml, mg = metrics(y, lopo), metrics(y, loto)
    c_lo, c_hi = jackknife_plus_interval(y, lopo, alpha=0.05)
    ci = bootstrap_ci(y, lopo, seed=0)
    p = permutation_test(y, lopo, n_perm=n_perm, seed=0)["p_value"]
    return {"model": name,
            "LOPO_R2": ml["R2"], "LOPO_rRMSE": ml["rRMSE_pct"], "LOPO_bias": ml["bias_pct"],
            "LOTO_R2": mg["R2"], "LOTO_rRMSE": mg["rRMSE_pct"], "LOTO_bias": mg["bias_pct"],
            "cov95": coverage(y, c_lo, c_hi), "int_score": interval_score(y, c_lo, c_hi),
            "perm_p": p, "rRMSE_ci_lo": ci["rRMSE_ci"][0], "rRMSE_ci_hi": ci["rRMSE_ci"][1]}


def mean_tree_ceiling(y):
    """Ceiling of the decomposition with perfect field inputs (field D̄,H̄,N + G2, mean-tree).
    Not a LiDAR result: it bounds how well N·v̄(D̄,H̄) can match the Schumacher-summed Vol_ha."""
    inv = pd.read_csv(config.DATA / "5-dados_campo/inventario_est.csv")
    g2, _, _, _ = fit_g2()
    inv["vbar"] = mv.g2_volume(inv.D_cm, inv.H_est, g2)
    inv["Vha_hat"] = inv.n_arv_ha * inv.vbar
    m = inv.set_index(["talhao", "parcela"])
    return m


def fit_g2():
    """Refit G2 from the destructive scaling data and sanity-check against the published R²/RMSE."""
    return mv.fit_g2_from_cubagem(config.DATA / "5-dados_campo/volumes.csv")


# ------------------------------------------------------------------------------------------------
def figure(df, y, rows, oof, g2_info, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tal = df.talhao.to_numpy()
    ut = np.unique(tal)
    cmap = plt.get_cmap("tab10")
    colors = {t: cmap(i % 10) for i, t in enumerate(ut)}
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))

    def scatter(a, pred, title, sub):
        lim = [min(y.min(), pred.min()) * 0.9, max(y.max(), pred.max()) * 1.05]
        a.plot(lim, lim, "k--", lw=1, alpha=0.6)
        for t in ut:
            m = tal == t
            a.scatter(y[m], pred[m], s=70, color=colors[t], edgecolor="k", lw=0.5,
                      label=f"T{t:03d}")
        a.set_xlabel("field Vol_ha (m³/ha)"); a.set_ylabel("predicted Vol_ha (m³/ha)")
        r = metrics(y, pred)
        a.set_title(f"{title}\n{sub}  R²={r['R2']:+.2f}  rRMSE={r['rRMSE_pct']:.1f}%  bias={r['bias_pct']:+.1f}%",
                    fontsize=10)
        a.set_xlim(lim); a.set_ylim(lim); a.grid(alpha=0.25)

    scatter(ax[0], oof["mech_loto"], "Mechanistic  N·v̄(D̄,H̄)", "leave-one-stand-out")
    scatter(ax[1], oof["base_loto"], "Empirical  Hdom+age", "leave-one-stand-out")
    ax[0].legend(fontsize=7, ncol=2, loc="upper left")

    # panel 3: the count calibration (FF3D observed vs field, + calibrated)
    a = ax[2]
    cal = mv.CountCalibrator("ratio").fit(df.ff3d_n_ha, df.field_n_ha)
    for t in ut:
        m = tal == t
        a.scatter(df.field_n_ha[m], df.ff3d_n_ha[m], s=70, color=colors[t], edgecolor="k",
                  lw=0.5, marker="o")
        a.scatter(df.field_n_ha[m], cal.predict(df.ff3d_n_ha[m]), s=70, color=colors[t],
                  edgecolor="k", lw=0.5, marker="^")
    lim = [df.field_n_ha.min() * 0.85, df.field_n_ha.max() * 1.1]
    a.plot(lim, lim, "k--", lw=1, alpha=0.6)
    a.set_xlabel("field stems/ha"); a.set_ylabel("FF3D stems/ha")
    rr = np.corrcoef(df.field_n_ha, df.ff3d_n_ha)[0, 1]
    a.set_title(f"FF3D count vs field (○ raw, △ calibrated)\nk={cal.k_:.2f}  corr(raw,field) r={rr:+.2f}",
                fontsize=10)
    a.set_xlim(lim); a.grid(alpha=0.25)

    fig.suptitle("Mechanistic volume decomposition — cross-stand validation (N=13 plots, 6 stands)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
def main():
    seed_everything(config.ABA.seed)

    g2_coef, g2_r2, g2_rmse, g2_n = fit_g2()
    print(f"G2 refit from volumes.csv (N={g2_n}): R²={g2_r2:.4f} RMSE={g2_rmse:.4f} "
          f"(report R²=0.9950/0.0190) | coef={tuple(round(c, 8) for c in g2_coef)}")
    print(f"Campos coef (b0,b1,b2)={mv.CAMPOS}  [Cubagem.R L110-112]")

    df = build_table()
    y = df.Vol_ha.to_numpy()
    print(f"\nN={len(df)} plots, {df.talhao.nunique()} stands | Vol_ha {y.min():.0f}–{y.max():.0f} m³/ha")
    print(f"  LiDAR Hdom {df.Hdom.min():.1f}–{df.Hdom.max():.1f} m | H̄ {df.Hmean.min():.1f}–{df.Hmean.max():.1f} m")
    print(f"  FF3D stems/ha {df.ff3d_n_ha.min():.0f}–{df.ff3d_n_ha.max():.0f} | "
          f"field stems/ha {df.field_n_ha.min():.0f}–{df.field_n_ha.max():.0f} | "
          f"detection {df.det_ratio.mean()*100:.0f}%")
    # mechanistic intermediates with the fitted-on-all model (illustrative, not the CV prediction)
    full = mv.MechanisticVolume(g2_coef=g2_coef).fit(df, y)
    df["Dbar_hat"] = full.dbar(df)
    print(f"  Campos-inverted D̄ {df.Dbar_hat.min():.1f}–{df.Dbar_hat.max():.1f} cm | "
          f"v̄ {full.vbar(df).min():.3f}–{full.vbar(df).max():.3f} m³ | bias term={full.bias_:.3f}")

    # --- perfect-inputs ceiling of the decomposition -------------------------------------------
    ceil = mean_tree_ceiling(y)
    cyhat = ceil["Vha_hat"].to_numpy()
    cy = ceil["Vol_ha"].to_numpy()
    mc = metrics(cy, cyhat)
    print(f"\n[ceiling] mean-tree G2 with PERFECT field D̄,H̄,N: R²={mc['R2']:+.2f} "
          f"rRMSE={mc['rRMSE_pct']:.1f}% bias={mc['bias_pct']:+.1f}%  "
          f"(Jensen + G2≠Schumacher — bounds the LiDAR result)")

    # --- model zoo ------------------------------------------------------------------------------
    def MK(cc, bias):
        return lambda: mv.make_mechanistic(g2_coef=g2_coef, count_correction=cc, fit_bias=bias)

    # diagnostic: perfect count (use field_n_ha as the FF3D count) to isolate detection error
    def MK_perfectN():
        return lambda: mv.make_mechanistic(g2_coef=g2_coef, count_correction="none", fit_bias=True,
                                           count_col="field_n_ha")

    # diagnostic: perfect count and field mean-height/Hdom → the decomposition's ceiling under CV
    def MK_fieldH():
        return lambda: mv.make_mechanistic(g2_coef=g2_coef, count_correction="none", fit_bias=True,
                                           count_col="field_n_ha", hmean_col="H_field",
                                           hdom_col="Hdom_field")

    specs = {
        "Mechanistic (ratio-count + bias)":  MK("ratio", True),
        "Mechanistic (ratio-count, no bias)": MK("ratio", False),
        "Mechanistic (linear-count + bias)": MK("linear", True),
        "Mechanistic (raw count, no bias)":  MK("none", False),
        "Mechanistic (PERFECT count, LiDAR H)": MK_perfectN(),
        "Mechanistic (field H + field N)":   MK_fieldH(),
        "Empirical Hdom+age (baseline)":     lambda: StandVolume(("Hdom", "age")),
        "Empirical Hdom only":               lambda: StandVolume(("Hdom",)),
    }

    rows = [evaluate(name, make, df, y) for name, make in specs.items()]
    res = pd.DataFrame(rows)

    # out-of-fold vectors for the figure (main mechanistic vs baseline)
    oof = {}
    mk_mech = MK("ratio", True); mk_base = lambda: StandVolume(("Hdom", "age"))
    oof["mech_lopo"] = abv._lopo(mk_mech, df)[0]; oof["mech_loto"] = abv._group_cv(mk_mech, df)
    oof["base_lopo"] = abv._lopo(mk_base, df)[0]; oof["base_loto"] = abv._group_cv(mk_base, df)

    print("\n=== Vol_ha decomposition vs empirical baseline (out-of-fold CV) ===")
    hdr = f"  {'model':36s} {'LOPO_R2':>7} {'LOPO%':>6} {'LOTO_R2':>7} {'LOTO%':>6} {'cov95':>6} {'perm_p':>7}"
    print(hdr)
    for _, r in res.iterrows():
        print(f"  {r['model']:36s} {r.LOPO_R2:+7.2f} {r.LOPO_rRMSE:6.1f} "
              f"{r.LOTO_R2:+7.2f} {r.LOTO_rRMSE:6.1f} {r.cov95:6.2f} {r.perm_p:7.3f}")

    mech = res[res.model == "Mechanistic (ratio-count + bias)"].iloc[0]
    base = res[res.model == "Empirical Hdom+age (baseline)"].iloc[0]
    better = mech.LOTO_R2 > base.LOTO_R2 and mech.LOTO_rRMSE < base.LOTO_rRMSE
    verdict = ("YES — mechanistic decomposition generalizes cross-stand BETTER than the empirical fit"
               if better else
               "NO — mechanistic does NOT beat the empirical Hdom+age fit cross-stand at N=13")
    print(f"\nVERDICT: mech LOTO R²={mech.LOTO_R2:+.2f}/{mech.LOTO_rRMSE:.1f}%  vs  "
          f"base LOTO R²={base.LOTO_R2:+.2f}/{base.LOTO_rRMSE:.1f}%  →  {verdict}")
    print("CAVEAT: FF3D count is stand-biased (dense stands under-detect) → count calibration still "
          "needs field plots; the mechanistic route removes the *empirical-surface* fit, not the "
          "need for some calibration.")

    # --- persist --------------------------------------------------------------------------------
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = config.OUT_DIR / "mechanistic_decomposition.csv"
    out_png = config.OUT_DIR / "mechanistic_decomposition.png"
    out_json = config.OUT_DIR / "mechanistic_decomposition.json"
    res.to_csv(out_csv, index=False)
    figure(df, y, res, oof, (g2_r2, g2_rmse), out_png)

    write_manifest(out_json, script="exp_mechanistic_decomposition.py",
                   config={"N": int(len(df)), "stands": int(df.talhao.nunique()),
                           "g2_coef": [float(c) for c in g2_coef], "g2_r2": g2_r2, "g2_rmse": g2_rmse,
                           "campos": list(mv.CAMPOS), "seed": config.ABA.seed,
                           "count_source": str(FF3D_CSV.name),
                           "hdom": "lidar_hdom top_frac=0.5", "hmean": "lidar_hdom top_frac=1.0"},
                   seed=config.ABA.seed,
                   metrics={"models": res.to_dict(orient="records"),
                            "perfect_input_ceiling": mc,
                            "verdict": verdict,
                            "mech_vs_base_LOTO": {"mech_R2": float(mech.LOTO_R2),
                                                  "mech_rRMSE": float(mech.LOTO_rRMSE),
                                                  "base_R2": float(base.LOTO_R2),
                                                  "base_rRMSE": float(base.LOTO_rRMSE)}},
                   extra={"provenance": {
                       "campos": "Codigos/Cubagem.R L110-118; Modelos_Hipsométricos.R (lm ln_H~inv_D+ln_H_dom)",
                       "g2": "Codigos/Cubagem.R L72 (lm V~D2+D2_H+H_m); refit from 5-dados_campo/volumes.csv; "
                             "report relatorio_volumetricos_completo.md R²=0.995"}})
    print(f"\nwrote {out_csv}\n      {out_png}\n      {out_json}")

    # optional W&B (offline-friendly)
    try:
        from greenvista.tracking import log_experiment
        log_experiment("mechanistic-decomposition", group="volume", tags=["mechanistic", "N-stems-x-vbar"],
                       config={"task": "V_ha = N·v̄(D̄,H̄)", "N": int(len(df))},
                       metrics={"mech_LOTO_R2": float(mech.LOTO_R2), "mech_LOTO_rRMSE": float(mech.LOTO_rRMSE),
                                "base_LOTO_R2": float(base.LOTO_R2), "verdict": verdict},
                       table_csv=str(out_csv), artifacts=[str(out_png)])
    except Exception as e:
        print(f"(W&B logging skipped: {e})")

    return res


if __name__ == "__main__":
    main()

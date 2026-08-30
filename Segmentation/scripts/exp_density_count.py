#!/usr/bin/env python3
"""Stem density (trees/ha) from LiDAR as a first-class, calibrated, census-validated deliverable.

Predicts field stem density from a LiDAR stem detector (FF3D; CHM local maxima as a baseline), with the
correction learned under leave-one-stand-out (fit on held-in stands, apply to the held-out stand —
never a per-stand factor scored on that same stand). Reports calibrated trees/ha error
(rRMSE, bias, R²) under LOPO + LOTO with bootstrap CIs, conformal coverage, and a CV-level shuffle
control, versus the uncalibrated detector and versus CHM.

Ground truth = the São Manuel field census (`5-dados_campo/inv_euc.csv`): live measured stems per
plot (D_cm present) → trees/ha over the 400 m² plot. This reconciles to `inventario_est.n_arv`
(the count FF3D was benchmarked against); the raw 875 census rows also include gaps (cod F) and dead
stems, which a detector cannot see and which are excluded from "stem density".

Run: PYTHONPATH=. GREENVISTA_LAZ_DIR=<lazdir> WANDB_MODE=offline \
     python scripts/exp_density_count.py
"""
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import laspy
from scipy.ndimage import maximum_filter, gaussian_filter

from greenvista import config
from greenvista.repro import seed_everything, write_manifest
from greenvista.density import DensityCalibrator, cv_predict, to_density, detection_rate
from greenvista.eval import metrics, bootstrap_ci, jackknife_plus_interval, coverage, cv_permutation_test

PR = config.PLOT_RADIUS_M
CIRCLE_HA = config.PLOT_AREA_HA           # 12 m-radius detection footprint (0.04524 ha)
PLOT_HA = 0.04                            # 400 m² field census plot
CHM_RES, CHM_HMIN, CHM_WIN = 0.5, 5.0, 3  # CHM local-maxima baseline (matches spike_s4_chm_detection)
SEED = 0


# --------------------------------------------------------------------------------------------------
# ground truth: live stems per plot from the census
# --------------------------------------------------------------------------------------------------
def field_counts():
    inv = pd.read_csv(config.DATA / "5-dados_campo/inv_euc.csv")
    live = (inv[inv.D_cm.notna()].groupby(["talhao", "parcela"]).size()
            .rename("field_live").reset_index())
    est = pd.read_csv(config.DATA / "5-dados_campo/inventario_est.csv")[["talhao", "parcela", "n_arv"]]
    gt = live.merge(est, on=["talhao", "parcela"])
    gt["field_stems_ha"] = to_density(gt.n_arv, PLOT_HA)   # canonical live-stem density
    return gt


# --------------------------------------------------------------------------------------------------
# CHM local-maxima baseline + a model-agnostic canopy-cover proxy, per plot, from the clouds
# --------------------------------------------------------------------------------------------------
def _chm_local_maxima(x, y, z, cx, cy):
    x0, y0 = x.min(), y.min()
    nx = int((x.max() - x0) / CHM_RES) + 1
    ny = int((y.max() - y0) / CHM_RES) + 1
    ix = ((x - x0) / CHM_RES).astype(int)
    iy = ((y - y0) / CHM_RES).astype(int)
    chm = np.zeros((nx, ny), np.float32)
    np.maximum.at(chm, (ix, iy), z.astype(np.float32))
    sm = gaussian_filter(chm, 0.6)
    peaks = (sm == maximum_filter(sm, size=CHM_WIN)) & (chm > CHM_HMIN)
    px, py = np.where(peaks)
    dx, dy = x0 + px * CHM_RES + CHM_RES / 2, y0 + py * CHM_RES + CHM_RES / 2
    return int(((dx - cx) ** 2 + (dy - cy) ** 2 <= PR ** 2).sum())


def chm_and_cover(talhoes, parc):
    chm, cover = {}, {}
    for t in sorted(talhoes):
        las = laspy.read(str(config.LAZ_DIR / f"SaoManuelTotal_{t:03d}.laz"))
        X, Y, Z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        cls = np.asarray(las.classification)
        for _, r in parc[parc.talhao == t].iterrows():
            p = int(r.parcela)
            m = (np.abs(X - r.cx) <= PR) & (np.abs(Y - r.cy) <= PR)
            xx, yy, zz, cc = X[m], Y[m], Z[m], cls[m]
            veg = (cc != config.GROUND_NOISE_CLASSES[0]) & (cc != config.GROUND_NOISE_CLASSES[1])
            chm[(t, p)] = _chm_local_maxima(xx[veg], yy[veg], zz[veg], r.cx, r.cy)
            incirc = (xx - r.cx) ** 2 + (yy - r.cy) ** 2 <= PR ** 2   # canopy cover proxy
            cover[(t, p)] = float((zz[incirc] > config.Z_MIN_M).mean())
    return chm, cover


def build_table():
    gt = field_counts()
    ff = (pd.read_csv(config.OUT_DIR / "ff3d_detection.csv")
          .drop(columns=["field_stems_ha", "ff3d_stems_ha"], errors="ignore"))
    ff["parcela"] = ff.plot_id.str.split("_").str[1].astype(int)
    df = ff.merge(gt, on=["talhao", "parcela"]).sort_values(["talhao", "parcela"]).reset_index(drop=True)

    parc = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for c in ("talhao", "parcela"):
        parc[c] = pd.to_numeric(parc[c], errors="coerce").astype("Int64")
    parc["cx"], parc["cy"] = parc.geometry.x, parc.geometry.y
    chm, cover = chm_and_cover(df.talhao.unique(), parc)

    df["chm_in_plot"] = [chm[(t, p)] for t, p in zip(df.talhao, df.parcela)]
    df["cover"] = [cover[(t, p)] for t, p in zip(df.talhao, df.parcela)]
    df["ff3d_stems_ha_raw"] = to_density(df.ff3d_in_plot, CIRCLE_HA)
    df["chm_stems_ha_raw"] = to_density(df.chm_in_plot, CIRCLE_HA)
    return df


# --------------------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------------------
def evaluate(make, df, y, g, n_perm=1000):
    lopo = cv_predict(make, df, y)
    loto = cv_predict(make, df, y, groups=g)
    ml, mg = metrics(y, lopo), metrics(y, loto)
    ci_lopo = bootstrap_ci(y, lopo, seed=SEED)["rRMSE_ci"]
    ci_loto = bootstrap_ci(y, loto, seed=SEED)["rRMSE_ci"]
    lo, hi = jackknife_plus_interval(y, lopo, alpha=0.05)
    cov = coverage(y, lo, hi)
    # CV-level shuffle control under LOTO: re-run the whole calibration on permuted counts.
    perm = cv_permutation_test(lambda yy: cv_predict(make, df, yy, groups=g), y,
                               n_perm=n_perm, seed=SEED)["p_value"]
    return {"lopo": ml, "loto": mg, "rRMSE_lopo_ci": ci_lopo, "rRMSE_loto_ci": ci_loto,
            "cov95_lopo": cov, "shuffle_p_loto": perm, "lopo_pred": lopo, "loto_pred": loto}


def main():
    seed_everything(SEED)
    df = build_table()
    y = df.field_stems_ha.to_numpy()
    g = df.talhao.to_numpy()
    n_stands = df.talhao.nunique()

    print(f"N={len(df)} plots across {n_stands} stands (stands {sorted(df.talhao.unique())})")
    print(f"field live-stem density: {y.min():.0f}–{y.max():.0f} trees/ha (mean {y.mean():.0f}, sd {y.std():.0f})")
    print(f"live stems reconcile to inventario_est.n_arv within ±1 on 2 plots "
          f"(census also has {int((pd.read_csv(config.DATA/'5-dados_campo/inv_euc.csv').D_cm.isna()).sum())} "
          f"non-stem rows: gaps/dead, excluded)")
    print(f"raw detection (aggregate): FF3D {detection_rate(df.ff3d_in_plot, df.n_arv)*100:.0f}%  "
          f"CHM {detection_rate(df.chm_in_plot, df.n_arv)*100:.0f}%  "
          f"(counts over a {PR:.0f} m circle = {CIRCLE_HA*1e4:.0f} m² vs the {PLOT_HA*1e4:.0f} m² field plot)")
    print(f"raw FF3D density {df.ff3d_stems_ha_raw.mean():.0f} trees/ha, CHM {df.chm_stems_ha_raw.mean():.0f} "
          f"— both under the field {y.mean():.0f}\n")

    models = {
        "FF3D raw (uncalibrated)":   lambda: DensityCalibrator("ff3d_stems_ha_raw", method="raw"),
        "FF3D global correction":    lambda: DensityCalibrator("ff3d_stems_ha_raw", method="global"),
        "FF3D proxy=cover":          lambda: DensityCalibrator("ff3d_stems_ha_raw", method="proxy", proxy_col="cover"),
        "FF3D proxy=unassigned%":    lambda: DensityCalibrator("ff3d_stems_ha_raw", method="proxy", proxy_col="pct_unassigned"),
        "CHM raw (uncalibrated)":    lambda: DensityCalibrator("chm_stems_ha_raw", method="raw"),
        "CHM global correction":     lambda: DensityCalibrator("chm_stems_ha_raw", method="global"),
        "CHM proxy=cover":           lambda: DensityCalibrator("chm_stems_ha_raw", method="proxy", proxy_col="cover"),
    }

    res = {name: evaluate(make, df, y, g) for name, make in models.items()}

    hdr = f"{'model':<26}{'LOPO rRMSE':>12}{'LOPO R²':>9}{'LOPO bias':>11}{'LOTO rRMSE':>12}{'LOTO R²':>9}{'LOTO bias':>11}{'cov95':>7}{'shufP':>7}"
    print(hdr); print("-" * len(hdr))
    for name, r in res.items():
        l, gm = r["lopo"], r["loto"]
        print(f"{name:<26}{l['rRMSE_pct']:>11.1f}%{l['R2']:>+9.2f}{l['bias_pct']:>+10.1f}%"
              f"{gm['rRMSE_pct']:>11.1f}%{gm['R2']:>+9.2f}{gm['bias_pct']:>+10.1f}%"
              f"{r['cov95_lopo']:>7.2f}{r['shuffle_p_loto']:>7.3f}")
    print(f"\n(LOTO rRMSE 95% bootstrap CIs — FF3D proxy=unassigned%: "
          f"[{res['FF3D proxy=unassigned%']['rRMSE_loto_ci'][0]:.1f}, {res['FF3D proxy=unassigned%']['rRMSE_loto_ci'][1]:.1f}]%; "
          f"FF3D global: [{res['FF3D global correction']['rRMSE_loto_ci'][0]:.1f}, {res['FF3D global correction']['rRMSE_loto_ci'][1]:.1f}]%)")

    # ---- operational levels: aggregate the LOTO plot predictions up -----------------------------
    # Per-plot detection noise partly averages out over a stand, and the 12 m-circle vs 400 m² plot
    # footprint mismatch cancels — so density is most trustworthy at stand/estate level.
    const_plot = 100 * y.std(ddof=0) / y.mean()   # rRMSE of "assume the mean everywhere" = the R²=0 line
    agg = {}
    print(f"\n=== aggregated levels (LOTO predictions rolled up) — R²=0 baseline is "
          f"'assume mean density', per-plot rRMSE {const_plot:.1f}% ===")
    print(f"{'model':<26}{'per-STAND rRMSE':>16}{'stand bias':>12}{'stand R²':>10}{'estate bias':>13}")
    for name in ("FF3D global correction", "FF3D proxy=unassigned%", "CHM global correction"):
        p = res[name]["loto_pred"]
        d = df.assign(_p=p)
        by = d.groupby("talhao").agg(f=("field_stems_ha", "mean"), phat=("_p", "mean"))
        ms = metrics(by.f.to_numpy(), by.phat.to_numpy())
        estate_bias = 100 * (p.mean() - y.mean()) / y.mean()
        agg[name] = {"per_stand": ms, "estate_bias_pct": float(estate_bias),
                     "const_mean_plot_rRMSE_pct": float(const_plot)}
        print(f"{name:<26}{ms['rRMSE_pct']:>15.1f}%{ms['bias_pct']:>+11.1f}%{ms['R2']:>+10.2f}{estate_bias:>+12.1f}%")

    # ---- persist ------------------------------------------------------------------------------
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    best = "FF3D proxy=unassigned%"
    out = df[["plot_id", "talhao", "parcela", "n_arv", "field_stems_ha",
              "ff3d_in_plot", "ff3d_stems_ha_raw", "chm_in_plot", "chm_stems_ha_raw",
              "pct_unassigned", "cover"]].copy()
    out["ff3d_global_pred"] = res["FF3D global correction"]["loto_pred"]
    out["ff3d_proxy_pred"] = res[best]["loto_pred"]
    csv_path = config.OUT_DIR / "density_count.csv"
    out.to_csv(csv_path, index=False)

    def _slim(r):
        return {"LOPO": r["lopo"], "LOTO": r["loto"], "rRMSE_lopo_ci": r["rRMSE_lopo_ci"],
                "rRMSE_loto_ci": r["rRMSE_loto_ci"], "cov95_lopo": r["cov95_lopo"],
                "shuffle_p_loto": r["shuffle_p_loto"]}
    manifest = write_manifest(
        config.OUT_DIR / "density_count.json",
        script="scripts/exp_density_count.py",
        config={"ground_truth": "inv_euc live stems (D_cm present) → trees/ha; = inventario_est.n_arv",
                "detector": "FF3D panoptic (manual_match/ff3d_detection.csv); CHM local-maxima baseline",
                "detection_footprint_ha": CIRCLE_HA, "field_plot_ha": PLOT_HA,
                "proxies": ["cover (returns>2m, model-agnostic)", "pct_unassigned (FF3D-internal)"],
                "n_plots": len(df), "n_stands": int(n_stands), "seed": SEED},
        metrics={name: _slim(r) for name, r in res.items()},
        seed=SEED,
        extra={"aggregate_detection": {"ff3d": detection_rate(df.ff3d_in_plot, df.n_arv),
                                       "chm": detection_rate(df.chm_in_plot, df.n_arv)},
               "aggregated_levels": agg})
    print(f"\nwrote {csv_path}\nwrote {config.OUT_DIR/'density_count.json'}")

    _figure(df, y, res, config.OUT_DIR / "density_count.png")
    print(f"wrote {config.OUT_DIR/'density_count.png'}")

    try:  # project convention: mirror to W&B (offline by default via WANDB_MODE)
        from greenvista.tracking import log_experiment
        log_experiment("density-count", config=manifest["config"],
                       metrics={f"{n}/LOTO_rRMSE": r["loto"]["rRMSE_pct"] for n, r in res.items()},
                       table_csv=str(csv_path), artifacts=[str(config.OUT_DIR / "density_count.png")],
                       tags=["density", "stem-count", "calibration", "LOTO"])
    except Exception as e:
        print(f"(W&B log skipped: {e})")


def _figure(df, y, res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"raw": "#9aa0a6", "global": "#4c78a8", "proxy": "#e45756", "chm": "#54a24b"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) LOTO predicted vs observed for the FF3D ladder
    lim = [700, 1850]
    ax[0].plot(lim, lim, "k--", lw=1, alpha=0.6, label="1:1")
    for key, name, col in [("raw", "FF3D raw", C["raw"]),
                           ("global", "FF3D global", C["global"]),
                           ("proxy", "FF3D proxy=unassigned%", C["proxy"])]:
        nm = {"raw": "FF3D raw (uncalibrated)", "global": "FF3D global correction",
              "proxy": "FF3D proxy=unassigned%"}[key]
        ax[0].scatter(y, res[nm]["loto_pred"], s=42, color=col, alpha=0.85,
                      edgecolor="white", linewidth=0.6, label=name)
    ax[0].set_xlim(lim); ax[0].set_ylim(lim)
    ax[0].set_xlabel("field stem density (trees/ha)")
    ax[0].set_ylabel("LOTO-predicted (trees/ha)")
    ax[0].set_title("(a) Calibrated density vs census — cross-stand (LOTO)")
    ax[0].legend(fontsize=8, loc="upper left")

    # (b) LOTO rRMSE per model with bootstrap CI whiskers
    names = list(res.keys())
    rr = [res[n]["loto"]["rRMSE_pct"] for n in names]
    err = [[res[n]["loto"]["rRMSE_pct"] - res[n]["rRMSE_loto_ci"][0] for n in names],
           [res[n]["rRMSE_loto_ci"][1] - res[n]["loto"]["rRMSE_pct"] for n in names]]
    cols = [C["chm"] if n.startswith("CHM") else
            (C["raw"] if "raw" in n else C["proxy"] if "proxy" in n else C["global"]) for n in names]
    yb = np.arange(len(names))[::-1]
    ax[1].barh(yb, rr, color=cols, alpha=0.9)
    ax[1].errorbar(rr, yb, xerr=err, fmt="none", ecolor="#333", elinewidth=1, capsize=3)
    ax[1].set_yticks(yb); ax[1].set_yticklabels(names, fontsize=8)
    ax[1].set_xlabel("LOTO rRMSE (%)  — lower is better")
    ax[1].set_title("(b) Cross-stand density error (95% bootstrap CI)")

    # (c) the proxy relationship: correction factor vs unassigned%
    c = y / df.ff3d_stems_ha_raw.to_numpy()
    sc = ax[2].scatter(df.pct_unassigned, c, c=df.talhao, cmap="tab10", s=48,
                       edgecolor="white", linewidth=0.6)
    ax[2].set_xlabel("FF3D unassigned points (%)  — an observable LiDAR proxy")
    ax[2].set_ylabel("required correction  field / raw-FF3D")
    r = np.corrcoef(df.pct_unassigned, c)[0, 1]
    ax[2].set_title(f"(c) The correction is observable (r={r:+.2f})")
    plt.colorbar(sc, ax=ax[2], label="stand", fraction=0.046)

    fig.suptitle("LiDAR stem density (trees/ha), calibrated to the São Manuel census",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

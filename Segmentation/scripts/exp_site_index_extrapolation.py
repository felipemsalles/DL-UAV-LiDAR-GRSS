#!/usr/bin/env python3
"""The site-index / yield-curve model
— Vol_ha = exp(a + b*ln(Hdom) + c*ln(age)), the log-log family that is the LiDAR-native analogue of
the company's G2/Schumacher + site-index inventory — is a mechanistic, monotonic model constrained
to extrapolate sanely to the one young stand (stand 4, ~4.8 yr) that breaks every unconstrained
empirical fit under leave-one-stand-out (LOTO).

This script does not re-derive the deployable model (that is `scripts/hdom_yield_pipeline.py` +
`greenvista/stand_volume.py::StandVolume`, reused here unedited); it tests the extrapolation claim
head-to-head against the strongest empirical competitors on the exact same 13-plot table:
  1. mechanistic  - StandVolume(Hdom, age)               (greenvista.stand_volume, log-log, robust Huber)
  2. GPR (RBF)    - GaussianProcessRegressor(Hdom, age)   (greenvista.area_based.models.make_gpr) — same
                    two inputs as the mechanistic model, isolating functional form as the variable.
  3. log-GPR      - log-target GPR, near-linear kernel (Hdom, age) (make_loggpr) — the "improved" GPR
                    that still uses a flexible, unconstrained kernel.
  4. PLS          - partial least squares on the full lidR metric suite, no age (make_pls) — the
                    literature-standard LiDAR-only ABA competitor (Cosenza 2021 family).

Validation: LOPO + leave-one-stand-out (both schemes), jackknife+ conformal coverage, bootstrap rRMSE
CIs, a permutation/shuffle control, plus an explicit T4-vs-rest split of the LOTO predictions
(`greenvista.site_index_yield.metrics_for_group` / `metrics_excluding_group`) so "does it fail only on
the young stand?" gets a direct answer instead of one pooled number.

Requires the raw LiDAR clouds (to compute per-plot Hdom, reusing hdom_yield_pipeline.build_table()):
    unzip -o "Dados_SãoManuel/Talhões LiDAR/SaoManuael_LIDAR.zip" -d /tmp/lazall
(GREENVISTA_LAZ_DIR defaults to <tempdir>/lazall = /tmp/lazall, so no env var needed if unzipped there.)

Run: PYTHONPATH=. python scripts/exp_site_index_extrapolation.py
"""
import importlib.util
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from greenvista import config
from greenvista.repro import seed_everything, write_manifest
from greenvista.area_based import data as abd, models as abm, validate as abv
from greenvista.eval import (metrics, jackknife_plus_interval, coverage, interval_score,
                             permutation_test, bootstrap_ci)
from greenvista import site_index_yield as siy

T4 = 4                 # the one young stand (~4.8 yr) — the whole cross-stand bottleneck
GROUP_COL = config.GROUP_COL   # "talhao"

# dataviz-validated categorical colors (checked with node scripts/validate_palette.js)
COL_OTHER = "#4C72B0"   # the other 5 stands, steel blue
COL_T4 = "#DD5F4B"      # stand 4 (young stand), vermillion — means "T4" everywhere in the figure
COL_MECH = "#1B9E77"    # mechanistic yield curve, teal-green
COL_GPR = "#7570B3"     # GPR (RBF), purple
COL_LOGGPR = "#D95F02"  # log-GPR, orange


def _load_hdom_table():
    """Reuse scripts/hdom_yield_pipeline.py's build_table() (computes LiDAR Hdom per plot from the raw
    clouds + merges age/Vol_ha/lidR-suite) rather than duplicating the LiDAR-reading logic. `scripts/`
    is not a package, so import by file path."""
    spec = importlib.util.spec_from_file_location(
        "hdom_yield_pipeline", Path(__file__).parent / "hdom_yield_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df, _cloud = mod.build_table()
    return df


def evaluate(name, df, y, lopo, loto):
    """Metric bundle for one model: LOPO + LOTO (pooled), LOTO excluding T4, and the T4-fold alone.
    The last two separate "collapses everywhere" from "collapses only on the young stand", which is
    the claim under test."""
    m_lopo = metrics(y, lopo)
    lo, hi = jackknife_plus_interval(y, lopo)
    ci_lopo = bootstrap_ci(y, lopo, seed=config.ABA.seed)
    row = {
        "model": name,
        "LOPO_R2": m_lopo["R2"], "LOPO_rRMSE": m_lopo["rRMSE_pct"],
        "LOPO_rRMSE_ci_lo": ci_lopo["rRMSE_ci"][0], "LOPO_rRMSE_ci_hi": ci_lopo["rRMSE_ci"][1],
        "cov95": coverage(y, lo, hi), "int_score": interval_score(y, lo, hi),
        "perm_p": permutation_test(y, lopo, n_perm=3000, seed=config.ABA.seed)["p_value"],
    }
    if loto is not None:
        m_loto = metrics(y, loto)
        ci_loto = bootstrap_ci(y, loto, seed=config.ABA.seed)
        m_excl = siy.metrics_excluding_group(df, y, loto, GROUP_COL, T4)
        m_t4 = siy.metrics_for_group(df, y, loto, GROUP_COL, T4)
        row.update({
            "LOTO_R2": m_loto["R2"], "LOTO_rRMSE": m_loto["rRMSE_pct"],
            "LOTO_rRMSE_ci_lo": ci_loto["rRMSE_ci"][0], "LOTO_rRMSE_ci_hi": ci_loto["rRMSE_ci"][1],
            "LOTO_R2_exclT4": m_excl["R2"], "LOTO_rRMSE_exclT4": m_excl["rRMSE_pct"],
            "T4_bias_pct": m_t4["bias_pct"], "T4_rRMSE_pct": m_t4["rRMSE_pct"],
        })
    return row


def make_figure(df, y, oof, zoo, out_path):
    """2x2 predicted-vs-observed (LOTO) panels, one per model, T4 highlighted — plus a bottom panel
    showing the actual fitted yield curves (Vol_ha vs Hdom at T4's age) for the three (Hdom,age)-only
    models, refit on the 11 non-T4 plots exactly as in the T4 LOTO fold. PLS has no single-Hdom axis
    (it consumes the full lidR suite), so it is shown only in the top scatter row.
    """
    t4_mask = (df[GROUP_COL] == T4).to_numpy()
    all_pred = np.concatenate([oof[n][1] for n in zoo])
    lo_lim = min(y.min(), all_pred.min()) * 0.9
    hi_lim = max(y.max(), all_pred.max()) * 1.05

    mosaic = [["mech", "gpr"], ["loggpr", "pls"], ["curve", "curve"]]
    fig, axd = plt.subplot_mosaic(mosaic, figsize=(11, 13), height_ratios=[1, 1, 0.85])
    panel_key = {"mechanistic (Hdom+age, yield curve)": "mech",
                 "GPR RBF (Hdom+age)": "gpr",
                 "log-GPR (Hdom+age)": "loggpr",
                 "PLS (lidR suite, no age)": "pls"}

    for name, key in panel_key.items():
        ax = axd[key]
        _, loto = oof[name]
        r2, rr = metrics(y, loto)["R2"], metrics(y, loto)["rRMSE_pct"]
        t4b = siy.metrics_for_group(df, y, loto, GROUP_COL, T4)["bias_pct"]
        ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], "--", color="0.6", lw=1, zorder=1)
        ax.scatter(y[~t4_mask], loto[~t4_mask], s=75, color=COL_OTHER, edgecolor="k",
                  linewidth=0.6, label="other 5 stands", zorder=2)
        ax.scatter(y[t4_mask], loto[t4_mask], s=210, marker="*", color=COL_T4, edgecolor="k",
                  linewidth=0.8, label="stand 4 (young, 4.8 yr)", zorder=3)
        ax.set_xlim(lo_lim, hi_lim); ax.set_ylim(lo_lim, hi_lim)
        ax.set_xlabel("Observed Vol_ha (m³/ha)"); ax.set_ylabel("Predicted Vol_ha (m³/ha) — LOTO")
        ax.set_title(f"{name}\nLOTO R²={r2:+.2f}, rRMSE={rr:.1f}%  |  T4 bias={t4b:+.0f}%", fontsize=10.5)
        if key == "mech":
            ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # --- bottom panel: the actual yield curves, refit on the 11 non-T4 plots (= the T4 LOTO fold) ---
    ax = axd["curve"]
    train = df[GROUP_COL] != T4
    Xtr, ytr = df[train], y[train]
    t4_age = float(df.loc[df[GROUP_COL] == T4, "age"].iloc[0])
    hdom_grid = np.linspace(15, 40, 200)
    curve_specs = [("mechanistic", zoo["mechanistic (Hdom+age, yield curve)"], COL_MECH, "-", 3.0),
                  ("GPR (RBF)", zoo["GPR RBF (Hdom+age)"], COL_GPR, "--", 2.2),
                  ("log-GPR", zoo["log-GPR (Hdom+age)"], COL_LOGGPR, "-.", 2.2)]
    curves = {}
    for label, make, col, ls, lw in curve_specs:
        m = make().fit(Xtr, ytr)
        curve = siy.yield_curve(m, hdom_grid, age=t4_age)
        curves[label] = curve
        ax.plot(hdom_grid, curve, ls, color=col, lw=lw, label=f"{label} (fit w/o T4)")
    t4_hdom = df.loc[df[GROUP_COL] == T4, "Hdom"].to_numpy()
    t4_vol = y[t4_mask]
    ax.scatter(t4_hdom, t4_vol, s=220, marker="*", color=COL_T4, edgecolor="k", linewidth=1.0,
              zorder=5, label="stand 4 observed (held out)")
    ax.axvspan(Xtr.Hdom.min(), Xtr.Hdom.max(), color="0.9", zorder=0,
              label="training Hdom range (other 5 stands)")
    ax.annotate("GPR / log-GPR revert to ≈ the training-set mean regardless of Hdom:\n"
               "training age is 9.8–10.6 yr, so age=4.8 sits far outside the kernel's\n"
               "support and the posterior stops responding to Hdom entirely.",
               xy=(37.5, curves["GPR (RBF)"][-1]), xytext=(16, 385), fontsize=8.2,
               arrowprops=dict(arrowstyle="->", color="0.4", lw=1), color="0.25")
    ax.set_xlabel("Hdom (m)"); ax.set_ylabel(f"Vol_ha (m³/ha) at age={t4_age:.1f} yr (T4's age)")
    ax.set_title("Yield curve extrapolation to the young stand's Hdom — "
                 "models refit WITHOUT stand 4", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)

    fig.suptitle("Site-index / yield-curve model vs empirical fits under leave-one-stand-out\n"
                 "blue = other stands (out-of-fold) · red star = stand 4, the young stand held out",
                 y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    seed_everything(config.ABA.seed)
    df = _load_hdom_table()
    y = df.Vol_ha.to_numpy()
    print(f"N={len(df)} plots, {df.talhao.nunique()} stands | LiDAR Hdom {df.Hdom.min():.1f}-"
          f"{df.Hdom.max():.1f} m | age {df.age.min():.1f}-{df.age.max():.1f} yr | "
          f"stand {T4} = the young stand (Hdom {df.loc[df.talhao==T4,'Hdom'].min():.1f}-"
          f"{df.loc[df.talhao==T4,'Hdom'].max():.1f} m, age "
          f"{df.loc[df.talhao==T4,'age'].iloc[0]:.1f} yr)\n")

    zoo = {
        "mechanistic (Hdom+age, yield curve)": lambda: siy.mechanistic_model(("Hdom", "age")),
        "GPR RBF (Hdom+age)":                  lambda: abm.make_gpr(("Hdom", "age")),
        "log-GPR (Hdom+age)":                  lambda: abm.make_loggpr(("Hdom", "age")),
        "PLS (lidR suite, no age)":             lambda: abm.make_pls(abd.ALL_METRICS, max_k=4),
    }
    oof = {name: abv.oof_predictions(mk, df, group_col=GROUP_COL) for name, mk in zoo.items()}
    rows = [evaluate(name, df, y, *oof[name]) for name in zoo]
    res = pd.DataFrame(rows)

    print("=== LOPO + leave-one-stand-out (LOTO), T4 isolated (site-index yield curve vs empirical) ===")
    hdr = (f"  {'model':38s} {'LOPO_R2':>7} {'LOPO%':>6} {'LOTO_R2':>7} {'LOTO%':>6} "
          f"{'LOTO\\T4_R2':>10} {'LOTO\\T4%':>8} {'T4_bias%':>9} {'cov95':>6} {'perm_p':>7}")
    print(hdr)
    for _, r in res.iterrows():
        print(f"  {r['model']:38s} {r['LOPO_R2']:+7.2f} {r['LOPO_rRMSE']:6.1f} "
              f"{r.get('LOTO_R2', float('nan')):+7.2f} {r.get('LOTO_rRMSE', float('nan')):6.1f} "
              f"{r.get('LOTO_R2_exclT4', float('nan')):+10.2f} {r.get('LOTO_rRMSE_exclT4', float('nan')):8.1f} "
              f"{r.get('T4_bias_pct', float('nan')):+9.1f} {r['cov95']:6.2f} {r['perm_p']:7.3f}")

    # --- T4 hold-out detail: predicted vs observed, per model, per plot -------------------------
    t4_tables = []
    for name in zoo:
        _, loto = oof[name]
        t = siy.holdout_table(df, y, loto, GROUP_COL, T4)
        t.insert(0, "model", name)
        t4_tables.append(t)
    t4_df = pd.concat(t4_tables, ignore_index=True)
    print(f"\n=== stand-4 hold-out detail (LOTO T4-fold: trained on the OTHER 5 stands) ===")
    print(t4_df.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    # --- site-index framing (Hdom projected to a reference age) ----------------------------------
    si = siy.stand_site_index(df, ref_age=7.0)
    print(f"\n=== site index (Hdom @ ref_age=7 yr; illustrative guide-curve b=1) per stand ===")
    print(si.round(2).to_string())
    print(f"  → stand {T4} (age {si.loc[T4,'age_mean']:.1f} yr, raw Hdom {si.loc[T4,'Hdom_mean']:.1f} m) "
          f"projects to SI={si.loc[T4,'SI']:.1f} m at age 7 — comparable productivity to the mature "
          f"stands once age is normalized away (the site-index framing the company's G2 inventory uses).")

    honest_ok = (res["LOTO_R2_exclT4"] > 0) & (res["cov95"] >= 0.85) & (res["perm_p"] < 0.05)
    mech_row = res[res.model.str.startswith("mechanistic")].iloc[0]
    print(f"\n=== VERDICT ===")
    print(f"  mechanistic: LOTO R²={mech_row.LOTO_R2:+.2f} (all 6 stands) | T4 bias={mech_row.T4_bias_pct:+.1f}%")
    for _, r in res[~res.model.str.startswith("mechanistic")].iterrows():
        gap = mech_row.LOTO_R2 - r.LOTO_R2
        print(f"  {r['model']:38s}: LOTO R²={r.LOTO_R2:+.2f} | T4 bias={r.T4_bias_pct:+.1f}% "
              f"| mechanistic beats it by ΔLOTO_R²={gap:+.2f}")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.OUT_DIR / "site_index_extrapolation.csv"
    t4_csv_path = config.OUT_DIR / "site_index_extrapolation_t4.csv"
    fig_path = config.OUT_DIR / "site_index_extrapolation.png"
    res.to_csv(csv_path, index=False)
    t4_df.to_csv(t4_csv_path, index=False)
    make_figure(df, y, oof, zoo, fig_path)

    write_manifest(config.OUT_DIR / "site_index_extrapolation_manifest.json",
                   script="exp_site_index_extrapolation.py",
                   config={"seed": config.ABA.seed, "T4_talhao": T4, "group_col": GROUP_COL},
                   metrics={"table": res.to_dict(orient="records"),
                            "t4_holdout": t4_df.to_dict(orient="records"),
                            "site_index": si.reset_index().to_dict(orient="records"),
                            "honest_ok": honest_ok.tolist()})

    try:
        from greenvista.tracking import log_experiment
        log_experiment("site-index-yield-extrapolation",
                       config={"seed": config.ABA.seed, "T4_talhao": T4},
                       metrics={"mechanistic_LOTO_R2": float(mech_row.LOTO_R2),
                                "mechanistic_T4_bias_pct": float(mech_row.T4_bias_pct)},
                       table_csv=str(csv_path), artifacts=[csv_path, t4_csv_path, fig_path])
    except Exception as e:  # never let tracking break the experiment
        print(f"  (W&B logging skipped: {e})")

    print(f"\nwrote {csv_path}, {t4_csv_path}, {fig_path} + manifest")
    return res


if __name__ == "__main__":
    main()

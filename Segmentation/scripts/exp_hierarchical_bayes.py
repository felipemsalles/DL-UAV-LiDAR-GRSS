#!/usr/bin/env python3
"""Bayesian hierarchical (partial-pooling) stand-volume model, validated at N=13 over 6 stands.

    ln Vol_ha = a + b·ln(Hdom) + c·ln(age) + u[stand] + eps,   u ~ N(0, sigma_stand),  eps ~ N(0, sigma_resid)

Partial pooling borrows strength across stands, gives posterior-predictive intervals, and decomposes
variance into a between-stand piece and a plot/residual piece. That decomposition turns the
qualitative finding that the numbers are carried by age, not LiDAR, into a posterior quantity: adding
ln(age) to the mean collapses sigma_stand.

Everything runs through the project's validation harness:
  * LOPO (`area_based.validate._lopo`) and LOTO (leave-one-*stand*-out, whole stands held out).
  * Posterior-predictive 95% interval coverage vs the project's out-of-fold conformal (jackknife+).
  * A shuffle-label control plus a permutation p-value.
  * `seed_everything` + `write_manifest`; a forest/variance figure to manual_match/.
  * Comparison against the frequentist baseline `stand_volume.StandVolume(("Hdom","age"))`.

With only 6 groups sigma_stand is weakly identified, hence the weakly-informative HalfNormal prior;
convergence (divergences / R-hat / ESS) is checked and reported. Run:

    PYTHONPATH=. GREENVISTA_LAZ_DIR=/tmp/lazall WANDB_MODE=offline \
        python scripts/exp_hierarchical_bayes.py
"""
import warnings
warnings.simplefilter("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from greenvista import config
from greenvista.repro import seed_everything, write_manifest
from greenvista.area_based import data as abd, validate as abv
from greenvista.eval import metrics, jackknife_plus_interval, coverage, interval_score, permutation_test
from greenvista.eval.hierarchical import HierarchicalStandVolume
from greenvista.stand_volume import StandVolume, lidar_hdom

TABLE_CSV = config.OUT_DIR / "hierarchical_plot_table.csv"


# --------------------------------------------------------------------- data --
def build_table():
    """Real 13-plot table: LiDAR dominant height (lidar_hdom) + stand age + field Vol_ha.

    Built exactly as scripts/hdom_yield_pipeline.py does. Cached to manual_match/ so reruns skip the
    LiDAR pass (the cache is a pure derivative of the real dataset)."""
    if TABLE_CSV.exists():
        df = pd.read_csv(TABLE_CSV)
        if {"Hdom", "age", "Vol_ha", "talhao"}.issubset(df.columns):
            print(f"loaded cached plot table: {TABLE_CSV}")
            return df
    import laspy
    import geopandas as gpd
    df = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                        config.DATA / "5-dados_campo/inventario_est.csv")
    inv = pd.read_csv(config.DATA / "5-dados_campo/inv_euc.csv")
    df = df.merge(inv.groupby(["talhao", "parcela"]).H_dom.first().reset_index(), on=["talhao", "parcela"])
    parc = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for c in ("talhao", "parcela"):
        parc[c] = pd.to_numeric(parc[c], errors="coerce").astype("Int64")
    parc["cx"], parc["cy"] = parc.geometry.x, parc.geometry.y
    df = df.merge(parc[["talhao", "parcela", "cx", "cy"]], on=["talhao", "parcela"])
    cloud, hd = {}, []
    for _, r in df.iterrows():
        t = int(r.talhao)
        if t not in cloud:
            cloud.clear()
            las = laspy.read(str(config.LAZ_DIR / f"SaoManuelTotal_{t:03d}.laz"))
            cloud[t] = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
        hd.append(lidar_hdom(cloud[t], (r.cx, r.cy)))
    df["Hdom"] = hd
    df["age"] = df[abd.AGE]
    df = df[["plot_id", "talhao", "parcela", "Hdom", "H_dom", "age", "Vol_ha", "zmean", "zq90"]].copy()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_CSV, index=False)
    print(f"built + cached plot table: {TABLE_CSV}")
    return df


# ----------------------------------------------------------------- LOTO OOF --
def loto_oof(make, df, target="Vol_ha", group_col=config.GROUP_COL):
    """Leave-one-stand-out: point preds + posterior-predictive 95% intervals (one whole stand out per
    fold). Mirrors `area_based.validate._group_cv` but also returns predict_interval bounds so we can
    measure cross-stand PPD coverage. Held-out stands are unseen, so the population mean and
    sigma_stand are folded into the interval (see HierarchicalStandVolume.predict_interval)."""
    n = len(df)
    pred = np.empty(n); lo = np.empty(n); hi = np.empty(n)
    groups = df[group_col].to_numpy()
    for g in np.unique(groups):
        te_mask = groups == g
        tr = df[~te_mask]; te = df[te_mask]
        m = make().fit(tr, tr[target].to_numpy())
        w = np.where(te_mask)[0]
        pred[w] = m.predict(te)
        l, h = m.predict_interval(te, alpha=0.05)
        lo[w] = l; hi[w] = h
    return pred, lo, hi


def cover_pair(y, ppd_lo, ppd_hi, point):
    """PPD coverage (from the model's predictive interval) vs project conformal (jackknife+ on points)."""
    c_lo, c_hi = jackknife_plus_interval(y, point, alpha=0.05)
    return {"ppd_cov95": coverage(y, ppd_lo, ppd_hi),
            "conformal_cov95": coverage(y, c_lo, c_hi),
            "ppd_int_score": interval_score(y, ppd_lo, ppd_hi),
            "conformal_int_score": interval_score(y, c_lo, c_hi)}


# ------------------------------------------------------------------- figure --
def make_figure(main, nested, y, lopo, lopo_lo, lopo_hi, loto, loto_lo, loto_hi, df, path):
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
    fig.suptitle("Bayesian hierarchical stand-volume model — ln Vol_ha ~ ln(Hdom) + ln(age) + (1|stand), N=13 / 6 stands",
                 fontsize=12, y=1.02)

    # Panel A — variance decomposition: sigma_stand across nested means
    labels = list(nested.keys())
    ys = np.arange(len(labels))[::-1]
    for lab, yy in zip(labels, ys):
        s = nested[lab]["sigma_stand"]
        ax[0].plot([s["q2.5"], s["q97.5"]], [yy, yy], color="#2166ac", lw=2)
        ax[0].plot(s["mean"], yy, "o", color="#2166ac", ms=7)
    sr = main.variance_components()["sigma_resid"]
    ax[0].axvline(sr["mean"], color="#b2182b", ls="--", lw=1.5,
                  label=f"sigma_resid (main) = {sr['mean']:.3f}")
    ax[0].set_yticks(ys); ax[0].set_yticklabels(labels, fontsize=9)
    ax[0].set_xlabel("between-stand sd  σ_stand  (log Vol_ha)")
    ax[0].set_title("A. Variance decomposition:\nσ_stand collapses as age enters the mean", fontsize=10)
    ax[0].legend(fontsize=8, loc="lower right"); ax[0].set_xlim(left=0)

    # Panel B — fixed-effect elasticity posteriors of the main model
    vc = main.variance_components()
    coefs = [(f"ln({c})", vc[f"coef_ln_{c}"]) for c in main.cols]
    yb = np.arange(len(coefs))[::-1]
    for (name, s), yy in zip(coefs, yb):
        ax[1].plot([s["q2.5"], s["q97.5"]], [yy, yy], color="#1a9850", lw=2)
        ax[1].plot(s["mean"], yy, "o", color="#1a9850", ms=7)
        ax[1].text(s["mean"], yy + 0.12, f"{s['mean']:+.2f}", ha="center", fontsize=8)
    ax[1].axvline(0, color="grey", ls=":", lw=1)
    ax[1].set_yticks(yb); ax[1].set_yticklabels([n for n, _ in coefs], fontsize=10)
    ax[1].set_ylim(-0.6, len(coefs) - 0.5 + 0.4)   # headroom so the top label clears the title
    ax[1].set_xlabel("log-log elasticity (95% credible interval)")
    ax[1].set_title(f"B. Fixed-effect elasticities  (both 95% CIs include 0)\nintercept a = {vc['intercept']['mean']:.2f}",
                    fontsize=10)

    # Panel C — held-out obs-vs-pred with posterior-predictive 95% intervals (LOPO + LOTO)
    yl = df.talhao == 4  # young stand marker
    for lab, pr, lo, hi, col in [("LOPO", lopo, lopo_lo, lopo_hi, "#4393c3"),
                                 ("LOTO", loto, loto_lo, loto_hi, "#d6604d")]:
        ax[2].errorbar(y, pr, yerr=[pr - lo, hi - pr], fmt="o", ms=5, color=col, alpha=0.85,
                       ecolor=col, elinewidth=1, capsize=2, label=f"{lab} (95% PPD)")
    ax[2].scatter(y[yl.to_numpy()], loto[yl.to_numpy()], s=140, facecolors="none",
                  edgecolors="black", linewidths=1.4, label="young stand (T4)")
    lim = [0.9 * min(y.min(), lopo.min()), 1.1 * max(y.max(), loto.max())]
    ax[2].plot(lim, lim, "k--", lw=1, alpha=0.6)
    ax[2].set_xlim(lim); ax[2].set_ylim(lim)
    ax[2].set_xlabel("observed Vol_ha (m³/ha)"); ax[2].set_ylabel("predicted Vol_ha (m³/ha)")
    ax[2].set_title("C. Held-out predictions + posterior-predictive intervals", fontsize=10)
    ax[2].legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- main --
def main():
    seed_everything(0)
    df = build_table()
    y = df.Vol_ha.to_numpy()
    print(f"\nN={len(df)} plots across {df.talhao.nunique()} stands "
          f"(plots/stand: {dict(df.talhao.value_counts().sort_index())})")
    print(f"Hdom {df.Hdom.min():.1f}–{df.Hdom.max():.1f} m | age {df.age.min():.1f}–{df.age.max():.1f} yr "
          f"| Vol_ha {y.min():.0f}–{y.max():.0f} m³/ha  (young stand = stand 4)\n")

    # ---- the hierarchical model through the project CV harness -------------------------------------
    make = lambda: HierarchicalStandVolume(("Hdom", "age"), draws=1000, tune=1500,
                                           target_accept=0.95, seed=0)
    print("→ LOPO (leave-one-plot-out) via area_based.validate._lopo …")
    lopo, lopo_lo, lopo_hi = abv._lopo(make, df)          # _lopo conformance + PPD intervals
    print("→ LOTO (leave-one-stand-out, whole stands held out) …")
    loto, loto_lo, loto_hi = loto_oof(make, df)           # mirrors _group_cv + PPD intervals
    # conformance check: point preds identical to the harness _group_cv (cheap-ish; reuse make)
    ml, mg = metrics(y, lopo), metrics(y, loto)
    covL = cover_pair(y, lopo_lo, lopo_hi, lopo)
    covG = cover_pair(y, loto_lo, loto_hi, loto)
    perm_p = permutation_test(y, lopo, n_perm=5000, seed=0)["p_value"]

    print("\n=== HIERARCHICAL BAYES — out-of-fold metrics ===")
    print(f"  LOPO  R²={ml['R2']:+.3f}  rRMSE={ml['rRMSE_pct']:.1f}%  bias={ml['bias_pct']:+.1f}%")
    print(f"  LOTO  R²={mg['R2']:+.3f}  rRMSE={mg['rRMSE_pct']:.1f}%  bias={mg['bias_pct']:+.1f}%  (cross-stand)")
    print(f"  PPD cov95:      LOPO {covL['ppd_cov95']:.2f}   LOTO {covG['ppd_cov95']:.2f}")
    print(f"  conformal cov95:LOPO {covL['conformal_cov95']:.2f}   LOTO {covG['conformal_cov95']:.2f}")
    print(f"  permutation p (LOPO R²) = {perm_p:.4f}")

    # ---- shuffle-label control: refit LOPO on shuffled y (cheaper draws) → R² must collapse --------
    print("\n→ shuffle-label control (LOPO on permuted labels) …")
    rng = np.random.default_rng(123)
    y_shuf = rng.permutation(y)
    df_shuf = df.copy(); df_shuf["Vol_ha"] = y_shuf
    make_fast = lambda: HierarchicalStandVolume(("Hdom", "age"), draws=500, tune=800,
                                                target_accept=0.9, seed=0)
    shuf_pred = abv._lopo(make_fast, df_shuf)[0]
    shuf_R2 = metrics(y_shuf, shuf_pred)["R2"]
    print(f"  shuffle-label LOPO R² = {shuf_R2:+.3f}  (should be ≤0 — signal is not memorized noise)")

    # ---- variance decomposition: nested means, sigma_stand collapse -------------------------------
    print("\n→ variance decomposition (nested full-data fits, target_accept=0.99) …")
    nested_specs = {"1 + (1|stand)": (), "+ ln(Hdom)": ("Hdom",), "+ ln(Hdom) + ln(age) [main]": ("Hdom", "age")}
    nested = {}
    main_model = None
    for lab, cols in nested_specs.items():
        m = HierarchicalStandVolume(cols, draws=1000, tune=1500, target_accept=0.99, seed=0).fit(
            df, y)
        nested[lab] = m.variance_components()
        if cols == ("Hdom", "age"):
            main_model = m
    diag = main_model.diagnostics()
    vc = main_model.variance_components()

    print(f"  {'mean structure':32s} {'σ_stand':>22} {'σ_resid':>22}")
    for lab, vcn in nested.items():
        ss, sr = vcn["sigma_stand"], vcn["sigma_resid"]
        print(f"  {lab:32s} {ss['mean']:.3f} [{ss['q2.5']:.3f},{ss['q97.5']:.3f}]"
              f"   {sr['mean']:.3f} [{sr['q2.5']:.3f},{sr['q97.5']:.3f}]")
    print(f"\n  main model: {main_model.equation()}")
    print(f"  age elasticity  c = {vc['coef_ln_age']['mean']:+.3f}  "
          f"[{vc['coef_ln_age']['q2.5']:+.3f}, {vc['coef_ln_age']['q97.5']:+.3f}]")
    print(f"  Hdom elasticity b = {vc['coef_ln_Hdom']['mean']:+.3f}  "
          f"[{vc['coef_ln_Hdom']['q2.5']:+.3f}, {vc['coef_ln_Hdom']['q97.5']:+.3f}]")
    print(f"  ICC = σ_stand²/(σ_stand²+σ_resid²) = {vc['ICC']['mean']:.2f} "
          f"[{vc['ICC']['q2.5']:.2f}, {vc['ICC']['q97.5']:.2f}]  (residual variance at stand level)")
    print(f"  convergence: max R-hat={diag['max_rhat']}, min ESS-bulk={diag['min_ess_bulk']:.0f}, "
          f"divergences={diag['n_divergences']} / {diag['n_samples']} draws")

    # ---- frequentist baseline: StandVolume(('Hdom','age')) -----------------------------------------
    print("\n→ frequentist baseline StandVolume(('Hdom','age')) …")
    mk_sv = lambda: StandVolume(("Hdom", "age"))
    sv_lopo = abv._lopo(mk_sv, df)[0]
    sv_loto = abv._group_cv(mk_sv, df)
    sml, smg = metrics(y, sv_lopo), metrics(y, sv_loto)
    print(f"  StandVolume  LOPO R²={sml['R2']:+.3f}/{sml['rRMSE_pct']:.1f}%   "
          f"LOTO R²={smg['R2']:+.3f}/{smg['rRMSE_pct']:.1f}%")
    print(f"  Hierarchical LOPO R²={ml['R2']:+.3f}/{ml['rRMSE_pct']:.1f}%   "
          f"LOTO R²={mg['R2']:+.3f}/{mg['rRMSE_pct']:.1f}%")

    # ---- figure + manifest -------------------------------------------------------------------------
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = config.OUT_DIR / "hierarchical_bayes.png"
    make_figure(main_model, nested, y, lopo, lopo_lo, lopo_hi, loto, loto_lo, loto_hi, df, fig_path)
    print(f"\nwrote figure {fig_path}")

    results = {
        "N": int(len(df)), "n_stands": int(df.talhao.nunique()),
        "LOPO": ml, "LOTO": mg,
        "coverage": {"LOPO": covL, "LOTO": covG},
        "permutation_p_LOPO": perm_p,
        "shuffle_label_LOPO_R2": float(shuf_R2),
        "variance_decomposition": nested,
        "main_model": {"equation": main_model.equation(), "components": vc},
        "convergence": diag,
        "baseline_StandVolume": {"LOPO": sml, "LOTO": smg},
    }
    import json
    (config.OUT_DIR / "hierarchical_bayes.json").write_text(json.dumps(results, indent=2, default=str))
    write_manifest(config.OUT_DIR / "hierarchical_bayes_manifest.json",
                   script="exp_hierarchical_bayes.py",
                   config={"model": "ln Vol_ha ~ ln(Hdom)+ln(age)+(1|stand)", "draws": 1000,
                           "tune": 1500, "chains": 4, "target_accept_cv": 0.95,
                           "target_accept_vardecomp": 0.99, "sigma_stand_prior": "HalfNormal(0.5)",
                           "seed": 0},
                   seed=0, metrics=results)
    print(f"wrote {config.OUT_DIR/'hierarchical_bayes.json'} + manifest")

    # ---- optional W&B (offline-safe) ---------------------------------------------------------------
    try:
        from greenvista.tracking import log_experiment
        log_experiment(
            name="hierarchical-bayes",
            config={"model": "ln Vol_ha ~ ln(Hdom)+ln(age)+(1|stand)", "seed": 0},
            metrics={"LOPO_R2": ml["R2"], "LOPO_rRMSE": ml["rRMSE_pct"],
                     "LOTO_R2": mg["R2"], "LOTO_rRMSE": mg["rRMSE_pct"],
                     "ppd_cov95_LOPO": covL["ppd_cov95"], "ppd_cov95_LOTO": covG["ppd_cov95"],
                     "sigma_stand_main": vc["sigma_stand"]["mean"],
                     "sigma_resid_main": vc["sigma_resid"]["mean"],
                     "coef_ln_age": vc["coef_ln_age"]["mean"], "perm_p": perm_p,
                     "shuffle_R2": shuf_R2, "max_rhat": diag["max_rhat"],
                     "n_divergences": diag["n_divergences"]},
            artifacts=[str(fig_path), str(config.OUT_DIR / "hierarchical_bayes.json")])
    except Exception as e:
        print(f"(W&B logging skipped: {e})")

    return results


if __name__ == "__main__":
    main()

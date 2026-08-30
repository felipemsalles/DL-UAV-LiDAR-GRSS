#!/usr/bin/env python3
"""Spike S10 — upgraded area-based ladder: full model set, with vs without stand age.

Tests the literature's top recommendation (Silva 2016: age/site stratification is the lever from
~10% → ~5% rRMSE) on our 13 plots, plus the completed model ladder (OLS/GPR/RF/kNN/PCA-OLS). Reports
R², rRMSE, bias and conformal-style coverage under identical leave-one-plot-out CV.

Run (repo root, greenvista env): PYTHONPATH=. python scripts/spike_s10_aba_upgrade.py
"""
import pandas as pd
from greenvista import config
from greenvista.area_based import data as abd, models as abm, validate as abv
from greenvista.repro import write_manifest, seed_everything

F, FA = abd.FEATURES, abd.FEATURES_AGE


def main():
    seed_everything(config.ABA.seed)
    df = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                        config.DATA / "5-dados_campo/inventario_est.csv")
    print(f"plots N={len(df)} | Vol_ha {df.Vol_ha.min():.0f}–{df.Vol_ha.max():.0f} | "
          f"age {df[abd.AGE].min():.1f}–{df[abd.AGE].max():.1f} yr")

    models = {
        # base ladder (6 curated metrics, no age)
        "gpr":            lambda: abm.make_gpr(F),
        "rf":             lambda: abm.make_rf(F),
        "elasticnet":     lambda: abm.make_elasticnet(F),
        "knn":            lambda: abm.make_knn(F),
        "log_allometric": lambda: abm.make_log_allometric(("zmean", "zq90")),
        "pca_ols(35 metrics)": abm.make_pca_ols,
        "baseline_lm":    abm.make_baseline_lm,
        # age-aware variants (6 metrics + idade_anos)
        "gpr+age":        lambda: abm.make_gpr(FA),
        "rf+age":         lambda: abm.make_rf(FA),
        "elasticnet+age": lambda: abm.make_elasticnet(FA),
        "log_allom+age":  lambda: abm.make_log_allometric(("zmean", "zq90", abd.AGE)),
    }
    rep = abv.run(df, models=models, group_col=config.GROUP_COL, perm=2000)

    rows = sorted(rep.items(), key=lambda kv: kv[1]["rRMSE_pct"])
    print("\n=== LOPO Vol_ha — upgraded ladder (sorted by rRMSE); cov95=conformal, LOTO=leave-stand-out ===")
    print(f"  {'model':22s} {'R2':>6} {'rRMSE%':>7} {'bias%':>7} {'cov95':>6} {'LOTO-R2':>8} {'perm-p':>7}")
    for name, mm in rows:
        cov = mm.get("coverage95", float("nan"))
        print(f"  {name:22s} {mm['R2']:+6.2f} {mm['rRMSE_pct']:7.1f} {mm['bias_pct']:+7.1f} "
              f"{cov:6.2f} {mm.get('R2_group', float('nan')):+8.2f} {mm.get('perm_p', float('nan')):7.3f}")

    # age effect on the main models
    print("\n=== age effect (Δ rRMSE, with age − without) ===")
    for base, aged in [("gpr", "gpr+age"), ("rf", "rf+age"), ("elasticnet", "elasticnet+age"),
                       ("log_allometric", "log_allom+age")]:
        d = rep[aged]["rRMSE_pct"] - rep[base]["rRMSE_pct"]
        print(f"  {base:14s} {rep[base]['rRMSE_pct']:5.1f}%  →  {aged:16s} {rep[aged]['rRMSE_pct']:5.1f}%   "
              f"Δ={d:+.1f} pts  {'(age helps)' if d < -0.3 else '(no gain)' if abs(d) <= 0.3 else '(age hurts)'}")

    best = rows[0]
    print(f"\nBEST: {best[0]} — rRMSE {best[1]['rRMSE_pct']:.1f}%, R²={best[1]['R2']:+.2f}, "
          f"bias {best[1]['bias_pct']:+.1f}%")
    print(f"  ≈ AGB {abd.volume_to_agb(df.Vol_ha.mean()):.0f} Mg/ha mean stem (density {abd.EUC_BASIC_DENSITY})")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(model=k, **{m: v for m, v in val.items() if m != "R2_ci"}) for k, val in rep.items()]
                 ).to_csv(config.OUT_DIR / "s10_aba_upgrade.csv", index=False)
    write_manifest(config.OUT_DIR / "s10_manifest.json", script="spike_s10_aba_upgrade.py",
                   config=config.ABA, seed=config.ABA.seed,
                   metrics={k: {mk: mv for mk, mv in v.items() if mk != "R2_ci"} for k, v in rep.items()})
    print(f"\nwrote {config.OUT_DIR / 's10_aba_upgrade.csv'} + s10_manifest.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Two orthogonal questions about what the LiDAR adds and what N=13 can support.

Q1. Does the LiDAR explain anything the planting date does not?
    "LiDAR alone" cannot separate canopy structure from stand development, and
    "LiDAR + age jointly" is wrecked by collinearity
    (canopy metrics correlate 0.75..0.91 with age here), which degrades leave-one-stand-out
    monotonically as features are added. The orthogonalised test fits the age model first and asks
    the LiDAR to predict its held-out error, so a null here is a genuine null.

Q2. Is there a metric that 13 plots can support?
    Per-plot R² is not informative at this N. Model-assisted (GREG) estimation reframes the
    deliverable as the stand total with a design-based interval, and scores the flight by how much
    variance it removes, i.e. how many extra field plots the LiDAR is worth. A weak model still
    buys a real gain.

Run: PYTHONPATH=. python scripts/exp_lidar_beyond_age.py
"""
import warnings

import pandas as pd

from greenvista import config, estimation
from greenvista.area_based import data as abd, models as abm, validate as abv
from greenvista.eval import metrics
from greenvista.eval.orthogonal import lidar_beyond_base
from greenvista.repro import seed_everything, write_manifest

warnings.simplefilter("ignore")

AGE = "idade_anos"


def plot_table():
    df = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                        config.DATA / "5-dados_campo/inventario_est.csv")
    struct = config.OUT_DIR / "plot_structural.csv"
    if struct.exists():
        keep = [c for c in ("plot_id", "rumple", "cover", "penetration")
                if c in pd.read_csv(struct).columns]
        df = df.merge(pd.read_csv(struct)[keep], on="plot_id", how="left")
    return df


def q1_orthogonal(df):
    """Does LiDAR predict the age model's held-out error?"""
    print("=" * 78)
    print("Q1  does the LiDAR explain what age does NOT explain?")
    print("=" * 78)
    # the residual target is signed, so the log-allometric family is unusable here; stage 2 uses
    # PLS / ridge, both valid on signed targets and low-variance at N=13
    base = lambda: abm.make_huber_allom((AGE,))
    candidates = {
        "PLS (suite completa)": lambda: abm.make_pls(),
        "ridge (zmean, zq90)": lambda: abm.make_ridge(("zmean", "zq90")),
        "ridge (zmax)": lambda: abm.make_ridge(("zmax",)),
        "ridge (rumple, cover)": lambda: abm.make_ridge(("rumple", "cover")),
    }
    rows = []
    for name, make_resid in candidates.items():
        r = lidar_beyond_base(base, make_resid, df, group_col=config.GROUP_COL)
        rows.append({
            "residual_model": name,
            "base_R2": r["base"]["R2"], "base_rRMSE": r["base"]["rRMSE_pct"],
            "combined_R2": r["combined"]["R2"], "combined_rRMSE": r["combined"]["rRMSE_pct"],
            "resid_r": r["resid_r"], "resid_R2": r["resid_R2"],
            "null_r_mean": r["null_r_mean"], "null_r_sd": r["null_r_sd"],
            "p_vs_null": r["p_vs_null"], "z_vs_null": r["z_vs_null"],
        })
        print(f"\n  LiDAR on the age-model residual  [{name}]")
        print(f"    age only          R2={r['base']['R2']:+.2f}  error={r['base']['rRMSE_pct']:.1f}%"
              f"  CI {r['base_rRMSE_ci'][0]:.1f} to {r['base_rRMSE_ci'][1]:.1f}")
        print(f"    age + residual    R2={r['combined']['R2']:+.2f}  error={r['combined']['rRMSE_pct']:.1f}%"
              f"  CI {r['combined_rRMSE_ci'][0]:.1f} to {r['combined_rRMSE_ci'][1]:.1f}")
        print(f"    does the predicted residual track the real one?  r={r['resid_r']:+.2f}  R2={r['resid_R2']:+.2f}")
        print(f"    null by shuffled refit  {r['null_r_mean']:+.2f} +/- {r['null_r_sd']:.2f}"
              f"   z={r['z_vs_null']:+.2f}  p={r['p_vs_null']:.3f}")
    return pd.DataFrame(rows)


def q2_model_assisted(df):
    """Reframe the deliverable: how many field plots is the flight worth?"""
    print()
    print("=" * 78)
    print("Q2  how many field plots does the flight save (model-assisted estimation)")
    print("=" * 78)
    y = df.Vol_ha.to_numpy(float)
    field = estimation.field_only_mean(y)
    print(f"\n  field only, 13 plots   mean {field['mean_ha']:.0f} m3/ha  "
          f"standard error {field['se']:.1f}  95% CI {field['lo']:.0f} to {field['hi']:.0f}")

    models = {
        "PLS so LiDAR": lambda: abm.make_pls(),
        "altura + idade": lambda: abm.make_huber_allom(("zmean", AGE)),
        "so idade": lambda: abm.make_huber_allom((AGE,)),
    }
    rows = []
    for name, make in models.items():
        lopo, loto = abv.oof_predictions(make, df, group_col=config.GROUP_COL)
        for scheme, pred in (("parcela nova", lopo), ("talhao novo", loto)):
            eff = estimation.relative_efficiency(y, pred)
            ci = estimation.bootstrap_relative_efficiency(y, pred)
            m = metrics(y, pred)
            rows.append({"model": name, "scheme": scheme, "rRMSE_pct": m["rRMSE_pct"],
                         "RE": eff["RE"], "RE_lo": ci[0], "RE_hi": ci[1],
                         "equivalent_plots": eff["equivalent_plots"],
                         "se_field_only": eff["se_field_only"],
                         "se_model_assisted": eff["se_model_assisted"],
                         "se_reduction_pct": eff["se_reduction_pct"]})
            print(f"\n  {name:16s} [{scheme}]  error {m['rRMSE_pct']:.1f}%")
            print(f"    relative efficiency {eff['RE']:.2f}  (CI {ci[0]:.2f} to {ci[1]:.2f})"
                  f"  worth {eff['equivalent_plots']:+.1f} extra plots")
            print(f"    standard error of the mean  {eff['se_field_only']:.1f} -> {eff['se_model_assisted']:.1f}"
                  f"  ({eff['se_reduction_pct']:+.0f}%)")
    return pd.DataFrame(rows)


def main():
    seed_everything(0)
    df = plot_table()
    print(f"{len(df)} plots, {df[config.GROUP_COL].nunique()} stands, "
          f"Vol_ha {df.Vol_ha.min():.0f} to {df.Vol_ha.max():.0f} (mean {df.Vol_ha.mean():.0f})\n")

    q1 = q1_orthogonal(df)
    q2 = q2_model_assisted(df)

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    q1.to_csv(config.OUT_DIR / "lidar_beyond_age.csv", index=False)
    q2.to_csv(config.OUT_DIR / "model_assisted_efficiency.csv", index=False)
    write_manifest(config.OUT_DIR / "lidar_beyond_age_manifest.json",
                   script="exp_lidar_beyond_age",
                   config={"base": "huber_allom(idade_anos)",
                           "group_col": config.GROUP_COL, "n": len(df)},
                   metrics={"q1": q1.to_dict("records"), "q2": q2.to_dict("records")})
    print(f"\nwrote to {config.OUT_DIR}")


if __name__ == "__main__":
    main()

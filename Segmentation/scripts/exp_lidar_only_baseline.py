#!/usr/bin/env python3
"""Iterate LiDAR-only volume models (no age, no field DBH, no stocking) to find the best baseline.

Applies the operational ABA toolbox, LiDAR-only:
 (1) ABA regression — several learners on the canopy-metric suite;
 (2) dominant height + (3) density/cover/vertical-structure features;
 (4) LiDAR-derived site stratification (a canopy site index replacing age);
 (5) model-assisted stand totals with design CIs;
 (6) validation by LOPO + leave-one-stand-out + conformal + permutation.

Reports the best baseline by leave-one-stand-out rRMSE among models that are cross-stand-positive.

Run: PYTHONPATH=. python scripts/exp_lidar_only_baseline.py
"""
import numpy as np, pandas as pd
from greenvista import config
from greenvista.repro import seed_everything, write_manifest
from greenvista.area_based import data as abd, models as abm, validate as abv
from greenvista.eval import (metrics, jackknife_plus_interval, coverage, permutation_test)

# LiDAR-only feature sets: no idade_anos, no D_cm/H_est, no n_arv_ha
LIDAR_METRICS = [c for c in abd.ALL_METRICS]                     # full lidR .stdmetrics_z
STRUCT = ["rumple", "cover", "penetration"]
DOMH = ["zmax", "zq95", "zq90"]
ALLOM_LIDAR = ("zmean", "zq90", "zmax", "zq50", "zq70", "zsd", "pzabovezmean")


def lidar_site_index(df):
    """A LiDAR-derived stand site index replacing age: stand-mean dominant height (zq95). Constant
    within a stand and varying across stands, it captures the development/site axis that age would,
    but purely from the cloud."""
    si = df.groupby("talhao")["zq95"].transform("mean") if "zq95" in df else df.groupby("talhao")["zmax"].transform("mean")
    return si


def ev(df, y, name, make, group_col="talhao"):
    lopo = abv._lopo(make, df)[0]
    loto = abv._group_cv(make, df, group_col=group_col)
    a, b = metrics(y, lopo), metrics(y, loto)
    lo, hi = jackknife_plus_interval(y, lopo)
    return {"model": name, "LOPO_R2": a["R2"], "LOPO_rRMSE": a["rRMSE_pct"],
            "LOTO_R2": b["R2"], "LOTO_rRMSE": b["rRMSE_pct"], "cov95": coverage(y, lo, hi),
            "perm_p": permutation_test(y, lopo, n_perm=3000)["p_value"], "_lopo": lopo, "_loto": loto}


def main():
    seed_everything(0)
    df = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                        config.DATA / "5-dados_campo/inventario_est.csv")
    s = config.OUT_DIR / "plot_structural.csv"
    if s.exists():
        df = df.merge(pd.read_csv(s)[["plot_id", *STRUCT]], on="plot_id", how="left")
    df["site_lidar"] = lidar_site_index(df)          # method 4: LiDAR site index (no age)
    y = df.Vol_ha.to_numpy()
    ALL = [c for c in LIDAR_METRICS if c in df] + [c for c in STRUCT if c in df]

    zoo = {
        "dominant-H allom":            lambda: abm.make_huber_allom(("zmax", "zq90")),
        "PLS (full suite)":            lambda: abm.make_pls(LIDAR_METRICS, max_k=3),
        "PLS (suite+struct)":          lambda: abm.make_pls([c for c in ALL], max_k=3),
        "RF (full suite)":             lambda: abm.make_rf([c for c in LIDAR_METRICS if c in df]),
        "ElasticNet (full suite)":     lambda: abm.make_elasticnet([c for c in LIDAR_METRICS if c in df]),
        "GPR (curated)":               lambda: abm.make_gpr(("zmax", "zmean", "zsd", "zq50", "zq90", "pzabovezmean")),
        "bss allom (LiDAR, LOTO-sel)": lambda: abm.make_bestsubset_allom(ALLOM_LIDAR, select_by="group"),
        "bss allom+struct (LOTO-sel)": lambda: abm.make_bestsubset_allom((*ALLOM_LIDAR, "rumple", "cover"), select_by="group"),
        # method 4: LiDAR site-index as a covariate (replaces age)
        "dom-H + LiDAR-site":          lambda: abm.make_huber_allom(("zmax", "zq90", "site_lidar")),
        "PLS + LiDAR-site":            lambda: abm.make_pls([*LIDAR_METRICS, "site_lidar"], max_k=4),
    }
    rows = [ev(df, y, n, mk) for n, mk in zoo.items()]

    # method 1+3 ensemble: average OOF preds of the cross-stand generalizers
    oof = {r["model"]: r for r in rows}
    def ens(names, label):
        lo = np.mean([oof[n]["_lopo"] for n in names], axis=0)
        gt = np.mean([oof[n]["_loto"] for n in names], axis=0)
        a, b = metrics(y, lo), metrics(y, gt)
        l, h = jackknife_plus_interval(y, lo)
        return {"model": label, "LOPO_R2": a["R2"], "LOPO_rRMSE": a["rRMSE_pct"], "LOTO_R2": b["R2"],
                "LOTO_rRMSE": b["rRMSE_pct"], "cov95": coverage(y, l, h),
                "perm_p": permutation_test(y, lo, n_perm=3000)["p_value"], "_lopo": lo, "_loto": gt}
    rows.append(ens(["PLS (full suite)", "dominant-H allom"], "ENS PLS+domH"))
    rows.append(ens(["PLS (full suite)", "dom-H + LiDAR-site"], "ENS PLS+domH-site"))
    rows.append(ens(["PLS (full suite)", "RF (full suite)", "dominant-H allom"], "ENS PLS+RF+domH"))

    res = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    res = res.sort_values("LOTO_rRMSE").reset_index(drop=True)
    print("=== LiDAR-ONLY volume baseline search (no age, no field DBH) — sorted by cross-stand rRMSE ===")
    print(f"  {'model':28s} {'LOPO_R2':>7} {'LOPO%':>6} {'LOTO_R2':>7} {'LOTO%':>6} {'cov95':>6} {'perm_p':>7}")
    for _, r in res.iterrows():
        print(f"  {r['model']:28s} {r.LOPO_R2:+7.2f} {r.LOPO_rRMSE:6.1f} {r.LOTO_R2:+7.2f} {r.LOTO_rRMSE:6.1f} "
              f"{r.cov95:6.2f} {r.perm_p:7.3f}")

    ok = res[(res.LOTO_R2 > 0) & (res.perm_p < 0.05)]
    best = ok.iloc[0] if len(ok) else res.iloc[0]
    print(f"\nBEST LiDAR-ONLY BASELINE (cross-stand): {best['model']} — cross-stand R²={best.LOTO_R2:+.2f} "
          f"(rRMSE {best.LOTO_rRMSE:.1f}%), within-range LOPO R²={best.LOPO_R2:+.2f} ({best.LOPO_rRMSE:.1f}%), "
          f"cov95={best.cov95:.2f}, p={best.perm_p:.3f}")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.OUT_DIR / "lidar_only_baseline.csv", index=False)
    write_manifest(config.OUT_DIR / "lidar_only_baseline_manifest.json", script="exp_lidar_only_baseline.py",
                   config={"seed": 0, "note": "no age / no field DBH / no stocking"},
                   metrics=res.to_dict(orient="records"))
    print(f"wrote {config.OUT_DIR/'lidar_only_baseline.csv'}")


if __name__ == "__main__":
    main()

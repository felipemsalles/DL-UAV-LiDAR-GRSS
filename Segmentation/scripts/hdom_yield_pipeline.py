#!/usr/bin/env python3
"""LiDAR dominant height + stand age → Vol_ha (yield-model style).

Uses only inputs available wall-to-wall or on record: LiDAR Hdom (grid dominant height) + stand age
(planting records). Validates under LOPO and leave-one-stand-out, expresses the result in the
site-index terms used by the G2 inventory, and produces a wall-to-wall map for one stand.

Run: PYTHONPATH=. GREENVISTA_LAZ_DIR=<lazdir> python scripts/hdom_yield_pipeline.py
"""
import warnings
import numpy as np, pandas as pd, laspy, geopandas as gpd
warnings.simplefilter("ignore")

from greenvista import config
from greenvista.repro import seed_everything, write_manifest
from greenvista.stand_volume import StandVolume, lidar_hdom, site_index
from greenvista.area_based import data as abd, validate as abv
from greenvista.eval import metrics, jackknife_plus_interval, coverage, permutation_test


def build_table():
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
    return df, cloud


def main():
    seed_everything(0)
    df, cloud = build_table()
    y = df.Vol_ha.to_numpy()
    print(f"N={len(df)} plots | LiDAR Hdom {df.Hdom.min():.1f}–{df.Hdom.max():.1f} m | age {df.age.min():.1f}–{df.age.max():.1f} yr")

    make = lambda: StandVolume(("Hdom", "age"))
    lopo = abv._lopo(make, df)[0]
    loto = abv._group_cv(make, df)
    lo, hi = jackknife_plus_interval(y, lopo)
    ml, mg = metrics(y, lopo), metrics(y, loto)
    p = permutation_test(y, lopo, n_perm=3000)["p_value"]
    print("\n=== Vol_ha ~ LiDAR Hdom + age (the deployable model) ===")
    print(f"  LOPO  R²={ml['R2']:+.2f} rRMSE={ml['rRMSE_pct']:.1f}% bias={ml['bias_pct']:+.1f}%")
    print(f"  LOTO  R²={mg['R2']:+.2f} rRMSE={mg['rRMSE_pct']:.1f}%   (cross-stand)")
    print(f"  conformal cov95={coverage(y, lo, hi):.2f}  permutation p={p:.3f}")
    m = make().fit(df, y)
    print(f"  fitted: {m.equation()}")

    # site index: projects Hdom to a reference age. It is an age-independent productivity measure,
    # used to stratify stands and to feed the yield curves; it does not track current standing
    # volume (age does that). Fitting the guide-curve exponent b requires data from multiple ages;
    # b=1 here is illustrative only.
    df["SI7"] = site_index(df.Hdom, df.age, ref_age=7.0)
    print(f"\n  site index (Hdom @ 7 yr, b=1 illustrative): {df.SI7.min():.1f}–{df.SI7.max():.1f} m "
          f"— a productivity stratifier for the yield curves (not a direct volume predictor)")

    # wall-to-wall map of stand 1 (age taken from the planting records)
    from greenvista.stand_volume import volume_map
    t = 1
    age_t = float(df[df.talhao == t].age.iloc[0])
    gdf = volume_map(m, cloud.get(t, next(iter(cloud.values()))), ages=age_t, cell=20.0)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUT_DIR / "hdom_yield_map_t001.gpkg"
    gdf.to_file(str(out), driver="GPKG")
    print(f"\nwall-to-wall map (stand {t}, age {age_t:.1f}): {len(gdf)} cells, "
          f"mean {gdf.vol_ha.mean():.0f} m³/ha, stand total≈{(gdf.vol_ha.mean()*25):.0f} m³ over 25 ha  → {out}")

    write_manifest(config.OUT_DIR / "hdom_yield_manifest.json", script="hdom_yield_pipeline.py",
                   config={"model": "Vol_ha ~ ln(Hdom) + ln(age)", "seed": 0},
                   metrics={"LOPO": ml, "LOTO": mg, "cov95": coverage(y, lo, hi), "perm_p": p,
                            "equation": m.equation()})
    print(f"wrote {out.parent/'hdom_yield_manifest.json'}")


if __name__ == "__main__":
    main()

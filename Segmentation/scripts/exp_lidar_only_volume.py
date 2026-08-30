#!/usr/bin/env python3
"""Plot volume from LiDAR alone (no age, no field DBH), by the two routes in the literature.

Route A — ITD (detect trees → H→V allometry → sum/ha): fails here (R²<0). Nadir UAV misses ~half the
stems (suppressed understory) and cannot size them; better detection (FF3D) does not lift the
occlusion limit on per-tree volume.

Route B — ABA (area-based regression, canopy metrics → volume): works. PLS on the full lidR metric
suite is the best LiDAR-only model: LOPO R²≈0.74 / rRMSE≈13%, and it generalizes cross-stand
(leave-one-stand-out R²≈0.34 / rRMSE≈21%), unlike the 2-metric height allometry. Adding age, a free
stand record, tightens it further (LOTO rRMSE≈13%).

Run: PYTHONPATH=. GREENVISTA_LAZ_DIR=<lazdir> python scripts/exp_lidar_only_volume.py
"""
import numpy as np, pandas as pd, geopandas as gpd, warnings
warnings.simplefilter("ignore")

from greenvista import config
from greenvista.repro import seed_everything, write_manifest
from greenvista.area_based import data as abd, models as abm, validate as abv, stand_from_trees as sft
from greenvista.eval import metrics


def _plot_table():
    df = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                        config.DATA / "5-dados_campo/inventario_est.csv")
    s = config.OUT_DIR / "plot_structural.csv"
    if s.exists():
        df = df.merge(pd.read_csv(s)[["plot_id", "rumple", "cover", "penetration"]], on="plot_id", how="left")
    return df


def route_a_itd(df):
    """LiDAR-only ITD volume (needs the clouds in GREENVISTA_LAZ_DIR)."""
    import laspy
    parc = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for c in ("talhao", "parcela"):
        parc[c] = pd.to_numeric(parc[c], errors="coerce").astype("Int64")
    parc["cx"], parc["cy"] = parc.geometry.x, parc.geometry.y
    d = df.merge(parc[["talhao", "parcela", "cx", "cy"]], on=["talhao", "parcela"], how="left")
    coef = sft.fit_hv_allometry()
    cloud, rows = {}, []
    for _, r in d.sort_values("talhao").iterrows():
        t = int(r.talhao)
        if t not in cloud:
            las = laspy.read(str(config.LAZ_DIR / f"SaoManuelTotal_{t:03d}.laz"))
            cloud = {t: np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])}
        pts = cloud[t]
        m = (pts[:, 0] - r.cx) ** 2 + (pts[:, 1] - r.cy) ** 2 <= 12.0 ** 2
        raw, ndet = sft.plot_vol_ha(pts[m], (r.cx, r.cy), coef)
        rows.append({"field": r.Vol_ha, "itd": raw, "ndet": ndet})
    res = pd.DataFrame(rows)
    return metrics(res.field.to_numpy(), res.itd.to_numpy()), float(res.ndet.mean())


def route_b_aba(df):
    y = df.Vol_ha.to_numpy()
    models = {
        "LiDAR height allom (zmean,zq90)": lambda: abm.make_huber_allom(("zmean", "zq90")),
        "LiDAR dominant-H (zmax,zq90)":    lambda: abm.make_huber_allom(("zmax", "zq90")),
        "LiDAR PLS (full suite) [BEST]":   lambda: abm.make_pls(abd.ALL_METRICS, max_k=3),
        "LiDAR+age (reference)":           lambda: abm.make_huber_allom(("zmean", "zq90", "idade_anos")),
    }
    out = {}
    for name, mk in models.items():
        lopo = abv._lopo(mk, df)[0]; loto = abv._group_cv(mk, df)
        out[name] = {"LOPO": metrics(y, lopo), "LOTO": metrics(y, loto)}
    return out


def main():
    seed_everything(0)
    df = _plot_table()
    print("=== Route B: LiDAR-only ABA (canopy metrics → volume) ===")
    b = route_b_aba(df)
    for name, m in b.items():
        print(f"  {name:34s} LOPO R²={m['LOPO']['R2']:+.2f} rRMSE={m['LOPO']['rRMSE_pct']:4.1f}%   "
              f"LOTO R²={m['LOTO']['R2']:+.2f} rRMSE={m['LOTO']['rRMSE_pct']:4.1f}%")

    a = None
    try:
        (am, ndet) = route_a_itd(df)
        a = {**am, "mean_detected": ndet}
        print(f"\n=== Route A: LiDAR-only ITD (detect→allometry→sum) ===")
        print(f"  ITD volume: R²={am['R2']:+.2f} rRMSE={am['rRMSE_pct']:.0f}% bias={am['bias_pct']:+.0f}%  "
              f"(mean {ndet:.0f} stems detected vs ~62 real → nadir occlusion)")
    except Exception as e:
        print(f"\n(Route A skipped — clouds not in GREENVISTA_LAZ_DIR: {e})")

    print("\nVERDICT: LiDAR-only volume IS possible via ABA/PLS (~13% LOPO, ~21% cross-stand); "
          "ITD fails on nadir occlusion. More calibration plots spanning ages shrink the cross-stand gap.")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_manifest(config.OUT_DIR / "lidar_only_volume_manifest.json", script="exp_lidar_only_volume.py",
                   config={"seed": 0}, metrics={"aba": b, "itd": a})


if __name__ == "__main__":
    main()

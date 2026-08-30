#!/usr/bin/env python3
"""Per-plot canopy structural features (rumple, cover, gap, penetration) from the clouds.

For each of the 13 field plots, extracts its 12 m footprint from the stand cloud and computes the
metrics in greenvista.features.structure. Writes manual_match/plot_structural.csv (merged by
plot_id), including a recomputed zmean for comparison against the lidR CSV.

Run: PYTHONPATH=. GREENVISTA_LAZ_DIR=<dir with SaoManuelTotal_00X.laz> python scripts/build_structural_features.py
"""
import warnings
import numpy as np, pandas as pd, laspy, geopandas as gpd

from greenvista import config
from greenvista.image_cnn import footprint
from greenvista.features import structure
from greenvista.segmentation import build_chm
from greenvista.area_based import data as abd


def plot_structural(xyz, center, radius=config.PLOT_RADIUS_M, zmin=config.Z_MIN_M, zmax=config.Z_MAX_M):
    pts = footprint.extract_footprint(xyz, center, radius)
    if len(pts) == 0:
        return {k: np.nan for k in structure.STRUCTURE_KEYS} | {"zmean_rec": np.nan, "n": 0}
    z_all = pts[:, 2]
    canopy = pts[(z_all >= zmin) & (z_all <= zmax)]
    if len(canopy) < 10:
        return {k: np.nan for k in structure.STRUCTURE_KEYS} | {"zmean_rec": np.nan, "n": len(canopy)}
    chm, *_ = build_chm(canopy[:, :2], canopy[:, 2], res=0.5)   # 0.5 m CHM for rumple
    struct = structure.structural_scalars(chm, z_all, res=0.5)   # cover/gap from all returns (>2 m)
    struct["penetration"] = structure.penetration_ratio(canopy[:, 2])
    struct["zmean_rec"] = float(canopy[:, 2].mean())
    struct["n"] = int(len(canopy))
    return struct


def main():
    df = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                        config.DATA / "5-dados_campo/inventario_est.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parc = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    parc["talhao"] = pd.to_numeric(parc.talhao, errors="coerce").astype("Int64")
    parc["parcela"] = pd.to_numeric(parc.parcela, errors="coerce").astype("Int64")
    parc["cx"], parc["cy"] = parc.geometry.x, parc.geometry.y
    df = df.merge(parc[["talhao", "parcela", "cx", "cy"]], on=["talhao", "parcela"], how="left")

    rows, cloud = [], {}
    for _, r in df.sort_values("talhao").iterrows():
        t = int(r.talhao)
        if t not in cloud:
            print(f"loading stand {t:03d} …", flush=True)
            las = laspy.read(str(config.LAZ_DIR / f"SaoManuelTotal_{t:03d}.laz"))
            cloud = {t: np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])}  # 1 at a time (RAM)
        s = plot_structural(cloud[t], (r.cx, r.cy))
        s["plot_id"] = r.plot_id
        s["zmean_csv"] = float(r.zmean)
        rows.append(s)
        print(f"  {r.plot_id}: rumple={s['rumple']:.2f} cover={s['cover']:.2f} "
              f"penet={s['penetration']:.2f} | zmean rec={s['zmean_rec']:.1f} csv={r.zmean:.1f}", flush=True)

    out = pd.DataFrame(rows)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = config.OUT_DIR / "plot_structural.csv"
    out.to_csv(dst, index=False)
    dz = (out.zmean_rec - out.zmean_csv).abs()
    print(f"\nzmean footprint-vs-lidR check: max |Δ|={dz.max():.2f} m (should be small if centers align)")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()

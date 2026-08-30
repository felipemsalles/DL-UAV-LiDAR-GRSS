#!/usr/bin/env python3
"""Render the 13 real plots to a labeled CHM grid for visual inspection.

For each labeled plot: load its stand cloud, extract the footprint at the parcelas.shp centre,
rasterize (greenvista.image_cnn.render), and tile the CHM channel with Vol_ha + DBH titles.

Run (repo root, greenvista env): PYTHONPATH=. python scripts/preview_plot_rasters.py
"""
import warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from greenvista import config
from greenvista.image_cnn import footprint, render

RADIUS_M = 12.0  # lidR metric-extraction radius (area-based plan); area_parc=400 → confirm vs 11.28 m


def labeled_plots():
    import geopandas as gpd
    pm = pd.read_csv(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parc = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    parc["x"], parc["y"] = parc.geometry.x, parc.geometry.y
    # parcelas.shp stores keys as strings; the CSV as ints — coerce before joining
    for col in ("talhao", "parcela"):
        parc[col] = pd.to_numeric(parc[col], errors="coerce").astype("Int64")
        pm[col] = pd.to_numeric(pm[col], errors="coerce").astype("Int64")
    df = pm.merge(parc[["talhao", "parcela", "x", "y"]], on=["talhao", "parcela"], how="inner")
    return df.sort_values(["talhao", "parcela"]).reset_index(drop=True)


def main():
    df = labeled_plots()
    print(f"matched {len(df)} labeled plots to parcelas.shp centres (radius {RADIUS_M} m)")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    n = len(df); cols = 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.2 * rows))
    axes = np.array(axes).ravel()

    cloud_cache = {}
    for i, r in df.iterrows():
        t = int(r.talhao)
        if t not in cloud_cache:
            import laspy
            las = laspy.read(str(config.LAZ_DIR / f"SaoManuelTotal_{t:03d}.laz"))
            cloud_cache[t] = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
        pts = footprint.extract_footprint(cloud_cache[t], (r.x, r.y), RADIUS_M)
        raster, scal = render.to_raster(pts, (r.x, r.y), RADIUS_M)
        chm = raster[render.CHM]
        ax = axes[i]
        im = ax.imshow(np.where(chm > 0, chm, np.nan), cmap="viridis", origin="lower")
        ax.set_title(f"T{t}P{int(r.parcela)}  Vol={r.Vol_ha:.0f}  DBH={r.D_cm:.1f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        print(f"  T{t}P{int(r.parcela)}: pts={len(pts):>6}  CHMmax={chm.max():5.1f}m  "
              f"Vol_ha={r.Vol_ha:6.1f}  DBH={r.D_cm:4.1f}  (scalar zmax={scal['zmax']:.1f})")
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("São Manuel — per-plot CHM (metres) with field Vol_ha & DBH", y=1.005)
    fig.tight_layout()
    out = config.OUT_DIR / "plot_rasters_chm_grid.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

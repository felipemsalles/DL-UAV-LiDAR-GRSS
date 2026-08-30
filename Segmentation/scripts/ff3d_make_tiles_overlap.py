#!/usr/bin/env python3
"""Build an overlapping set of 32 m tiles for the worst-detection plots (T1, T5).

The single-tile FF3D benchmark uses one plot-centered 32 m tile per plot and counts instances within
the 12 m plot radius. A 32 m tile has a 16 m half-width, so the counted region keeps only a 4 m
margin to the tile edge and plot-boundary trees (≈12 m from centre) sit where FF3D may lose them
(cropped crowns, no context). This builds, for each plot, a 3×3 grid of 32 m tiles at stride 16 m
(50 % overlap). Tile size is unchanged, so per-tile VRAM is unchanged; every point within the 12 m
plot radius ends up ≥ ~7.5 m inside at least one tile.

The detection parse (`ff3d_detection_overlap.py`) pools instance centroids across all 9 offset tiles
of a plot, NMS-merges duplicates, and counts merged instances inside the 12 m plot → detection vs
field. Isolates edge loss; a modest or null gain is a valid outcome.

Scope: spike on the 2 worst stands only (T5 43 %, T1 54 %) = 4 plots × 9 tiles = 36 tiles.

Run: PYTHONPATH=. GREENVISTA_LAZ_DIR=<lazdir> python scripts/ff3d_make_tiles_overlap.py [size] [stride] [bucket_in_dir]
"""
import os, sys, shutil, itertools
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, laspy, warnings
warnings.simplefilter("ignore")

from greenvista import config
from greenvista.area_based import data as abd

SIZE = float(sys.argv[1]) if len(sys.argv) > 1 else 32.0
STRIDE = float(sys.argv[2]) if len(sys.argv) > 2 else 16.0
BUCKET_IN = Path(sys.argv[3]) if len(sys.argv) > 3 else \
    config.REPO / "project/models/FF3D_inference/FF3D_oracle/bucket_in_folder"
BACKUP = Path(config.LAZ_DIR).parent / "ff3d_tiles_overlap"

# Stands to tile. Default = the two with the worst detection. To scale to the 13 plots, use
# OVERLAP_TALHOES="2,4,6,7"; the manifest is merged, so the T1/T5 tiles are preserved.
SPIKE_TALHOES = tuple(int(t) for t in os.environ.get("OVERLAP_TALHOES", "1,5").split(","))


def load_plot_centres():
    df = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                        config.DATA / "5-dados_campo/inventario_est.csv")
    inv = pd.read_csv(config.DATA / "5-dados_campo/inventario_est.csv")
    df = df.merge(inv[["talhao", "parcela", "n_arv", "n_arv_ha"]], on=["talhao", "parcela"], how="left")
    parc = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for c in ("talhao", "parcela"):
        parc[c] = pd.to_numeric(parc[c], errors="coerce").astype("Int64")
    parc["cx"], parc["cy"] = parc.geometry.x, parc.geometry.y
    df = df.merge(parc[["talhao", "parcela", "cx", "cy"]], on=["talhao", "parcela"], how="left")
    return df[df.talhao.isin(SPIKE_TALHOES)].sort_values(["talhao", "parcela"]).reset_index(drop=True)


def main():
    df = load_plot_centres()
    BUCKET_IN.mkdir(parents=True, exist_ok=True)
    BACKUP.mkdir(parents=True, exist_ok=True)
    # 3x3 grid of centre offsets at ±STRIDE (and 0).
    offs = [-STRIDE, 0.0, STRIDE]
    grid = list(itertools.product(offs, offs))  # 9 (dx,dy) pairs
    h = SIZE / 2.0

    cloud = {}
    rows = []
    for _, r in df.iterrows():
        t = int(r.talhao)
        if t not in cloud:
            cloud.clear()
            las = laspy.read(str(config.LAZ_DIR / f"SaoManuelTotal_{t:03d}.laz"))
            cloud[t] = (las, np.asarray(las.x), np.asarray(las.y))
        las, x, y = cloud[t]
        for gi, (dx, dy) in enumerate(grid):
            tcx, tcy = r.cx + dx, r.cy + dy
            m = (np.abs(x - tcx) <= h) & (np.abs(y - tcy) <= h)
            sub = laspy.LasData(las.header)
            sub.points = las.points[m].copy()
            name = f"t{t:03d}_p{int(r.parcela):03d}_g{gi}"
            sub.write(str(BUCKET_IN / f"{name}.laz"))
            shutil.copy(BUCKET_IN / f"{name}.laz", BACKUP / f"{name}.laz")
            sz = (BUCKET_IN / f"{name}.laz").stat().st_size / 1e6
            rows.append({"tile": name, "plot_id": r.plot_id, "talhao": t,
                         "parcela": int(r.parcela), "gi": gi, "dx": dx, "dy": dy,
                         "tcx": float(tcx), "tcy": float(tcy),
                         "cx": float(r.cx), "cy": float(r.cy), "n_pts": int(m.sum()),
                         "field_n_arv": float(r.n_arv), "field_n_arv_ha": float(r.n_arv_ha),
                         "laz_MB": round(sz, 2)})
        print(f"  plot {r.plot_id}: 9 tiles, field {int(r.n_arv)} stems, "
              f"pts {min(x['n_pts'] for x in rows[-9:])}-{max(x['n_pts'] for x in rows[-9:])}")
    man = pd.DataFrame(rows)
    mpath = BACKUP / "tiles_manifest.csv"
    if mpath.exists():  # merge, to preserve the T1/T5 rows from an earlier run
        prev = pd.read_csv(mpath)
        man = pd.concat([prev, man], ignore_index=True).drop_duplicates("tile", keep="last")
        man = man.sort_values(["talhao", "parcela", "gi"]).reset_index(drop=True)
    man.to_csv(mpath, index=False)
    print(f"\n{len(rows)} tiles built → {BUCKET_IN}\n manifest now {len(man)} rows, backup → {BACKUP}")
    print(f" tile MB range {man.laz_MB.min()}-{man.laz_MB.max()}, "
          f"pts {man.n_pts.min()}-{man.n_pts.max()}")
    big = man[man.laz_MB > 10]
    if len(big):
        print(f" {len(big)} tiles > 10 MB (OOM watch): {big.tile.tolist()}")


if __name__ == "__main__":
    main()

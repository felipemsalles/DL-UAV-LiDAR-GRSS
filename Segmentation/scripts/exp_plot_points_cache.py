#!/usr/bin/env python3
"""Extract the points of the 13 plots from each LAZ once and cache them in an .npz file.

Rationale: the two new routes (return/intensity metrics, and planting-grid occupancy) need the same
per-plot points. Reading 812 MB of LAZ twice would be wasteful, and neither route needs a GPU. This
step is the only slow one; after it both analyses run in seconds.

It also stores intensity, return_number and number_of_returns, which the current pipeline does not
use. All ~40 metrics in `greenvista/area_based/data.py` are height metrics (zmax, zmean, zq*,
zpcum*, ...), with no return, intensity or RGB metric, even though the data carries all three
(point format 3, 5 returns).

Plot centres come from Dados_SaoManuel/2-shapes/Parcelas/parcelas.shp (EPSG:31982), radius
config.PLOT_RADIUS_M = 12 m, the same footprint used to compute the height metrics.

Run: PYTHONPATH=. python scripts/exp_plot_points_cache.py
"""
import os
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from greenvista import config  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LAZ_DIR = Path(os.environ.get("GREENVISTA_LAZ_DIR", REPO / "work" / "lazall"))
SHP = config.DATA / "2-shapes" / "Parcelas" / "parcelas.shp"
OUT = REPO / "work" / "plot_points_cache.npz"

# the 13 eucalyptus plots used throughout the project (015 is Pinus, 003 has no point cloud)
PLOTS = [(1, 1), (1, 2), (2, 6), (2, 7), (4, 4), (4, 5), (5, 8), (5, 9),
         (6, 10), (6, 11), (6, 12), (7, 13), (7, 14)]

g = gpd.read_file(SHP).to_crs(config.CRS)
g["talhao_i"] = g.talhao.astype(int)
g["parcela_i"] = g.parcela.astype(int)
cen = {(int(r.talhao_i), int(r.parcela_i)): (r.geometry.centroid.x, r.geometry.centroid.y)
       for r in g.itertuples()}

R = config.PLOT_RADIUS_M
store, faltando = {}, []

for talhao in sorted({t for t, _ in PLOTS}):
    laz = LAZ_DIR / f"SaoManuelTotal_{talhao:03d}.laz"
    if not laz.exists():
        faltando.append(str(laz))
        continue
    las = laspy.read(str(laz))
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    inten = np.asarray(las.intensity).astype(np.float32)
    rnum = np.asarray(las.return_number).astype(np.int8)
    nret = np.asarray(las.number_of_returns).astype(np.int8)
    cls = np.asarray(las.classification).astype(np.int8)
    print(f"stand {talhao}: {len(x):,} points", flush=True)

    for t, p in PLOTS:
        if t != talhao:
            continue
        cx, cy = cen[(t, p)]
        sel = (x - cx) ** 2 + (y - cy) ** 2 <= R ** 2
        k = f"{t}_{p}"
        store[k + "|xyz"] = np.column_stack([x[sel] - cx, y[sel] - cy, z[sel]]).astype(np.float32)
        store[k + "|inten"] = inten[sel]
        store[k + "|rnum"] = rnum[sel]
        store[k + "|nret"] = nret[sel]
        store[k + "|cls"] = cls[sel]
        store[k + "|centro"] = np.array([cx, cy], dtype=np.float64)
        print(f"  plot {p}: {int(sel.sum()):,} points within the {R:.0f} m radius", flush=True)
    del las, x, y, z, inten, rnum, nret, cls

if faltando:
    print("MISSING POINT CLOUDS:", faltando, file=sys.stderr)
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT, **store)
print("wrote", OUT, f"({OUT.stat().st_size / 1e6:.0f} MB)")

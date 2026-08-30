#!/usr/bin/env python3
"""Closes the impossible-geometry test on the published product: the centroids merged at 1.5 m.

`exp_split_merge.py` measured the single pass (60.8%) and estimated, per window, what would survive
the merge. Nothing is estimated here: it reuses exactly the functions of `ff3d_detection_overlap.py`
to rebuild the merged centroids that yield the 72.8%, and re-measures.

Question: after the merge, how many detections still have a nearest neighbour closer than the
planting spacing? Two real trees are never that close, so each such pair is an impossible one and a
floor on commission, measured without a reference map and without registration.

The merge uses a 1.5 m radius, so nothing survives below that by construction. The observable window
runs from 1.5 m up to the plot spacing (1.41 to 2.38 m in ours). In plots whose spacing is smaller
than 1.5 m the window is empty and the plot is uninformative, which is also reported.

Inputs, all already on disk:
* PLYs of the 117 overlapping tiles in ~/ff3d_regen13_out/round_2_after_remove_noise_200
* backup .laz tiles in work/ff3d_tiles_overlap (regenerated; the UTM shift agrees to 0.000 m)

Run: PYTHONPATH=. GREENVISTA_LAZ_DIR=$PWD/work/lazall python scripts/exp_split_merge_fundido.py [ply_dir]
"""
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ff3d_detection_overlap import instance_centroids, nms_merge  # noqa: E402

from greenvista import config  # noqa: E402

PLY_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path.home() / "ff3d_regen13_out" / "round_2_after_remove_noise_200"
BACKUP = Path(config.LAZ_DIR).parent / "ff3d_tiles_overlap"
SHP = config.DATA / "2-shapes" / "Parcelas" / "parcelas.shp"
INV = config.DATA / "5-dados_campo" / "inv_euc.csv"
OUT = config.OUT_DIR / "split_merge_fundido.csv"
MERGE_R = 1.5


def espac_menor(s):
    n = re.findall(r"\d+[.,]?\d*", str(s).replace(",", "."))
    return min(float(n[0]), float(n[1])) if len(n) >= 2 else None


gdf = gpd.read_file(SHP).to_crs(config.CRS)
CEN = {(int(r.talhao), int(r.parcela)): (r.geometry.centroid.x, r.geometry.centroid.y)
       for r in gdf.itertuples()}
inv = pd.read_csv(INV, sep=None, engine="python")
viv = inv[inv.D_cm.notna()]

linhas = []
for (t, p), g in viv.groupby(["talhao", "parcela"]):
    chave = f"{int(t)}_{int(p)}"
    sy = espac_menor(next((v for v in g.espac if isinstance(v, str)), ""))
    if sy is None or (int(t), int(p)) not in CEN:
        continue
    cx, cy = CEN[(int(t), int(p))]

    todos_c, todos_s = [], []
    for gi in range(9):
        nome = f"t{int(t):03d}_p{int(p):03d}_g{gi}"
        ply = PLY_DIR / f"{nome}_round2.ply"
        laz = BACKUP / f"{nome}.laz"
        if not (ply.exists() and laz.exists()):
            continue
        c, s, _ = instance_centroids(ply, laz)
        if len(c):
            todos_c.append(c)
            todos_s.append(s)
    if not todos_c:
        print(f"  {chave}: no tiles")
        continue

    c = np.vstack(todos_c)
    s = np.concatenate(todos_s)
    cf = nms_merge(c, s, MERGE_R)   # returns only the centroids
    dentro = (cf[:, 0] - cx) ** 2 + (cf[:, 1] - cy) ** 2 <= config.PLOT_RADIUS_M ** 2
    cf = cf[dentro]
    if len(cf) < 3:
        continue

    d = cKDTree(cf).query(cf, k=2)[0][:, 1]
    janela_existe = sy > MERGE_R
    viol = float((d < sy * 0.98).mean()) if janela_existe else np.nan
    linhas.append(dict(chave=chave, campo=len(g), fundidos=len(cf),
                       taxa=len(cf) / len(g), espac=sy,
                       janela_util=janela_existe, nn_min=float(d.min()),
                       nn_mediana=float(np.median(d)), frac_impossivel=viol,
                       n_impossivel=int((d < sy * 0.98).sum()) if janela_existe else 0))
    print(f"  {chave}: field {len(g)}, merged {len(cf)} ({len(cf)/len(g):.0%}), "
          f"spacing {sy:.2f} m, NN min {d.min():.2f} m, "
          f"{'impossible ' + format(viol, '.1%') if janela_existe else 'empty window'}", flush=True)

d = pd.DataFrame(linhas)
print("\n" + d.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

u = d[d.janela_util]
print(f"\n=== {d.fundidos.sum()} merged detections, {d.campo.sum()} field trees, "
      f"detection {d.fundidos.sum() / d.campo.sum():.1%}")
print(f"global minimum NN after the merge: {d.nn_min.min():.2f} m (merge radius {MERGE_R} m)")
print(f"plots with a usable window (spacing > {MERGE_R} m): {len(u)} of {len(d)}")
if len(u):
    n_imp, n_tot = u.n_impossivel.sum(), u.fundidos.sum()
    print(f"impossible pairs in those plots: {n_imp} of {n_tot} ({n_imp / n_tot:.1%})")
d.to_csv(OUT, index=False)
print(f"wrote {OUT}")

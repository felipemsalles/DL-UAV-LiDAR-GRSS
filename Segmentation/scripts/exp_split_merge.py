#!/usr/bin/env python3
"""Are omission and commission cancelling out inside the 72.8%? A test with no field tree positions.

Per-cell counting, the initial idea, only catches large-scale compensation, of the "lost one in corner
A and invented one in corner B" kind. The likely compensation mechanism here is local: one crown split
in two (commission) and two crowns merged into one (omission). That happens between immediate
neighbours, inside the same cell, and would cancel within it.

The test used here starts from the planting geometry and needs neither registration nor field
positions. The trees are planted on a regular grid; in stand 1, 2.87 m between rows and 1.88 m within
the row, so two real trees are never less than ~1.88 m apart. Every detection whose nearest neighbour
is closer than the planting spacing is, by geometry, a pair that cannot correspond to two distinct
trees, which gives a floor on commission from crown splitting, measured with no reference map at all.

These PLYs are from the single pass, with no overlapping tiles and no NMS merge. They support the
60.8% of the reference condition, and not the published 72.8%, so what this test measures is the
commission of the baseline. To estimate what would survive the merge, the fraction of pairs in the
window between 1.5 m (the NMS radius) and the planting spacing is also measured.

It also reconstructs the field grid (row + tree + spacing) and compares the nearest-neighbour distance
distribution against a null of random thinning of that same grid at the observed rate. If the
detections have more mass at short distances than the null, there is clustering the grid does not
explain.

Run: PYTHONPATH=. python scripts/exp_split_merge.py
"""
import re
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from greenvista import config  # noqa: E402
from greenvista.segmentation.ff3d import load_panoptic_ply  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PLY_DIR = REPO / "work" / "ff3d_degraded_out" / "full"
INV = config.DATA / "5-dados_campo" / "inv_euc.csv"
TILE_DIR = REPO / "work" / "ff3d_degraded" / "full"
SHP = config.DATA / "2-shapes" / "Parcelas" / "parcelas.shp"
OUT = config.OUT_DIR / "split_merge.csv"
RNG = np.random.default_rng(20260730)
N_SIM = 300


def espac(s):
    if not isinstance(s, str):
        return None
    n = re.findall(r"\d+[.,]?\d*", s.replace(",", "."))
    return tuple(sorted((float(n[0]), float(n[1])), reverse=True)) if len(n) >= 2 else None


def centroides(ply, tile_laz, cx, cy):
    """Centroids of the tile instances, in UTM, restricted to the plot circle.

    The relevant crop is spatial, and not by point count: the tile covers a far larger area than the
    400 m2 plot, and filtering only by >= 30 points gives 943 detections against 717 field trees,
    with pairs 5 cm apart. It follows the official scorer (scripts/ff3d_detection_numbers.py), which
    converts the centroid to UTM by adding the tile shift and keeps only what falls inside the plot
    circle, so that the number is the same one that supports the 72.8%."""
    t = load_panoptic_ply(ply)
    if not t:
        return np.zeros((0, 2))
    c = np.array([v["points"][:, :2].mean(0) for v in t.values()])
    lo = np.array([min(v["points"][:, 0].min() for v in t.values()),
                   min(v["points"][:, 1].min() for v in t.values())])
    las = laspy.read(str(tile_laz))
    off = np.array([float(np.asarray(las.x).min()), float(np.asarray(las.y).min())]) - lo
    c = c + off
    dentro = (c[:, 0] - cx) ** 2 + (c[:, 1] - cy) ** 2 <= config.PLOT_RADIUS_M ** 2
    return c[dentro]


def grade_de_campo(g, sx, sy):
    """Relative position of each tree. `arvore` is sequential and contiguous within each `linha`,
    checked against the inventory, so the position along the row is (arvore - first of the row)."""
    pos = []
    for linha, gl in g.groupby("linha"):
        a0 = gl.arvore.min()
        for a in gl.arvore:
            pos.append(((a - a0) * sy, (linha - 1) * sx))
    p = np.array(pos, dtype=float)
    return p - p.mean(0)


def nn(p):
    if len(p) < 2:
        return np.array([])
    d, _ = cKDTree(p).query(p, k=2)
    return d[:, 1]


gdf = gpd.read_file(SHP).to_crs(config.CRS)
CEN = {(int(r.talhao), int(r.parcela)): (r.geometry.centroid.x, r.geometry.centroid.y)
       for r in gdf.itertuples()}

inv = pd.read_csv(INV, sep=None, engine="python")
viv = inv[inv.D_cm.notna()]

linhas = []
for (t, parc), g in viv.groupby(["talhao", "parcela"]):
    ply = PLY_DIR / f"t{int(t):03d}_p{int(parc):03d}_round2.ply"
    sp = espac(next((v for v in g.espac if isinstance(v, str)), None))
    tile = TILE_DIR / f"t{int(t):03d}_p{int(parc):03d}.laz"
    if not ply.exists() or sp is None or not tile.exists() or (int(t), int(parc)) not in CEN:
        print(f"  skipping {int(t)}_{int(parc)}")
        continue
    cx, cy = CEN[(int(t), int(parc))]
    sx, sy = sp                                   # sx between rows, sy within the row (the smaller)
    campo = grade_de_campo(g, sx, sy)
    det = centroides(ply, tile, cx, cy)
    if len(det) < 20:
        continue

    nn_campo, nn_det = nn(campo), nn(det)
    taxa = len(det) / len(campo)

    # null: thin the field grid down to the number of detections, omission only, zero commission
    nulo_min, nulo_sub = [], []
    k = min(len(det), len(campo))
    for _ in range(N_SIM):
        sub = campo[RNG.choice(len(campo), k, replace=False)]
        d = nn(sub)
        nulo_min.append(d.min())
        nulo_sub.append((d < sy * 0.98).mean())

    linhas.append(dict(
        chave=f"{int(t)}_{int(parc)}", talhao=int(t),
        n_campo=len(campo), n_det=len(det), taxa=taxa,
        espac_linha=sx, espac_arvore=sy,
        nn_campo_min=float(nn_campo.min()), nn_campo_med=float(np.median(nn_campo)),
        nn_det_min=float(nn_det.min()), nn_det_med=float(np.median(nn_det)),
        # pairs closer than the planting spacing: floor on commission from splitting
        frac_abaixo_espac=float((nn_det < sy * 0.98).mean()),
        # what would survive the 1.5 m NMS merge: pairs between 1.5 m and the spacing
        frac_janela_nms=float(((nn_det >= 1.5) & (nn_det < sy * 0.98)).mean()),
        frac_abaixo_1m5=float((nn_det < 1.5).mean()),
        nulo_frac_abaixo=float(np.mean(nulo_sub)),
        nulo_nn_min=float(np.mean(nulo_min))))
    print(f"  {linhas[-1]['chave']}: field {len(campo)}, det {len(det)} ({taxa:.0%}), "
          f"spacing {sy:.2f} m, NN min det {nn_det.min():.2f} m, "
          f"below the spacing {linhas[-1]['frac_abaixo_espac']:.1%} "
          f"(null {linhas[-1]['nulo_frac_abaixo']:.1%})", flush=True)

d = pd.DataFrame(linhas)
print("\n" + d[["chave", "n_campo", "n_det", "taxa", "espac_arvore", "nn_det_min",
                "frac_abaixo_espac", "nulo_frac_abaixo"]].to_string(
    index=False, float_format=lambda v: f"{v:.3f}"))

tot_det = d.n_det.sum()
tot_sob = (d.frac_abaixo_espac * d.n_det).sum()
tot_nulo = (d.nulo_frac_abaixo * d.n_det).sum()
print(f"\n=== {tot_det} detections in total")
print(f"pairs closer than the planting spacing: {tot_sob:.0f} ({tot_sob/tot_det:.1%})")
print(f"expected under random thinning of the grid (omission only): {tot_nulo:.0f} ({tot_nulo/tot_det:.1%})")
jan = (d.frac_janela_nms * d.n_det).sum(); sob15 = (d.frac_abaixo_1m5 * d.n_det).sum()
print(f"  of these, below 1.5 m (the MERGE would remove them): {sob15:.0f} ({sob15/tot_det:.1%})")
print(f"  of these, between 1.5 m and the spacing (WOULD SURVIVE the merge): {jan:.0f} ({jan/tot_det:.1%})")
print(f"minimum NN observed among the detections: {d.nn_det_min.min():.2f} m")
print(f"smallest planting spacing across the plots: {d.espac_arvore.min():.2f} m")
print("\nThese PLYs are from the single pass (60.8%), with no merge. The number above is the BASELINE.")
d.to_csv(OUT, index=False)
print(f"wrote {OUT}")

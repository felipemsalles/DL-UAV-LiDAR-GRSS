#!/usr/bin/env python3
"""Re-measure the per-tree localisation floor with FF3D instance centroids (vs CHM treetops).

The CHM spike (exp_grid_reconstruction.py) found treetops sit a median 1.64 m from the 26 surveyed
trees, with only ~30 % of stems detected on the closed canopy. FF3D segments the closed canopy far
better, and — because it labels wood vs leaf — allows a stem-base proxy (low wood points) that
should match the stem GPS better than a crown centroid.

For stand 001's two segmented 32 m plot tiles the FF3D local->UTM shift is recovered the same way
as in ff3d_detection_numbers.py (min-corner alignment against a regenerated tile crop), then three
localisers are compared on the same surveyed trees that fall inside the tiles:
  (a) FF3D crown centroid  (mean xy of all instance points)
  (b) FF3D stem-base proxy (median xy of the instance's low wood points, semantic_pred==1, z<3 m)
  (c) CHM treetop          (local-maxima detection, restricted to the same footprints)

Run: PYTHONPATH=. python scripts/exp_ff3d_centroid_floor.py
"""
import numpy as np
import laspy
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from plyfile import PlyData

from greenvista import config, segmentation as seg
from greenvista.io.geoloc import load_geoloc

CLOUD = config.TILE_LAZ
PLY_DIR = config.REPO / "work" / "ff3d_degraded_out" / "full"
PLYS = {p: PLY_DIR / f"t001_p{p:03d}_round2.ply" for p in (1, 2)}
HALF = 16.0            # tiles are 32 m squares (|x-cx|<=16)
WOOD = 1               # semantic_pred wood class
BASE_Z = 3.0           # m above ground: "low" wood = stem base
NOMINAL_TREE = 1.88    # within-row spacing (same-tree threshold ~ half of this)


def read_ply(path):
    el = PlyData.read(path).elements[0]
    D = {p.name: np.array(el[p.name]) for p in el.properties}
    return D["x"], D["y"], D["z"], D["instance_pred"], D["semantic_pred"]


def tile_min(cx, cy, X, Y, xyz_x, xyz_y):
    m = (np.abs(xyz_x - cx) <= HALF) & (np.abs(xyz_y - cy) <= HALF)
    return xyz_x[m].min(), xyz_y[m].min()


def main():
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    # plot centres for stand-1 plots 1,2
    parc = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    import pandas as pd
    for c in ("talhao", "parcela"):
        parc[c] = pd.to_numeric(parc[c], errors="coerce").astype("Int64")
    ctr = {int(r.parcela): (r.geometry.x, r.geometry.y)
           for _, r in parc[parc.talhao == 1].iterrows() if int(r.parcela) in PLYS}

    las = laspy.read(CLOUD)
    cx_all, cy_all = np.asarray(las.x), np.asarray(las.y)

    crown_c, stem_c = [], []       # UTM centroids across both tiles
    for pid, ply in PLYS.items():
        cx, cy = ctr[pid]
        txmin, tymin = tile_min(cx, cy, None, None, cx_all, cy_all)
        lx, ly, lz, inst, sem = read_ply(ply)
        sx, sy = txmin - lx.min(), tymin - ly.min()   # local -> UTM (min-corner alignment)
        for iid in np.unique(inst):
            if iid < 0:
                continue
            m = inst == iid
            crown_c.append([lx[m].mean() + sx, ly[m].mean() + sy])
            wm = m & (sem == WOOD) & (lz < BASE_Z)     # low wood = stem base
            if wm.sum() < 5:
                wm = m & (sem == WOOD)                 # fall back to all wood
            if wm.sum() < 5:
                wm = m                                 # fall back to crown
            stem_c.append([np.median(lx[wm]) + sx, np.median(ly[wm]) + sy])
    crown_c = np.array(crown_c); stem_c = np.array(stem_c)
    print(f"FF3D instances across the 2 stand-001 tiles: {len(crown_c)}")

    # surveyed trees inside either 32 m tile footprint
    geo = load_geoloc()
    gt = geo[["Easting", "Northing"]].to_numpy()
    inside = np.zeros(len(gt), bool)
    for cx, cy in ctr.values():
        inside |= (np.abs(gt[:, 0] - cx) <= HALF) & (np.abs(gt[:, 1] - cy) <= HALF)
    gt_in = gt[inside]
    print(f"surveyed trees inside the tiled footprints: {inside.sum()}/{len(gt)} "
          f"(the rest of the stand was never segmented)")

    # CHM treetops restricted to the same footprints (apples-to-apples)
    keep = las.classification != 18
    chm, x0, y0, res = seg.build_chm(np.column_stack([cx_all[keep], cy_all[keep]]).astype(float),
                                     np.asarray(las.z[keep], float), res=0.5)
    tops = seg.detect_treetops(chm, x0, y0, res, window=3, hmin=10.0, smooth=0.6)[:, :2]
    tin = np.zeros(len(tops), bool)
    for cx, cy in ctr.values():
        tin |= (np.abs(tops[:, 0] - cx) <= HALF) & (np.abs(tops[:, 1] - cy) <= HALF)
    tops_in = tops[tin]

    def floor(name, cents):
        d, _ = cKDTree(cents).query(gt_in, k=1)
        within = (d <= NOMINAL_TREE / 2).mean() * 100
        print(f"  {name:22s} n_det={len(cents):3d} | to surveyed tree: "
              f"median={np.median(d):.2f} m  mean={d.mean():.2f} m  p90={np.percentile(d,90):.2f} m  "
              f"| within {NOMINAL_TREE/2:.2f} m: {within:.0f}%")
        return d

    print(f"\n=== localisation floor on the SAME {inside.sum()} surveyed trees ===")
    d_crown = floor("FF3D crown centroid", crown_c)
    d_stem = floor("FF3D stem-base (wood)", stem_c)
    d_chm = floor("CHM treetop", tops_in)

    # figure
    fig, ax = plt.subplots(figsize=(9, 8))
    for cx, cy in ctr.values():
        ax.add_patch(plt.Rectangle((cx - HALF, cy - HALF), 2 * HALF, 2 * HALF,
                     fill=False, ec="gray", ls=":", lw=1))
    ax.scatter(tops_in[:, 0], tops_in[:, 1], s=20, c="green", alpha=0.5, label=f"CHM treetops ({len(tops_in)})")
    ax.scatter(crown_c[:, 0], crown_c[:, 1], s=14, c="orange", alpha=0.6, label=f"FF3D crown centroids ({len(crown_c)})")
    ax.scatter(stem_c[:, 0], stem_c[:, 1], s=14, c="purple", alpha=0.6, label=f"FF3D stem-base ({len(stem_c)})")
    ax.scatter(gt_in[:, 0], gt_in[:, 1], s=90, c="red", marker="x", lw=2, label=f"surveyed trees ({inside.sum()})")
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Stand 001: FF3D centroids vs CHM treetops vs surveyed trees (tiled plots)")
    figp = config.OUT_DIR / "ff3d_centroid_floor_talhao001.png"
    fig.tight_layout(); fig.savefig(figp, dpi=110)
    print(f"\nwrote {figp}")

    best = min([("crown centroid", np.median(d_crown)), ("stem-base", np.median(d_stem))], key=lambda x: x[1])
    print("\n" + "=" * 78)
    print(f"FF3D best localiser: {best[0]} at {best[1]:.2f} m median vs CHM {np.median(d_chm):.2f} m "
          f"(spike whole-strip 1.64 m).")
    if best[1] < NOMINAL_TREE / 2:
        print("→ FF3D gets INSIDE half-spacing: per-tree labels become plausible on tiled plots.")
    elif best[1] < np.median(d_chm):
        print("→ FF3D improves on CHM but still ~tree-spacing scale: positions approximate, not clean.")
    else:
        print("→ FF3D no better here; localisation stays the wall (crown≠stem offset dominates).")
    print("=" * 78)


if __name__ == "__main__":
    main()

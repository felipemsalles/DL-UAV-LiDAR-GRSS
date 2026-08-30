"""Per-tree DAP (DBH) dataset from UAV-LiDAR crowns.

Reproduces the per-tree deep-learning setup of a peer's São Manuel study (CNN / PointNet /
PointTransformer → DAP). The binding constraint: only the 23 geolocated cubed trees (stand 001) have
both a position (to isolate the crown) and a measured DAP, so N=23 in a single stand. That is enough
to train the nets, but only LOOCV with a permutation control can validate them, and no cross-stand
test is possible since every labeled tree sits in a single stand.

Crowns are extracted as a cylinder around each tree's (E,N); an FF3D panoptic PLY can be supplied
instead (segmented instances) via `load_tree_crowns_ff3d` when available.
"""
import numpy as np

from greenvista import config
from greenvista.io.geoloc import load_geoloc, link_geoloc_volumes
from greenvista.io.ground_truth import load_volumes
from greenvista.image_cnn import footprint


def load_tree_crowns(radius=2.0, zmin=config.Z_MIN_M, zmax=config.Z_MAX_M, talhao=1,
                     laz=None, min_pts=64):
    """Return {arv: {points (N,3) float32, D_cm, H_m, center (2,)}} for the geolocated cubed trees."""
    import laspy
    geo = load_geoloc()
    vol = load_volumes()
    linked = link_geoloc_volumes(geo, vol, talhao=talhao)
    las = laspy.read(str(laz or config.LAZ_DIR / f"SaoManuelTotal_{talhao:03d}.laz"))
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
    trees = {}
    for _, t in linked.iterrows():
        pts = footprint.extract_footprint(xyz, (t.Easting, t.Northing), radius)
        c = pts[(pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)]
        if len(c) < min_pts:
            continue
        trees[int(t.arv)] = {"points": c.astype(np.float32), "D_cm": float(t.D_cm),
                             "H_m": float(t.H_m), "center": np.array([t.Easting, t.Northing], float)}
    return trees


def normalize_points(pts, center_xy, n_sample, rng, augment=False, scale=10.0, jitter=0.02):
    """Center XY on the tree, keep Z as absolute height (DAP depends on height, so do not
    z-normalize), resample to `n_sample`, optional azimuthal rotation + jitter (DAP is
    rotation-invariant), scale to ~O(1). Returns (n_sample, 3) float32."""
    p = np.asarray(pts, np.float32).copy()
    p[:, 0] -= center_xy[0]
    p[:, 1] -= center_xy[1]
    idx = rng.choice(len(p), n_sample, replace=len(p) < n_sample)
    p = p[idx]
    if augment:
        th = rng.uniform(0, 2 * np.pi)
        ct, st = np.cos(th), np.sin(th)
        x, y = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0] = x * ct - y * st
        p[:, 1] = x * st + y * ct
        p = p + rng.normal(0, jitter, p.shape).astype(np.float32)
    p[:, :2] /= scale
    p[:, 2] /= scale
    return p.astype(np.float32)


def crown_scalars(pts):
    """Simple crown geometry used by the allometry baselines: zmax, zmean, crown_area (2D hull)."""
    from scipy.spatial import ConvexHull, QhullError
    z = pts[:, 2]
    area = np.nan
    if len(pts) >= 3:
        try:
            area = float(ConvexHull(pts[:, :2]).volume)   # 2D hull .volume == area
        except (QhullError, ValueError):
            area = np.nan
    return {"zmax": float(z.max()), "zmean": float(z.mean()),
            "crown_area": area if np.isfinite(area) else np.pi * 2.0 ** 2}

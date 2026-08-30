"""Wall-to-wall CNN volume map.

Slide a plot-sized (12 m radius) window over the cloud on a regular grid; for each cell, render the
window exactly like a training plot and predict Vol_ha with the full-fit CNN. Cell volume (m³) =
Vol_ha × cell area. Intervals use the model's in-sample residual std, which is deliberately loose
at N=13.
"""
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import box

from greenvista import config
from . import render, footprint
from .train import predict_raster

CRS = config.CRS


def volume_map(net, nrm, resid_sd, xyz, cell=config.ABA.map_cell_m, radius=footprint.DEFAULT_RADIUS_M,
               size=128, min_pts=400, zmin=config.Z_MIN_M, zmax=config.Z_MAX_M, dev=None,
               scalar_envelope=None):
    """Return a GeoDataFrame of `cell`-m square cells with vol_ha / volume_m3 / vol_lo / vol_hi.

    `scalar_envelope` (dict {scalar: (min,max)} from the training plots) flags cells whose scalar
    inputs fall outside the training envelope as `extrapolated=True`. Cells with non-finite scalars
    are skipped and counted in `.attrs['n_skipped_nan']`.
    """
    x, y = xyz[:, 0], xyz[:, 1]
    tree = cKDTree(np.column_stack([x, y]))
    x0, y0, x1, y1 = x.min(), y.min(), x.max(), y.max()
    cell_area_ha = cell * cell / 1e4
    zci = 1.96 * float(resid_sd[0])                       # Vol_ha residual std → ±95% half-width
    rows, geoms, n_skipped = [], [], 0
    cxs = np.arange(x0 + cell / 2, x1, cell)
    cys = np.arange(y0 + cell / 2, y1, cell)
    for cx in cxs:
        for cy in cys:
            idx = tree.query_ball_point([cx, cy], radius)
            if len(idx) < min_pts:
                continue
            raster, scal = render.to_raster(xyz[idx], (cx, cy), radius, size=size, zmin=zmin, zmax=zmax)
            svec = np.array([scal["zmax"], scal["zmean"], scal["zsd"]], np.float32)
            if not np.all(np.isfinite(svec)):            # NaN guard
                n_skipped += 1
                continue
            vha = max(0.0, float(predict_raster(net, nrm, raster, svec, dev)[0]))
            extrap = _outside_envelope(scal, scalar_envelope)
            rows.append({"vol_ha": vha, "volume_m3": vha * cell_area_ha,
                         "vol_lo": max(0.0, vha - zci) * cell_area_ha, "vol_hi": (vha + zci) * cell_area_ha,
                         "extrapolated": extrap})
            geoms.append(box(cx - cell / 2, cy - cell / 2, cx + cell / 2, cy + cell / 2))
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=CRS)
    gdf.attrs["n_skipped_nan"] = n_skipped
    return gdf


def _outside_envelope(scal, envelope):
    if not envelope:
        return False
    for k, (lo, hi) in envelope.items():
        v = scal.get(k, np.nan)
        if not np.isfinite(v) or v < lo or v > hi:
            return True
    return False

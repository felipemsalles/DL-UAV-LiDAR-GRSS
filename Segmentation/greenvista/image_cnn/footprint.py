"""Extract a plot's circular footprint from a LiDAR cloud.

Plots are 12 m-radius circles centred on the points in 2-shapes/Parcelas/parcelas.shp. This matches
the canonical lidR extraction (`metricas_LiDAR.R`: `clip_roi(..., radius = 12)` and
`plot_metrics(..., .stdmetrics_z, radius = 12)`), so the rasters share the footprint over which the
plot height metrics (the strong Vol_ha predictors) were computed. (The field-inventory `area_parc` is
a nominal 400 m²; the LiDAR metrics use radius 12.)
Kept array-pure (operates on an (N,3) ndarray) so it is trivially testable; laz loading is a thin
wrapper around it.
"""
from pathlib import Path
import numpy as np

# Canonical lidR plot radius (metricas_LiDAR.R). Footprint over which .stdmetrics_z were computed.
DEFAULT_RADIUS_M = 12.0


def extract_footprint(xyz, center, radius=DEFAULT_RADIUS_M):
    """Return the subset of `xyz` (N,3) whose XY lies within `radius` of `center` (cx, cy)."""
    pts = np.asarray(xyz, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 3)
    dx = pts[:, 0] - float(center[0])
    dy = pts[:, 1] - float(center[1])
    return pts[dx * dx + dy * dy <= float(radius) ** 2]


def load_plot_points(laz_path, center, radius=DEFAULT_RADIUS_M):
    """Read a (normalized) .laz and return the (M,3) points inside the plot footprint."""
    import laspy
    las = laspy.read(str(Path(laz_path)))
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
    return extract_footprint(xyz, center, radius)

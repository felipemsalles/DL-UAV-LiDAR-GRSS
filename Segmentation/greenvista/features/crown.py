"""Crown geometry features from a tree's (segmented) point cloud.

Volume is scale-dependent, so these features keep metric units: do not unit-normalize the points
before calling. Points are XYZ with Z already height-above-ground (or pass z0 to subtract).

The richer 41 height-distribution metrics and the voxel metrics live in
project/scripts/common.py:height_distribution_metrics() / voxel_crown_metrics(); this module keeps
the core geometric set self-contained and tested.
"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError

CROWN_FEATURE_KEYS = (
    "n_points", "H_max", "H_min", "H_mean", "H_std", "z99",
    "crown_area", "crown_volume", "crown_diameter",
)

def crown_features(points: np.ndarray, z0: float = 0.0) -> dict:
    """Compute crown features for one tree. `points`: (N,3) XYZ in metres. Returns a dict."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    z = pts[:, 2] - z0
    feats = {
        "n_points": int(len(pts)),
        "H_max": float(z.max()) if len(pts) else np.nan,
        "H_min": float(z.min()) if len(pts) else np.nan,
        "H_mean": float(z.mean()) if len(pts) else np.nan,
        "H_std": float(z.std()) if len(pts) else np.nan,
        "z99": float(np.percentile(z, 99)) if len(pts) else np.nan,
    }
    xy = pts[:, :2]
    feats["crown_area"] = _hull_measure(xy)        # 2D hull -> area (m²)
    feats["crown_volume"] = _hull_measure(pts)     # 3D hull -> volume (m³)
    if len(xy) >= 1:
        c = xy.mean(axis=0)
        feats["crown_diameter"] = float(2 * np.linalg.norm(xy - c, axis=1).max())
    else:
        feats["crown_diameter"] = np.nan
    return feats

def _hull_measure(pts: np.ndarray) -> float:
    """ConvexHull .volume: area in 2D, volume in 3D. NaN if degenerate."""
    if len(pts) < pts.shape[1] + 1:
        return np.nan
    try:
        return float(ConvexHull(pts).volume)
    except (QhullError, ValueError):
        return np.nan

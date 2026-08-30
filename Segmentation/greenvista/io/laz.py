"""LiDAR I/O helpers. Clouds are height-normalized; classes are {1=unclass, 2=ground, 18=noise}."""
from pathlib import Path
import numpy as np
from greenvista import config

def canopy_mask(classification: np.ndarray) -> np.ndarray:
    """Boolean mask of canopy/veg points (everything that is not ground or noise)."""
    cls = np.asarray(classification)
    return ~np.isin(cls, config.GROUND_NOISE_CLASSES)

def load_canopy(path: Path | str):
    """Read a .laz/.las and return (xy[N,2], z[N]) for canopy points (Z = height above ground)."""
    import laspy
    las = laspy.read(str(path))
    m = canopy_mask(np.asarray(las.classification))
    xy = np.column_stack([np.asarray(las.x)[m], np.asarray(las.y)[m]])
    return xy, np.asarray(las.z)[m]

def points_in_cylinder(xy: np.ndarray, z: np.ndarray, e: float, n: float, radius: float):
    """Subset points within `radius` (m) of (e, n) in XY. Returns (xy_sub, z_sub)."""
    m = (xy[:, 0] - e) ** 2 + (xy[:, 1] - n) ** 2 <= radius * radius
    return xy[m], z[m]

"""CHM local-maxima individual-tree detection (classical baseline).

What runs here is smoothed-CHM local maxima (Popescu-style), a different algorithm from watershed in
the ITD literature (Guangxi 2024 benchmarks them separately as LMA vs WA). Some CSVs and result
pages label it "CHM watershed"; actual watershed lives in `watershed.py` next to this file, and
`scripts/exp_watershed_baseline.py` runs the head-to-head.

On dense closed eucalyptus canopy this detects only ~30-43% of stems and structurally misses
overtopped/suppressed trees (no local maximum on the top surface); it is kept as a baseline and for
comparison. No CHM raster was delivered, so it is regenerated from the point cloud here.
"""
import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter

def build_chm(xy: np.ndarray, z: np.ndarray, res: float = 0.5):
    """Rasterize a canopy height model (max-Z per cell). Returns (chm, x0, y0, res)."""
    x0, y0 = float(xy[:, 0].min()), float(xy[:, 1].min())
    nx = int((xy[:, 0].max() - x0) / res) + 1
    ny = int((xy[:, 1].max() - y0) / res) + 1
    ix = ((xy[:, 0] - x0) / res).astype(int)
    iy = ((xy[:, 1] - y0) / res).astype(int)
    chm = np.zeros((nx, ny), np.float32)
    np.maximum.at(chm, (ix, iy), z.astype(np.float32))
    return chm, x0, y0, res

def detect_treetops(chm, x0, y0, res, window: int = 5, hmin: float = 5.0, smooth: float = 0.6):
    """Local-maxima tree-top detection. Returns array (M,3) of [x, y, height]."""
    sm = gaussian_filter(chm, smooth) if smooth else chm
    peaks = (sm == maximum_filter(sm, size=window)) & (chm > hmin)
    ix, iy = np.where(peaks)
    return np.column_stack([x0 + ix * res + res / 2, y0 + iy * res + res / 2, chm[ix, iy]])

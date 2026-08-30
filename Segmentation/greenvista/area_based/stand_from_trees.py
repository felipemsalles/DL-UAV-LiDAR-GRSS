"""ITD → stand-aggregate volume: the hybrid path — mediocre per tree, more robust stand totals.

Detect treetops (CHM local maxima), predict each tree's volume from its LiDAR height via an H→V
allometry fit on the destructive-scaling data (lnV = a + b·lnH, R²≈0.96 — DBH never used), and sum per plot → Vol_ha.
This is the standard way to turn per-tree detection into an inventory estimate (Zhou 2024 recovered
90.86% stand accuracy from per-tree DBH R²=0.84; Bilek 2025 kept population bias ≤5%), and it holds
only while detection is good. On a closed eucalyptus canopy the CHM structurally misses suppressed
stems (~30–43% detected here), so the raw aggregate under-counts and the estimate is negatively
biased; an optional stocking correction scales the sum to the expected stem count to gauge the
detection-limited ceiling.
"""
import numpy as np
import pandas as pd

from greenvista import config
from greenvista.segmentation import build_chm, detect_treetops

RADIUS_M = config.PLOT_RADIUS_M
PLOT_AREA_HA = config.PLOT_AREA_HA           # 12 m radius → 0.04524 ha


def fit_hv_allometry(volumes_csv=None):
    """ln V = a + b·ln H from the destructive-scaling data (per-tree height→volume; DBH not used). Returns [a, b]."""
    vol = pd.read_csv(volumes_csv or config.VOLUMES_CSV)
    h = pd.to_numeric(vol["H_m"], errors="coerce"); v = pd.to_numeric(vol["V"], errors="coerce")
    ok = h.notna() & v.notna() & (h > 0) & (v > 0)
    h, v = h[ok].to_numpy(), v[ok].to_numpy()
    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(h), np.log(h)]), np.log(v), rcond=None)
    return coef


def tree_volume(H, coef):
    a, b = coef
    return np.exp(a + b * np.log(np.clip(H, 1e-3, None)))


def plot_vol_ha(points, center, coef, radius=RADIUS_M, res=0.5, window=5, hmin=5.0,
                stocking_ha=None):
    """Detect treetops in the footprint, sum per-tree volumes → (vol_ha, n_detected).

    If `stocking_ha` is given, scale the sum to that expected stem density (the detection-limit
    correction) — use the field n_arv_ha to see what the hybrid could reach with perfect detection.
    """
    xy, z = points[:, :2], points[:, 2]
    chm, x0, y0, r = build_chm(xy, z, res)
    tops = detect_treetops(chm, x0, y0, r, window=window, hmin=hmin)
    d2 = (tops[:, 0] - center[0]) ** 2 + (tops[:, 1] - center[1]) ** 2
    tops = tops[d2 <= radius ** 2]
    if len(tops) == 0:
        return 0.0, 0
    plot_vol = float(tree_volume(tops[:, 2], coef).sum())
    if stocking_ha is not None:
        plot_vol *= (stocking_ha * PLOT_AREA_HA) / len(tops)   # scale detected → expected stems
    return plot_vol / PLOT_AREA_HA, len(tops)

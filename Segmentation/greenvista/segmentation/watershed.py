"""CHM watershed individual-tree detection — the second classical baseline.

Marker-controlled watershed on the (negated, smoothed) canopy height model, the `lidR::watershed()`
recipe: markers come from extended maxima (h-maxima) of the smoothed CHM, so basins shallower than
`tol` metres are merged before flooding; segmentation is constrained to the canopy mask `chm > hmin`.

`local_maxima.py` holds the local-maxima detector; this module is the actual watershed, so both
classical baselines can be measured with the same ruler on the same data.

Result on the 13 São Manuel plots (`scripts/exp_watershed_baseline.py`, 12 m counting circle, 717 field
stems): local maxima 414 (57.7%); watershed 416 (58.0%) with no basin merging, then 309 / 264 / 199 /
147 / 118 (43.1% → 16.5%) at tol = 0.25 / 0.5 / 1.0 / 1.5 / 2.0 m.

Two caveats:
  1. With markers = local maxima the watershed count is the marker count, so the two baselines are
     the same detector for counting and differ only in crown delineation; the 58.0% is that tie.
  2. Basin merging strictly hurts here. A closed eucalyptus canopy has no clear valley between
     neighbouring crowns, so adjacent basins fuse and two trees become one; at lidR's default
     tol=1 m detection more than halves. Watershed should therefore be reported at its best setting
     (58.0%); reporting the default would mirror the per-plot tuning criticised in Guangxi 2024.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, label

DEFAULT_TOL_M = 1.0     # lidR::watershed() default; see caveat 2 above


def detect_watershed(chm, x0, y0, res, hmin: float = 5.0, smooth: float = 0.6,
                     tol: float | None = DEFAULT_TOL_M):
    """Watershed crowns on a CHM. Returns array (M,3) of [x, y, height] at each segment's apex.

    `tol` is the h-maxima merge depth in metres: basins whose peak rises less than `tol` above the
    saddle joining them to a taller neighbour are merged. `tol=None` disables merging, giving one
    basin per local maximum (the configuration that ties the local-maxima baseline on this data).
    """
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed as _watershed
    from scipy.ndimage import maximum_filter

    sm = gaussian_filter(chm, smooth) if smooth else chm
    mask = chm > hmin
    if not mask.any():
        return np.empty((0, 3))

    if tol is None:
        seeds = (sm == maximum_filter(sm, size=3)) & mask
    else:
        seeds = h_maxima(sm * mask, tol) > 0
    markers, n = label(seeds)
    if n == 0:
        return np.empty((0, 3))

    lab = _watershed(-sm, markers=markers, mask=mask)
    out = []
    for i in range(1, lab.max() + 1):
        sel = lab == i
        if not sel.any():
            continue
        ix, iy = np.unravel_index(np.where(sel, chm, -1.0).argmax(), chm.shape)
        out.append([x0 + ix * res + res / 2, y0 + iy * res + res / 2, float(chm[ix, iy])])
    return np.array(out) if out else np.empty((0, 3))

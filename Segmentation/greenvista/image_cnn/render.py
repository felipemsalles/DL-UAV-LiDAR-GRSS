"""Render a plot's points into a multi-channel canopy image for the CNN.

The CHM channel stays in metres (clipped to the lidR window [zmin, zmax]) and is not per-plot
min-max normalized. Volume is dominated by absolute height (r(zmax, Vol_ha)=0.88), so normalizing it
away — which a DAP-only model can afford to do — discards the strongest signal. Standardization, if
any, is done globally at train time (dataset.py), never per image.

Channels (size×size each, masked to the inscribed circle):
  CHM     – max z per pixel, metres, clipped [0, zmax]   (canopy height surface)
  DENSITY – log1p(point count per pixel)                  (cover / gaps)
  SPREAD  – per-pixel z std                                (vertical structure)
"""
import numpy as np

from greenvista.features import structure

CHM, DENSITY, SPREAD = 0, 1, 2
N_CHANNELS = 3


def to_raster(points, center, radius, size=128, zmin=2.0, zmax=45.0, with_rumple=False):
    """Rasterize plot points → (raster (C,size,size) float32, scalars dict).

    `scalars` carries the absolute-height side-features the hybrid CNN concatenates at its dense
    head: {zmax, zmean, zsd} plus the structural proxies {rumple, cover, gap_fraction, penetration},
    which are near-free from the CHM/point heights and are strong volume predictors. Extra keys are
    ignored by callers that select a fixed subset (e.g. SCALAR_KEYS).

    `with_rumple=True` appends a 4th per-pixel canopy-slope channel (the pixel-level basis of the
    rumple index); N_CHANNELS stays 3 for the default 3-channel model contract.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    cx, cy = float(center[0]), float(center[1])
    radius = float(radius)
    n_ch = N_CHANNELS + (1 if with_rumple else 0)
    raster = np.zeros((n_ch, size, size), dtype=np.float32)
    empty_scalars = {"zmax": 0.0, "zmean": 0.0, "zsd": 0.0,
                     "rumple": 0.0, "cover": 0.0, "gap_fraction": 0.0, "penetration": 0.0}

    if len(pts) == 0:
        return raster, empty_scalars

    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    in_circle = dx * dx + dy * dy <= radius * radius
    in_window = (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
    p = pts[in_circle & in_window]
    if len(p) == 0:
        return raster, empty_scalars

    # map XY over the footprint bbox [c-r, c+r] to pixel indices
    px = np.clip(((p[:, 0] - (cx - radius)) / (2 * radius) * size).astype(int), 0, size - 1)
    py = np.clip(((p[:, 1] - (cy - radius)) / (2 * radius) * size).astype(int), 0, size - 1)
    z = p[:, 2].astype(np.float32)

    chm = np.zeros((size, size), np.float32)
    np.maximum.at(chm, (py, px), z)
    cnt = np.zeros((size, size), np.float32)
    np.add.at(cnt, (py, px), 1.0)
    s = np.zeros((size, size), np.float32)
    ss = np.zeros((size, size), np.float32)
    np.add.at(s, (py, px), z)
    np.add.at(ss, (py, px), z * z)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, s / cnt, 0.0)
        var = np.clip(np.where(cnt > 0, ss / cnt - mean * mean, 0.0), 0.0, None)
    chm_clipped = np.clip(chm, 0.0, zmax)
    raster[CHM] = chm_clipped                      # metres, not per-plot normalized
    raster[DENSITY] = np.log1p(cnt)
    raster[SPREAD] = np.sqrt(var).astype(np.float32)
    if with_rumple:                                # 4th channel = per-pixel canopy slope magnitude
        gy, gx = np.gradient(chm_clipped)
        raster[N_CHANNELS] = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)

    # circular mask (zero the corners outside the inscribed circle)
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    outside = (xx - c) ** 2 + (yy - c) ** 2 > (size / 2.0) ** 2
    raster[:, outside] = 0.0

    # structural scalars: rumple from the CHM surface, cover/gap/penetration from heights
    struct = structure.structural_scalars(chm_clipped, z, res=(2 * radius / size))
    scalars = {
        "zmax": float(z.max()),
        "zmean": float(z.mean()),
        "zsd": float(z.std(ddof=1)) if len(z) > 1 else 0.0,
        **{k: float(v) for k, v in struct.items()},
    }
    return raster, scalars

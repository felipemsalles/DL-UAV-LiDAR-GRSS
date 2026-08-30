"""Canopy structural metrics that proxy volume: rumple, cover / gap fraction, penetration ratio.

These are near-free from a rasterized CHM or a plot's height points and are established biomass /
volume predictors (surface roughness + cover fraction; e.g. Kane 2010 rumple, cover in every ABA
study). Array-pure, so they are trivially unit-tested and reusable by both the area-based feature
suite and the CNN raster/scalar branch.
"""
import numpy as np


def rumple_index(chm: np.ndarray, res: float = 1.0) -> float:
    """Rumple = canopy 3-D surface area / ground planar area (>= 1; higher = rougher canopy).

    `chm` is a 2-D max-height raster in metres, `res` the cell size (m). Triangulated-surface
    approximation via per-cell slope: area_cell = sqrt(1 + (dz/dx)^2 + (dz/dy)^2) * cell_area.
    """
    chm = np.asarray(chm, float)
    if chm.ndim != 2 or min(chm.shape) < 2:
        return float("nan")
    dzdy, dzdx = np.gradient(chm, res)
    cell_area = res * res
    surface = np.sqrt(1.0 + dzdx ** 2 + dzdy ** 2) * cell_area
    return float(surface.sum() / (chm.size * cell_area))


def canopy_cover(z: np.ndarray, height_break: float = 2.0) -> float:
    """Fraction of returns above `height_break` (m) — first-return canopy cover proxy."""
    z = np.asarray(z, float)
    return float((z > height_break).mean()) if z.size else float("nan")


def gap_fraction(z: np.ndarray, height_break: float = 2.0) -> float:
    """1 - canopy_cover: fraction of returns at/below the break (gaps / understory / ground)."""
    c = canopy_cover(z, height_break)
    return float("nan") if np.isnan(c) else 1.0 - c


def penetration_ratio(z: np.ndarray, lower_frac: float = 0.5) -> float:
    """Fraction of canopy returns in the lower `lower_frac` of the height range — how deep the
    pulse penetrates the crown (openness / occlusion proxy). Higher = more open canopy."""
    z = np.asarray(z, float)
    if z.size == 0:
        return float("nan")
    lo, hi = z.min(), z.max()
    if hi <= lo:
        return 1.0
    thr = lo + lower_frac * (hi - lo)
    return float((z <= thr).mean())


STRUCTURE_KEYS = ("rumple", "cover", "gap_fraction", "penetration")


def structural_scalars(chm: np.ndarray, z: np.ndarray, res: float = 1.0, height_break: float = 2.0) -> dict:
    """Bundle the structural metrics into a dict keyed by STRUCTURE_KEYS (order-stable)."""
    return {
        "rumple": rumple_index(chm, res),
        "cover": canopy_cover(z, height_break),
        "gap_fraction": gap_fraction(z, height_break),
        "penetration": penetration_ratio(z),
    }

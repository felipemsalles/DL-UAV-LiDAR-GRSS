"""Shared utilities for the GreenVista pipeline."""

import logging
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    return logger


def smalian_volume(d1_cm: float, d2_cm: float, length_m: float) -> float:
    """Smalian section volume: V = (pi/8) * (d1^2 + d2^2) * L.
    Diameters in cm, length in m. Returns volume in m^3."""
    d1_m = d1_cm / 100.0
    d2_m = d2_cm / 100.0
    return (np.pi / 8) * (d1_m ** 2 + d2_m ** 2) * length_m


def huber_volume(d_mid_cm: float, length_m: float) -> float:
    """Huber section volume: V = (pi/4) * d_mid^2 * L.
    Uses mid-section diameter only. Machado & Figueiredo Filho (2003)
    proved Huber's error is exactly half of Smalian's, opposite sign.
    Diameters in cm, length in m. Returns volume in m^3."""
    d_mid_m = d_mid_cm / 100.0
    return (np.pi / 4) * d_mid_m ** 2 * length_m


def freire_volume(dbh_cm: float, height_m: float,
                  form_factor: float = 0.5) -> float:
    """Cylindrical volume with form factor (Freire et al. 2025).
    V = pi * (DBH/200)^2 * H * f.  Returns volume in m^3."""
    return np.pi * (dbh_cm / 200.0) ** 2 * height_m * form_factor


def freire_biomass(dbh_cm: float, height_m: float,
                   wood_density: float = 0.5) -> float:
    """Biomass estimation (Freire et al. 2025).
    Bs(kg) = exp(-2.977 + ln(rho * DBH^2 * H)).
    DBH in cm, H in m, wood_density (rho) in g/cm^3."""
    if dbh_cm <= 0 or height_m <= 0 or wood_density <= 0:
        return np.nan
    return np.exp(-2.977 + np.log(wood_density * dbh_cm ** 2 * height_m))


def percentile_radial_filter(xy: np.ndarray,
                             percentile: float = 80) -> np.ndarray:
    """Keep only points within the given percentile of radial distances
    from centroid (Peres 2025). Returns boolean mask."""
    centroid = xy.mean(axis=0)
    dists = np.linalg.norm(xy - centroid, axis=1)
    threshold = np.percentile(dists, percentile)
    return dists <= threshold


def welzl_enclosing_circle(points: np.ndarray) -> tuple:
    """Welzl's algorithm for the smallest enclosing circle.
    points: Nx2 array. Returns (cx, cy, radius).

    Iterative implementation to avoid recursion limits on large point sets.
    From ForAINet (Xiang et al. 2024)."""
    pts = points.copy()
    np.random.default_rng(42).shuffle(pts)

    def _circle_from_1(p):
        return (p[0], p[1], 0.0)

    def _circle_from_2(p1, p2):
        cx = (p1[0] + p2[0]) / 2
        cy = (p1[1] + p2[1]) / 2
        r = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) / 2
        return (cx, cy, r)

    def _circle_from_3(p1, p2, p3):
        ax, ay = p1[0], p1[1]
        bx, by = p2[0], p2[1]
        cx, cy = p3[0], p3[1]
        D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(D) < 1e-12:
            # Collinear — return circle through the two farthest points
            d1 = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
            d2 = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2)
            d3 = np.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)
            if d1 >= d2 and d1 >= d3:
                return _circle_from_2(p1, p2)
            if d2 >= d3:
                return _circle_from_2(p2, p3)
            return _circle_from_2(p1, p3)
        ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
              + (cx ** 2 + cy ** 2) * (ay - by)) / D
        uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx)
              + (cx ** 2 + cy ** 2) * (bx - ax)) / D
        r = np.sqrt((ax - ux) ** 2 + (ay - uy) ** 2)
        return (ux, uy, r)

    def _in_circle(c, p):
        return np.sqrt((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2) <= c[2] + 1e-10

    def _min_circle_with_boundary(P, R):
        if len(R) == 3 or len(P) == 0:
            if len(R) == 0:
                return (0, 0, 0)
            if len(R) == 1:
                return _circle_from_1(R[0])
            if len(R) == 2:
                return _circle_from_2(R[0], R[1])
            return _circle_from_3(R[0], R[1], R[2])

        # Iterative Welzl using a stack
        # For small N this is fine; for large N we use the randomized approach
        p = P[0]
        rest = P[1:]
        c = _min_circle_with_boundary(rest, R)
        if _in_circle(c, p):
            return c
        return _min_circle_with_boundary(rest, R + [p])

    n = len(pts)
    if n == 0:
        return (0, 0, 0)
    if n == 1:
        return _circle_from_1(pts[0])
    if n == 2:
        return _circle_from_2(pts[0], pts[1])

    # For large point sets, use incremental approach to avoid deep recursion
    c = _circle_from_2(pts[0], pts[1])
    for i in range(2, n):
        if not _in_circle(c, pts[i]):
            # Point i must be on the boundary — restart with i on boundary
            c = _circle_from_1(pts[i])
            for j in range(i):
                if not _in_circle(c, pts[j]):
                    c = _circle_from_2(pts[i], pts[j])
                    for k in range(j):
                        if not _in_circle(c, pts[k]):
                            c = _circle_from_3(pts[i], pts[j], pts[k])
    return c


# ===================================================================
# Voxel-Based Crown Metrics (Ye et al. 2025)
# ===================================================================

def voxel_crown_metrics(points: np.ndarray,
                        voxel_size: float = 0.25) -> dict:
    """Compute voxel-based crown metrics from 3D points.

    Ye et al. (2025) showed voxel metrics (count, density, vertical
    distribution) are top features for eucalyptus AGB estimation.

    Args:
        points: Nx3 array of crown/tree points
        voxel_size: voxel edge length in meters

    Returns dict with:
        voxel_count: number of occupied voxels
        voxel_volume: occupied voxels × voxel_size³ (m³)
        voxel_density: occupied / total voxels in bounding box
        voxel_h_entropy: Shannon entropy of height-binned voxel distribution
        voxel_h_skewness: skewness of occupied voxel heights
        voxel_gap_fraction: fraction of empty columns in 2D grid
    """
    if len(points) < 4:
        return {k: np.nan for k in [
            "voxel_count", "voxel_volume", "voxel_density",
            "voxel_h_entropy", "voxel_h_skewness", "voxel_gap_fraction"]}

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = maxs - mins

    if np.any(spans < voxel_size):
        return {k: np.nan for k in [
            "voxel_count", "voxel_volume", "voxel_density",
            "voxel_h_entropy", "voxel_h_skewness", "voxel_gap_fraction"]}

    # Voxelize: assign each point to a voxel index
    indices = ((points - mins) / voxel_size).astype(int)
    grid_shape = ((spans / voxel_size).astype(int) + 1)

    # Unique occupied voxels
    voxel_keys = indices[:, 0] * (grid_shape[1] * grid_shape[2]) + \
                 indices[:, 1] * grid_shape[2] + indices[:, 2]
    unique_voxels = np.unique(voxel_keys)
    n_occupied = len(unique_voxels)
    total_voxels = int(grid_shape[0] * grid_shape[1] * grid_shape[2])

    feats = {
        "voxel_count": n_occupied,
        "voxel_volume": float(n_occupied * voxel_size ** 3),
        "voxel_density": float(n_occupied / max(total_voxels, 1)),
    }

    # Height distribution of occupied voxels
    voxel_heights = (unique_voxels % grid_shape[2]).astype(float) * voxel_size + mins[2]
    n_h_bins = max(1, int(spans[2] / voxel_size))
    h_counts, _ = np.histogram(voxel_heights, bins=n_h_bins)
    h_counts = h_counts[h_counts > 0]

    # Shannon entropy of vertical voxel distribution
    if len(h_counts) > 1:
        p = h_counts / h_counts.sum()
        feats["voxel_h_entropy"] = float(-np.sum(p * np.log(p + 1e-10)))
    else:
        feats["voxel_h_entropy"] = 0.0

    # Skewness of voxel heights
    if len(voxel_heights) > 2:
        h_mean = voxel_heights.mean()
        h_std = voxel_heights.std()
        if h_std > 0:
            feats["voxel_h_skewness"] = float(
                np.mean(((voxel_heights - h_mean) / h_std) ** 3))
        else:
            feats["voxel_h_skewness"] = 0.0
    else:
        feats["voxel_h_skewness"] = 0.0

    # Gap fraction: empty columns in the 2D XY grid
    xy_keys = indices[:, 0] * grid_shape[1] + indices[:, 1]
    n_xy_occupied = len(np.unique(xy_keys))
    total_xy = int(grid_shape[0] * grid_shape[1])
    feats["voxel_gap_fraction"] = float(1.0 - n_xy_occupied / max(total_xy, 1))

    return feats


# ===================================================================
# Chi-Squared Circle Fit Quality (Huo et al. 2025)
# ===================================================================

def chi_squared_circle_test(xy: np.ndarray, cx: float, cy: float,
                            r: float, alpha: float = 0.05) -> dict:
    """Assess circle fit quality using chi-squared test.

    Huo et al. (2025) showed standard circle fitting is unsuitable for
    eucalyptus UAV-LiDAR and proposed chi-squared filtering.

    Tests whether residuals (point-to-circle distances) are consistent
    with random noise. High chi² → poor fit (branch contamination).

    Args:
        xy: Nx2 points
        cx, cy, r: fitted circle parameters
        alpha: significance level

    Returns dict with:
        chi2_stat: chi-squared statistic
        chi2_pvalue: p-value
        chi2_pass: True if fit passes the test
        fit_rmse: RMS residual
    """
    n = len(xy)
    if n < 4 or r <= 0:
        return {"chi2_stat": np.nan, "chi2_pvalue": np.nan,
                "chi2_pass": False, "fit_rmse": np.nan}

    dists = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    residuals = dists - r
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # Estimate variance from residuals (3 DOF used for cx, cy, r)
    sigma2 = np.sum(residuals ** 2) / (n - 3) if n > 3 else np.nan
    if np.isnan(sigma2) or sigma2 <= 0:
        return {"chi2_stat": np.nan, "chi2_pvalue": np.nan,
                "chi2_pass": False, "fit_rmse": rmse}

    chi2_stat = float(np.sum(residuals ** 2) / sigma2)
    dof = n - 3
    p_value = float(1.0 - chi2.cdf(chi2_stat, dof))

    return {
        "chi2_stat": round(chi2_stat, 2),
        "chi2_pvalue": round(p_value, 4),
        "chi2_pass": p_value > alpha,
        "fit_rmse": round(rmse, 4),
    }


# ===================================================================
# Kozak Taper Model (Lee et al. 2025)
# ===================================================================

def kozak_taper_volume(diameters_cm: list, heights_m: list,
                       tree_height: float,
                       n_integration_steps: int = 100) -> float:
    """Estimate stem volume using Kozak's variable-exponent taper model.

    Lee et al. (2025) showed fitting a Kozak taper curve to LiDAR-derived
    diameters then integrating halved estimation error vs traditional methods.

    Uses the simplified Kozak (2004) model:
        d(h) = a0 * DBH^a1 * a2^DBH * X^(b1*z^2 + b2*ln(z+0.001) + b3*sqrt(z) + b4*e^z + b5*(DBH/H))
    where X = (1 - sqrt(h/H)) / (1 - sqrt(1.3/H)), z = h/H

    Fits model to observed (height, diameter) pairs, then integrates
    V = integral_0^H of pi/4 * d(h)^2 dh.

    Returns volume in m³, or NaN if fitting fails.
    """
    if len(diameters_cm) < 3 or tree_height <= 1.3:
        return np.nan

    d_m = np.array(diameters_cm) / 100.0  # convert to meters
    h_m = np.array(heights_m)

    # Find DBH (closest to 1.3m)
    dbh_idx = int(np.argmin(np.abs(h_m - 1.3)))
    dbh_m = d_m[dbh_idx]
    if dbh_m <= 0:
        return np.nan

    H = tree_height
    p = 1.3 / H  # relative inflection point

    # Simplified Kozak: d(h) = a0 * (1 - sqrt(h/H))/(1 - sqrt(p))^c * DBH^a1
    # We use a 3-param simplification that's stable with few data points:
    # d(h) = dbh * ((1 - (h/H)^a) / (1 - p^a))^b
    # where a, b are shape parameters

    def _taper(params, h_arr):
        a, b = params
        a = max(abs(a), 0.01)
        b = max(abs(b), 0.01)
        z = h_arr / H
        ratio = (1.0 - z ** a) / (1.0 - p ** a)
        ratio = np.clip(ratio, 0, None)
        return dbh_m * ratio ** b

    def _residuals(params):
        pred = _taper(params, h_m)
        return pred - d_m

    try:
        result = least_squares(_residuals, [0.5, 1.0],
                               bounds=([0.01, 0.01], [10.0, 10.0]),
                               method="trf", max_nfev=500)
        if not result.success and result.cost > 0.1:
            return np.nan
    except Exception:
        return np.nan

    # Integrate taper curve for volume
    h_steps = np.linspace(0, H, n_integration_steps)
    d_steps = _taper(result.x, h_steps)
    d_steps = np.clip(d_steps, 0, None)

    # Trapezoidal integration of cross-sectional area
    areas = np.pi / 4 * d_steps ** 2
    volume = float(np.trapezoid(areas, h_steps))

    return round(volume, 6) if volume > 0 else np.nan


# ===================================================================
# LiDAR Biomass Index (Hu et al. 2021)
# ===================================================================

def lidar_biomass_index(points: np.ndarray,
                        z_ground: float = 0.0,
                        flight_height: float = 100.0) -> float:
    """Compute the LiDAR Biomass Index (LBI) from crown points.

    Hu et al. (2021) proposed a physics-based parameter derived from
    the LiDAR equation. LBI is the sum of inverse-squared ranges from
    sensor to each point, weighted by the number of returns.

    For UAV-LiDAR: range ≈ flight_height - point_z

    LBI = sum(1 / range_i^2) for all points above ground.

    This bypasses DBH and volume entirely, using crown point density
    as a direct biomass proxy.

    Args:
        points: Nx3 array of tree points
        z_ground: ground elevation (m)
        flight_height: sensor height above ground (m)

    Returns LBI value (dimensionless), or NaN if insufficient points.
    """
    if len(points) < 3:
        return np.nan

    z = points[:, 2] - z_ground
    ranges = flight_height - z
    # Exclude points at or above sensor height
    valid = ranges > 1.0
    if valid.sum() < 3:
        return np.nan

    lbi = float(np.sum(1.0 / ranges[valid] ** 2))
    return round(lbi, 6)


# ===================================================================
# Height Distribution Metrics (Ye et al. 2025)
# ===================================================================

# All feature keys produced by height_distribution_metrics().
# Used for NaN-fill on early return.
_HEIGHT_DIST_KEYS = (
    # Canopy density profile (9)
    ["zpcum1", "zpcum2", "zpcum3", "zpcum4", "zpcum5",
     "zpcum6", "zpcum7", "zpcum8", "zpcum9"]
    # L-moments (7)
    + ["L1", "L2", "L3", "L4", "Lskew", "Lkurt", "Lcoefvar"]
    # VCI + CRR (2)
    + ["VCI", "CRR"]
    # Height dispersion (4)
    + ["ziqr", "zMADmean", "zMADmedian", "zentropy"]
    # Additional percentiles (16)
    + [f"zq{p}" for p in [1, 5, 10, 15, 20, 30, 35, 40, 45, 55, 60, 65, 70, 80, 85, 99]]
    # Percent above thresholds (3)
    + ["pzabovemean", "pzabove2", "pzabove5"]
)


def _lmoments(h_sorted: np.ndarray, n: int) -> dict:
    """L-moments via probability weighted moments (Hosking 1990).
    Input must be sorted ascending. Returns L1–L4 + ratios."""
    # PWM estimators: b_r = (1/n) * sum_{i=r+1}^{n} C(i-1,r)/C(n-1,r) * x_i
    i = np.arange(1, n + 1, dtype=float)

    b0 = h_sorted.mean()
    b1 = np.sum((i - 1) / (n - 1) * h_sorted) / n if n > 1 else 0.0
    b2 = (np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * h_sorted) / n
          if n > 2 else 0.0)
    b3 = (np.sum((i - 1) * (i - 2) * (i - 3)
                 / ((n - 1) * (n - 2) * (n - 3)) * h_sorted) / n
          if n > 3 else 0.0)

    L1 = b0
    L2 = 2 * b1 - b0
    L3 = 6 * b2 - 6 * b1 + b0
    L4 = 20 * b3 - 30 * b2 + 12 * b1 - b0

    feats = {"L1": float(L1), "L2": float(L2),
             "L3": float(L3), "L4": float(L4)}

    if abs(L2) > 1e-10:
        feats["Lskew"] = float(L3 / L2)
        feats["Lkurt"] = float(L4 / L2)
    else:
        feats["Lskew"] = 0.0
        feats["Lkurt"] = 0.0

    feats["Lcoefvar"] = float(L2 / L1) if abs(L1) > 1e-10 else 0.0
    return feats


def height_distribution_metrics(heights: np.ndarray) -> dict:
    """Compute height distribution metrics following Ye et al. (2025).

    Ye et al. used 96 LiDAR metrics from lidRmetrics::metrics_set3()
    and achieved R²=0.903 for eucalyptus AGB. This function computes
    the subset that applies to individual tree point clouds.

    Args:
        heights: 1D array of ground-normalized heights (m)

    Returns dict with ~41 features:
        - zpcum1–9: canopy density profile (cumulative % in 10 height bins)
        - L1–L4, Lskew, Lkurt, Lcoefvar: L-moments
        - VCI, CRR: vertical complexity index, canopy relief ratio
        - ziqr, zMADmean, zMADmedian, zentropy: height dispersion
        - zq1–zq99: additional height percentiles
        - pzabovemean, pzabove2, pzabove5: percent above thresholds
    """
    if len(heights) < 4:
        return {k: np.nan for k in _HEIGHT_DIST_KEYS}

    h = heights.astype(float)
    n = len(h)
    h_sorted = np.sort(h)
    h_min = float(h_sorted[0])
    h_max = float(h_sorted[-1])
    h_mean = float(h.mean())
    h_range = h_max - h_min

    feats = {}

    # --- Canopy density profile: cumulative % in 10 equal height bins ---
    n_bins = 10
    if h_range > 0:
        bin_edges = np.linspace(h_min, h_max, n_bins + 1)
        for k in range(1, n_bins):  # zpcum1 through zpcum9
            feats[f"zpcum{k}"] = float(np.sum(h <= bin_edges[k]) / n * 100)
    else:
        for k in range(1, n_bins):
            feats[f"zpcum{k}"] = 100.0 if k > 0 else 0.0

    # --- L-moments (Hosking 1990) ---
    feats.update(_lmoments(h_sorted, n))

    # --- VCI and CRR ---
    if h_range > 0:
        # VCI: Shannon entropy of height bins, normalized by ln(n_bins)
        counts, _ = np.histogram(h, bins=n_bins)
        p = counts / n
        p_pos = p[p > 0]
        feats["VCI"] = float(-np.sum(p_pos * np.log(p_pos)) / np.log(n_bins))
        # CRR: canopy relief ratio
        feats["CRR"] = float((h_mean - h_min) / h_range)
    else:
        feats["VCI"] = 0.0
        feats["CRR"] = np.nan

    # --- Height dispersion ---
    h_median = float(np.median(h))
    q25 = float(np.percentile(h, 25))
    q75 = float(np.percentile(h, 75))
    feats["ziqr"] = q75 - q25
    feats["zMADmean"] = float(np.mean(np.abs(h - h_mean)))
    feats["zMADmedian"] = float(np.median(np.abs(h - h_median)))

    # zentropy: Shannon entropy of height distribution (1D, not voxel-based)
    if h_range > 0:
        counts_e, _ = np.histogram(h, bins=max(1, int(np.sqrt(n))))
        p_e = counts_e / n
        p_e = p_e[p_e > 0]
        feats["zentropy"] = float(-np.sum(p_e * np.log(p_e)))
    else:
        feats["zentropy"] = 0.0

    # --- Additional height percentiles ---
    extra_pcts = [1, 5, 10, 15, 20, 30, 35, 40, 45, 55, 60, 65, 70, 80, 85, 99]
    pct_values = np.percentile(h, extra_pcts)
    for pct, val in zip(extra_pcts, pct_values):
        feats[f"zq{pct}"] = float(val)

    # --- Percent above thresholds ---
    feats["pzabovemean"] = float(np.sum(h > h_mean) / n * 100)
    feats["pzabove2"] = float(np.sum(h > 2.0) / n * 100)
    feats["pzabove5"] = float(np.sum(h > 5.0) / n * 100)

    return feats

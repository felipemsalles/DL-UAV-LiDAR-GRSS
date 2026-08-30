#!/usr/bin/env python3
"""
Post-segmentation pipeline for FF3D output.

Trunk audit, diameter extraction and feature table: reads a panoptic PLY from
ForestFormer3D and produces a per-tree metrics CSV plus plots.

Usage:
    conda run -n greenvista python scripts/tree_analysis.py \
        --input /tmp/ff3d_output/Yuchen_..._panoptic_test_round2.ply \
        --output /tmp/ff3d_output/analysis/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plyfile import PlyData
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError

import hdbscan

from common import (setup_logger, smalian_volume, huber_volume,
                     freire_volume, freire_biomass,
                     percentile_radial_filter, welzl_enclosing_circle,
                     voxel_crown_metrics, chi_squared_circle_test,
                     kozak_taper_volume, lidar_biomass_index,
                     height_distribution_metrics)

log = setup_logger("tree_analysis")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BIN_SIZE = 0.5  # meters — height bin for trunk slicing
MIN_POINTS_VIABLE = 10  # minimum wood points per bin for diameter fitting
OUTLIER_STD = 2.0  # std-dev threshold for XY outlier removal
MIN_ANGULAR_COVERAGE = 180  # degrees — skip bin if angular spread is less
ROUNDNESS_THRESHOLD = 0.06  # meters (6 cm) — circle vs ellipse
MAX_PLAUSIBLE_DIAMETER = 2.0  # meters — reject fits above this

# RANSAC parameters
RANSAC_ITERATIONS = 1000  # number of random samples
RANSAC_INLIER_THRESHOLD = 0.02  # meters — max distance from circle to be inlier
RANSAC_MIN_INLIER_RATIO = 0.5  # fraction of points that must be inliers

# Outlier filter mode: "std" (2-std-dev), "p80" (percentile 80), "hdbscan"
OUTLIER_FILTER = "p80"  # default to Peres (2025) method
P80_PERCENTILE = 80  # for percentile filter

# Adaptive bin expansion (ForAINet)
ADAPTIVE_BIN_EXPANSION = True  # try wider bins when sparse
ADAPTIVE_MAX_WIDTH = 1.0  # max bin width in meters

# Brazilian rigorous scaling (cubagem rigorosa) section heights
# Base sections: 0, 0.1, 0.3, 0.7, 1.3m, then every 1m up to tree height
SCALING_BASE_HEIGHTS = [0.0, 0.1, 0.3, 0.7, 1.3]
SCALING_TOLERANCE = 0.25  # meters — half-window around target height for point selection

# Semantic labels from FF3D
SEM_GROUND = 0
SEM_WOOD = 1
SEM_LEAF = 2


# ===================================================================
# Step 0: Load & Extract Individual Trees
# ===================================================================

def load_ply(path: str) -> dict:
    """Read PLY and split into per-tree dicts."""
    ply = PlyData.read(path)
    el = ply.elements[0]
    data = {p.name: np.array(el[p.name]) for p in el.properties}
    xyz = np.column_stack([data["x"], data["y"], data["z"]])
    instance = data["instance_pred"]
    semantic = data["semantic_pred"]
    score = data["score"]

    trees = {}
    for iid in np.unique(instance):
        if iid == -1:
            continue
        mask = instance == iid
        pts = xyz[mask]
        sem = semantic[mask]
        trees[int(iid)] = {
            "points": pts,
            "semantic": sem,
            "score": float(score[mask][0]),
            "wood_points": pts[sem == SEM_WOOD],
            "leaf_points": pts[sem == SEM_LEAF],
            "ground_points": pts[sem == SEM_GROUND],
        }
    log.info("Loaded %d trees from %s", len(trees), Path(path).name)
    return trees


# ===================================================================
# Step 1: Trunk Point Audit — density profiles
# ===================================================================

def normalize_height(tree: dict) -> float:
    """Return ground-level z for this tree. Uses ground points if available,
    otherwise min z of all tree points."""
    gp = tree["ground_points"]
    if len(gp) > 0:
        return float(gp[:, 2].min())
    return float(tree["points"][:, 2].min())


def filter_wood_hdbscan(wood_points: np.ndarray,
                        min_cluster_size: int = 10) -> np.ndarray:
    """Use HDBSCAN to identify the dominant trunk cluster among wood points.
    Returns filtered points belonging to the largest cluster.
    From ForAINet (Xiang et al. 2024) — HDBSCAN separates true stem points
    from branch/noise points before computing diameters.

    Falls back to returning all points if HDBSCAN finds no clusters or
    if there are too few points."""
    if len(wood_points) < min_cluster_size:
        return wood_points

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=3)
    labels = clusterer.fit_predict(wood_points)

    if labels.max() < 0:
        # No clusters found — return all points
        return wood_points

    # Find the largest cluster (most likely the trunk)
    unique_labels = [l for l in np.unique(labels) if l >= 0]
    if not unique_labels:
        return wood_points

    largest = max(unique_labels, key=lambda l: (labels == l).sum())
    mask = labels == largest
    filtered = wood_points[mask]

    if len(filtered) < min_cluster_size:
        return wood_points

    return filtered


def trunk_density_profile(tree: dict) -> dict:
    """Slice wood points into height bins, return profile."""
    z0 = normalize_height(tree)
    wood = tree["wood_points"]
    if len(wood) == 0:
        return {"bins": np.array([]), "counts": np.array([]),
                "viable_mask": np.array([], dtype=bool), "z0": z0}

    h = wood[:, 2] - z0
    h_max = h.max()
    edges = np.arange(0, h_max + BIN_SIZE, BIN_SIZE)
    counts, _ = np.histogram(h, bins=edges)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    viable = counts >= MIN_POINTS_VIABLE

    return {
        "bins": bin_centers,
        "counts": counts,
        "edges": edges,
        "viable_mask": viable,
        "z0": z0,
    }


# ===================================================================
# Step 2: Multi-Height Diameter Extraction
# ===================================================================

def angular_coverage(xy: np.ndarray) -> float:
    """Compute angular spread (degrees) of XY points around centroid."""
    c = xy.mean(axis=0)
    angles = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
    angles_sorted = np.sort(angles)
    # Find the largest gap between consecutive angles
    diffs = np.diff(angles_sorted)
    # Also check wraparound gap
    wraparound = 2 * np.pi - (angles_sorted[-1] - angles_sorted[0])
    max_gap = max(np.max(diffs), wraparound)
    coverage = 360.0 - np.degrees(max_gap)
    return coverage


def fit_circle_kasa(xy: np.ndarray) -> tuple:
    """Algebraic circle fit (Kasa method), then one Nelder-Mead refinement.
    Returns (cx, cy, r, residual)."""
    n = len(xy)
    A_mat = np.column_stack([xy, np.ones(n)])
    b_vec = xy[:, 0] ** 2 + xy[:, 1] ** 2
    result, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    cx0 = result[0] / 2
    cy0 = result[1] / 2
    r0 = np.sqrt(result[2] + cx0 ** 2 + cy0 ** 2)

    def cost(params):
        cx, cy, r = params
        dists = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
        return np.sum((dists - r) ** 2)

    res = minimize(cost, [cx0, cy0, r0], method="Nelder-Mead",
                   options={"maxiter": 2000})
    cx, cy, r = res.x
    residual = np.sqrt(res.fun / n)
    return cx, cy, abs(r), residual


def _circle_from_3pts(p1, p2, p3):
    """Compute circle (cx, cy, r) through 3 non-collinear points."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    D = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(D) < 1e-12:
        return None
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
          + (cx ** 2 + cy ** 2) * (ay - by)) / D
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx)
          + (cx ** 2 + cy ** 2) * (bx - ax)) / D
    r = np.sqrt((ax - ux) ** 2 + (ay - uy) ** 2)
    return ux, uy, r


def fit_circle_ransac(xy: np.ndarray) -> tuple:
    """RANSAC circle fit (ForAINet/Peres). Falls back to Kasa+NM if RANSAC
    fails to find a good consensus. Returns (cx, cy, r, residual)."""
    n = len(xy)
    if n < 3:
        return fit_circle_kasa(xy)

    best_inliers = None
    best_count = 0
    rng = np.random.default_rng(42)

    for _ in range(RANSAC_ITERATIONS):
        idx = rng.choice(n, 3, replace=False)
        result = _circle_from_3pts(xy[idx[0]], xy[idx[1]], xy[idx[2]])
        if result is None:
            continue
        cx, cy, r = result
        if r > MAX_PLAUSIBLE_DIAMETER / 2:
            continue
        dists = np.abs(np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2) - r)
        inlier_mask = dists < RANSAC_INLIER_THRESHOLD
        count = inlier_mask.sum()
        if count > best_count:
            best_count = count
            best_inliers = inlier_mask

    # Fall back to Kasa if RANSAC didn't find enough inliers
    if best_inliers is None or best_count < RANSAC_MIN_INLIER_RATIO * n:
        return fit_circle_kasa(xy)

    # Refine on inliers using Kasa+NM
    return fit_circle_kasa(xy[best_inliers])


def fit_ellipse(xy: np.ndarray) -> tuple:
    """Least-squares ellipse fit. Returns (cx, cy, a, b, theta, residual).
    Falls back to circle fit if optimizer diverges."""
    c0 = xy.mean(axis=0)
    dists = np.linalg.norm(xy - c0, axis=1)
    a0 = np.max(dists)
    b0 = np.median(dists)
    r_max = MAX_PLAUSIBLE_DIAMETER  # bound semi-axes

    def cost(params):
        cx, cy, a, b, theta = params
        a, b = abs(a), abs(b)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx = xy[:, 0] - cx
        dy = xy[:, 1] - cy
        u = cos_t * dx + sin_t * dy
        v = -sin_t * dx + cos_t * dy
        ang = np.arctan2(v, u)
        denom = np.sqrt((b * np.cos(ang)) ** 2 + (a * np.sin(ang)) ** 2)
        denom = np.maximum(denom, 1e-10)
        r_ellipse = a * b / denom
        r_point = np.sqrt(u ** 2 + v ** 2)
        return np.sum((r_point - r_ellipse) ** 2)

    res = minimize(cost, [c0[0], c0[1], a0, b0, 0.0], method="Nelder-Mead",
                   options={"maxiter": 5000})
    cx, cy, a, b, theta = res.x
    a, b = abs(a), abs(b)
    if b > a:
        a, b = b, a
        theta += np.pi / 2

    # If ellipse fit diverged, fall back to circle
    if a > r_max or b > r_max:
        cx, cy, r, residual = fit_circle_ransac(xy)
        return cx, cy, r, r, 0.0, residual

    residual = np.sqrt(res.fun / len(xy))
    return cx, cy, a, b, theta, residual


def _fit_slice(xy_raw: np.ndarray, height_m: float) -> dict | None:
    """Fit a circle or ellipse to a 2D cross-section slice.
    Returns measurement dict or None if fitting fails quality checks.

    RANSAC handles outlier rejection internally, so no pre-cleaning is done
    before circle fitting. Outlier removal only applies before ellipse fitting
    (which uses least-squares and is sensitive to outliers).
    """
    if len(xy_raw) < MIN_POINTS_VIABLE:
        return None

    xy = xy_raw.copy()

    # Angular coverage check (on raw points)
    cov = angular_coverage(xy)
    if cov < MIN_ANGULAR_COVERAGE:
        return None

    # Roundness check -> circle or ellipse
    centroid = xy.mean(axis=0)
    radii = np.linalg.norm(xy - centroid, axis=1)
    roundness = radii.max() - radii.min()

    if roundness > ROUNDNESS_THRESHOLD:
        # Ellipse fit uses least-squares — remove outliers before fitting
        if OUTLIER_FILTER == "p80":
            mask = percentile_radial_filter(xy, P80_PERCENTILE)
        else:  # "std"
            dists = np.linalg.norm(xy - centroid, axis=1)
            mask = (dists < dists.mean() + OUTLIER_STD * dists.std()
                    if dists.std() > 0 else np.ones(len(xy), dtype=bool))
        xy_clean = xy[mask]
        if len(xy_clean) < MIN_POINTS_VIABLE:
            return None
        xy = xy_clean
        cx, cy, a, b, theta, residual = fit_ellipse(xy)
        diameter = a + b  # mean diameter (semi-axes -> diameter = 2*mean_semi)
        fit_type = "ellipse"
    else:
        # RANSAC handles outliers internally — pass raw points
        cx, cy, r, residual = fit_circle_ransac(xy)
        diameter = 2 * r
        fit_type = "circle"

    # Reject implausible fits
    if diameter > MAX_PLAUSIBLE_DIAMETER:
        return None

    # Chi-squared quality test (Huo et al. 2025)
    if fit_type == "circle":
        chi2_result = chi_squared_circle_test(xy, cx, cy, diameter / 2)
    else:
        # For ellipse, test against equivalent circle
        chi2_result = chi_squared_circle_test(xy, cx, cy, diameter / 2)

    return {
        "height_m": round(height_m, 2),
        "diameter_cm": round(diameter * 100, 1),
        "fit_type": fit_type,
        "residual_cm": round(residual * 100, 2),
        "n_points": len(xy),
        "angular_coverage_deg": round(cov, 1),
        "chi2_pass": chi2_result["chi2_pass"],
        "chi2_pvalue": chi2_result["chi2_pvalue"],
    }


def extract_diameters(tree: dict, profile: dict) -> list:
    """Fit circles/ellipses at each viable height bin.
    Supports HDBSCAN pre-filtering and adaptive bin expansion."""
    results = []
    wood = tree["wood_points"]
    if len(wood) == 0 or len(profile["bins"]) == 0:
        return results

    # Optional HDBSCAN pre-filtering of wood points
    if OUTLIER_FILTER == "hdbscan":
        wood = filter_wood_hdbscan(wood)

    z0 = profile["z0"]
    edges = profile["edges"]
    h = wood[:, 2] - z0

    for i, viable in enumerate(profile["viable_mask"]):
        lo, hi = edges[i], edges[i + 1]
        height_mid = (lo + hi) / 2

        if viable:
            mask = (h >= lo) & (h < hi)
            result = _fit_slice(wood[mask][:, :2], height_mid)
            if result is not None:
                results.append(result)
                continue

        # Adaptive bin expansion (ForAINet): try wider bins when sparse
        if ADAPTIVE_BIN_EXPANSION:
            bin_width = hi - lo
            while bin_width < ADAPTIVE_MAX_WIDTH:
                bin_width += BIN_SIZE
                expand = (bin_width - (hi - lo)) / 2
                lo_exp = lo - expand
                hi_exp = hi + expand
                mask = (h >= lo_exp) & (h < hi_exp)
                if mask.sum() >= MIN_POINTS_VIABLE:
                    result = _fit_slice(wood[mask][:, :2], height_mid)
                    if result is not None:
                        results.append(result)
                        break

    return results


def scaling_section_heights(tree_height: float) -> list:
    """Generate Brazilian rigorous scaling section heights for a given tree.
    Base sections: 0, 0.1, 0.3, 0.7, 1.3m, then every 1m to tree top."""
    heights = [h for h in SCALING_BASE_HEIGHTS if h < tree_height]
    h = 2.0
    while h < tree_height:
        heights.append(h)
        h += 1.0
    return heights


def extract_diameters_at_sections(tree: dict, z0: float) -> list:
    """Fit circles/ellipses at standard scaling section heights.
    Uses a tolerance window around each target height."""
    wood = tree["wood_points"]
    if len(wood) == 0:
        return []

    h = wood[:, 2] - z0
    tree_height = tree["points"][:, 2].max() - z0
    sections = scaling_section_heights(tree_height)
    tol = SCALING_TOLERANCE

    results = []
    for target_h in sections:
        mask = (h >= target_h - tol) & (h < target_h + tol)
        pts_slice = wood[mask]
        result = _fit_slice(pts_slice[:, :2], target_h)
        if result is not None:
            results.append(result)

    return results


# ===================================================================
# Step 3: Crown + Trunk Feature Extraction
# ===================================================================

def crown_features(tree: dict, z0: float) -> dict:
    """Compute crown-related features from leaf points."""
    pts = tree["points"]
    leaf = tree["leaf_points"]
    wood = tree["wood_points"]
    h_all = pts[:, 2] - z0

    feats = {
        "H_max": float(h_all.max()),
        "H_min": float(h_all.min()),
        "n_leaf_points": len(leaf),
        "n_wood_points": len(wood),
        "n_total_points": len(pts),
        "wood_ratio": len(wood) / max(len(pts), 1),
    }

    # LBI from all tree points (Hu et al. 2021 — physics-based biomass proxy)
    feats["lbi"] = lidar_biomass_index(pts, z_ground=z0)

    # Voxel metrics from all tree points (Ye et al. 2025)
    vox = voxel_crown_metrics(pts)
    feats.update(vox)

    if len(leaf) < 3:
        # Not enough leaf points for meaningful stats
        for k in ["H_mean", "H_std", "H_cv", "H_p25", "H_p50", "H_p75",
                   "H_p90", "H_p95", "crown_diameter", "crown_area",
                   "crown_volume", "crown_base_height", "crown_length"]:
            feats[k] = np.nan
        feats.update(height_distribution_metrics(np.array([])))
        return feats

    h_leaf = leaf[:, 2] - z0
    feats["H_mean"] = float(h_leaf.mean())
    feats["H_std"] = float(h_leaf.std())
    feats["H_cv"] = feats["H_std"] / feats["H_mean"] if feats["H_mean"] > 0 else np.nan
    for p, name in [(25, "H_p25"), (50, "H_p50"), (75, "H_p75"),
                     (90, "H_p90"), (95, "H_p95")]:
        feats[name] = float(np.percentile(h_leaf, p))

    feats["crown_base_height"] = float(h_leaf.min())
    feats["crown_length"] = feats["H_max"] - feats["crown_base_height"]

    # Height distribution metrics (Ye et al. 2025)
    feats.update(height_distribution_metrics(h_leaf))

    # 2D convex hull for crown area + Welzl's circle for crown diameter
    xy_leaf = leaf[:, :2]
    try:
        hull2d = ConvexHull(xy_leaf)
        feats["crown_area"] = float(hull2d.volume)  # in 2D, .volume = area
    except QhullError:
        feats["crown_area"] = np.nan

    # Welzl's smallest enclosing circle (ForAINet — tighter than max hull dist)
    if len(xy_leaf) >= 2:
        cx_w, cy_w, r_w = welzl_enclosing_circle(xy_leaf)
        feats["crown_diameter"] = float(2 * r_w)
    else:
        feats["crown_diameter"] = np.nan

    # 3D convex hull for volume
    if len(leaf) >= 4:
        try:
            hull3d = ConvexHull(leaf)
            feats["crown_volume"] = float(hull3d.volume)
        except QhullError:
            feats["crown_volume"] = np.nan
    else:
        feats["crown_volume"] = np.nan

    return feats


def compute_section_volumes(diameters: list) -> tuple:
    """Compute Smalian and Huber volumes between consecutive measurements.
    Returns (smalian_total_m3, huber_total_m3, list of per-section dicts)."""
    if len(diameters) < 2:
        return np.nan, np.nan, []

    # Sort by height
    sorted_d = sorted(diameters, key=lambda d: d["height_m"])
    sections = []
    smalian_total = 0.0
    huber_total = 0.0
    for i in range(len(sorted_d) - 1):
        d1 = sorted_d[i]
        d2 = sorted_d[i + 1]
        length = d2["height_m"] - d1["height_m"]
        if length <= 0:
            continue
        s_vol = smalian_volume(d1["diameter_cm"], d2["diameter_cm"], length)
        d_mid = (d1["diameter_cm"] + d2["diameter_cm"]) / 2
        h_vol = huber_volume(d_mid, length)
        smalian_total += s_vol
        huber_total += h_vol
        sections.append({
            "h_bottom": d1["height_m"],
            "h_top": d2["height_m"],
            "d_bottom_cm": d1["diameter_cm"],
            "d_top_cm": d2["diameter_cm"],
            "length_m": round(length, 2),
            "volume_m3": round(s_vol, 6),
            "huber_volume_m3": round(h_vol, 6),
        })
    return round(smalian_total, 6), round(huber_total, 6), sections


def trunk_features(diameters: list, tree_height: float = np.nan) -> dict:
    """Summarize diameter measurements, section volumes, simple volume,
    Hossfeldt diameter, and biomass."""
    feats = {}
    if not diameters:
        feats["d_1.3m"] = np.nan
        feats["d_hossfeldt"] = np.nan
        feats["d_max"] = np.nan
        feats["d_mean"] = np.nan
        feats["n_viable_heights"] = 0
        feats["viable_height_range"] = 0.0
        feats["smalian_volume_m3"] = np.nan
        feats["huber_volume_m3"] = np.nan
        feats["kozak_volume_m3"] = np.nan
        feats["n_smalian_sections"] = 0
        feats["freire_volume_m3"] = np.nan
        feats["biomass_kg"] = np.nan
        feats["n_chi2_passed"] = 0
        feats["chi2_pass_ratio"] = 0.0
        return feats

    heights = [d["height_m"] for d in diameters]
    diams = [d["diameter_cm"] for d in diameters]

    feats["n_viable_heights"] = len(diameters)
    feats["viable_height_range"] = max(heights) - min(heights)
    feats["d_max"] = max(diams)
    feats["d_mean"] = float(np.mean(diams))

    # DBH: find measurement closest to 1.3m
    closest_idx = int(np.argmin([abs(h - 1.3) for h in heights]))
    if abs(heights[closest_idx] - 1.3) <= BIN_SIZE:
        feats["d_1.3m"] = diams[closest_idx]
    else:
        feats["d_1.3m"] = np.nan

    # Hossfeldt diameter: at 1/3 of total height (Peres 2025)
    if not np.isnan(tree_height) and tree_height > 0:
        h_target = tree_height / 3.0
        hoss_idx = int(np.argmin([abs(h - h_target) for h in heights]))
        if abs(heights[hoss_idx] - h_target) <= BIN_SIZE * 2:
            feats["d_hossfeldt"] = diams[hoss_idx]
        else:
            feats["d_hossfeldt"] = np.nan
    else:
        feats["d_hossfeldt"] = np.nan

    # Chi-squared quality stats (Huo et al. 2025)
    n_passed = sum(1 for d in diameters if d.get("chi2_pass", False))
    feats["n_chi2_passed"] = n_passed
    feats["chi2_pass_ratio"] = round(n_passed / len(diameters), 2)

    # Section volumes (Smalian + Huber)
    smalian_vol, huber_vol, sections = compute_section_volumes(diameters)
    feats["smalian_volume_m3"] = smalian_vol
    feats["huber_volume_m3"] = huber_vol
    feats["n_smalian_sections"] = len(sections)

    # Kozak taper volume (Lee et al. 2025) — more robust than Smalian
    if not np.isnan(tree_height) and len(diameters) >= 3:
        feats["kozak_volume_m3"] = kozak_taper_volume(
            diams, heights, tree_height)
    else:
        feats["kozak_volume_m3"] = np.nan

    # Freire simple volume: V = pi*(DBH/200)^2*H*0.5
    if not np.isnan(feats["d_1.3m"]) and not np.isnan(tree_height):
        feats["freire_volume_m3"] = round(
            freire_volume(feats["d_1.3m"], tree_height), 6)
    else:
        feats["freire_volume_m3"] = np.nan

    # Biomass (Freire 2025): Bs = exp(-2.977 + ln(rho * DBH^2 * H))
    if not np.isnan(feats["d_1.3m"]) and not np.isnan(tree_height):
        feats["biomass_kg"] = round(
            freire_biomass(feats["d_1.3m"], tree_height), 2)
    else:
        feats["biomass_kg"] = np.nan

    return feats


def build_feature_table(trees: dict, profiles: dict,
                        all_diameters: dict) -> pd.DataFrame:
    """Build per-tree feature DataFrame with centroid coords and
    section-level diameter features for XGBoost."""
    rows = []
    for tid in sorted(trees.keys()):
        tree = trees[tid]
        prof = profiles[tid]
        z0 = prof["z0"]
        diams = all_diameters[tid]

        # Centroid from all points (for GPS matching)
        pts = tree["points"]
        row = {
            "instance_id": tid,
            "score": tree["score"],
            "centroid_x": float(pts[:, 0].mean()),
            "centroid_y": float(pts[:, 1].mean()),
            "centroid_z": float(pts[:, 2].mean()),
        }
        crown = crown_features(tree, z0)
        row.update(crown)
        row.update(trunk_features(diams, tree_height=crown["H_max"]))

        # Section-level diameter features (pivoted):
        # d_at_Xm for standard heights — these are the key XGBoost features
        # per Peres 2025 (section volumes > DBH for volume prediction)
        if diams:
            for d in diams:
                h = d["height_m"]
                # Round to nearest 0.5m for column naming
                h_key = round(h * 2) / 2
                col = f"d_at_{h_key:.1f}m"
                row[col] = d["diameter_cm"]

        rows.append(row)

    return pd.DataFrame(rows)


# ===================================================================
# Step 4: Visualization
# ===================================================================

def plot_trunk_profiles(trees: dict, profiles: dict, out_dir: Path):
    """Bar chart of wood point count vs height for each tree."""
    tree_ids = sorted(profiles.keys())
    n = len(tree_ids)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                             squeeze=False)

    for idx, tid in enumerate(tree_ids):
        ax = axes[idx // cols][idx % cols]
        prof = profiles[tid]
        if len(prof["bins"]) == 0:
            ax.text(0.5, 0.5, "No wood points", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            colors = ["#2ecc71" if v else "#bdc3c7"
                      for v in prof["viable_mask"]]
            ax.bar(prof["bins"], prof["counts"], width=BIN_SIZE * 0.8,
                   color=colors)
            ax.axhline(MIN_POINTS_VIABLE, color="red", ls="--", lw=0.8,
                       label=f"threshold={MIN_POINTS_VIABLE}")
            ax.legend(fontsize=7)
        n_wood = len(trees[tid]["wood_points"])
        ax.set_title(f"Tree {tid} ({n_wood} wood pts)", fontsize=9)
        ax.set_xlabel("Height (m)", fontsize=8)
        ax.set_ylabel("Wood points", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Trunk Point Density Profiles", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "trunk_profiles.png", dpi=150)
    plt.close(fig)
    log.info("Saved trunk_profiles.png")


def plot_diameter_vs_height(all_diameters: dict, out_dir: Path):
    """Scatter/line plot of diameter vs height for trees with measurements."""
    has_data = {tid: d for tid, d in all_diameters.items() if d}
    if not has_data:
        log.info("No diameter measurements — skipping diameter_vs_height.png")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for tid, diams in sorted(has_data.items()):
        heights = [d["height_m"] for d in diams]
        ds = [d["diameter_cm"] for d in diams]
        marker = "o" if diams[0]["fit_type"] == "circle" else "s"
        ax.plot(heights, ds, marker=marker, label=f"Tree {tid}", markersize=5)

    ax.set_xlabel("Height (m)")
    ax.set_ylabel("Diameter (cm)")
    ax.set_title("Diameter vs Height")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "diameter_vs_height.png", dpi=150)
    plt.close(fig)
    log.info("Saved diameter_vs_height.png")


def plot_feature_correlation(df: pd.DataFrame, out_dir: Path):
    """Correlation matrix of numeric features."""
    numeric = df.select_dtypes(include=[np.number]).drop(
        columns=["instance_id"], errors="ignore")
    # Drop columns that are all NaN
    numeric = numeric.dropna(axis=1, how="all")

    if numeric.shape[1] < 2:
        log.info("Not enough features for correlation — skipping")
        return

    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_correlation.png", dpi=150)
    plt.close(fig)
    log.info("Saved feature_correlation.png")


def plot_3d_pyvista(trees: dict, out_dir: Path):
    """Interactive 3D viz colored by instance and semantic class."""
    try:
        import pyvista as pv
    except ImportError:
        log.warning("PyVista not available — skipping 3D viz")
        return

    # Color by instance
    all_pts = []
    all_instance = []
    all_semantic = []
    for tid, tree in sorted(trees.items()):
        all_pts.append(tree["points"])
        all_instance.append(np.full(len(tree["points"]), tid))
        all_semantic.append(tree["semantic"])

    pts = np.vstack(all_pts)
    instances = np.concatenate(all_instance)
    semantics = np.concatenate(all_semantic)

    cloud = pv.PolyData(pts)
    cloud["instance"] = instances
    cloud["semantic"] = semantics

    # Save as VTK for later interactive use
    vtk_path = out_dir / "trees_colored.vtk"
    cloud.save(str(vtk_path))
    log.info("Saved %s (open in ParaView or PyVista)", vtk_path.name)


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-segmentation analysis of FF3D output")
    parser.add_argument("--input", required=True,
                        help="Path to FF3D panoptic PLY file")
    parser.add_argument("--output", required=True,
                        help="Output directory for results")
    parser.add_argument("--section-heights", action="store_true",
                        help="Also extract diameters at Brazilian scaling "
                             "section heights (0, 0.1, 0.3, 0.7, 1.3, 2, 3...m)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Load
    log.info("Step 0: Loading PLY...")
    trees = load_ply(args.input)

    # Step 1: Trunk audit
    log.info("Step 1: Trunk density profiles...")
    profiles = {}
    for tid in sorted(trees.keys()):
        profiles[tid] = trunk_density_profile(trees[tid])
        n_viable = profiles[tid]["viable_mask"].sum() if len(profiles[tid]["viable_mask"]) else 0
        n_wood = len(trees[tid]["wood_points"])
        log.info("  Tree %2d: %4d wood pts, %d viable height bins",
                 tid, n_wood, n_viable)

    # Step 2: Diameter extraction
    log.info("Step 2: Diameter extraction...")
    all_diameters = {}
    for tid in sorted(trees.keys()):
        diams = extract_diameters(trees[tid], profiles[tid])
        all_diameters[tid] = diams
        if diams:
            log.info("  Tree %2d: %d measurements, diameters %s cm",
                     tid, len(diams), [d['diameter_cm'] for d in diams])
        else:
            log.info("  Tree %2d: no viable measurements", tid)

    # Step 2b: Section-height diameter extraction (for validation vs ground truth)
    if args.section_heights:
        log.info("Step 2b: Diameter extraction at scaling section heights...")
        section_rows = []
        for tid in sorted(trees.keys()):
            z0 = profiles[tid]["z0"]
            diams = extract_diameters_at_sections(trees[tid], z0)
            n = len(diams)
            msg = f"  Tree {tid:2d}: {n} sections measured"
            if n:
                msg += f", heights {[d['height_m'] for d in diams]}"
            log.info(msg)
            for d in diams:
                d["instance_id"] = tid
                section_rows.append(d)
        if section_rows:
            df_sections = pd.DataFrame(section_rows)
            sec_path = out_dir / "diameters_at_sections.csv"
            df_sections.to_csv(sec_path, index=False)
            log.info("Saved %s (%d measurements)", sec_path, len(df_sections))
        else:
            log.info("No section-height measurements (not enough wood points)")

    # Step 3: Feature table
    log.info("Step 3: Building feature table...")
    df = build_feature_table(trees, profiles, all_diameters)
    csv_path = out_dir / "features.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved %s (%d trees, %d features)", csv_path, len(df),
             len(df.columns))

    # Step 4: Visualization
    log.info("Step 4: Generating plots...")
    plot_trunk_profiles(trees, profiles, out_dir)
    plot_diameter_vs_height(all_diameters, out_dir)
    plot_feature_correlation(df, out_dir)
    plot_3d_pyvista(trees, out_dir)

    # Summary
    log.info("=== Summary ===")
    log.info("\n%s", df[["instance_id", "score", "H_max", "n_wood_points",
              "n_viable_heights", "d_1.3m", "crown_diameter",
              "smalian_volume_m3", "freire_volume_m3"]].to_string(
        index=False))
    log.info("Done. Results in %s", out_dir)


if __name__ == "__main__":
    main()

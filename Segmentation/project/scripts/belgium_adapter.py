#!/usr/bin/env python3
"""
Belgium Destructive Biomass dataset adapter.

Loads individual TLS tree point clouds (XYZ only), extracts features
using the same functions as tree_analysis.py, matches to destructive
harvest ground truth, and runs the XGBoost volume prediction pipeline.

Validates the full pipeline on real data with measured volumes
(Demol et al. 2021, Zenodo 4557401).

Usage:
    cd project/scripts
    conda run -n greenvista python belgium_adapter.py \
        --data-dir ../data/belgium_biomass \
        --output ../data/belgium_biomass/results/
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import xgboost as xgb

# Import from production code — no duplication
import shap

from common import (setup_logger, smalian_volume, huber_volume,
                     freire_volume, freire_biomass,
                     welzl_enclosing_circle,
                     voxel_crown_metrics, kozak_taper_volume, lidar_biomass_index,
                     height_distribution_metrics)
from tree_analysis import (
    _fit_slice, MIN_POINTS_VIABLE, BIN_SIZE, ADAPTIVE_BIN_EXPANSION,
    ADAPTIVE_MAX_WIDTH,
)

log = setup_logger("belgium_adapter")

# ---------------------------------------------------------------------------
# Belgium-specific constants
# ---------------------------------------------------------------------------
# Height-based trunk/crown split (since no semantic labels)
# 40% gave R²=0.854 in the documented validation
CROWN_BASE_RATIO = 0.4


# ===================================================================
# Data Loading
# ===================================================================

def load_point_cloud(path: Path) -> np.ndarray:
    """Load a single tree point cloud (space-delimited XYZ)."""
    return np.loadtxt(path)


def parse_tree_name(filename: str) -> str:
    """Convert filename 'PSYLA1.txt' to ground truth name 'PSYLA-01'."""
    match = re.match(r"([A-Z]+)(\d+)\.txt", filename)
    if not match:
        return filename.replace(".txt", "")
    return f"{match.group(1)}-{int(match.group(2)):02d}"


def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    """Load Belgium ground truth CSV. Volume in dm³ → m³."""
    df = pd.read_csv(csv_path)
    for col in ["Volume_total_tree_harvested", "Volume_stem_harvested"]:
        if col in df.columns:
            df[col + "_m3"] = df[col] / 1000.0
    return df


# ===================================================================
# Height-Based Trunk/Crown Split
# ===================================================================

def split_trunk_crown(points: np.ndarray) -> dict:
    """Split tree point cloud into trunk and crown zones using height.

    Belgium data has no semantic labels, so we use height threshold:
    - Below CROWN_BASE_RATIO × tree_height → trunk (wood)
    - Above → crown (leaf)

    Returns a dict compatible with tree_analysis.py functions.
    """
    z_min = float(points[:, 2].min())
    z_max = float(points[:, 2].max())
    tree_height = z_max - z_min

    crown_base_z = z_min + CROWN_BASE_RATIO * tree_height
    trunk_mask = points[:, 2] <= crown_base_z
    crown_mask = ~trunk_mask

    return {
        "points": points,
        "wood_points": points[trunk_mask],
        "leaf_points": points[crown_mask],
        "ground_points": np.empty((0, 3)),
        "score": 1.0,
        "semantic": np.zeros(len(points), dtype=int),
    }


# ===================================================================
# Feature Extraction — reuses tree_analysis.py functions
# ===================================================================

def extract_diameters_belgium(tree: dict, z0: float) -> list:
    """Extract diameters from wood points with adaptive bin expansion.
    Uses _fit_slice from tree_analysis.py (includes P80, RANSAC, chi²)."""
    wood = tree["wood_points"]
    if len(wood) < MIN_POINTS_VIABLE:
        return []

    h = wood[:, 2] - z0
    h_max = h.max()
    if h_max <= 0:
        return []
    edges = np.arange(0, h_max + BIN_SIZE, BIN_SIZE)

    results = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        height_mid = (lo + hi) / 2
        mask = (h >= lo) & (h < hi)

        if mask.sum() >= MIN_POINTS_VIABLE:
            result = _fit_slice(wood[mask][:, :2], height_mid)
            if result is not None:
                results.append(result)
                continue

        # Adaptive bin expansion
        if ADAPTIVE_BIN_EXPANSION:
            bin_width = hi - lo
            while bin_width < ADAPTIVE_MAX_WIDTH:
                bin_width += BIN_SIZE
                expand = (bin_width - (hi - lo)) / 2
                mask = (h >= lo - expand) & (h < hi + expand)
                if mask.sum() >= MIN_POINTS_VIABLE:
                    result = _fit_slice(wood[mask][:, :2], height_mid)
                    if result is not None:
                        results.append(result)
                        break

    return results


def crown_features_belgium(tree: dict, z0: float) -> dict:
    """Crown features for Belgium trees. Uses all production functions."""
    pts = tree["points"]
    leaf = tree["leaf_points"]
    wood = tree["wood_points"]
    h_all = pts[:, 2] - z0

    feats = {
        "H_max": float(h_all.max()),
        "H_min": float(h_all.min()),
        "n_total_points": len(pts),
        "n_wood_points": len(wood),
        "n_leaf_points": len(leaf),
        "wood_ratio": len(wood) / max(len(pts), 1),
    }

    # LBI (Hu et al. 2021)
    feats["lbi"] = lidar_biomass_index(pts, z_ground=z0)

    # Voxel metrics (Ye et al. 2025)
    vox = voxel_crown_metrics(pts)
    feats.update(vox)

    if len(leaf) < 3:
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

    # Crown area from 2D convex hull
    xy_leaf = leaf[:, :2]
    try:
        hull2d = ConvexHull(xy_leaf)
        feats["crown_area"] = float(hull2d.volume)
    except QhullError:
        feats["crown_area"] = np.nan

    # Welzl's circle for crown diameter (ForAINet)
    if len(xy_leaf) >= 2:
        _, _, r_w = welzl_enclosing_circle(xy_leaf)
        feats["crown_diameter"] = float(2 * r_w)
    else:
        feats["crown_diameter"] = np.nan

    # 3D convex hull for crown volume
    if len(leaf) >= 4:
        try:
            hull3d = ConvexHull(leaf)
            feats["crown_volume"] = float(hull3d.volume)
        except QhullError:
            feats["crown_volume"] = np.nan
    else:
        feats["crown_volume"] = np.nan

    return feats


def trunk_features_belgium(diameters: list, tree_height: float) -> dict:
    """Trunk features including Hossfeldt, Kozak taper, chi² stats."""
    feats = {}
    if not diameters:
        for k in ["d_1.3m", "d_hossfeldt", "d_max", "d_mean",
                   "n_viable_heights", "viable_height_range",
                   "smalian_volume_m3", "huber_volume_m3", "kozak_volume_m3",
                   "n_smalian_sections", "freire_volume_m3",
                   "biomass_kg", "n_chi2_passed", "chi2_pass_ratio"]:
            feats[k] = np.nan if k not in ("n_viable_heights",
                                            "viable_height_range",
                                            "n_smalian_sections",
                                            "n_chi2_passed") else 0
        feats["chi2_pass_ratio"] = 0.0
        return feats

    heights = [d["height_m"] for d in diameters]
    diams = [d["diameter_cm"] for d in diameters]

    feats["n_viable_heights"] = len(diameters)
    feats["viable_height_range"] = max(heights) - min(heights)
    feats["d_max"] = max(diams)
    feats["d_mean"] = float(np.mean(diams))

    # DBH at 1.3m
    closest_idx = int(np.argmin([abs(h - 1.3) for h in heights]))
    feats["d_1.3m"] = diams[closest_idx] if abs(heights[closest_idx] - 1.3) <= BIN_SIZE else np.nan

    # Hossfeldt diameter at h/3 (Peres 2025)
    if tree_height > 0:
        h_target = tree_height / 3.0
        hoss_idx = int(np.argmin([abs(h - h_target) for h in heights]))
        feats["d_hossfeldt"] = diams[hoss_idx] if abs(heights[hoss_idx] - h_target) <= BIN_SIZE * 2 else np.nan
    else:
        feats["d_hossfeldt"] = np.nan

    # Chi-squared quality stats (Huo 2025)
    n_passed = sum(1 for d in diameters if d.get("chi2_pass", False))
    feats["n_chi2_passed"] = n_passed
    feats["chi2_pass_ratio"] = round(n_passed / len(diameters), 2)

    # Smalian volume
    sorted_d = sorted(diameters, key=lambda d: d["height_m"])
    total_vol = 0.0
    n_sections = 0
    for i in range(len(sorted_d) - 1):
        length = sorted_d[i + 1]["height_m"] - sorted_d[i]["height_m"]
        if length > 0:
            total_vol += smalian_volume(sorted_d[i]["diameter_cm"],
                                        sorted_d[i + 1]["diameter_cm"], length)
            n_sections += 1
    feats["smalian_volume_m3"] = round(total_vol, 6) if n_sections > 0 else np.nan
    feats["n_smalian_sections"] = n_sections

    # Huber volume — uses mid-section diameter (Machado 2003: half Smalian's error)
    huber_total = 0.0
    n_huber = 0
    for i in range(len(sorted_d) - 1):
        length = sorted_d[i + 1]["height_m"] - sorted_d[i]["height_m"]
        if length > 0:
            d_mid = (sorted_d[i]["diameter_cm"] + sorted_d[i + 1]["diameter_cm"]) / 2
            huber_total += huber_volume(d_mid, length)
            n_huber += 1
    feats["huber_volume_m3"] = round(huber_total, 6) if n_huber > 0 else np.nan

    # Kozak taper volume (Lee et al. 2025)
    if tree_height > 1.3 and len(diameters) >= 3:
        feats["kozak_volume_m3"] = kozak_taper_volume(diams, heights, tree_height)
    else:
        feats["kozak_volume_m3"] = np.nan

    # Freire volume
    if not np.isnan(feats["d_1.3m"]):
        feats["freire_volume_m3"] = round(freire_volume(feats["d_1.3m"], tree_height), 6)
    else:
        feats["freire_volume_m3"] = np.nan

    # Biomass (Freire 2025)
    if not np.isnan(feats["d_1.3m"]):
        feats["biomass_kg"] = round(freire_biomass(feats["d_1.3m"], tree_height), 2)
    else:
        feats["biomass_kg"] = np.nan

    return feats


def extract_all_features(tree: dict, tree_name: str) -> dict:
    """Extract all features for a single tree."""
    z0 = float(tree["points"][:, 2].min())
    diams = extract_diameters_belgium(tree, z0)
    h = float(tree["points"][:, 2].max()) - z0

    row = {"tree_name": tree_name}
    row.update(crown_features_belgium(tree, z0))
    row.update(trunk_features_belgium(diams, h))

    # Section-level diameter features (pivoted for XGBoost)
    for d in diams:
        h_key = round(d["height_m"] * 2) / 2
        row[f"d_at_{h_key:.1f}m"] = d["diameter_cm"]

    return row


# ===================================================================
# XGBoost Pipeline
# ===================================================================

EXCLUDE_FEATURES = {
    "tree_name", "gt_tree_name", "gt_site",
    # Volume estimates are targets/baselines, not features
    "smalian_volume_m3", "huber_volume_m3", "freire_volume_m3", "kozak_volume_m3",
    "biomass_kg",
    # Ground truth columns
    "gt_volume_m3", "gt_stem_volume_m3", "gt_dbh_cm",
    "gt_height_m", "gt_fresh_mass_kg", "gt_wsg",
}

# Trunk-derived features — excluded in crown-only mode (airborne UAV simulation)
TRUNK_FEATURES = {
    "d_1.3m", "d_hossfeldt", "d_max", "d_mean",
    "n_viable_heights", "viable_height_range",
    "n_smalian_sections", "n_chi2_passed", "chi2_pass_ratio",
}

XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}


def prepare_features(df: pd.DataFrame, target_col: str,
                     crown_only: bool = False) -> tuple:
    """Select features for XGBoost. Returns (X, y, feature_names).

    If crown_only=True, excludes all trunk-derived features and
    section-level diameter columns (d_at_*), simulating airborne UAV.
    """
    exclude = EXCLUDE_FEATURES | (TRUNK_FEATURES if crown_only else set())
    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and c != target_col
                    and not c.startswith("gt_")
                    and not (crown_only and c.startswith("d_at_"))]
    X = df[feature_cols].copy()
    y = df[target_col].values

    X = X[X.columns[X.notna().any()]]
    X = X[X.columns[X.notna().sum() >= len(X) * 0.5]]
    feature_names = list(X.columns)

    mode = "CROWN-ONLY" if crown_only else "ALL"
    log.info("Features (%s): %d — %s", mode, len(feature_names), feature_names)
    log.info("Target: %s, N=%d, range: %.4f – %.4f m³",
             target_col, len(y), y.min(), y.max())
    return X.values, y, feature_names


def train_and_evaluate(X, y, feature_names, out_dir: Path, target_name: str):
    """Train XGBoost with LOOCV."""
    model = xgb.XGBRegressor(**XGB_PARAMS)
    y_pred = cross_val_predict(model, X, y, cv=LeaveOneOut())

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    bias = np.mean(y_pred - y)
    rrmse = (rmse / np.mean(y)) * 100

    metrics = {"R2": round(r2, 4), "RMSE_m3": round(rmse, 6),
               "rRMSE_pct": round(rrmse, 2), "MAE_m3": round(mae, 6),
               "bias_m3": round(bias, 6), "N": len(y)}

    log.info("=" * 60)
    log.info("XGBoost LOOCV (%s, N=%d):", target_name, len(y))
    log.info("  R²=%.4f  RMSE=%.4f m³  rRMSE=%.1f%%  MAE=%.4f  Bias=%.4f",
             r2, rmse, rrmse, mae, bias)

    model.fit(X, y)
    perm = permutation_importance(model, X, y, n_repeats=30,
                                  random_state=42, scoring="r2")
    importance = sorted(zip(feature_names, perm.importances_mean),
                        key=lambda x: x[1], reverse=True)
    log.info("Feature Importance (top 10):")
    for name, imp in importance[:10]:
        log.info("  %30s  %.4f", name, imp)

    # Plots
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y, y_pred, s=40, alpha=0.7, edgecolors="k", linewidths=0.5)
    lims = [min(y.min(), y_pred.min()) * 0.9, max(y.max(), y_pred.max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    z = np.polyfit(y, y_pred, 1)
    ax.plot(np.linspace(*lims, 100), np.poly1d(z)(np.linspace(*lims, 100)),
            "r-", lw=1.5, alpha=0.7, label=f"y={z[0]:.2f}x+{z[1]:.4f}")
    ax.set(xlabel="Observed Volume (m³)", ylabel="Predicted Volume (m³)",
           xlim=lims, ylim=lims, aspect="equal")
    ax.set_title(f"XGBoost LOOCV — {target_name}\n"
                 f"R²={r2:.3f}  RMSE={rmse:.4f}  rRMSE={rrmse:.1f}%")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / f"pred_vs_obs_{target_name}.png", dpi=150)
    plt.close(fig)

    residuals = y_pred - y
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(y_pred, residuals, s=40, alpha=0.7, edgecolors="k", linewidths=0.5)
    axes[0].axhline(0, color="k", ls="--"); axes[0].set(xlabel="Predicted", ylabel="Residual")
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(residuals, bins=15, edgecolor="k", alpha=0.7)
    axes[1].axvline(0, color="k", ls="--"); axes[1].set(xlabel="Residual")
    axes[1].set_title(f"bias={bias:.4f} m³"); axes[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / f"residuals_{target_name}.png", dpi=150)
    plt.close(fig)

    top = importance[:15]
    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.4)))
    ax.barh([x[0] for x in top][::-1], [x[1] for x in top][::-1],
            color="#3498db", edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Permutation Importance")
    ax.set_title(f"Feature Importance — {target_name}")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / f"importance_{target_name}.png", dpi=150)
    plt.close(fig)

    return model, y_pred, metrics, importance


def compute_baselines(df, y_true, out_dir):
    """Compute baselines including Kozak taper."""
    baselines = {}

    # Freire formula with GT DBH
    if "gt_dbh_cm" in df.columns and "gt_height_m" in df.columns:
        dbh = df["gt_dbh_cm"].values
        h = df["gt_height_m"].values
        mask = ~(np.isnan(dbh) | np.isnan(h))
        if mask.sum() > 0:
            y_freire = np.pi * (dbh / 200) ** 2 * h * 0.5
            baselines["Freire (GT DBH)"] = {
                "R2": round(r2_score(y_true[mask], y_freire[mask]), 4),
                "RMSE_m3": round(np.sqrt(mean_squared_error(y_true[mask], y_freire[mask])), 6),
                "rRMSE_pct": round(np.sqrt(mean_squared_error(y_true[mask], y_freire[mask]))
                                   / np.mean(y_true[mask]) * 100, 2),
            }

    # Smalian from LiDAR
    if "smalian_volume_m3" in df.columns:
        smal = df["smalian_volume_m3"].values
        mask = ~np.isnan(smal)
        if mask.sum() > 2:
            baselines["Smalian (LiDAR)"] = {
                "R2": round(r2_score(y_true[mask], smal[mask]), 4),
                "RMSE_m3": round(np.sqrt(mean_squared_error(y_true[mask], smal[mask])), 6),
                "rRMSE_pct": round(np.sqrt(mean_squared_error(y_true[mask], smal[mask]))
                                   / np.mean(y_true[mask]) * 100, 2),
            }

    # Huber from LiDAR (Machado 2003)
    if "huber_volume_m3" in df.columns:
        hub = df["huber_volume_m3"].values
        mask = ~np.isnan(hub)
        if mask.sum() > 2:
            baselines["Huber (LiDAR)"] = {
                "R2": round(r2_score(y_true[mask], hub[mask]), 4),
                "RMSE_m3": round(np.sqrt(mean_squared_error(y_true[mask], hub[mask])), 6),
                "rRMSE_pct": round(np.sqrt(mean_squared_error(y_true[mask], hub[mask]))
                                   / np.mean(y_true[mask]) * 100, 2),
            }

    # Kozak taper from LiDAR (Lee 2025)
    if "kozak_volume_m3" in df.columns:
        koz = df["kozak_volume_m3"].values
        mask = ~np.isnan(koz)
        if mask.sum() > 2:
            baselines["Kozak taper (LiDAR)"] = {
                "R2": round(r2_score(y_true[mask], koz[mask]), 4),
                "RMSE_m3": round(np.sqrt(mean_squared_error(y_true[mask], koz[mask])), 6),
                "rRMSE_pct": round(np.sqrt(mean_squared_error(y_true[mask], koz[mask]))
                                   / np.mean(y_true[mask]) * 100, 2),
            }

    # Mean volume
    y_mean = np.full_like(y_true, y_true.mean())
    baselines["Mean volume"] = {
        "R2": 0.0,
        "RMSE_m3": round(np.sqrt(mean_squared_error(y_true, y_mean)), 6),
        "rRMSE_pct": round(np.sqrt(mean_squared_error(y_true, y_mean))
                           / np.mean(y_true) * 100, 2),
    }

    # Linear H_max
    if "H_max" in df.columns:
        h_max = df["H_max"].values.reshape(-1, 1)
        mask = ~np.isnan(h_max.ravel())
        if mask.sum() > 2:
            lr = LinearRegression()
            y_lr = cross_val_predict(lr, h_max[mask], y_true[mask], cv=LeaveOneOut())
            baselines["Linear H_max"] = {
                "R2": round(r2_score(y_true[mask], y_lr), 4),
                "RMSE_m3": round(np.sqrt(mean_squared_error(y_true[mask], y_lr)), 6),
                "rRMSE_pct": round(np.sqrt(mean_squared_error(y_true[mask], y_lr))
                                   / np.mean(y_true[mask]) * 100, 2),
            }

    # Linear CD+H
    if "crown_diameter" in df.columns and "H_max" in df.columns:
        X_lr = df[["crown_diameter", "H_max"]].values
        mask = ~np.isnan(X_lr).any(axis=1)
        if mask.sum() > 2:
            lr = LinearRegression()
            y_lr = cross_val_predict(lr, X_lr[mask], y_true[mask], cv=LeaveOneOut())
            baselines["Linear CD+H"] = {
                "R2": round(r2_score(y_true[mask], y_lr), 4),
                "RMSE_m3": round(np.sqrt(mean_squared_error(y_true[mask], y_lr)), 6),
                "rRMSE_pct": round(np.sqrt(mean_squared_error(y_true[mask], y_lr))
                                   / np.mean(y_true[mask]) * 100, 2),
            }

    log.info("Baselines:")
    for name, bl in baselines.items():
        log.info("  %25s  R²=%.3f  rRMSE=%.1f%%", name, bl["R2"], bl["rRMSE_pct"])

    return baselines


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Belgium biomass — end-to-end validation")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", default="gt_volume_m3",
                        choices=["gt_volume_m3", "gt_stem_volume_m3"])
    parser.add_argument("--crown-only", action="store_true",
                        help="Exclude trunk features (simulates airborne UAV)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="SHAP feature selection: retrain with top K features (0=disabled)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Ground truth
    log.info("Step 1: Loading ground truth...")
    gt = load_ground_truth(data_dir / "ground_truth.csv")
    log.info("  %d trees, sites: %s", len(gt), gt["site_name"].unique().tolist())

    # Step 2: Load + extract features
    log.info("Step 2: Feature extraction...")
    pc_dir = data_dir / "pointclouds_clean"
    rows = []
    for pc_file in sorted(pc_dir.glob("*.txt")):
        tree_name = parse_tree_name(pc_file.name)
        pts = load_point_cloud(pc_file)
        tree = split_trunk_crown(pts)
        row = extract_all_features(tree, tree_name)
        rows.append(row)

        dbh = row.get("d_1.3m", np.nan)
        n_d = row.get("n_viable_heights", 0)
        log.info("  %s: %dk pts, H=%.1fm, %d diams, DBH=%.1f cm, "
                 "chi2_pass=%.0f%%, voxels=%d, LBI=%.2f",
                 tree_name, len(pts) // 1000, row["H_max"], n_d,
                 dbh if not np.isnan(dbh) else 0,
                 row.get("chi2_pass_ratio", 0) * 100,
                 row.get("voxel_count", 0),
                 row.get("lbi", 0) if not np.isnan(row.get("lbi", np.nan)) else 0)

    features_df = pd.DataFrame(rows)
    features_df.to_csv(out_dir / "features.csv", index=False)
    log.info("Features: %d trees × %d columns", len(features_df), len(features_df.columns))

    # Step 3: Match to ground truth
    log.info("Step 3: Matching...")
    gt_cols = {
        "gt_tree_name": gt["tree_name"],
        "gt_site": gt["site_name"],
        "gt_dbh_cm": gt["DBH"],
        "gt_height_m": gt["TH_felled"],
        "gt_volume_m3": gt["Volume_total_tree_harvested_m3"],
        "gt_stem_volume_m3": gt["Volume_stem_harvested_m3"],
        "gt_fresh_mass_kg": gt["Fresh_mass_total_tree_harvested"],
        "gt_wsg": gt["WSG_base_disc"],
    }
    gt_slim = pd.DataFrame(gt_cols)
    merged = features_df.merge(gt_slim, left_on="tree_name",
                                right_on="gt_tree_name", how="inner")
    log.info("Matched %d / %d trees", len(merged), len(features_df))
    if len(merged) < 5:
        log.error("Too few matches."); return

    merged.to_csv(out_dir / "matched.csv", index=False)

    # Step 4: DBH validation
    log.info("Step 4: DBH validation...")
    mask_dbh = ~merged["d_1.3m"].isna()
    if mask_dbh.sum() > 0:
        lidar_dbh = merged.loc[mask_dbh, "d_1.3m"].values
        gt_dbh = merged.loc[mask_dbh, "gt_dbh_cm"].values
        dbh_rmse = np.sqrt(mean_squared_error(gt_dbh, lidar_dbh))
        dbh_r2 = r2_score(gt_dbh, lidar_dbh) if len(gt_dbh) > 1 else np.nan
        log.info("  DBH: N=%d, RMSE=%.2f cm, R²=%.3f", mask_dbh.sum(), dbh_rmse, dbh_r2)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(gt_dbh, lidar_dbh, s=40, alpha=0.7, edgecolors="k")
        lims = [min(gt_dbh.min(), lidar_dbh.min()) * 0.9,
                max(gt_dbh.max(), lidar_dbh.max()) * 1.1]
        ax.plot(lims, lims, "k--", label="1:1")
        ax.set(xlabel="GT DBH (cm)", ylabel="LiDAR DBH (cm)", aspect="equal")
        ax.set_title(f"DBH (N={mask_dbh.sum()}, RMSE={dbh_rmse:.2f} cm, R²={dbh_r2:.3f})")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "dbh_validation.png", dpi=150)
        plt.close(fig)

    # Step 5: XGBoost
    target_col = args.target
    crown_only = args.crown_only
    top_k = args.top_k
    tag = target_col.replace("gt_", "")
    if crown_only:
        tag += "_crown_only"

    log.info("Step 5: XGBoost (target=%s, crown_only=%s)...", target_col, crown_only)
    X, y, feat_names = prepare_features(merged, target_col, crown_only=crown_only)
    model, y_pred, metrics, importance = train_and_evaluate(
        X, y, feat_names, out_dir, tag)

    # Step 5b: SHAP analysis (Ye et al. 2025)
    log.info("Step 5b: SHAP analysis...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(pd.DataFrame(X, columns=feat_names))

    fig = plt.figure(figsize=(10, max(5, len(feat_names) * 0.3)))
    shap.plots.beeswarm(shap_values, show=False, max_display=15)
    plt.tight_layout()
    fig.savefig(out_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved shap_beeswarm.png")

    fig = plt.figure(figsize=(8, max(4, len(feat_names) * 0.3)))
    shap.plots.bar(shap_values, show=False, max_display=15)
    plt.tight_layout()
    fig.savefig(out_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved shap_bar.png")

    # Top SHAP features
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_ranking = sorted(zip(feat_names, mean_abs_shap),
                          key=lambda x: x[1], reverse=True)
    log.info("  SHAP feature ranking (top 10):")
    for name, val in shap_ranking[:10]:
        log.info("    %30s  %.4f", name, val)

    # Step 5c: SHAP top-K feature selection retrain
    if top_k > 0:
        selected = [name for name, _ in shap_ranking[:top_k]]
        log.info("Step 5c: Retrain with SHAP top-%d features: %s", top_k, selected)
        sel_idx = [feat_names.index(s) for s in selected]
        X_sel = X[:, sel_idx]

        tag_sel = f"{tag}_shap_top{top_k}"
        model_sel, y_pred_sel, metrics_sel, imp_sel = train_and_evaluate(
            X_sel, y, selected, out_dir, tag_sel)

        log.info("  Full (%d feat): R²=%.4f | SHAP top-%d: R²=%.4f  (delta=%+.4f)",
                 len(feat_names), metrics["R2"], top_k, metrics_sel["R2"],
                 metrics_sel["R2"] - metrics["R2"])

        # Use selected model if it's better
        if metrics_sel["R2"] > metrics["R2"]:
            log.info("  >>> SHAP top-%d WINS — using selected model", top_k)
            model, y_pred, metrics, importance = model_sel, y_pred_sel, metrics_sel, imp_sel
            feat_names = selected
            X = X_sel
    else:
        # Auto-search: try K = 5, 8, 10, 12, 15, 20 and report best
        log.info("Step 5c: Auto-search optimal feature count...")
        best_k, best_r2 = 0, metrics["R2"]
        results_k = [(len(feat_names), metrics["R2"], "all")]
        for k in [5, 8, 10, 12, 15, 20]:
            if k >= len(feat_names):
                continue
            sel = [name for name, _ in shap_ranking[:k]]
            sel_idx = [feat_names.index(s) for s in sel]
            X_k = X[:, sel_idx]
            m_k = xgb.XGBRegressor(**XGB_PARAMS)
            yp_k = cross_val_predict(m_k, X_k, y, cv=LeaveOneOut())
            r2_k = r2_score(y, yp_k)
            results_k.append((k, r2_k, ", ".join(sel[:3]) + "..."))
            if r2_k > best_r2:
                best_k, best_r2 = k, r2_k

        log.info("  Feature count sweep:")
        for k, r2_k, label in results_k:
            marker = " <<<" if r2_k == best_r2 else ""
            log.info("    K=%-3d  R²=%.4f  (%s)%s", k, r2_k, label, marker)

        if best_k > 0:
            log.info("  Optimal: K=%d (R²=%.4f, +%.4f vs all)",
                     best_k, best_r2, best_r2 - metrics["R2"])

    # Step 5d: Crown-based DBH estimation (Kukkonen 2021)
    log.info("Step 5d: Crown-based DBH estimation (Kukkonen 2021)...")
    crown_cols = ["H_max", "H_mean", "H_std", "H_p50", "H_p75", "H_p90",
                  "H_p95", "crown_diameter", "crown_area", "crown_volume",
                  "crown_base_height", "crown_length", "n_total_points",
                  "voxel_count", "voxel_volume", "voxel_density",
                  "voxel_h_entropy", "lbi"]
    crown_cols = [c for c in crown_cols if c in merged.columns]
    mask_dbh_gt = ~merged["gt_dbh_cm"].isna()
    if mask_dbh_gt.sum() > 5:
        X_crown = merged.loc[mask_dbh_gt, crown_cols].values
        y_dbh = merged.loc[mask_dbh_gt, "gt_dbh_cm"].values
        # Drop columns with NaN
        valid = ~np.isnan(X_crown).any(axis=0)
        X_crown = X_crown[:, valid]
        crown_cols_used = [c for c, v in zip(crown_cols, valid) if v]

        dbh_model = xgb.XGBRegressor(**XGB_PARAMS)
        y_dbh_pred = cross_val_predict(dbh_model, X_crown, y_dbh, cv=LeaveOneOut())
        dbh_r2 = r2_score(y_dbh, y_dbh_pred)
        dbh_rmse = np.sqrt(mean_squared_error(y_dbh, y_dbh_pred))
        log.info("  Crown→DBH: R²=%.3f, RMSE=%.2f cm (N=%d, %d features)",
                 dbh_r2, dbh_rmse, len(y_dbh), len(crown_cols_used))

        # Train final model + SHAP for crown→DBH
        dbh_model.fit(X_crown, y_dbh)
        perm_dbh = permutation_importance(dbh_model, X_crown, y_dbh,
                                          n_repeats=30, random_state=42, scoring="r2")
        dbh_imp = sorted(zip(crown_cols_used, perm_dbh.importances_mean),
                         key=lambda x: x[1], reverse=True)
        log.info("  Crown→DBH top features:")
        for name, imp in dbh_imp[:5]:
            log.info("    %25s  %.4f", name, imp)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_dbh, y_dbh_pred, s=40, alpha=0.7, edgecolors="k")
        lims = [min(y_dbh.min(), y_dbh_pred.min()) * 0.9,
                max(y_dbh.max(), y_dbh_pred.max()) * 1.1]
        ax.plot(lims, lims, "k--", label="1:1")
        ax.set(xlabel="GT DBH (cm)", ylabel="Predicted DBH (cm)", aspect="equal")
        ax.set_title(f"Crown→DBH (Kukkonen 2021)\n"
                     f"R²={dbh_r2:.3f}  RMSE={dbh_rmse:.2f} cm  N={len(y_dbh)}")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(out_dir / "crown_dbh_prediction.png", dpi=150)
        plt.close(fig)
        log.info("  Saved crown_dbh_prediction.png")

    # Step 6: Baselines
    log.info("Step 6: Baselines...")
    baselines = compute_baselines(merged, y, out_dir)

    # Step 7: Per-site
    log.info("Step 7: Per-site...")
    for site, group in merged.groupby("gt_site"):
        mask = merged["gt_site"] == site
        if mask.sum() < 3: continue
        y_s, yp_s = y[mask.values], y_pred[mask.values]
        log.info("  %s: N=%d, R²=%.3f, RMSE=%.4f",
                 site, mask.sum(),
                 r2_score(y_s, yp_s) if len(y_s) > 1 else np.nan,
                 np.sqrt(mean_squared_error(y_s, yp_s)))

    # Summary
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info("  Trees: %d matched, %d features", len(merged), len(feat_names))
    log.info("  XGBoost LOOCV: R²=%.4f, RMSE=%.4f m³, rRMSE=%.1f%%",
             metrics["R2"], metrics["RMSE_m3"], metrics["rRMSE_pct"])
    for name, bl in baselines.items():
        log.info("  %s: R²=%.3f, rRMSE=%.1f%%", name, bl["R2"], bl["rRMSE_pct"])
    log.info("  Results: %s", out_dir)


if __name__ == "__main__":
    main()

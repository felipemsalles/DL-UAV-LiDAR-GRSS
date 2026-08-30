#!/usr/bin/env python3
"""
Validation: Belgium Destructive Biomass (Zenodo 4557401)

Validates the XGBoost volume pipeline end-to-end against real destructive
volume measurements. Point clouds are TLS (XYZ only, no semantic labels), so
features are computed without trunk/crown separation.

Usage:
    conda run -n greenvista python scripts/validation/validate_belgium.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import setup_logger, smalian_volume, freire_volume
from tree_analysis import (
    _fit_slice,
    BIN_SIZE,
)

log = setup_logger("validate_belgium")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "belgium_biomass"
OUT_DIR = Path(__file__).resolve().parent / "results" / "belgium"

# XGBoost params (same as production)
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


def load_ground_truth() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "ground_truth.csv")
    # Volume is in dm³ in the CSV, convert to m³
    df["volume_m3"] = df["Volume_total_tree_harvested"] / 1000.0
    df["stem_volume_m3"] = df["Volume_stem_harvested"] / 1000.0
    # Build filename key for matching (PSYLA-01 -> PSYLA1)
    df["file_key"] = df["tree_name"].str.replace("-", "").str.replace(
        r"0(\d)$", r"\1", regex=True)
    log.info("Ground truth: %d trees, volume %.3f–%.3f m³, DBH %.1f–%.1f cm",
             len(df), df.volume_m3.min(), df.volume_m3.max(),
             df.DBH.min(), df.DBH.max())
    return df


def load_point_cloud(file_key: str) -> np.ndarray:
    txt_path = DATA_DIR / "pointclouds_clean" / f"{file_key}.txt"
    if not txt_path.exists():
        return np.empty((0, 3))
    return np.loadtxt(str(txt_path))


def extract_features(pts: np.ndarray) -> dict:
    """Extract features from a TLS point cloud without semantic labels.
    Uses height-based heuristic: points below 20% of tree height are
    trunk candidates, rest are crown."""
    z = pts[:, 2]
    z_min = z.min()
    h = z - z_min  # height above ground
    tree_height = h.max()

    feats = {
        "n_total_points": len(pts),
        "H_max": tree_height,
        "H_min": 0.0,
    }

    # Height stats from all points
    feats["H_mean"] = float(h.mean())
    feats["H_std"] = float(h.std())
    feats["H_cv"] = feats["H_std"] / feats["H_mean"] if feats["H_mean"] > 0 else np.nan
    for p, name in [(25, "H_p25"), (50, "H_p50"), (75, "H_p75"),
                     (90, "H_p90"), (95, "H_p95")]:
        feats[name] = float(np.percentile(h, p))

    # Crown geometry from upper portion (above 40% of height)
    crown_threshold = tree_height * 0.4
    crown_mask = h >= crown_threshold
    crown_pts = pts[crown_mask]

    if len(crown_pts) >= 4:
        xy_crown = crown_pts[:, :2]
        try:
            hull2d = ConvexHull(xy_crown)
            feats["crown_area"] = float(hull2d.volume)
            verts = xy_crown[hull2d.vertices]
            max_dist = 0.0
            for j in range(len(verts)):
                dists = np.linalg.norm(verts[j] - verts[j + 1:], axis=1)
                if len(dists) > 0:
                    max_dist = max(max_dist, dists.max())
            feats["crown_diameter"] = float(max_dist)
        except QhullError:
            feats["crown_area"] = np.nan
            feats["crown_diameter"] = np.nan

        feats["crown_base_height"] = float(h[crown_mask].min())
        feats["crown_length"] = tree_height - feats["crown_base_height"]

        try:
            hull3d = ConvexHull(crown_pts)
            feats["crown_volume"] = float(hull3d.volume)
        except QhullError:
            feats["crown_volume"] = np.nan
    else:
        for k in ["crown_area", "crown_diameter", "crown_base_height",
                   "crown_length", "crown_volume"]:
            feats[k] = np.nan

    # Diameter extraction at multiple heights (trunk = lower portion)
    diameters = []
    for target_h in np.arange(0.5, min(tree_height * 0.4, 15.0), 0.5):
        tol = 0.25
        mask = (h >= target_h - tol) & (h < target_h + tol)
        slice_pts = pts[mask]
        result = _fit_slice(slice_pts[:, :2], target_h)
        if result is not None:
            diameters.append(result)

    if diameters:
        d_values = [d["diameter_cm"] for d in diameters]
        heights = [d["height_m"] for d in diameters]
        feats["n_viable_heights"] = len(diameters)
        feats["d_max"] = max(d_values)
        feats["d_mean"] = float(np.mean(d_values))
        feats["viable_height_range"] = max(heights) - min(heights)

        # DBH closest to 1.3m
        closest_idx = int(np.argmin([abs(h_val - 1.3) for h_val in heights]))
        if abs(heights[closest_idx] - 1.3) <= BIN_SIZE:
            feats["d_1.3m"] = d_values[closest_idx]
        else:
            feats["d_1.3m"] = np.nan

        # Smalian volume from fitted diameters
        sorted_d = sorted(diameters, key=lambda d: d["height_m"])
        total_vol = 0.0
        n_sections = 0
        for i in range(len(sorted_d) - 1):
            length = sorted_d[i + 1]["height_m"] - sorted_d[i]["height_m"]
            if length > 0:
                total_vol += smalian_volume(
                    sorted_d[i]["diameter_cm"],
                    sorted_d[i + 1]["diameter_cm"],
                    length)
                n_sections += 1
        feats["smalian_volume_m3"] = round(total_vol, 6)
        feats["n_smalian_sections"] = n_sections

        # Freire volume
        if not np.isnan(feats.get("d_1.3m", np.nan)):
            feats["freire_volume_m3"] = round(
                freire_volume(feats["d_1.3m"], tree_height), 6)
        else:
            feats["freire_volume_m3"] = np.nan

        # Per-height diameter features
        for d in diameters:
            h_key = round(d["height_m"] * 2) / 2
            feats[f"d_at_{h_key:.1f}m"] = d["diameter_cm"]
    else:
        for k in ["n_viable_heights", "d_max", "d_mean", "viable_height_range",
                   "d_1.3m", "smalian_volume_m3", "n_smalian_sections",
                   "freire_volume_m3"]:
            feats[k] = np.nan if k != "n_viable_heights" else 0

    return feats


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth()

    log.info("Extracting features from %d point clouds...", len(gt))
    rows = []
    for _, row in gt.iterrows():
        pts = load_point_cloud(row.file_key)
        if len(pts) == 0:
            log.warning("No point cloud for %s", row.tree_name)
            continue

        feats = extract_features(pts)
        feats["tree_name"] = row.tree_name
        feats["site"] = row.site_name
        feats["gt_dbh_cm"] = row.DBH
        feats["gt_height_m"] = row.TH_felled
        feats["gt_volume_m3"] = row.volume_m3
        feats["gt_stem_volume_m3"] = row.stem_volume_m3
        feats["gt_fresh_mass_kg"] = row.Fresh_mass_total_tree_harvested
        rows.append(feats)

        log.info("  %s: %d pts, H=%.1fm, DBH_fit=%s, gt_vol=%.3f m³",
                 row.tree_name, len(pts), feats["H_max"],
                 f"{feats.get('d_1.3m', 'N/A'):.1f}cm"
                 if not np.isnan(feats.get("d_1.3m", np.nan)) else "N/A",
                 row.volume_m3)

    df = pd.DataFrame(rows)
    features_path = OUT_DIR / "features.csv"
    df.to_csv(features_path, index=False)
    log.info("Saved %s (%d trees, %d columns)", features_path, len(df),
             len(df.columns))

    # --- DBH validation ---
    valid_dbh = df.dropna(subset=["d_1.3m"])
    if len(valid_dbh) > 0:
        dbh_err = valid_dbh["d_1.3m"] - valid_dbh["gt_dbh_cm"]
        dbh_rmse = np.sqrt((dbh_err ** 2).mean())
        dbh_r2 = r2_score(valid_dbh["gt_dbh_cm"], valid_dbh["d_1.3m"])
        log.info("")
        log.info("=== DBH Validation (%d/%d trees) ===", len(valid_dbh), len(df))
        log.info("  RMSE = %.2f cm", dbh_rmse)
        log.info("  R²   = %.4f", dbh_r2)
        log.info("  Bias = %.2f cm", dbh_err.mean())

    # --- Smalian volume validation ---
    valid_smalian = df.dropna(subset=["smalian_volume_m3"])
    if len(valid_smalian) > 0:
        s_err = valid_smalian["smalian_volume_m3"] - valid_smalian["gt_volume_m3"]
        s_rmse = np.sqrt((s_err ** 2).mean())
        log.info("")
        log.info("=== Smalian Volume vs Destructive (%d trees) ===",
                 len(valid_smalian))
        log.info("  RMSE = %.4f m³", s_rmse)
        log.info("  Bias = %.4f m³", s_err.mean())
        if len(valid_smalian) > 2:
            log.info("  R²   = %.4f",
                     r2_score(valid_smalian["gt_volume_m3"],
                              valid_smalian["smalian_volume_m3"]))

    # --- XGBoost volume prediction ---
    log.info("")
    log.info("=== XGBoost LOOCV Volume Prediction ===")

    # Select features (exclude identifiers, targets, leaky columns)
    exclude = {"tree_name", "site", "gt_dbh_cm", "gt_height_m",
               "gt_volume_m3", "gt_stem_volume_m3", "gt_fresh_mass_kg",
               "smalian_volume_m3", "freire_volume_m3"}
    feature_cols = [c for c in df.columns
                    if c not in exclude and df[c].dtype in [np.float64, np.int64]]

    X = df[feature_cols].copy()
    y = df["gt_volume_m3"].values

    # Drop columns with >50% NaN
    valid_cols = X.columns[X.notna().sum() >= len(X) * 0.5]
    X = X[valid_cols]
    feature_names = list(X.columns)

    log.info("  Features: %d (%s)", len(feature_names), feature_names)
    log.info("  Target: gt_volume_m3 (N=%d)", len(y))

    X_arr = X.values
    model = xgb.XGBRegressor(**XGB_PARAMS)
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X_arr, y, cv=loo)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    bias = np.mean(y_pred - y)
    rrmse = (rmse / np.mean(y)) * 100

    log.info("")
    log.info("  XGBoost LOOCV Results:")
    log.info("    R²    = %.4f", r2)
    log.info("    RMSE  = %.4f m³", rmse)
    log.info("    rRMSE = %.1f%%", rrmse)
    log.info("    MAE   = %.4f m³", mae)
    log.info("    Bias  = %.4f m³", bias)

    # Train final model for feature importance
    model.fit(X_arr, y)
    perm = permutation_importance(model, X_arr, y, n_repeats=30,
                                  random_state=42, scoring="r2")
    importance = sorted(zip(feature_names, perm.importances_mean),
                        key=lambda x: x[1], reverse=True)
    log.info("")
    log.info("  Feature Importance (permutation, top 10):")
    for name, imp in importance[:10]:
        log.info("    %25s  %.4f", name, imp)

    # --- Baselines ---
    log.info("")
    log.info("=== Baselines ===")

    # Freire formula with GT DBH
    y_freire = np.pi * (df["gt_dbh_cm"].values / 200) ** 2 * \
               df["gt_height_m"].values * 0.5
    freire_r2 = r2_score(y, y_freire)
    freire_rmse = np.sqrt(mean_squared_error(y, y_freire))
    log.info("  Freire (GT DBH): R²=%.4f, RMSE=%.4f m³, rRMSE=%.1f%%",
             freire_r2, freire_rmse, freire_rmse / np.mean(y) * 100)

    # Linear H_max
    lr = LinearRegression()
    h_max = df["H_max"].values.reshape(-1, 1)
    y_lr_h = cross_val_predict(lr, h_max, y, cv=loo)
    lr_h_r2 = r2_score(y, y_lr_h)
    lr_h_rmse = np.sqrt(mean_squared_error(y, y_lr_h))
    log.info("  Linear H_max:   R²=%.4f, RMSE=%.4f m³, rRMSE=%.1f%%",
             lr_h_r2, lr_h_rmse, lr_h_rmse / np.mean(y) * 100)

    # Mean volume
    mean_rmse = np.sqrt(mean_squared_error(y, np.full_like(y, y.mean())))
    log.info("  Mean volume:    R²=0.0000, RMSE=%.4f m³, rRMSE=%.1f%%",
             mean_rmse, mean_rmse / np.mean(y) * 100)

    # --- Plots ---
    plot_predicted_vs_observed(y, y_pred, r2, rmse, rrmse, OUT_DIR)
    plot_feature_importance(importance, OUT_DIR)
    plot_baseline_comparison(
        {"XGBoost LOOCV": {"R2": r2, "rRMSE": rrmse},
         "Freire (GT DBH)": {"R2": freire_r2,
                              "rRMSE": freire_rmse / np.mean(y) * 100},
         "Linear H_max": {"R2": lr_h_r2,
                           "rRMSE": lr_h_rmse / np.mean(y) * 100},
         "Mean volume": {"R2": 0.0, "rRMSE": mean_rmse / np.mean(y) * 100}},
        OUT_DIR)

    # Save summary
    summary = {
        "n_trees": len(df),
        "xgb_r2": round(r2, 4), "xgb_rmse_m3": round(rmse, 4),
        "xgb_rrmse_pct": round(rrmse, 1),
        "freire_r2": round(freire_r2, 4),
        "n_features": len(feature_names),
        "features": feature_names,
        "importance_top5": importance[:5],
    }
    import json
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("")
    log.info("Done. Results in %s", OUT_DIR)


def plot_predicted_vs_observed(y_true, y_pred, r2, rmse, rrmse, out_dir):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, s=40, alpha=0.7, edgecolors="k", linewidths=0.5)
    lims = [0, max(y_true.max(), y_pred.max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    ax.set_xlabel("Observed Volume (m³)")
    ax.set_ylabel("Predicted Volume (m³)")
    ax.set_title(f"Belgium Biomass — XGBoost LOOCV\n"
                 f"R²={r2:.3f}  RMSE={rmse:.4f} m³  rRMSE={rrmse:.1f}%")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_vs_observed.png", dpi=150)
    plt.close(fig)
    log.info("Saved predicted_vs_observed.png")


def plot_feature_importance(importance, out_dir, top_n=15):
    top = importance[:top_n]
    names = [x[0] for x in top][::-1]
    values = [x[1] for x in top][::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
    ax.barh(names, values, color="#3498db", edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Permutation Importance")
    ax.set_title("Belgium Biomass — Feature Importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    log.info("Saved feature_importance.png")


def plot_baseline_comparison(models, out_dir):
    names = list(models.keys())
    r2_vals = [models[n]["R2"] for n in names]
    rrmse_vals = [models[n]["rRMSE"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#2ecc71" if "XGBoost" in n else "#95a5a6" for n in names]

    axes[0].barh(names, r2_vals, color=colors, edgecolor="k", linewidth=0.5)
    axes[0].set_xlabel("R²")
    axes[0].set_title("R² Comparison")
    axes[0].set_xlim(0, 1.05)
    axes[0].grid(True, axis="x", alpha=0.3)

    axes[1].barh(names, rrmse_vals, color=colors, edgecolor="k", linewidth=0.5)
    axes[1].set_xlabel("rRMSE (%)")
    axes[1].set_title("Relative RMSE")
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.suptitle("Belgium Biomass — Model Comparison", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close(fig)
    log.info("Saved model_comparison.png")


if __name__ == "__main__":
    run()

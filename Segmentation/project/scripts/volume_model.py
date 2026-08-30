#!/usr/bin/env python3
"""
XGBoost volume estimation pipeline.

Trains XGBoost to predict total tree volume directly from LiDAR-extracted
features (crown + trunk), bypassing DBH. Validates against rigorous scaling
ground truth (cubagem rigorosa).

Usage:
    # With ground truth available:
    conda run -n greenvista python scripts/volume_model.py \
        --features /tmp/analysis/features.csv \
        --ground-truth data/cubagem.csv \
        --output /tmp/analysis/model/

    # Prediction only (with trained model):
    conda run -n greenvista python scripts/volume_model.py \
        --features /tmp/analysis/features.csv \
        --model /tmp/analysis/model/xgb_volume.json \
        --output /tmp/analysis/predictions/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import xgboost as xgb

import shap

from common import setup_logger, smalian_volume

log = setup_logger("volume_model")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Ground truth CSV column names (rigorous-scaling / cubagem rigorosa format)
GT_TREE_ID = "nm_arvore"
GT_LAT = "nm_latitude"
GT_LON = "nm_longitude"
GT_DBH = "nm_d_c"
GT_HEIGHT = "nm_altura"
GT_SECTION_HEIGHT = "nm_seccao"
GT_D1 = "d1"
GT_D2 = "d2"

# GPS matching tolerance (meters) — UTM coordinates
GPS_MATCH_TOLERANCE = 3.0  # generous given ~1m GPS precision + crown offset

# Features to exclude from model input (identifiers, targets, leaky features)
EXCLUDE_FEATURES = {
    "instance_id", "score",
    # Volume columns are targets/estimates, not raw features
    "smalian_volume_m3", "huber_volume_m3", "freire_volume_m3", "kozak_volume_m3",
    # Biomass is derived from DBH (leaks)
    "biomass_kg",
    # These leak ground truth info if present
    "gt_volume_m3", "gt_dbh_cm",
}

# Trunk-derived features — excluded in crown-only mode (airborne UAV simulation)
TRUNK_FEATURES = {
    "d_1.3m", "d_hossfeldt", "d_max", "d_mean",
    "n_viable_heights", "viable_height_range",
    "n_smalian_sections", "n_chi2_passed", "chi2_pass_ratio",
}

# XGBoost hyperparameters tuned for small N (~60 trees)
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 3,          # shallow to prevent overfitting with N=60
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,   # conservative for small N
    "reg_alpha": 0.1,        # L1 regularization
    "reg_lambda": 1.0,       # L2 regularization
    "random_state": 42,
}


# ===================================================================
# Ground Truth Processing
# ===================================================================

def load_ground_truth(path: str) -> pd.DataFrame:
    """Load the rigorous-scaling (cubagem rigorosa) CSV and compute per-tree reference volumes.

    The CSV has one row per section per tree. We compute:
    - Total volume via Smalian formula across all sections
    - Per-tree summary: DBH, height, UTM coords
    """
    # Try common separators
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8")
            if len(df.columns) > 1:
                break
        except ValueError:
            continue
    else:
        df = pd.read_csv(path, encoding="latin-1")

    # Normalize column names: strip whitespace, lowercase
    df.columns = df.columns.str.strip()

    log.info("Loaded ground truth: %d rows, columns: %s",
             len(df), list(df.columns))

    # Compute Smalian volume per tree
    trees = []
    for tree_id, group in df.groupby(GT_TREE_ID):
        group = group.sort_values(GT_SECTION_HEIGHT)
        sections = group[GT_SECTION_HEIGHT].values
        # Use mean diameter (d1+d2)/2 at each section
        if "d_cm (media)" in group.columns:
            diams = group["d_cm (media)"].values
        else:
            diams = (group[GT_D1].values + group[GT_D2].values) / 2.0

        # Smalian volume across consecutive sections
        total_vol = 0.0
        for i in range(len(sections) - 1):
            d_bot = diams[i]
            d_top = diams[i + 1]
            length = sections[i + 1] - sections[i]
            if length > 0 and not (np.isnan(d_bot) or np.isnan(d_top)):
                total_vol += smalian_volume(d_bot, d_top, length)

        row = group.iloc[0]
        trees.append({
            "gt_tree_id": int(tree_id),
            "gt_lat": float(row[GT_LAT]) if GT_LAT in row.index else np.nan,
            "gt_lon": float(row[GT_LON]) if GT_LON in row.index else np.nan,
            "gt_dbh_cm": float(row[GT_DBH]) if GT_DBH in row.index else np.nan,
            "gt_height_m": float(row[GT_HEIGHT]) if GT_HEIGHT in row.index else np.nan,
            "gt_volume_m3": round(total_vol, 6),
            "gt_n_sections": len(sections),
        })

    gt = pd.DataFrame(trees)
    log.info("Computed volumes for %d trees", len(gt))
    log.info("  Volume range: %.4f - %.4f m³",
             gt['gt_volume_m3'].min(), gt['gt_volume_m3'].max())
    log.info("  DBH range: %.1f - %.1f cm",
             gt['gt_dbh_cm'].min(), gt['gt_dbh_cm'].max())
    return gt


# ===================================================================
# GPS Matching
# ===================================================================

def match_trees_gps(features: pd.DataFrame, gt: pd.DataFrame,
                    tree_coords: dict = None) -> pd.DataFrame:
    """Match LiDAR-segmented trees to ground truth via UTM coordinates.

    tree_coords: dict of {instance_id: (utm_x, utm_y)} from the point cloud.
    If not provided, attempts to use centroid columns in features if available.
    """
    if tree_coords is None:
        # Check if features have coordinate columns
        if "centroid_x" in features.columns and "centroid_y" in features.columns:
            tree_coords = {
                row["instance_id"]: (row["centroid_x"], row["centroid_y"])
                for _, row in features.iterrows()
            }
        else:
            log.warning("No tree coordinates available for GPS matching.")
            log.info("Falling back to manual ID matching (instance_id = gt_tree_id).")
            merged = features.merge(gt, left_on="instance_id",
                                    right_on="gt_tree_id", how="inner")
            log.info("Matched %d trees by ID", len(merged))
            return merged

    matches = []
    used_gt = set()
    for iid, (x, y) in tree_coords.items():
        best_dist = GPS_MATCH_TOLERANCE
        best_gt_id = None
        for _, gt_row in gt.iterrows():
            if gt_row["gt_tree_id"] in used_gt:
                continue
            dist = np.sqrt((x - gt_row["gt_lon"]) ** 2 +
                           (y - gt_row["gt_lat"]) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_gt_id = gt_row["gt_tree_id"]

        if best_gt_id is not None:
            matches.append((iid, best_gt_id, best_dist))
            used_gt.add(best_gt_id)

    log.info("GPS matching: %d/%d trees matched (tolerance=%.1fm)",
             len(matches), len(tree_coords), GPS_MATCH_TOLERANCE)
    for iid, gt_id, dist in matches:
        log.info("  Instance %s → GT tree %s (dist=%.2fm)", iid, gt_id, dist)

    if len(matches) == 0:
        log.warning("GPS matching found 0 matches — falling back to ID matching.")
        merged = features.merge(gt, left_on="instance_id",
                                right_on="gt_tree_id", how="inner")
        log.info("Matched %d trees by ID", len(merged))
        return merged

    match_df = pd.DataFrame(matches,
                            columns=["instance_id", "gt_tree_id", "match_dist"])
    merged = features.merge(match_df, on="instance_id")
    merged = merged.merge(gt, on="gt_tree_id")
    return merged


# ===================================================================
# Feature Selection & Preparation
# ===================================================================

def prepare_features(df: pd.DataFrame,
                     target_col: str = "gt_volume_m3",
                     crown_only: bool = False) -> tuple:
    """Select and prepare features for XGBoost.

    Returns (X, y, feature_names).
    If crown_only=True, excludes trunk-derived features and d_at_* columns.
    """
    exclude = EXCLUDE_FEATURES | (TRUNK_FEATURES if crown_only else set())
    feature_cols = [c for c in df.columns
                    if c not in exclude
                    and c != target_col
                    and not c.startswith("gt_")
                    and not c.startswith("match_")
                    and not (crown_only and c.startswith("d_at_"))]

    X = df[feature_cols].copy()
    y = df[target_col].values

    # Drop columns that are all NaN
    valid_cols = X.columns[X.notna().any()]
    X = X[valid_cols]

    # Drop columns with >50% missing
    threshold = len(X) * 0.5
    valid_cols = X.columns[X.notna().sum() >= threshold]
    X = X[valid_cols]

    feature_names = list(X.columns)

    mode = "CROWN-ONLY" if crown_only else "ALL"
    log.info("Feature preparation (%s):", mode)
    log.info("  Available features: %d", len(feature_names))
    log.info("  Features: %s", feature_names)
    log.info("  Target: %s (N=%d)", target_col, len(y))
    log.info("  Target range: %.4f - %.4f m³", y.min(), y.max())

    return X.values, y, feature_names


# ===================================================================
# Model Training & Evaluation
# ===================================================================

def train_xgboost_loocv(X: np.ndarray, y: np.ndarray,
                        feature_names: list) -> tuple:
    """Train XGBoost with Leave-One-Out Cross-Validation.

    LOOCV is optimal for small N (~60). Each tree is predicted by a model
    trained on the other 59 trees.

    Returns (model_trained_on_all, y_pred_loocv, metrics).
    """
    log.info("=" * 60)
    log.info("XGBoost LOOCV Training")
    log.info("=" * 60)

    model = xgb.XGBRegressor(**XGB_PARAMS)
    loo = LeaveOneOut()

    # LOOCV predictions
    y_pred = cross_val_predict(model, X, y, cv=loo)

    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    bias = np.mean(y_pred - y)
    rrmse = (rmse / np.mean(y)) * 100  # relative RMSE %

    metrics = {
        "R2": round(r2, 4),
        "RMSE_m3": round(rmse, 6),
        "rRMSE_pct": round(rrmse, 2),
        "MAE_m3": round(mae, 6),
        "bias_m3": round(bias, 6),
        "N": len(y),
    }

    log.info("LOOCV Results (N=%d):", len(y))
    log.info("  R²          = %s", metrics['R2'])
    log.info("  RMSE        = %.4f m³", metrics['RMSE_m3'])
    log.info("  rRMSE       = %.1f%%", metrics['rRMSE_pct'])
    log.info("  MAE         = %.4f m³", metrics['MAE_m3'])
    log.info("  Bias        = %.4f m³", metrics['bias_m3'])

    # Train final model on all data
    model.fit(X, y)

    # Permutation importance (more reliable than gain-based for correlated features)
    perm_result = permutation_importance(
        model, X, y, n_repeats=30, random_state=42, scoring="r2")
    importance_dict = {name: float(imp)
                       for name, imp in zip(feature_names,
                                            perm_result.importances_mean)}
    importance_sorted = sorted(importance_dict.items(),
                               key=lambda x: x[1], reverse=True)
    log.info("Feature Importance — permutation (top 10):")
    for name, imp in importance_sorted[:10]:
        log.info("  %30s  %.4f", name, imp)

    return model, y_pred, metrics, importance_sorted


def compute_baselines(df: pd.DataFrame, y_true: np.ndarray) -> dict:
    """Compute baseline volume predictions for comparison.

    Baseline 1: Freire formula V = pi*(DBH/200)^2*H*0.5 (requires GT DBH)
    Baseline 2: Mean volume (trivial baseline)
    Baseline 3: Linear H_max only
    """
    baselines = {}

    # Baseline 1: Freire formula with ground truth DBH
    if "gt_dbh_cm" in df.columns and "gt_height_m" in df.columns:
        dbh = df["gt_dbh_cm"].values
        h = df["gt_height_m"].values
        mask = ~(np.isnan(dbh) | np.isnan(h))
        if mask.sum() > 0:
            y_freire = np.pi * (dbh / 200) ** 2 * h * 0.5
            baselines["Freire (GT DBH)"] = {
                "R2": round(r2_score(y_true[mask], y_freire[mask]), 4),
                "RMSE_m3": round(np.sqrt(mean_squared_error(
                    y_true[mask], y_freire[mask])), 6),
                "rRMSE_pct": round(
                    np.sqrt(mean_squared_error(y_true[mask], y_freire[mask]))
                    / np.mean(y_true[mask]) * 100, 2),
            }

    # Baseline 2: Mean volume
    y_mean = np.full_like(y_true, y_true.mean())
    baselines["Mean volume"] = {
        "R2": 0.0,
        "RMSE_m3": round(np.sqrt(mean_squared_error(y_true, y_mean)), 6),
        "rRMSE_pct": round(
            np.sqrt(mean_squared_error(y_true, y_mean))
            / np.mean(y_true) * 100, 2),
    }

    # Baseline 3: Linear H_max
    if "H_max" in df.columns:
        from sklearn.linear_model import LinearRegression
        h_max = df["H_max"].values.reshape(-1, 1)
        mask = ~np.isnan(h_max.ravel())
        if mask.sum() > 2:
            lr = LinearRegression()
            y_lr = cross_val_predict(lr, h_max[mask], y_true[mask],
                                     cv=LeaveOneOut())
            baselines["Linear H_max"] = {
                "R2": round(r2_score(y_true[mask], y_lr), 4),
                "RMSE_m3": round(np.sqrt(mean_squared_error(
                    y_true[mask], y_lr)), 6),
                "rRMSE_pct": round(
                    np.sqrt(mean_squared_error(y_true[mask], y_lr))
                    / np.mean(y_true[mask]) * 100, 2),
            }

    # Baseline 4: Linear crown_diameter + H_max
    if "crown_diameter" in df.columns and "H_max" in df.columns:
        from sklearn.linear_model import LinearRegression
        X_lr = df[["crown_diameter", "H_max"]].values
        mask = ~np.isnan(X_lr).any(axis=1)
        if mask.sum() > 2:
            lr = LinearRegression()
            y_lr = cross_val_predict(lr, X_lr[mask], y_true[mask],
                                     cv=LeaveOneOut())
            baselines["Linear CD+H"] = {
                "R2": round(r2_score(y_true[mask], y_lr), 4),
                "RMSE_m3": round(np.sqrt(mean_squared_error(
                    y_true[mask], y_lr)), 6),
                "rRMSE_pct": round(
                    np.sqrt(mean_squared_error(y_true[mask], y_lr))
                    / np.mean(y_true[mask]) * 100, 2),
            }

    return baselines


# ===================================================================
# Visualization
# ===================================================================

def plot_predicted_vs_observed(y_true, y_pred, metrics, out_dir: Path,
                               title="XGBoost LOOCV"):
    """Scatter plot of predicted vs observed volume."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, s=40, alpha=0.7, edgecolors="k", linewidths=0.5)

    # 1:1 line
    lims = [min(y_true.min(), y_pred.min()) * 0.9,
            max(y_true.max(), y_pred.max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1, label="1:1")

    # Regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_line, p(x_line), "r-", lw=1.5, alpha=0.7,
            label=f"fit: y={z[0]:.2f}x+{z[1]:.4f}")

    ax.set_xlabel("Observed Volume (m³)", fontsize=12)
    ax.set_ylabel("Predicted Volume (m³)", fontsize=12)
    ax.set_title(f"{title}\n"
                 f"R²={metrics['R2']:.3f}  RMSE={metrics['RMSE_m3']:.4f} m³  "
                 f"rRMSE={metrics['rRMSE_pct']:.1f}%",
                 fontsize=11)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_vs_observed.png", dpi=150)
    plt.close(fig)
    log.info("Saved predicted_vs_observed.png")


def plot_residuals(y_true, y_pred, out_dir: Path):
    """Residual plot to check for systematic errors."""
    residuals = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, s=40, alpha=0.7,
                    edgecolors="k", linewidths=0.5)
    axes[0].axhline(0, color="k", ls="--", lw=1)
    axes[0].set_xlabel("Predicted Volume (m³)")
    axes[0].set_ylabel("Residual (m³)")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].grid(True, alpha=0.3)

    # Residual distribution
    axes[1].hist(residuals, bins=15, edgecolor="k", alpha=0.7)
    axes[1].axvline(0, color="k", ls="--", lw=1)
    axes[1].set_xlabel("Residual (m³)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Residual Distribution (bias={np.mean(residuals):.4f} m³)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "residuals.png", dpi=150)
    plt.close(fig)
    log.info("Saved residuals.png")


def plot_feature_importance(importance_sorted, out_dir: Path, top_n=15):
    """Horizontal bar chart of feature importance."""
    top = importance_sorted[:top_n]
    names = [x[0] for x in top][::-1]
    values = [x[1] for x in top][::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
    ax.barh(names, values, color="#3498db", edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Feature Importance (permutation)")
    ax.set_title("XGBoost Feature Importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)
    log.info("Saved feature_importance.png")


def plot_shap_analysis(model, X: np.ndarray, feature_names: list,
                       out_dir: Path):
    """SHAP beeswarm + bar plots (Ye et al. 2025 used SHAP for eucalyptus).
    Provides richer interpretation than permutation importance alone."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(pd.DataFrame(X, columns=feature_names))

    # Beeswarm plot (shows feature impact direction)
    fig = plt.figure(figsize=(10, max(5, len(feature_names) * 0.3)))
    shap.plots.beeswarm(shap_values, show=False, max_display=15)
    plt.tight_layout()
    fig.savefig(out_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved shap_beeswarm.png")

    # Bar plot (mean |SHAP|)
    fig = plt.figure(figsize=(8, max(4, len(feature_names) * 0.3)))
    shap.plots.bar(shap_values, show=False, max_display=15)
    plt.tight_layout()
    fig.savefig(out_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved shap_bar.png")


def plot_baseline_comparison(metrics_xgb, baselines, out_dir: Path):
    """Bar chart comparing XGBoost vs baselines."""
    all_models = {"XGBoost (LOOCV)": metrics_xgb}
    all_models.update(baselines)

    names = list(all_models.keys())
    r2_vals = [all_models[n].get("R2", 0) for n in names]
    rrmse_vals = [all_models[n].get("rRMSE_pct", 0) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # R² comparison
    colors = ["#2ecc71" if n.startswith("XGBoost") else "#95a5a6" for n in names]
    axes[0].barh(names, r2_vals, color=colors, edgecolor="k", linewidth=0.5)
    axes[0].set_xlabel("R²")
    axes[0].set_title("R² Comparison")
    axes[0].set_xlim(0, 1.05)
    axes[0].grid(True, axis="x", alpha=0.3)

    # rRMSE comparison
    axes[1].barh(names, rrmse_vals, color=colors, edgecolor="k", linewidth=0.5)
    axes[1].set_xlabel("rRMSE (%)")
    axes[1].set_title("Relative RMSE Comparison")
    axes[1].axvline(6.7, color="red", ls="--", lw=1.5,
                    label="IDEAL 2025 baseline (6.7%)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.suptitle("Model Comparison: XGBoost vs Baselines", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close(fig)
    log.info("Saved model_comparison.png")


# ===================================================================
# Prediction
# ===================================================================

def predict_new(model, features_path: str, output_path: str,
                feature_names: list):
    """Apply trained model to new trees (no ground truth)."""
    df = pd.read_csv(features_path)
    X = df[feature_names].values
    y_pred = model.predict(X)

    df["predicted_volume_m3"] = np.round(y_pred, 6)

    # Freire-style back-calculation of DBH from predicted volume + height
    # DBH = sqrt(V / (pi/4 * H * f)) * 100, where f=0.5
    if "H_max" in df.columns:
        h = df["H_max"].values
        mask = (h > 0) & ~np.isnan(h) & (y_pred > 0)
        dbh_back = np.full(len(df), np.nan)
        dbh_back[mask] = np.sqrt(
            y_pred[mask] / (np.pi / 4 * h[mask] * 0.5)) * 100
        df["backcalc_dbh_cm"] = np.round(dbh_back, 2)

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.csv"
    df.to_csv(pred_path, index=False)
    log.info("Predictions saved to %s", pred_path)
    log.info("\n%s", df[["instance_id", "H_max", "crown_diameter",
              "predicted_volume_m3"]].to_string(index=False))
    if "backcalc_dbh_cm" in df.columns:
        log.info("Back-calculated DBH (from predicted volume + height):")
        log.info("\n%s", df[["instance_id", "predicted_volume_m3", "H_max",
                  "backcalc_dbh_cm"]].to_string(index=False))


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="XGBoost volume estimation from LiDAR features")
    parser.add_argument("--features", required=True,
                        help="Path to features.csv from tree_analysis.py")
    parser.add_argument("--ground-truth",
                        help="Path to the rigorous-scaling (cubagem rigorosa) CSV (for training)")
    parser.add_argument("--model",
                        help="Path to trained model .json (for prediction)")
    parser.add_argument("--output", required=True,
                        help="Output directory for model/results")
    parser.add_argument("--target", default="gt_volume_m3",
                        help="Target column name (default: gt_volume_m3)")
    parser.add_argument("--crown-only", action="store_true",
                        help="Exclude trunk features (simulates airborne UAV)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="SHAP feature selection: retrain with top K features (0=disabled)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    log.info("Loading features...")
    features = pd.read_csv(args.features)
    log.info("  %d trees, %d features", len(features), len(features.columns))

    # Mode 1: Prediction with existing model
    if args.model and not args.ground_truth:
        log.info("Prediction mode (using trained model)")
        model = xgb.XGBRegressor()
        model.load_model(args.model)

        # Load feature names from model metadata
        meta_path = Path(args.model).parent / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            feature_names = meta["feature_names"]
        else:
            raise FileNotFoundError(
                f"Model metadata not found at {meta_path}. "
                "Retrain or provide model_metadata.json with feature_names.")

        predict_new(model, args.features, args.output, feature_names)
        return

    # Mode 2: Training with ground truth
    if not args.ground_truth:
        parser.error("Either --ground-truth (for training) or "
                     "--model (for prediction) is required.")

    log.info("Training mode")
    log.info("=" * 60)

    # Load and process ground truth
    log.info("Step 1: Loading ground truth...")
    gt = load_ground_truth(args.ground_truth)

    # Match trees
    log.info("Step 2: Matching LiDAR trees to ground truth...")
    matched = match_trees_gps(features, gt)

    if len(matched) == 0:
        log.error("No trees matched. Check coordinates or tree IDs.")
        return

    # Prepare features
    crown_only = args.crown_only
    top_k = args.top_k
    log.info("Step 3: Preparing features (crown_only=%s)...", crown_only)
    X, y, feature_names = prepare_features(matched, target_col=args.target,
                                            crown_only=crown_only)

    if len(y) < 5:
        log.error("Only %d matched trees. Need at least 5 for LOOCV.", len(y))
        return

    # Train and evaluate
    log.info("Step 4: Training XGBoost with LOOCV...")
    model, y_pred, metrics, importance = train_xgboost_loocv(
        X, y, feature_names)

    # SHAP analysis + feature selection
    log.info("Step 4b: SHAP analysis...")
    plot_shap_analysis(model, X, feature_names, out_dir)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(pd.DataFrame(X, columns=feature_names))
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_ranking = sorted(zip(feature_names, mean_abs_shap),
                          key=lambda x: x[1], reverse=True)
    log.info("  SHAP ranking (top 10):")
    for name, val in shap_ranking[:10]:
        log.info("    %30s  %.4f", name, val)

    if top_k > 0:
        selected = [name for name, _ in shap_ranking[:top_k]]
        log.info("Step 4c: Retrain with SHAP top-%d: %s", top_k, selected)
        sel_idx = [feature_names.index(s) for s in selected]
        X_sel = X[:, sel_idx]
        model_sel, y_pred_sel, metrics_sel, imp_sel = train_xgboost_loocv(
            X_sel, y, selected)

        log.info("  Full (%d feat): R²=%.4f | top-%d: R²=%.4f  (delta=%+.4f)",
                 len(feature_names), metrics["R2"], top_k, metrics_sel["R2"],
                 metrics_sel["R2"] - metrics["R2"])

        if metrics_sel["R2"] > metrics["R2"]:
            log.info("  >>> SHAP top-%d WINS — using selected model", top_k)
            model, y_pred, metrics, importance = model_sel, y_pred_sel, metrics_sel, imp_sel
            feature_names = selected
            X = X_sel

    # Baselines
    log.info("Step 5: Computing baselines...")
    baselines = compute_baselines(matched, y)
    log.info("Baseline Results:")
    for name, bl_metrics in baselines.items():
        log.info("  %25s  R²=%.3f  rRMSE=%.1f%%",
                 name, bl_metrics['R2'], bl_metrics['rRMSE_pct'])

    # Plots
    log.info("Step 6: Generating plots...")
    plot_predicted_vs_observed(y, y_pred, metrics, out_dir)
    plot_residuals(y, y_pred, out_dir)
    plot_feature_importance(importance, out_dir)
    plot_baseline_comparison(metrics, baselines, out_dir)

    # Save model and metadata
    model_path = out_dir / "xgb_volume.json"
    model.save_model(str(model_path))
    log.info("Saved model to %s", model_path)

    meta = {
        "feature_names": feature_names,
        "metrics_loocv": metrics,
        "baselines": baselines,
        "xgb_params": XGB_PARAMS,
        "importance": {name: imp for name, imp in importance},
    }
    meta_path = out_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Saved metadata to %s", meta_path)

    # Save matched data with predictions
    matched["predicted_volume_m3"] = y_pred
    matched_path = out_dir / "matched_predictions.csv"
    matched.to_csv(matched_path, index=False)
    log.info("Saved matched predictions to %s", matched_path)

    # Summary
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info("  Trees matched:  %d", len(y))
    log.info("  Features used:  %d", len(feature_names))
    log.info("  XGBoost LOOCV:")
    log.info("    R²    = %s", metrics['R2'])
    log.info("    RMSE  = %.4f m³", metrics['RMSE_m3'])
    log.info("    rRMSE = %.1f%%", metrics['rRMSE_pct'])
    log.info("  Baseline to beat (IDEAL 2025): 6.7%% volume error")
    if metrics["rRMSE_pct"] < 6.7:
        log.info("  >>> XGBoost BEATS the baseline! (%.1f%% < 6.7%%)",
                 metrics['rRMSE_pct'])
    else:
        log.info("  >>> Baseline not yet beaten (%.1f%% vs 6.7%%)",
                 metrics['rRMSE_pct'])
    log.info("  Results in %s", out_dir)


if __name__ == "__main__":
    main()

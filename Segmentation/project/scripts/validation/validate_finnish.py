#!/usr/bin/env python3
"""
Validation: Finnish Scots Pine (Zenodo 3712900)

Validates diameter fitting (RANSAC + Kasa) on 230 stem-only TLS point clouds
against field-measured DBH. Pure geometry: recovering correct diameters from
clean stem cross-sections is a precondition for the rest of the pipeline.

Usage:
    conda run -n greenvista python scripts/validation/validate_finnish.py
"""

import sys
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import from production pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import setup_logger
from tree_analysis import (
    angular_coverage,
    fit_circle_ransac,
    fit_circle_kasa,
    _fit_slice,
    MIN_POINTS_VIABLE,
)

log = setup_logger("validate_finnish")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "finnish_scots_pine"
OUT_DIR = Path(__file__).resolve().parent / "results" / "finnish"


def load_ground_truth() -> pd.DataFrame:
    gt = pd.read_csv(DATA_DIR / "Scots_pines.txt", sep="\t")
    gt.columns = ["tree_id", "dbh_cm", "height_m"]
    log.info("Ground truth: %d trees, DBH %.1f–%.1f cm, height %.1f–%.1f m",
             len(gt), gt.dbh_cm.min(), gt.dbh_cm.max(),
             gt.height_m.min(), gt.height_m.max())
    return gt


def load_stem_points(tree_id: int) -> np.ndarray:
    laz_path = DATA_DIR / f"stempoints_tree{tree_id}.laz"
    if not laz_path.exists():
        return np.empty((0, 3))
    las = laspy.read(str(laz_path))
    return np.column_stack([las.x, las.y, las.z])


def extract_dbh_at_height(pts: np.ndarray, target_h: float = 1.3,
                           tol: float = 0.25) -> dict:
    """Extract diameter at a target height using production _fit_slice."""
    z = pts[:, 2]
    mask = (z >= target_h - tol) & (z < target_h + tol)
    slice_pts = pts[mask]
    if len(slice_pts) < MIN_POINTS_VIABLE:
        return None
    result = _fit_slice(slice_pts[:, :2], target_h)
    return result


def extract_dbh_raw(pts: np.ndarray, target_h: float = 1.3,
                    tol: float = 0.25) -> dict:
    """Extract diameter using both RANSAC and Kasa directly (no quality filters)
    for comparison."""
    z = pts[:, 2]
    mask = (z >= target_h - tol) & (z < target_h + tol)
    xy = pts[mask][:, :2]
    if len(xy) < 3:
        return None
    cx_r, cy_r, r_r, res_r = fit_circle_ransac(xy)
    cx_k, cy_k, r_k, res_k = fit_circle_kasa(xy)
    return {
        "n_points": len(xy),
        "ransac_diam_cm": round(2 * r_r * 100, 2),
        "ransac_residual_cm": round(res_r * 100, 3),
        "kasa_diam_cm": round(2 * r_k * 100, 2),
        "kasa_residual_cm": round(res_k * 100, 3),
        "angular_coverage": round(angular_coverage(xy), 1),
    }


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth()

    rows = []
    for _, row in gt.iterrows():
        tid = int(row.tree_id)
        pts = load_stem_points(tid)
        if len(pts) == 0:
            log.warning("Tree %d: no point cloud found", tid)
            continue

        # Production pipeline result
        prod = extract_dbh_at_height(pts, 1.3)
        # Raw fitting comparison
        raw = extract_dbh_raw(pts, 1.3)

        entry = {
            "tree_id": tid,
            "gt_dbh_cm": row.dbh_cm,
            "gt_height_m": row.height_m,
            "n_total_pts": len(pts),
        }

        if raw:
            entry.update({
                "n_slice_pts": raw["n_points"],
                "angular_coverage": raw["angular_coverage"],
                "ransac_dbh_cm": raw["ransac_diam_cm"],
                "ransac_residual_cm": raw["ransac_residual_cm"],
                "kasa_dbh_cm": raw["kasa_diam_cm"],
                "kasa_residual_cm": raw["kasa_residual_cm"],
            })
        else:
            entry.update({
                "n_slice_pts": 0,
                "angular_coverage": np.nan,
                "ransac_dbh_cm": np.nan,
                "ransac_residual_cm": np.nan,
                "kasa_dbh_cm": np.nan,
                "kasa_residual_cm": np.nan,
            })

        if prod:
            entry["pipeline_dbh_cm"] = prod["diameter_cm"]
            entry["pipeline_fit_type"] = prod["fit_type"]
        else:
            entry["pipeline_dbh_cm"] = np.nan
            entry["pipeline_fit_type"] = "skipped"

        rows.append(entry)

    df = pd.DataFrame(rows)

    # Compute errors
    for method in ["ransac", "kasa", "pipeline"]:
        col = f"{method}_dbh_cm"
        if col in df.columns:
            df[f"{method}_error_cm"] = df[col] - df["gt_dbh_cm"]
            df[f"{method}_pct_error"] = (
                (df[col] - df["gt_dbh_cm"]) / df["gt_dbh_cm"] * 100
            )

    # Save results
    csv_path = OUT_DIR / "dbh_comparison.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved %s (%d trees)", csv_path, len(df))

    # Summary stats
    valid = df.dropna(subset=["ransac_dbh_cm"])
    log.info("")
    log.info("=== RESULTS ===")
    log.info("Trees with valid fits: %d / %d (%.0f%%)",
             len(valid), len(df), 100 * len(valid) / len(df))
    log.info("")

    for method in ["ransac", "kasa", "pipeline"]:
        err_col = f"{method}_error_cm"
        pct_col = f"{method}_pct_error"
        if err_col not in df.columns:
            continue
        v = df.dropna(subset=[err_col])
        if len(v) == 0:
            continue
        rmse = np.sqrt((v[err_col] ** 2).mean())
        mae = v[err_col].abs().mean()
        bias = v[err_col].mean()
        r2 = 1 - ((v[err_col] ** 2).sum() /
                   ((v["gt_dbh_cm"] - v["gt_dbh_cm"].mean()) ** 2).sum())
        log.info("%s (N=%d):", method.upper(), len(v))
        log.info("  RMSE  = %.2f cm", rmse)
        log.info("  MAE   = %.2f cm", mae)
        log.info("  Bias  = %.2f cm", bias)
        log.info("  R²    = %.4f", r2)
        log.info("  Mean %% error = %.1f%%", v[pct_col].mean())
        log.info("")

    # Plots
    plot_scatter(df, OUT_DIR)
    plot_error_distribution(df, OUT_DIR)
    log.info("Done. Results in %s", OUT_DIR)


def plot_scatter(df: pd.DataFrame, out_dir: Path):
    valid = df.dropna(subset=["ransac_dbh_cm"])
    if len(valid) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, method, label in [
        (axes[0], "ransac_dbh_cm", "RANSAC"),
        (axes[1], "kasa_dbh_cm", "Kasa+NM"),
    ]:
        v = df.dropna(subset=[method])
        ax.scatter(v["gt_dbh_cm"], v[method], s=15, alpha=0.6, edgecolors="k",
                   linewidths=0.3)
        lims = [0, max(v["gt_dbh_cm"].max(), v[method].max()) * 1.1]
        ax.plot(lims, lims, "k--", lw=1, label="1:1")

        err = v[method] - v["gt_dbh_cm"]
        rmse = np.sqrt((err ** 2).mean())
        r2 = 1 - ((err ** 2).sum() /
                   ((v["gt_dbh_cm"] - v["gt_dbh_cm"].mean()) ** 2).sum())

        ax.set_xlabel("Field DBH (cm)")
        ax.set_ylabel(f"{label} DBH (cm)")
        ax.set_title(f"{label}: R²={r2:.3f}, RMSE={rmse:.2f} cm (N={len(v)})")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Finnish Scots Pine — DBH Validation (stem-only TLS)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "dbh_scatter.png", dpi=150)
    plt.close(fig)
    log.info("Saved dbh_scatter.png")


def plot_error_distribution(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, method, label in [
        (axes[0], "ransac_error_cm", "RANSAC"),
        (axes[1], "kasa_error_cm", "Kasa+NM"),
    ]:
        v = df[method].dropna()
        if len(v) == 0:
            continue
        ax.hist(v, bins=30, edgecolor="k", alpha=0.7)
        ax.axvline(0, color="r", ls="--", lw=1.5)
        ax.axvline(v.mean(), color="blue", ls="-", lw=1.5,
                   label=f"bias={v.mean():.2f} cm")
        ax.set_xlabel("Error (cm)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} Error Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "error_distribution.png", dpi=150)
    plt.close(fig)
    log.info("Saved error_distribution.png")


if __name__ == "__main__":
    run()

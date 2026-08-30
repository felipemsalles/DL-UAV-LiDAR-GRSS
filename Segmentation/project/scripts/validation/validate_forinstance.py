#!/usr/bin/env python3
"""
Validation: FOR-instanceV2

Validates that tree_analysis.py produces physically plausible features
when given properly segmented data with real semantic labels
(stem/branch/leaf). Runs on a subset of FOR-instanceV2 PLY files.

Usage:
    conda run -n greenvista python scripts/validation/validate_forinstance.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import setup_logger
from tree_analysis import (
    trunk_density_profile,
    extract_diameters,
    crown_features,
    trunk_features,
    normalize_height,
    SEM_GROUND,
    SEM_WOOD,
    SEM_LEAF,
)

log = setup_logger("validate_forinstance")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / \
    "models" / "FF3D_inference" / "ff3d_forestsens" / "data" / \
    "ForAINetV2" / "train_val_data"
OUT_DIR = Path(__file__).resolve().parent / "results" / "forinstance"

# FOR-instanceV2 semantic labels differ from FF3D output:
# FOR-instance: 1=terrain/ground, 2=wood/stem, 3=leaf/live branches
# FF3D output:  0=ground, 1=wood, 2=leaf
# We need to remap
FORINSTANCE_GROUND = 1
FORINSTANCE_WOOD = 2
FORINSTANCE_LEAF = 3

# Limit number of plots to process (FOR-instance has 65 train+val files)
MAX_PLOTS = 15
# Skip PLY files larger than this (BlueCat_train is 7.6 GB — too slow)
MAX_FILE_SIZE_MB = 200


def load_forinstance_ply(path: str) -> dict:
    """Load a FOR-instanceV2 PLY and split into per-tree dicts,
    remapping semantic labels to match FF3D convention."""
    ply = PlyData.read(path)
    el = ply.elements[0]

    xyz = np.column_stack([np.array(el["x"]), np.array(el["y"]),
                           np.array(el["z"])])
    semantic = np.array(el["semantic_seg"])
    instance = np.array(el["treeID"])

    # Remap semantic labels: FOR-instance → FF3D convention
    sem_remapped = np.zeros_like(semantic)
    sem_remapped[semantic == FORINSTANCE_GROUND] = SEM_GROUND
    sem_remapped[semantic == FORINSTANCE_WOOD] = SEM_WOOD
    sem_remapped[semantic == FORINSTANCE_LEAF] = SEM_LEAF

    trees = {}
    for iid in np.unique(instance):
        if iid == 0:  # 0 = unassigned/background in FOR-instance
            continue
        mask = instance == iid
        pts = xyz[mask]
        sem = sem_remapped[mask]

        # Skip very small trees
        if len(pts) < 50:
            continue

        trees[int(iid)] = {
            "points": pts,
            "semantic": sem,
            "score": 1.0,
            "wood_points": pts[sem == SEM_WOOD],
            "leaf_points": pts[sem == SEM_LEAF],
            "ground_points": pts[sem == SEM_GROUND],
        }
    return trees


def process_plot(ply_path: Path) -> list:
    """Process one FOR-instance plot, return list of per-tree feature dicts."""
    plot_name = ply_path.stem
    size_mb = ply_path.stat().st_size / (1024 * 1024)
    log.info("  Loading %s (%.1f MB)...", plot_name, size_mb)
    trees = load_forinstance_ply(str(ply_path))
    log.info("  %s: %d trees found", plot_name, len(trees))

    rows = []
    for i, (tid, tree) in enumerate(sorted(trees.items())):
        profile = trunk_density_profile(tree)
        z0 = normalize_height(tree)
        diams = extract_diameters(tree, profile)
        crown = crown_features(tree, z0)
        trunk = trunk_features(diams, tree_height=crown["H_max"])

        log.info("    tree %d/%d (id=%d, %d pts, %d wood)",
                 i + 1, len(trees), tid, len(tree["points"]),
                 len(tree["wood_points"]))
        row = {
            "plot": plot_name,
            "tree_id": tid,
            "n_points": len(tree["points"]),
            "n_wood": len(tree["wood_points"]),
            "n_leaf": len(tree["leaf_points"]),
        }
        row.update(crown)
        row.update(trunk)
        rows.append(row)

    return rows


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(DATA_DIR.glob("*.ply"))
    if not ply_files:
        log.error("No PLY files found in %s", DATA_DIR)
        return

    # Filter out huge files and sort by size (smallest first)
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    ply_files = [f for f in ply_files if f.stat().st_size <= max_bytes]
    ply_files.sort(key=lambda f: f.stat().st_size)
    log.info("Files under %d MB: %d", MAX_FILE_SIZE_MB, len(ply_files))

    # Process a subset
    ply_subset = ply_files[:MAX_PLOTS]
    log.info("Processing %d / %d FOR-instanceV2 plots...",
             len(ply_subset), len(ply_files))

    all_rows = []
    for ply_path in ply_subset:
        rows = process_plot(ply_path)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    csv_path = OUT_DIR / "features.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved %s (%d trees from %d plots)", csv_path, len(df),
             len(ply_subset))

    # === Plausibility checks ===
    log.info("")
    log.info("=== PLAUSIBILITY CHECKS ===")
    log.info("")

    # 1. Height
    log.info("Height (H_max):")
    log.info("  Range: %.1f – %.1f m", df["H_max"].min(), df["H_max"].max())
    log.info("  Mean:  %.1f m", df["H_max"].mean())
    tall = (df["H_max"] > 50).sum()
    short = (df["H_max"] < 1).sum()
    log.info("  Implausible: %d > 50m, %d < 1m", tall, short)
    height_ok = tall == 0 and short == 0
    log.info("  PASS" if height_ok else "  FAIL")

    # 2. DBH
    valid_dbh = df.dropna(subset=["d_1.3m"])
    log.info("")
    log.info("DBH (d_1.3m):")
    log.info("  Trees with DBH: %d / %d (%.0f%%)",
             len(valid_dbh), len(df), 100 * len(valid_dbh) / len(df))
    if len(valid_dbh) > 0:
        log.info("  Range: %.1f – %.1f cm",
                 valid_dbh["d_1.3m"].min(), valid_dbh["d_1.3m"].max())
        log.info("  Mean:  %.1f cm", valid_dbh["d_1.3m"].mean())
        # Plausible range for forest trees: 5-100 cm
        implausible = ((valid_dbh["d_1.3m"] < 5) |
                       (valid_dbh["d_1.3m"] > 100)).sum()
        log.info("  Implausible (<5 or >100 cm): %d", implausible)
        dbh_ok = implausible / len(valid_dbh) < 0.1  # <10% implausible
        log.info("  PASS" if dbh_ok else "  FAIL")
    else:
        dbh_ok = False
        log.info("  FAIL (no DBH measurements)")

    # 3. Crown diameter
    valid_cd = df.dropna(subset=["crown_diameter"])
    log.info("")
    log.info("Crown diameter:")
    if len(valid_cd) > 0:
        log.info("  Range: %.1f – %.1f m",
                 valid_cd["crown_diameter"].min(),
                 valid_cd["crown_diameter"].max())
        log.info("  Mean:  %.1f m", valid_cd["crown_diameter"].mean())
        cd_ok = valid_cd["crown_diameter"].max() < 30  # no crown > 30m
        log.info("  PASS" if cd_ok else "  FAIL")
    else:
        cd_ok = False
        log.info("  FAIL (no crown diameter)")

    # 4. Wood point ratio
    log.info("")
    log.info("Wood point ratio:")
    log.info("  Range: %.3f – %.3f",
             df["wood_ratio"].min(), df["wood_ratio"].max())
    log.info("  Mean:  %.3f", df["wood_ratio"].mean())

    # 5. Viable heights
    log.info("")
    log.info("Viable height bins (for diameter fitting):")
    has_diams = (df["n_viable_heights"] > 0).sum()
    log.info("  Trees with ≥1 viable bin: %d / %d (%.0f%%)",
             has_diams, len(df), 100 * has_diams / len(df))
    log.info("  Mean viable heights: %.1f", df["n_viable_heights"].mean())

    # 6. Smalian volume
    valid_smalian = df.dropna(subset=["smalian_volume_m3"])
    valid_smalian = valid_smalian[valid_smalian["smalian_volume_m3"] > 0]
    log.info("")
    log.info("Smalian volume:")
    if len(valid_smalian) > 0:
        log.info("  Trees with volume: %d", len(valid_smalian))
        log.info("  Range: %.4f – %.4f m³",
                 valid_smalian["smalian_volume_m3"].min(),
                 valid_smalian["smalian_volume_m3"].max())
        # Plausible: 0.001 to 10 m³
        smalian_ok = valid_smalian["smalian_volume_m3"].max() < 10
        log.info("  PASS" if smalian_ok else "  FAIL")
    else:
        smalian_ok = True  # no data to fail on
        log.info("  No Smalian volumes computed (OK if few trunk points)")

    # === Summary ===
    log.info("")
    log.info("=== SUMMARY ===")
    all_pass = height_ok and dbh_ok and cd_ok
    checks = {"height": height_ok, "dbh": dbh_ok, "crown_diameter": cd_ok,
              "smalian": smalian_ok}
    for name, passed in checks.items():
        log.info("  %20s: %s", name, "PASS" if passed else "FAIL")
    log.info("")
    if all_pass:
        log.info("  ALL CHECKS PASSED — feature extraction produces "
                 "plausible values on properly segmented data.")
    else:
        failed = [k for k, v in checks.items() if not v]
        log.info("  FAILED CHECKS: %s — investigate before trusting "
                 "pipeline results.", failed)

    # Plots
    plot_distributions(df, OUT_DIR)
    log.info("Done. Results in %s", OUT_DIR)


def plot_distributions(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # H_max
    axes[0, 0].hist(df["H_max"].dropna(), bins=30, edgecolor="k", alpha=0.7)
    axes[0, 0].set_xlabel("Tree Height (m)")
    axes[0, 0].set_title(f"Height (N={len(df)})")

    # DBH
    valid_dbh = df["d_1.3m"].dropna()
    if len(valid_dbh) > 0:
        axes[0, 1].hist(valid_dbh, bins=30, edgecolor="k", alpha=0.7)
    axes[0, 1].set_xlabel("DBH (cm)")
    axes[0, 1].set_title(f"DBH at 1.3m (N={len(valid_dbh)})")

    # Crown diameter
    valid_cd = df["crown_diameter"].dropna()
    if len(valid_cd) > 0:
        axes[0, 2].hist(valid_cd, bins=30, edgecolor="k", alpha=0.7)
    axes[0, 2].set_xlabel("Crown Diameter (m)")
    axes[0, 2].set_title(f"Crown Diameter (N={len(valid_cd)})")

    # Wood ratio
    axes[1, 0].hist(df["wood_ratio"].dropna(), bins=30, edgecolor="k", alpha=0.7)
    axes[1, 0].set_xlabel("Wood Point Ratio")
    axes[1, 0].set_title("Wood Ratio")

    # N viable heights
    axes[1, 1].hist(df["n_viable_heights"], bins=range(0, 30), edgecolor="k",
                    alpha=0.7)
    axes[1, 1].set_xlabel("Viable Height Bins")
    axes[1, 1].set_title("Diameter Measurement Count")

    # N points
    axes[1, 2].hist(np.log10(df["n_points"].clip(lower=1)), bins=30,
                    edgecolor="k", alpha=0.7)
    axes[1, 2].set_xlabel("log10(N points)")
    axes[1, 2].set_title("Point Count Distribution")

    fig.suptitle("FOR-instanceV2 — Feature Distributions", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "distributions.png", dpi=150)
    plt.close(fig)
    log.info("Saved distributions.png")


if __name__ == "__main__":
    run()

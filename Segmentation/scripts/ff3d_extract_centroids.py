#!/usr/bin/env python3
"""Persist the raw FF3D per-tile instance centroids to a committable CSV.

The FF3D output bucket is cleared at the start of every run (``CLEAR_OUTPUT_BEFORE_RUN``), so the raw
detection PLYs do not survive a later run, and the per-plot counts alone
(``ff3d_detection_overlap.csv``) do not carry them. This script re-derives the raw per-instance
detections from the round2 PLYs and writes one row per FF3D instance to a tracked CSV under
``data/detections/`` (not ``manual_match/``, which is gitignored). Every downstream score
(NMS-merge, in-plot count, position accuracy) can then be reproduced from this CSV alone.

For each round2 PLY the FF3D local→UTM shift is recovered from the backup tile (the laz the overlap
tiler copies aside before ``bucket_in`` auto-clears), using the same recovery the 6b scorer uses
(``ff3d_detection_overlap.instance_centroids``, the converter's mean shift), cross-checked against the
baseline min-matching (``ff3d_detection_numbers.centroids_utm``); the two agree to ~0 m in practice.
Re-merging this CSV at a given radius therefore reproduces ``ff3d_detection_overlap.py`` counts exactly.

Output columns (one row per FF3D instance):
    tile, plot_id, talhao, parcela, rotation, cx, cy, z_top, n_pts, instance_id
where (cx, cy) = the instance centroid in absolute UTM (EPSG:31982), z_top = max z of the instance's
points, n_pts = instance point count, instance_id = FF3D ``instance_pred`` label, rotation = the tile's
z-axis rotation in degrees (0 for the plain overlap regen; carried from the manifest if present).

Run (regen 13-plot persist — the committable record):
    PYTHONPATH=. GREENVISTA_LAZ_DIR=/tmp/lazall python scripts/ff3d_extract_centroids.py \
        --manifest <backup>/ff3d_tiles_overlap/tiles_manifest.csv \
        --ply-dir  <round2 PLY dir> \
        --out      data/detections/ff3d_overlap_centroids_13plot.csv --verify 1.5

Verify only (re-merge an existing CSV, no PLYs needed):
    PYTHONPATH=. python scripts/ff3d_extract_centroids.py --verify-csv <csv> --manifest <manifest> --radius 1.5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ff3d_detection_numbers import find_plys, read_panoptic  # noqa: E402
from ff3d_detection_overlap import instance_centroids, nms_merge, in_plot_count  # noqa: E402

from greenvista import config  # noqa: E402

# Defaults: the 13-plot overlap backup manifest + the FF3D output bucket + the committable CSV.
DEFAULT_MANIFEST = Path(config.LAZ_DIR).parent / "ff3d_tiles_overlap" / "tiles_manifest.csv"
DEFAULT_PLY_DIR = config.REPO / "project/models/FF3D_inference/FF3D_oracle/bucket_out_folder"
DEFAULT_OUT = config.REPO / "data" / "detections" / "ff3d_overlap_centroids_13plot.csv"
PLOT_R = config.PLOT_RADIUS_M

COLUMNS = ["tile", "plot_id", "talhao", "parcela", "rotation",
           "cx", "cy", "z_top", "n_pts", "instance_id"]


def instance_rows(ply_path, backup_laz):
    """One dict per FF3D instance: (instance_id, cx, cy [UTM], z_top, n_pts), plus the shift disc.

    cx/cy/n_pts come from the same recovery the 6b scorer uses (``instance_centroids``, converter mean
    shift), so re-merging reproduces ``ff3d_detection_overlap.py`` exactly. z_top and instance_id are
    read in the matching instance order (``np.unique(inst)`` skipping negatives). ``disc`` is the
    mean-vs-min shift discrepancy (metres), for the cross-check log."""
    cents, sizes, disc = instance_centroids(ply_path, backup_laz)  # scorer's exact UTM recovery
    lx, ly, lz, inst = read_panoptic(ply_path)
    ids = [int(i) for i in np.unique(inst) if i >= 0]
    if len(ids) != len(cents):
        raise AssertionError(f"instance-count mismatch in {Path(ply_path).name}: "
                             f"{len(ids)} ids vs {len(cents)} centroids")
    rows = []
    for k, iid in enumerate(ids):
        m = inst == iid
        rows.append({"instance_id": iid,
                     "cx": float(cents[k, 0]), "cy": float(cents[k, 1]),
                     "z_top": float(lz[m].max()), "n_pts": int(sizes[k])})
    return rows, disc


def _rotation_of(row):
    """z-axis rotation (deg, int) for a manifest row — 0 for plain overlap; carried if present."""
    for key in ("rotation", "angle_deg"):
        if key in row and pd.notna(row[key]):
            return int(round(float(row[key])))
    return 0


def extract(manifest, backup_dir, ply_dir, out_path, rot0_only=False):
    man = pd.read_csv(manifest)
    if rot0_only and {"angle_deg", "flip"}.issubset(man.columns):
        before = len(man)
        man = man[(man.angle_deg == 0.0) & (man.flip == 0)].reset_index(drop=True)
        print(f"rot0-only filter: kept {len(man)}/{before} manifest rows (angle_deg==0 & flip==0)")

    plys = find_plys(ply_dir)
    print(f"found {len(plys)} PLY(s) under {ply_dir}")
    print(f"manifest {manifest} → {len(man)} tiles ; backup laz dir {backup_dir}")
    if not plys:
        print("no FF3D output found — run the pipeline first."); return None

    rows, discs, missing = [], [], []
    for _, t in man.iterrows():
        cand = [p for p in plys if str(t.tile) in Path(p).name]
        if not cand:
            missing.append(str(t.tile)); continue
        backup_laz = Path(backup_dir) / f"{t.tile}.laz"
        if not backup_laz.exists():
            missing.append(f"{t.tile} (no backup laz)"); continue
        try:
            irows, disc = instance_rows(cand[0], backup_laz)
        except Exception as e:
            print(f"  {t.tile}: FAILED — {e}"); missing.append(str(t.tile)); continue
        if not np.isnan(disc):
            discs.append(disc)
        rot = _rotation_of(t)
        for r in irows:
            rows.append({"tile": str(t.tile), "plot_id": t.plot_id,
                         "talhao": int(t.talhao), "parcela": int(t.parcela),
                         "rotation": rot, **r})

    df = pd.DataFrame(rows, columns=COLUMNS)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nwrote {len(df)} instance rows from {df.tile.nunique()} tiles → {out_path}")
    if missing:
        print(f"  {len(missing)} manifest tile(s) had no PLY/backup: {missing[:12]}"
              + (" ..." if len(missing) > 12 else ""))
    if discs:
        print(f"  shift cross-check (mean vs min recovery): max {max(discs):.4f} m, "
              f"median {np.median(discs):.4f} m over {len(discs)} tiles")
    print("  instances per plot:")
    per = df.groupby("plot_id").agg(tiles=("tile", "nunique"), instances=("instance_id", "size"))
    print(per.to_string())
    return df


def verify(df, manifest, radius):
    """Re-merge the CSV per plot (pool all its tiles' centroids, NMS at `radius`, count in the 12 m
    plot) → reproduces ``ff3d_detection_overlap.py``'s merged_in_plot. Uses n_pts as the NMS size and
    the plot centre from the manifest (its cx/cy columns are the PLOT centre, not the instance one)."""
    man = pd.read_csv(manifest)
    centre = (man.drop_duplicates("plot_id").set_index("plot_id")[["cx", "cy", "field_n_arv"]])
    print(f"\n=== verify: re-merge CSV at radius {radius} m (reproduces ff3d_detection_overlap) ===")
    out = []
    for pid, g in df.groupby("plot_id"):
        cents = g[["cx", "cy"]].to_numpy(float)
        sizes = g["n_pts"].to_numpy(int)
        merged = nms_merge(cents, sizes, radius)
        pcx, pcy = float(centre.loc[pid, "cx"]), float(centre.loc[pid, "cy"])
        merged_in = in_plot_count(merged, pcx, pcy)
        field = float(centre.loc[pid, "field_n_arv"])
        out.append({"plot_id": pid, "pooled_inst": len(cents), "merged_total": len(merged),
                    "merged_in_plot": merged_in, "field": int(field),
                    "det_%": round(100 * merged_in / field, 0)})
    res = pd.DataFrame(out)
    print(res.to_string(index=False))
    tot_in = int(res.merged_in_plot.sum()); tot_field = int(res.field.sum())
    print(f" TOTAL merged-in-plot {tot_in} / field {tot_field} = {100*tot_in/tot_field:.1f}%")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--backup-dir", default=None, help="dir with the backup <tile>.laz (default: manifest's dir)")
    ap.add_argument("--ply-dir", default=str(DEFAULT_PLY_DIR))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--rot0-only", action="store_true",
                    help="keep only angle_deg==0 & flip==0 rows (for testing on 7a/tta leftovers)")
    ap.add_argument("--verify", type=float, default=None, metavar="RADIUS",
                    help="after extracting, re-merge at RADIUS m and print per-plot in-plot counts")
    ap.add_argument("--verify-csv", default=None,
                    help="skip extraction; just re-merge this existing CSV (needs --manifest, --radius)")
    ap.add_argument("--radius", type=float, default=1.5, help="merge radius for --verify-csv")
    args = ap.parse_args()

    if args.verify_csv:
        df = pd.read_csv(args.verify_csv)
        verify(df, args.manifest, args.radius)
        return

    backup_dir = args.backup_dir or str(Path(args.manifest).parent)
    df = extract(args.manifest, backup_dir, args.ply_dir, args.out, rot0_only=args.rot0_only)
    if df is not None and args.verify is not None and len(df):
        verify(df, args.manifest, args.verify)


if __name__ == "__main__":
    main()

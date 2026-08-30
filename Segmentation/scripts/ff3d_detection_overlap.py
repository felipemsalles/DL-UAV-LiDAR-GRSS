#!/usr/bin/env python3
"""Idea #6b spike: parse overlapping-tile FF3D output → NMS-merged detection for T1 & T5.

For each plot we ran a 3×3 grid of 32 m tiles at stride 16 m (see `ff3d_make_tiles_overlap.py`).
Here we pool the instance centroids from all 9 offset tiles into absolute UTM, greedily NMS-merge
duplicates (a tree detected in several overlapping tiles) within a merge radius, then count the
merged instances inside the 12 m plot radius → detection vs field count. Compared head-to-head with
the single-tile baseline (`manual_match/ff3d_detection.csv`).

Merge radius: eucalyptus nearest-neighbour spacing ≈ 1.8 m, so a radius < 1.8 m cannot merge two
distinct trees; we use 1.2 m (task-specified). NMS keeps the largest instance (most complete crown)
in each cluster, which is the interior-tile detection rather than a cropped edge-tile one.

UTM recovery: the laz→ply converter shifts each tile by subtracting that tile's own mean(x),mean(y)
(`run_oracle_pipeline.sh`). We recover the exact shift from the backup tile's mean (robust across
tiles), and cross-check it against the baseline min-matching in `centroids_utm` (reported as
`max_shift_disc_m`). Reusing `read_panoptic` / `centroids_utm` / `find_plys` from the baseline parser.

Run: PYTHONPATH=. GREENVISTA_LAZ_DIR=<lazdir> python scripts/ff3d_detection_overlap.py [ply_dir] [merge_radius]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, laspy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ff3d_detection_numbers import find_plys, read_panoptic, centroids_utm  # noqa: E402

from greenvista import config  # noqa: E402

BACKUP = Path(config.LAZ_DIR).parent / "ff3d_tiles_overlap"
PLOT_R = config.PLOT_RADIUS_M
PLOT_AREA_HA = config.PLOT_AREA_HA
MERGE_RADIUS = 1.2  # metres; < NN spacing (~1.8 m) so distinct trees are never merged
BASELINE_CSV = config.OUT_DIR / "ff3d_detection.csv"


def instance_centroids(ply_path, backup_laz):
    """Instance centroids in absolute UTM via the exact converter shift (mean of the backup tile).

    Returns (cents Nx2 UTM, sizes N point-counts, max_disc vs baseline min-matching)."""
    lx, ly, lz, inst = read_panoptic(ply_path)
    las = laspy.read(str(backup_laz))
    mx, my = float(np.asarray(las.x).mean()), float(np.asarray(las.y).mean())
    cents, sizes = [], []
    for iid in np.unique(inst):
        if iid < 0:
            continue
        m = inst == iid
        cents.append([lx[m].mean() + mx, ly[m].mean() + my])
        sizes.append(int(m.sum()))
    cents = np.array(cents).reshape(-1, 2)
    sizes = np.array(sizes, dtype=int)
    # Cross-check against the baseline min-matching shift.
    disc = np.nan
    try:
        c_base, *_ = centroids_utm(ply_path, backup_laz)
        if len(c_base) == len(cents) and len(cents):
            disc = float(np.abs(c_base - cents).max())
    except Exception:
        pass
    return cents, sizes, disc


def nms_merge(cents, sizes, radius):
    """Greedy NMS: keep the largest instance in each cluster, suppress others within `radius`."""
    if len(cents) == 0:
        return cents
    order = np.argsort(-sizes)  # biggest crown first
    kept = []
    for i in order:
        p = cents[i]
        if not kept or np.min(np.hypot(*(np.asarray(kept) - p).T)) > radius:
            kept.append(p)
    return np.asarray(kept).reshape(-1, 2)


def in_plot_count(cents, cx, cy):
    if len(cents) == 0:
        return 0
    d2 = (cents[:, 0] - cx) ** 2 + (cents[:, 1] - cy) ** 2
    return int((d2 <= PLOT_R ** 2).sum())


def main():
    ply_dir = sys.argv[1] if len(sys.argv) > 1 else \
        str(config.REPO / "project/models/FF3D_inference/FF3D_oracle/bucket_out_folder")
    merge_r = float(sys.argv[2]) if len(sys.argv) > 2 else MERGE_RADIUS

    man = pd.read_csv(BACKUP / "tiles_manifest.csv")
    base = pd.read_csv(BASELINE_CSV).set_index("plot_id")
    plys = find_plys(ply_dir)
    print(f"found {len(plys)} PLY(s) under {ply_dir}  (merge radius = {merge_r} m)")
    if not plys:
        print("no FF3D output yet — run the pipeline first."); return

    rows, discs = [], []
    for (t, p), g in man.groupby(["talhao", "parcela"]):
        cx, cy = float(g.cx.iloc[0]), float(g.cy.iloc[0])
        field = float(g.field_n_arv.iloc[0])
        plot_id = g.plot_id.iloc[0]
        pooled_c, pooled_s = [], []
        raw_in_plot, n_found = 0, 0
        for _, tr in g.iterrows():
            cand = [pp for pp in plys if tr.tile in Path(pp).name]
            if not cand:
                continue
            n_found += 1
            c, s, disc = instance_centroids(cand[0], BACKUP / f"{tr.tile}.laz")
            if not np.isnan(disc):
                discs.append(disc)
            if len(c):
                pooled_c.append(c); pooled_s.append(s)
                raw_in_plot += in_plot_count(c, cx, cy)  # pre-merge (shows duplication)
        if pooled_c:
            allc = np.vstack(pooled_c); alls = np.concatenate(pooled_s)
        else:
            allc = np.empty((0, 2)); alls = np.empty((0,), int)
        merged = nms_merge(allc, alls, merge_r)
        merged_in_plot = in_plot_count(merged, cx, cy)
        base_in_plot = int(base.loc[plot_id, "ff3d_in_plot"]) if plot_id in base.index else np.nan
        rows.append({
            "plot_id": plot_id, "talhao": int(t), "parcela": int(p), "field_n_arv": field,
            "n_tiles_found": n_found, "pooled_instances": len(allc),
            "raw_in_plot_premerge": raw_in_plot, "merged_in_plot": merged_in_plot,
            "baseline_in_plot": base_in_plot,
            "overlap_det": round(merged_in_plot / field, 3),
            "baseline_det": round(base_in_plot / field, 3) if not np.isnan(base_in_plot) else np.nan,
            "merged_stems_ha": round(merged_in_plot / PLOT_AREA_HA, 0),
        })
        print(f"  {plot_id}: {n_found}/9 tiles, pooled {len(allc)} inst → raw {raw_in_plot} in-plot "
              f"→ NMS-merged {merged_in_plot} vs baseline {base_in_plot} vs field {int(field)} "
              f"({100*merged_in_plot/field:.0f}% vs base {100*base_in_plot/field:.0f}%)")

    res = pd.DataFrame(rows)
    print("\n=== per-plot (overlap NMS vs single-tile baseline) ===")
    print(res[["plot_id", "field_n_arv", "baseline_in_plot", "merged_in_plot",
               "baseline_det", "overlap_det"]].to_string(index=False))

    print("\n=== per-stand (spike comparison) ===")
    by = res.groupby("talhao").agg(field=("field_n_arv", "sum"),
                                   baseline=("baseline_in_plot", "sum"),
                                   overlap=("merged_in_plot", "sum")).reset_index()
    by["baseline_det_%"] = (100 * by.baseline / by.field).round(0)
    by["overlap_det_%"] = (100 * by.overlap / by.field).round(0)
    by["delta_pts"] = (by["overlap_det_%"] - by["baseline_det_%"]).round(0)
    print(by.to_string(index=False))

    if discs:
        print(f"\nUTM shift cross-check (exact-mean vs baseline min-matching): "
              f"max {max(discs):.3f} m, median {np.median(discs):.3f} m over {len(discs)} tiles")

    gain = by["delta_pts"].max()
    print(f"\nVERDICT: best per-stand gain = {gain:+.0f} pts "
          f"({'>= +5 → overlap helps, scale to 13' if gain >= 5 else '< +5 → null, edge-loss not the ceiling'})")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.OUT_DIR / "ff3d_detection_overlap.csv", index=False)
    print(f"\nwrote {config.OUT_DIR / 'ff3d_detection_overlap.csv'}")


if __name__ == "__main__":
    main()

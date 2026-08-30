#!/usr/bin/env python3
"""Adversarial validation of the overlapping-tile + NMS FF3D detection gain.

The overlap spike (`ff3d_detection_overlap.py`) reported large in-plot detection gains over the
single-tile baseline (1_1 62->79 %, 1_2 46->77 %, 5_9 22->43 %). This script decides whether those
gains are real distinct-tree recovery or an NMS under-merge artifact: the same physical tree seen in
several overlapping tiles, its centroids landing more than merge_radius apart and therefore counted
several times, which inflates recall.

Two tests, both CPU-only (no new inference):

1. Merge-radius sensitivity. Re-score merged in-plot detection at radii 1.0/1.2/1.5/2.0/2.5 m.
   Eucalyptus nearest-neighbour spacing is ~1.8 m, so a radius up to ~1.6 m cannot merge two distinct
   trees. If merged detection stays roughly stable across 1.2->1.6 m the gain is real; if it collapses
   toward the baseline as the radius grows, it was duplicates. Any plot whose merged count exceeds its
   field stem count is flagged as over-detection.

2. Position match (stand 1 only -- the one stand with surveyed cubed-tree GPS positions). The
   baseline single-tile PLYs were overwritten by the overlap run, but the overlap grid's center tile
   g4 (dx=dy=0) reproduces the baseline plot-centered 32 m tile at identical FF3D settings and
   identical coordinate recovery, so g4 serves as the single-tile reference (its in-plot count is
   cross-checked against the documented baseline CSV). The g4 centroids and the 9-tile NMS-merged
   centroids are matched to the known cubed-tree positions (restricted to trees inside the tiled
   footprints) at radii 3 and 4 m, as in ff3d_detection_numbers.py::main(). `match_detections` counts
   distinct ground-truth trees hit (bounded by n_gt), so duplicate detections cannot inflate it: more
   distinct known trees matched means real recall; an in-plot count that rose without more distinct
   matches means duplicate inflation.

Run: PYTHONPATH=. GREENVISTA_LAZ_DIR=/tmp/lazall python scripts/ff3d_overlap_validate.py [ply_dir]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Baseline parser (min-matching UTM recovery, as used by ff3d_detection_numbers.py::main()).
from ff3d_detection_numbers import find_plys, centroids_utm  # noqa: E402
# Overlap machinery: mean-based UTM recovery + greedy NMS + in-plot counter.
from ff3d_detection_overlap import instance_centroids, nms_merge, in_plot_count, BACKUP  # noqa: E402

from greenvista import config  # noqa: E402
from greenvista.io.geoloc import load_geoloc  # noqa: E402
from greenvista.segmentation import match_detections  # noqa: E402

# Radii from the task, plus two extra points at and just below the 1.8 m planting NN spacing. Two
# distinct eucalyptus (3.0x1.8 m grid) are never < ~1.6 m apart, so an NMS radius of ~1.6-1.7 m removes
# duplicates without merging distinct trees; its kept count is the best distinct-tree estimate.
RADII = [1.0, 1.2, 1.5, 1.6, 1.8, 2.0, 2.5]
MATCH_RADII = [3.0, 4.0]
PLOT_R = config.PLOT_RADIUS_M
NN_SPACING = 1.8   # eucalyptus within-row planting spacing (m); distinct trees are >= this apart
DUP_THRESH = 1.6   # kept centroids closer than this cannot be two distinct trees -> residual duplicate


def in_plot_subset(cents, cx, cy):
    if len(cents) == 0:
        return cents
    d2 = (cents[:, 0] - cx) ** 2 + (cents[:, 1] - cy) ** 2
    return cents[d2 <= PLOT_R ** 2]


def residual_dupes(cents, thresh):
    """Count kept centroids that have another kept centroid within `thresh`.

    Distinct eucalyptus (3.0x1.8 m grid) are >= ~1.8 m apart, so any kept pair closer than ~1.6 m are
    two detections of the same tree that survived NMS (residual under-merge). Returns
    (n_centroids_with_close_neighbour, n_such_pairs); n_pairs approximates the excess detections."""
    n = len(cents)
    if n < 2:
        return 0, 0
    close = np.zeros(n, bool)
    pairs = 0
    for i in range(n):
        d = np.hypot(cents[i, 0] - cents[:, 0], cents[i, 1] - cents[:, 1])
        d[i] = np.inf
        near = d < thresh
        if near.any():
            close[i] = True
        pairs += int((near[i + 1:]).sum())  # each unordered pair once
    return int(close.sum()), pairs


def pool_plot(g, plys):
    """Pool instance centroids across all found tiles of one plot (mean-based UTM recovery).

    Returns (pooled_cents Nx2, pooled_sizes N, per-tile dict {gi: (cents,sizes)}, n_tiles_found, discs)."""
    pooled_c, pooled_s, per_tile, discs = [], [], {}, []
    n_found = 0
    for _, tr in g.iterrows():
        cand = [pp for pp in plys if tr.tile in Path(pp).name]
        if not cand:
            continue
        n_found += 1
        c, s, disc = instance_centroids(cand[0], BACKUP / f"{tr.tile}.laz")
        if not np.isnan(disc):
            discs.append(disc)
        per_tile[int(tr.gi)] = (c, s)
        if len(c):
            pooled_c.append(c); pooled_s.append(s)
    if pooled_c:
        allc = np.vstack(pooled_c); alls = np.concatenate(pooled_s)
    else:
        allc = np.empty((0, 2)); alls = np.empty((0,), int)
    return allc, alls, per_tile, n_found, discs


def sensitivity(man, plys, base):
    print("\n" + "=" * 78)
    print("TEST 1 -- MERGE-RADIUS SENSITIVITY (merged in-plot detection per plot)")
    print("=" * 78)
    rows = []
    all_discs = []
    for (t, p), g in man.groupby(["talhao", "parcela"]):
        cx, cy = float(g.cx.iloc[0]), float(g.cy.iloc[0])
        field = int(g.field_n_arv.iloc[0])
        plot_id = g.plot_id.iloc[0]
        allc, alls, per_tile, n_found, discs = pool_plot(g, plys)
        all_discs += discs
        raw_in = in_plot_count(allc, cx, cy) if len(allc) else 0
        # single-tile reference = the center tile g4 (dx=dy=0 == baseline geometry)
        g4c, g4s = per_tile.get(4, (np.empty((0, 2)), np.empty((0,), int)))
        g4_in = in_plot_count(g4c, cx, cy) if len(g4c) else 0
        base_in = int(base.loc[plot_id, "ff3d_in_plot"]) if plot_id in base.index else np.nan
        row = {"plot_id": plot_id, "field": field, "n_tiles": n_found,
               "pooled_inst": len(allc), "raw_premerge_in_plot": raw_in,
               "g4_center_in_plot": g4_in, "baseline_csv_in_plot": base_in}
        for r in RADII:
            merged = nms_merge(allc, alls, r)
            row[f"m{r}"] = in_plot_count(merged, cx, cy)
        rows.append(row)
    df = pd.DataFrame(rows)

    # Detection-count table
    cols = ["plot_id", "field", "n_tiles", "pooled_inst", "raw_premerge_in_plot",
            "g4_center_in_plot", "baseline_csv_in_plot"] + [f"m{r}" for r in RADII]
    print("\nMerged in-plot detection (count) at each merge radius:")
    print(df[cols].to_string(index=False))

    # As % of field
    pct = df[["plot_id", "field"]].copy()
    for r in RADII:
        pct[f"m{r}_%"] = (100 * df[f"m{r}"] / df["field"]).round(0).astype(int)
    pct["g4_%"] = (100 * df["g4_center_in_plot"] / df["field"]).round(0).astype(int)
    pct["base_%"] = (100 * df["baseline_csv_in_plot"] / df["field"]).round(0)
    print("\nSame, as % of field stem count (g4_% and base_% = single-tile reference):")
    print(pct[["plot_id", "field", "base_%", "g4_%"] + [f"m{r}_%" for r in RADII]].to_string(index=False))

    # Collapse metric: how much of the 1.2 m gain over the single tile survives to 2.5 m?
    print("\nDuplicate-collapse read (per plot):")
    for _, rr in df.iterrows():
        ref = rr["g4_center_in_plot"]  # single-tile reference
        gain12 = rr["m1.2"] - ref
        gain25 = rr["m2.5"] - ref
        flag = "  <-- OVER-DETECTION (>field)" if rr["m1.2"] > rr["field"] else ""
        print(f"  {rr['plot_id']}: single-tile={ref}  m1.2={rr['m1.2']} (+{gain12})  "
              f"m2.0={rr['m2.0']} m2.5={rr['m2.5']} (+{gain25} vs single); "
              f"1.2->2.5 loses {rr['m1.2']-rr['m2.5']}{flag}")

    # residual-duplicate diagnostic on the canonical 1.2 m merged, in-plot centroids
    print(f"\nResidual under-merge in the m1.2 in-plot set (pairs closer than {DUP_THRESH} m cannot be "
          f"two distinct trees at {NN_SPACING} m spacing):")
    print(f"{'plot':>7} | {'m1.2':>4} | {'centroids w/ <1.6m nbr':>22} | {'dup pairs':>9} | "
          f"{'m1.6':>4} {'m1.8':>4} | corrected(=m1.6) vs single")
    for (t, p), g in man.groupby(["talhao", "parcela"]):
        cx, cy = float(g.cx.iloc[0]), float(g.cy.iloc[0])
        allc, alls, per_tile, n_found, _ = pool_plot(g, plys)
        merged12 = nms_merge(allc, alls, 1.2)
        ip = in_plot_subset(merged12, cx, cy)
        nclose, npairs = residual_dupes(ip, DUP_THRESH)
        rr = df[df.plot_id == g.plot_id.iloc[0]].iloc[0]
        ref = rr["g4_center_in_plot"]
        print(f"  {g.plot_id.iloc[0]:>5} | {len(ip):>4} | {nclose:>22} | {npairs:>9} | "
              f"{rr['m1.6']:>4} {rr['m1.8']:>4} | +{rr['m1.6']-ref} over single-tile {ref}")
    if all_discs:
        print(f"\nUTM recovery cross-check (mean-based vs min-matching): max {max(all_discs):.3f} m, "
              f"median {np.median(all_discs):.3f} m over {len(all_discs)} tiles "
              f"(small => the two recoveries agree; radius choice is not a coordinate artifact)")
    return df


def position_match(man, plys):
    print("\n" + "=" * 78)
    print("TEST 2 -- POSITION MATCH vs KNOWN CUBED-TREE GPS (stand 1, the definitive test)")
    print("=" * 78)
    geo = load_geoloc()
    gt = geo[["Easting", "Northing"]].to_numpy()

    m1 = man[man.talhao == 1]
    # single-tile reference centroids = union of the g4 center tiles across T1 plots.
    # overlap centroids = all-9-tile pool, NMS-merged.
    g4_cents, pool_cents_raw, pool_sizes = [], [], []
    g4_cents_min = []  # centroids_utm (min-matching) on g4, to tie to main()'s documented numbers
    inside_box, inside_plot = np.zeros(len(gt), bool), np.zeros(len(gt), bool)
    for (t, p), g in m1.groupby(["talhao", "parcela"]):
        cx, cy = float(g.cx.iloc[0]), float(g.cy.iloc[0])
        allc, alls, per_tile, n_found, _ = pool_plot(g, plys)
        if len(allc):
            pool_cents_raw.append(allc); pool_sizes.append(alls)
        g4c, g4s = per_tile.get(4, (np.empty((0, 2)), np.empty((0,), int)))
        if len(g4c):
            g4_cents.append(g4c)
        # min-matching recovery on the g4 tile, exactly as main()
        g4_tile = g[g.gi == 4]
        if len(g4_tile):
            tname = g4_tile.tile.iloc[0]
            cand = [pp for pp in plys if tname in Path(pp).name]
            if cand:
                c_min, *_ = centroids_utm(cand[0], BACKUP / f"{tname}.laz")
                if len(c_min):
                    g4_cents_min.append(c_min)
        # footprint masks: 16 m box == the single 32 m tile (main() uses this); 12 m == the counted plot
        inside_box |= (np.abs(gt[:, 0] - cx) <= 16) & (np.abs(gt[:, 1] - cy) <= 16)
        inside_plot |= ((gt[:, 0] - cx) ** 2 + (gt[:, 1] - cy) ** 2) <= PLOT_R ** 2

    g4_all = np.vstack(g4_cents) if g4_cents else np.empty((0, 2))
    g4_all_min = np.vstack(g4_cents_min) if g4_cents_min else np.empty((0, 2))
    pool_all = np.vstack(pool_cents_raw) if pool_cents_raw else np.empty((0, 2))
    pool_sz = np.concatenate(pool_sizes) if pool_sizes else np.empty((0,), int)

    print(f"\nKnown cubed trees: {len(gt)} total; {inside_box.sum()} inside the 16 m tile boxes, "
          f"{inside_plot.sum()} inside the 12 m plot radii.")
    print(f"single-tile (g4) centroids: {len(g4_all)} (mean-recovery) / {len(g4_all_min)} (min-recovery, main()-style)")
    print(f"9-tile pooled centroids: {len(pool_all)} raw")

    for label, gt_mask in [("16 m tile box (main()-style)", inside_box),
                           ("12 m plot radius (in-plot trees)", inside_plot)]:
        gt_sub = gt[gt_mask]
        if not len(gt_sub):
            continue
        print(f"\n--- ground-truth restricted to: {label}  (N={len(gt_sub)} known trees) ---")
        print(f"{'radius':>7} | {'single-tile g4':>26} | {'overlap NMS@1.2':>16} | "
              f"{'overlap NMS@1.5':>16} | verdict")
        for mr in MATCH_RADII:
            r_g4 = match_detections(g4_all, gt_sub, radius=mr)
            r_g4m = match_detections(g4_all_min, gt_sub, radius=mr)
            # overlap merged at 1.2 and 1.5 m (the distinct-tree merge band)
            merged12 = nms_merge(pool_all, pool_sz, 1.2)
            merged15 = nms_merge(pool_all, pool_sz, 1.5)
            r_o12 = match_detections(merged12, gt_sub, radius=mr)
            r_o15 = match_detections(merged15, gt_sub, radius=mr)
            gain = r_o12["detected"] - r_g4["detected"]
            verdict = ("REAL recall (+%d distinct)" % gain if gain > 0
                       else "NO distinct-tree gain -> in-plot rise was DUPLICATES" if gain == 0
                       else "overlap WORSE (%d)" % gain)
            print(f"  r={mr:>3.0f}m | {r_g4['detected']:>3}/{r_g4['n_gt']} distinct "
                  f"(min:{r_g4m['detected']}) | {r_o12['detected']:>3}/{r_o12['n_gt']} distinct | "
                  f"{r_o15['detected']:>3}/{r_o15['n_gt']} distinct | {verdict}")
    return g4_all, pool_all


def main():
    ply_dir = sys.argv[1] if len(sys.argv) > 1 else \
        str(config.REPO / "project/models/FF3D_inference/FF3D_oracle/bucket_out_folder")
    man = pd.read_csv(BACKUP / "tiles_manifest.csv")
    base = pd.read_csv(config.OUT_DIR / "ff3d_detection.csv").set_index("plot_id")
    plys = find_plys(ply_dir)
    print(f"found {len(plys)} PLY(s) under {ply_dir}")
    have = sorted({Path(p).name.split('_round2')[0] for p in plys})
    want = sorted(man.tile.tolist())
    missing = [w for w in want if w not in have]
    if missing:
        print(f"WARNING: {len(missing)} manifest tile(s) have no PLY yet: {missing}")
    if not plys:
        print("no FF3D output yet -- run the pipeline first."); return

    sensitivity(man, plys, base)
    position_match(man, plys)


if __name__ == "__main__":
    main()

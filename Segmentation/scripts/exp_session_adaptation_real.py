#!/usr/bin/env python3
"""Idea #7(b1): the real 6-stand evaluation of per-stand adaptive NMS merge radius.

GPU-free. Consumes the committed, durable per-tile FF3D detection centroids
(`data/detections/ff3d_overlap_centroids_13plot.csv` — one row per FF3D instance, 9
overlap tiles g0..g8 per plot, `cx,cy` = instance centroid in UTM, `n_pts` = crown size)
and asks the question the synthetic only settled in principle:

    Does a per-stand adaptive NMS merge radius (unsupervised, from each stand's own
    detected-centroid spacing) beat a fixed global radius across all 6 real stands,
    pulling the over-counting stands (duplicate leakage at a tight radius) toward 100 %
    of field without lowering recall on the dense/under-counting stands?

For each of the 13 plots we pool the 9 overlap tiles' centroids in UTM and count the
NMS-merged instances inside the 12 m plot vs the field stem count, under:
  (a) global r=1.5 m  — the 6b conservative headline,
  (b) global r=1.2 m  — the optimistic radius that over-counts easy stands,
  (c) per-stand adaptive r = clip(0.80 * within-tile spacing, 1.0, 2.0), from
      `greenvista.session_adapt` (batch + causal/streaming), estimated from each tile's
      interior detections (one-per-tree, free of the edge-crop centroid shift).

Stand 001 additionally gets a position-accuracy recall control vs the surveyed stems
(@2/3 m), checking that adaptation does not change which true stems are found.

Ground truth (plot centre + field stem count + per-tile centres) comes from the 6b
overlap tiling manifest; field counts cross-check against `inv_euc.csv` (fuste==1) within
a few trees. This does not touch the committed `scripts/exp_session_adaptation.py`.

Run:  PYTHONPATH=. python scripts/exp_session_adaptation_real.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from greenvista import config  # noqa: E402
from greenvista.session_adapt import (  # noqa: E402
    merge_stand_batch, merge_stand_streaming, merge_stand_global, count_within,
)

# --- shipped adaptation hyperparameters (identical to the synthetic/prototype script) ---
FRACTION, R_LO, R_HI = 0.80, 1.0, 2.0
INTERIOR_HALF = 11.0   # keep detections within 11 m of the tile centre (drops edge-crop centroids)
PLOT_R = config.PLOT_RADIUS_M

CSV = config.REPO / "data" / "detections" / "ff3d_overlap_centroids_13plot.csv"
MANIFEST = Path("/tmp/ff3d_tiles_overlap/tiles_manifest.csv")
OUT = config.OUT_DIR


def _gi(tile: str) -> int:
    """Grid index 0..8 from a tile name ending in g<k> (deployment order for streaming)."""
    return int(tile.rsplit("g", 1)[1])


def load_stands():
    """Yield per-plot dicts with the tile-ordered centroids, sizes, interior sets + GT.

    tile_cents/tile_sizes/tile_interior are lists aligned by grid index g0..g8.
    """
    det = pd.read_csv(CSV)
    man = pd.read_csv(MANIFEST)
    tile_centre = {r.tile: (float(r.tcx), float(r.tcy)) for r in man.itertuples()}
    gt = man.groupby("plot_id").agg(talhao=("talhao", "first"), parcela=("parcela", "first"),
                                    cx=("cx", "first"), cy=("cy", "first"),
                                    field=("field_n_arv", "first")).reset_index()

    stands = []
    for _, g in gt.iterrows():
        pid = g.plot_id
        sub = det[det.plot_id == pid]
        tiles = sorted(sub.tile.unique(), key=_gi)
        tile_cents, tile_sizes, tile_interior = [], [], []
        for t in tiles:
            st = sub[sub.tile == t]
            c = st[["cx", "cy"]].to_numpy(float).reshape(-1, 2)
            s = st["n_pts"].to_numpy(float).reshape(-1)
            tcx, tcy = tile_centre[t]
            if len(c):
                interior = (np.abs(c[:, 0] - tcx) <= INTERIOR_HALF) & \
                           (np.abs(c[:, 1] - tcy) <= INTERIOR_HALF)
            else:
                interior = np.zeros(0, bool)
            tile_cents.append(c)
            tile_sizes.append(s)
            tile_interior.append(c[interior].reshape(-1, 2))
        stands.append(dict(plot_id=pid, talhao=int(g.talhao), parcela=int(g.parcela),
                           cx=float(g.cx), cy=float(g.cy), field=int(g.field),
                           n_tiles=len(tiles), tile_cents=tile_cents, tile_sizes=tile_sizes,
                           tile_interior=tile_interior))
    return stands


def eval_stand(s):
    """Global-1.5 / global-1.2 / adaptive-batch / adaptive-stream counts in the 12 m plot."""
    tc, ts, ti = s["tile_cents"], s["tile_sizes"], s["tile_interior"]
    ctr = (s["cx"], s["cy"])
    g15 = merge_stand_global(tc, ts, radius=1.5)
    g12 = merge_stand_global(tc, ts, radius=1.2)
    b = merge_stand_batch(tc, ts, spacing_cents=ti, fraction=FRACTION, lo=R_LO, hi=R_HI)
    st = merge_stand_streaming(tc, ts, spacing_cents=ti, fraction=FRACTION, lo=R_LO, hi=R_HI)
    return {
        "count_global_1.5": count_within(g15["merged"], ctr, PLOT_R),
        "count_global_1.2": count_within(g12["merged"], ctr, PLOT_R),
        "count_adapt_batch": count_within(b["merged"], ctr, PLOT_R),
        "count_adapt_stream": count_within(st["merged"], ctr, PLOT_R),
        "within_tile_spacing_m": round(b["spacing"], 2),
        "adapt_radius_m": round(b["radius"], 2),
        "n_pooled": int(sum(len(x) for x in tc)),
        "_merged": {"global_1.5": g15["merged"], "global_1.2": g12["merged"],
                    "adapt_batch": b["merged"]},
    }


def talhao1_recall(stands, results):
    """Position recall vs the surveyed stand-001 stems (@2/3 m), per strategy, per T1 plot."""
    from greenvista.io.geoloc import load_geoloc
    from greenvista.segmentation import match_detections
    geo = load_geoloc()
    gt_all = geo[["Easting", "Northing"]].to_numpy(float)
    rows = []
    for s in stands:
        if s["talhao"] != 1:
            continue
        cx, cy = s["cx"], s["cy"]
        box = (np.abs(gt_all[:, 0] - cx) <= 32) & (np.abs(gt_all[:, 1] - cy) <= 32)
        gt_in = gt_all[box]
        merged = results[s["plot_id"]]["_merged"]
        for label, det in merged.items():
            for rr in (2.0, 3.0):
                res = match_detections(det, gt_in, radius=rr)
                rows.append(dict(plot_id=s["plot_id"], n_surveyed=int(len(gt_in)),
                                 strategy=label, radius_m=rr,
                                 recall=f"{res['detected']}/{res['n_gt']}",
                                 recall_pct=round(100 * res["detection_rate"], 0)))
    return pd.DataFrame(rows)


def main():
    if not MANIFEST.exists():
        print(f"ERROR: overlap manifest not found at {MANIFEST} (needed for plot centres + "
              f"field counts). Re-run the 6b overlap tiler or point MANIFEST at it.")
        return
    stands = load_stands()

    rows, results = [], {}
    for s in stands:
        r = eval_stand(s)
        results[s["plot_id"]] = r
        fld = s["field"]
        counts = {"global_1.5": r["count_global_1.5"], "global_1.2": r["count_global_1.2"],
                  "adaptive": r["count_adapt_batch"]}
        errs = {k: abs(v - fld) / fld for k, v in counts.items()}
        closest = min(errs, key=errs.get)
        rows.append({
            "plot_id": s["plot_id"], "talhao": s["talhao"], "field": fld,
            "n_tiles": s["n_tiles"], "n_pooled": r["n_pooled"],
            "within_tile_spacing_m": r["within_tile_spacing_m"],
            "adapt_radius_m": r["adapt_radius_m"],
            "global_1.5": r["count_global_1.5"], "global_1.2": r["count_global_1.2"],
            "adapt_batch": r["count_adapt_batch"], "adapt_stream": r["count_adapt_stream"],
            "det_g1.5": round(r["count_global_1.5"] / fld, 3),
            "det_g1.2": round(r["count_global_1.2"] / fld, 3),
            "det_adapt": round(r["count_adapt_batch"] / fld, 3),
            "err_g1.5": round(errs["global_1.5"], 3), "err_g1.2": round(errs["global_1.2"], 3),
            "err_adapt": round(errs["adaptive"], 3), "closest": closest,
        })
    df = pd.DataFrame(rows).sort_values(["talhao", "plot_id"]).reset_index(drop=True)

    # --- overall error = mean over plots of |count - field| / field ---
    overall = {s: round(df[f"err_{s}"].mean(), 4)
               for s in ("g1.5", "g1.2", "adapt")}
    # aggregate (estate) count vs field, and estate-level relative error
    tot_field = int(df.field.sum())
    tot = {s: int(df[col].sum()) for s, col in
           [("g1.5", "global_1.5"), ("g1.2", "global_1.2"), ("adapt", "adapt_batch")]}

    # --- print ---
    pd.set_option("display.width", 200)
    print("=" * 96)
    print("REAL 6-STAND EVAL — per-stand adaptive NMS merge radius vs fixed global (13 plots)")
    print(f"adaptive r = clip({FRACTION}*within-tile spacing, {R_LO}, {R_HI}) m; count in {PLOT_R:.0f} m plot")
    print("=" * 96)
    show = df[["plot_id", "talhao", "field", "within_tile_spacing_m", "adapt_radius_m",
               "global_1.5", "global_1.2", "adapt_batch", "adapt_stream",
               "det_g1.5", "det_g1.2", "det_adapt", "closest"]]
    print(show.to_string(index=False))

    print("\n--- per stand (summed counts vs summed field) ---")
    by = df.groupby("talhao").agg(field=("field", "sum"), g15=("global_1.5", "sum"),
                                  g12=("global_1.2", "sum"), adapt=("adapt_batch", "sum")).reset_index()
    by["g1.5_%"] = (100 * by.g15 / by.field).round(0)
    by["g1.2_%"] = (100 * by.g12 / by.field).round(0)
    by["adapt_%"] = (100 * by.adapt / by.field).round(0)
    print(by.to_string(index=False))

    print("\n--- OVERALL error (mean over 13 plots of |count-field|/field) ---")
    print(f"  global r=1.5 : {overall['g1.5']:.4f}")
    print(f"  global r=1.2 : {overall['g1.2']:.4f}")
    print(f"  adaptive     : {overall['adapt']:.4f}   "
          f"({'BEATS' if overall['adapt'] < min(overall['g1.5'], overall['g1.2']) else 'does NOT beat'} both globals)")
    print(f"\n  estate totals (field {tot_field}):  "
          f"g1.5={tot['g1.5']} ({100*tot['g1.5']/tot_field:.0f}%) | "
          f"g1.2={tot['g1.2']} ({100*tot['g1.2']/tot_field:.0f}%) | "
          f"adapt={tot['adapt']} ({100*tot['adapt']/tot_field:.0f}%)")
    # over-count diagnosis: which stands exceed 100% of field, and does adaptive pull them down?
    over = df[(df["det_g1.2"] > 1.0) | (df["det_g1.5"] > 1.0)]
    print(f"\n  OVER-counters (det > 100% at some global radius): "
          f"{', '.join(over.plot_id) if len(over) else 'NONE'}")
    if len(over):
        print(over[["plot_id", "field", "global_1.2", "det_g1.2", "global_1.5", "det_g1.5",
                    "adapt_batch", "det_adapt"]].to_string(index=False))
    closest_counts = df["closest"].value_counts()
    print(f"\n  closest-to-field wins per plot: {dict(closest_counts)}")

    # --- stand-001 position recall control ---
    print("\n--- stand-001 position-accuracy recall vs surveyed stems (control) ---")
    rec = talhao1_recall(stands, results)
    print(rec.to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "session_adapt_real_13plot.csv", index=False)
    rec.to_csv(OUT / "session_adapt_real_13plot_recall.csv", index=False)
    print(f"\nwrote {OUT / 'session_adapt_real_13plot.csv'}")
    print(f"wrote {OUT / 'session_adapt_real_13plot_recall.csv'}")


if __name__ == "__main__":
    main()

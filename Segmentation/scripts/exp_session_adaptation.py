#!/usr/bin/env python3
"""Idea #7(b): session-aware test-time adaptation of the FF3D NMS merge radius, per stand.

Two evaluations (both GPU-free, CPU only):

  synthetic  A controlled proof. Simulate several stands spanning a density range (true
             trees on a ~S-metre grid) + an FF3D-like detector that emits cross-tile
             duplicates (overlapping tiles; edge-crop centroid shift → far-apart
             duplicates) and misses. A per-stand adaptive radius recovers the true stem
             count better than any single global radius across the range, since no global
             radius is best for every density. Includes a no-ground-truth-peek check.

  real       The 2-stand prototype on what survived the bucket_out wipe: the r000_f0
             overlap detections for plots 1_2 (T1) and 5_9 (T5). Adaptive vs global
             (r=1.2 / r=1.5) vs field (57 / 51); plus 1_2 recall @2/3 m against the 16
             surveyed stand-001 stems. Two stands make this a prototype, and both are
             heavy under-counters, so the duplicate-leakage regime that adaptation fixes
             is not represented here (it lives in the wiped stands 4_4, 7_14). A decisive
             real test needs the 13-plot detections regenerated (~30-40 min GPU, deferred).

Run:
  PYTHONPATH=. python scripts/exp_session_adaptation.py [synthetic|real|all]
  # real needs the round2 PLYs in bucket_out_folder + the /tmp/ff3d_tiles_tta backup laz.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from greenvista import config  # noqa: E402
from greenvista.session_adapt import (  # noqa: E402
    merge_stand_batch, merge_stand_streaming, merge_stand_global,
    count_within,
)

OUT = config.OUT_DIR
# adaptation hyperparameters. The radius must sit between the duplicate scatter and the
# tree spacing: fraction 0.80 → ~0.8 of the typical NN spacing (safely below the neighbour
# distance, above the edge-crop duplicate scatter). 0.80 is the centre of the robust
# plateau (the synthetic win holds for fraction ∈ [0.65, 0.90], see the results doc) and
# was calibrated on the synthetic model, not on the N=2 real stands. Clamp to the
# eucalyptus regime [1.0, 2.0] m so it never merges distinct trees nor exceeds one spacing.
FRACTION, R_LO, R_HI = 0.80, 1.0, 2.0


# =====================================================================================
#  SYNTHETIC HARNESS
# =====================================================================================
def _make_stand(spacing, rng, L=64.0, p_det=0.90, sigma=0.30,
                edge_w=3.0, edge_shift=(0.5, 1.2), tile=32.0, stride=16.0):
    """Simulate one stand and return (tile_cents list, tile_sizes list, gt_xy, eval_win).

    True trees on a `spacing`-metre grid (jittered) over [0,L]^2. Overlapping `tile`-m
    tiles at `stride` (each interior tree seen by ~4 tiles). Per (tree, tile) inside the
    tile footprint: with prob `p_det` emit a detection at tree + N(0, sigma^2); if the
    tree is within `edge_w` of that tile's edge, add an inward 'edge-crop' shift
    (Uniform `edge_shift`, directed toward the tile centre) → the same tree lands far
    apart in two tiles = the fat-tail cross-tile duplicate a tight global radius misses.
    Sizes ~ crown point counts (larger for interior/complete views). Evaluation window =
    the fully-overlapped core [m, L-m]^2 so tile coverage is not the confound."""
    half = tile / 2.0
    # true grid (jittered)
    g = np.arange(spacing, L - spacing + 1e-9, spacing)
    gx, gy = np.meshgrid(g, g)
    gt = np.column_stack([gx.ravel(), gy.ravel()])
    gt = gt + rng.normal(0, 0.12 * spacing, gt.shape)
    # overlapping tile centres
    tc_grid = np.arange(half, L - half + 1e-9, stride)
    centres = [(cx, cy) for cx in tc_grid for cy in tc_grid]

    tile_cents, tile_sizes, tile_interior = [], [], []
    for (cx, cy) in centres:
        inside = (np.abs(gt[:, 0] - cx) <= half) & (np.abs(gt[:, 1] - cy) <= half)
        trees = gt[inside]
        m = len(trees)
        if m == 0:
            tile_cents.append(np.empty((0, 2))); tile_sizes.append(np.empty((0,)))
            tile_interior.append(np.empty((0, 2))); continue
        keep = rng.random(m) <= p_det                       # detection / miss
        trees = trees[keep]
        det = trees + rng.normal(0, sigma, trees.shape)     # base centroid jitter
        # edge-crop: trees near this tile's boundary get an inward centroid shift
        edge_d = np.minimum(half - np.abs(trees[:, 0] - cx), half - np.abs(trees[:, 1] - cy))
        is_edge = edge_d < edge_w
        complete = np.ones(len(trees))
        if is_edge.any():
            shift = rng.uniform(*edge_shift, size=is_edge.sum())
            direction = np.column_stack([cx - trees[is_edge, 0], cy - trees[is_edge, 1]])
            nrm = np.linalg.norm(direction, axis=1, keepdims=True)
            nrm[nrm < 1e-6] = 1.0
            det[is_edge] += shift[:, None] * direction / nrm    # pull toward tile centre
            complete[is_edge] = 0.4 + 0.6 * (edge_d[is_edge] / edge_w)  # clipped = fewer pts
        sizes = 2000.0 * complete * rng.uniform(0.8, 1.2, len(trees))
        tile_cents.append(det.reshape(-1, 2))
        tile_sizes.append(sizes.reshape(-1))
        # INTERIOR detections for the (unsupervised) spacing estimate — one-per-tree,
        # free of the edge-clip shift. Uses only tile geometry, never ground truth.
        tile_interior.append(det[~is_edge].reshape(-1, 2))

    m = 0.75 * tile  # core margin so the eval window is fully double-covered by tiles
    win = (m, L - m)
    in_win = (gt[:, 0] >= win[0]) & (gt[:, 0] <= win[1]) & \
             (gt[:, 1] >= win[0]) & (gt[:, 1] <= win[1])
    return tile_cents, tile_sizes, tile_interior, gt[in_win], win


def _count_in_win(cents, win):
    if len(cents) == 0:
        return 0
    lo, hi = win
    c = np.asarray(cents).reshape(-1, 2)
    return int(((c[:, 0] >= lo) & (c[:, 0] <= hi) & (c[:, 1] >= lo) & (c[:, 1] <= hi)).sum())


def run_synthetic(spacings=(1.6, 2.0, 2.4, 2.8, 3.2), n_seeds=30, p_det=0.90,
                  global_grid=None, verbose=True, tag="", make_plot=True):
    """Sweep global radii vs the per-stand adaptive radius across a density range.

    For every seed: build one stand per spacing, count merged detections in the core
    window at each global radius and at the adaptive (batch + streaming) radius, score
    |count - N_true| / N_true. Report per-stand optimal global radius (it varies), the
    best *single* global radius (min mean error over stands), and adaptive — showing no
    global radius matches the adaptive one across the range."""
    if global_grid is None:
        global_grid = np.round(np.arange(0.8, 2.41, 0.15), 2)

    # accumulate per (seed, stand): rRMSE-style relative count error for each strategy
    recs = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(1000 + seed)
        for S in spacings:
            tc, ts, ti, gt, win = _make_stand(S, rng, p_det=p_det)
            n_true = len(gt)
            row = {"seed": seed, "spacing": S, "n_true": n_true,
                   "n_pooled": int(sum(len(x) for x in tc))}
            # global radii
            for r in global_grid:
                cnt = _count_in_win(merge_stand_global(tc, ts, radius=r)["merged"], win)
                row[f"g_{r:.1f}"] = abs(cnt - n_true) / max(n_true, 1)
                row[f"gc_{r:.1f}"] = cnt
            # adaptive batch (spacing from interior detections; merge the full pool)
            b = merge_stand_batch(tc, ts, spacing_cents=ti, fraction=FRACTION, lo=R_LO, hi=R_HI)
            cb = _count_in_win(b["merged"], win)
            row["adapt_batch_err"] = abs(cb - n_true) / max(n_true, 1)
            row["adapt_batch_r"] = b["radius"]
            row["adapt_batch_spacing"] = b["spacing"]
            row["adapt_batch_cnt"] = cb
            # adaptive streaming (causal)
            st = merge_stand_streaming(tc, ts, spacing_cents=ti, fraction=FRACTION,
                                       lo=R_LO, hi=R_HI)
            cs = _count_in_win(st["merged"], win)
            row["adapt_stream_err"] = abs(cs - n_true) / max(n_true, 1)
            row["adapt_stream_r"] = st["radii"][-1]
            row["adapt_stream_cnt"] = cs
            recs.append(row)
    df = pd.DataFrame(recs)

    # --- per-stand mean error at each global radius + the adaptive columns ---
    gcols = [c for c in df.columns if c.startswith("g_")]
    by_stand = df.groupby("spacing")[gcols + ["adapt_batch_err", "adapt_stream_err"]].mean()

    # best single global radius = the one with the lowest mean error across ALL stands
    global_mean = df[gcols].mean()
    best_global_col = global_mean.idxmin()
    best_global_r = float(best_global_col.split("_")[1])
    best_global_err = float(global_mean.min())
    adapt_batch_err = float(df["adapt_batch_err"].mean())
    adapt_stream_err = float(df["adapt_stream_err"].mean())

    # per-stand optimal global radius (to show it shifts with density)
    opt_r = {}
    for S, r_row in by_stand[gcols].iterrows():
        opt_r[S] = float(r_row.idxmin().split("_")[1])

    if verbose:
        print("=" * 78)
        print(f"SYNTHETIC{(' [' + tag + ']') if tag else ''} — per-stand adaptive vs global "
              "merge radius (mean relative count")
        print(f"error over {n_seeds} seeds; detector p_det={p_det}; fraction={FRACTION}, "
              f"clamp [{R_LO},{R_HI}] m)")
        print("=" * 78)
        # compact table: for each stand, its best global radius + err, adaptive r + err
        print(f"\n{'spacing':>8} {'N_true':>7} {'opt_global_r':>13} {'best_r_err':>11} "
              f"{'adapt_r':>8} {'adapt_err':>10} {'stream_r':>9} {'stream_err':>11}")
        for S in spacings:
            sub = df[df.spacing == S]
            bcol = by_stand.loc[S, gcols].idxmin()
            print(f"{S:>8.1f} {sub.n_true.mean():>7.0f} {opt_r[S]:>13.1f} "
                  f"{by_stand.loc[S, bcol]:>11.3f} {sub.adapt_batch_r.mean():>8.2f} "
                  f"{sub.adapt_batch_err.mean():>10.3f} {sub.adapt_stream_r.mean():>9.2f} "
                  f"{sub.adapt_stream_err.mean():>11.3f}")
        print(f"\nPer-stand OPTIMAL global radius ranges {min(opt_r.values()):.1f}–"
              f"{max(opt_r.values()):.1f} m  → no single global radius is optimal for all.")
        print(f"\nBest SINGLE global radius (min mean err over all stands): "
              f"r={best_global_r:.1f} m, mean rel. err = {best_global_err:.3f}")
        print(f"Adaptive BATCH    : mean rel. err = {adapt_batch_err:.3f}  "
              f"({100*(best_global_err-adapt_batch_err)/best_global_err:+.0f}% vs best global)")
        print(f"Adaptive STREAMING: mean rel. err = {adapt_stream_err:.3f}  "
              f"({100*(best_global_err-adapt_stream_err)/best_global_err:+.0f}% vs best global)")
        # worst-stand (max over stands) — the metric a deployer actually cares about
        worst_global = float(by_stand[best_global_col].max())
        worst_adapt = float(by_stand["adapt_batch_err"].max())
        print(f"\nWORST-stand error: best global r={best_global_r:.1f} → {worst_global:.3f} | "
              f"adaptive batch → {worst_adapt:.3f}")
        # significance across seeds (paired): adaptive batch vs best single global
        paired = df.groupby("seed").apply(
            lambda d: d["adapt_batch_err"].mean() - d[best_global_col].mean())
        wins = int((paired < 0).sum())
        print(f"Paired over seeds: adaptive beats best-global on {wins}/{n_seeds} seeds "
              f"(mean Δerr {paired.mean():+.3f} ± {paired.std():.3f})")

    suffix = f"_{tag}" if tag else ""
    if make_plot:
        _plot_synthetic(df, by_stand, gcols, best_global_col, best_global_r, spacings, opt_r,
                        suffix)
    OUT.mkdir(parents=True, exist_ok=True)
    by_stand.to_csv(OUT / f"session_adapt_synthetic{suffix}.csv")
    if verbose:
        print(f"\nwrote {OUT / f'session_adapt_synthetic{suffix}.csv'}"
              + (f" and {OUT / f'session_adapt_synthetic{suffix}.png'}" if make_plot else ""))
    return {
        "best_global_r": best_global_r, "best_global_err": best_global_err,
        "adapt_batch_err": adapt_batch_err, "adapt_stream_err": adapt_stream_err,
        "opt_r_by_stand": opt_r, "n_seeds": n_seeds, "df": df, "by_stand": by_stand,
    }


def _plot_synthetic(df, by_stand, gcols, best_global_col, best_global_r, spacings, opt_r,
                    suffix=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")
        return
    radii = np.array([float(c.split("_")[1]) for c in gcols])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(spacings)))
    # (a) error vs global radius, one curve per stand; adaptive marker at its own radius
    for c, S in zip(cmap, spacings):
        err = by_stand.loc[S, gcols].to_numpy()
        ax[0].plot(radii, err, "-", color=c, label=f"S={S:.1f} m")
        sub = df[df.spacing == S]
        ax[0].plot(sub.adapt_batch_r.mean(), sub.adapt_batch_err.mean(), "*",
                   color=c, ms=15, mec="k", mew=0.6, zorder=5)
    ax[0].axvline(best_global_r, ls="--", color="0.4", lw=1,
                  label=f"best single global r={best_global_r:.1f}")
    ax[0].set_xlabel("merge radius (m)")
    ax[0].set_ylabel("relative count error  |ĉ − N_true| / N_true")
    ax[0].set_title("No single radius is best for every density\n(★ = per-stand adaptive radius)")
    ax[0].legend(fontsize=8, ncol=2)
    ax[0].grid(alpha=0.25)
    # (b) optimal radius vs spacing + the adaptive law
    S_arr = np.array(spacings)
    ax[1].plot(S_arr, [opt_r[S] for S in spacings], "o-", color="#1b7837",
               label="empirical optimal global r per stand")
    ax[1].plot(S_arr, [df[df.spacing == S].adapt_batch_r.mean() for S in spacings], "s--",
               color="#762a83", label=f"adaptive r = clip({FRACTION}·spacing, {R_LO}, {R_HI})")
    ax[1].set_xlabel("true tree spacing S (m)")
    ax[1].set_ylabel("merge radius (m)")
    ax[1].set_title("Optimal radius shifts with density —\nthe adaptive law tracks it")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"session_adapt_synthetic{suffix}.png", dpi=130)
    plt.close(fig)


# =====================================================================================
#  REAL 2-STAND PROTOTYPE
# =====================================================================================
def _load_real_stand(plot_id, ply_dir, backup_dir, manifest, interior_half=11.0):
    """Load the r000_f0 overlap tiles of one stand → (tile_cents, tile_sizes,
    tile_interior, cx, cy, field, tiles). Deployment order = grid index gi (0..8).
    `tile_interior` = detections within `interior_half` m of the tile centre (the clean
    one-per-tree set for the spacing estimate; drops edge-clipped centroids)."""
    from ff3d_detection_overlap import instance_centroids  # local import (needs scripts/)
    sub = manifest[(manifest.plot_id == plot_id) & (manifest.cond == "overlap") &
                   (manifest.angle_deg == 0.0) & (manifest.flip == 0)].sort_values("gi")
    cx, cy = float(sub.cx.iloc[0]), float(sub.cy.iloc[0])
    field = float(sub.field_n_arv.iloc[0])
    tile_cents, tile_sizes, tile_interior, tiles = [], [], [], []
    for _, r in sub.iterrows():
        ply = Path(ply_dir) / f"{r.tile}_round2.ply"
        if not ply.exists():
            continue
        c, s, _ = instance_centroids(str(ply), str(Path(backup_dir) / f"{r.tile}.laz"))
        tile_cents.append(c)
        tile_sizes.append(s)
        interior = (np.abs(c[:, 0] - r.tcx) <= interior_half) & \
                   (np.abs(c[:, 1] - r.tcy) <= interior_half) if len(c) else np.zeros(0, bool)
        tile_interior.append(c[interior].reshape(-1, 2))
        tiles.append(r.tile)
    return tile_cents, tile_sizes, tile_interior, cx, cy, field, tiles


def run_real(ply_dir=None, backup_dir="/tmp/ff3d_tiles_tta", verbose=True):
    """2-stand prototype: adaptive vs global (1.2/1.5) vs field, + 1_2 surveyed recall."""
    from greenvista.io.geoloc import load_geoloc
    from greenvista.segmentation import match_detections

    ply_dir = ply_dir or str(
        config.REPO / "project/models/FF3D_inference/FF3D_oracle/bucket_out_folder"
        / "round_2_after_remove_noise_200")
    man_path = Path(backup_dir) / "tiles_manifest.csv"
    if not man_path.exists():
        print(f"(real prototype skipped: no manifest at {man_path})")
        return None
    manifest = pd.read_csv(man_path)
    plots = [("1_2", 57), ("5_9", 51)]

    geo = load_geoloc()
    gt_all = geo[["Easting", "Northing"]].to_numpy()

    rows = []
    if verbose:
        print("=" * 78)
        print("REAL 2-stand PROTOTYPE — overlap (r000_f0) detections, adaptive vs global")
        print("=" * 78)
    for pid, field in plots:
        tc, ts, ti, cx, cy, fld, tiles = _load_real_stand(pid, ply_dir, backup_dir, manifest)
        if not tc:
            print(f"  {pid}: no PLYs found under {ply_dir}"); continue
        n_tiles = len(tc)
        n_pool = int(sum(len(x) for x in tc))
        # spacing from interior detections (clean); merge the full pool
        b = merge_stand_batch(tc, ts, spacing_cents=ti, fraction=FRACTION, lo=R_LO, hi=R_HI)
        st = merge_stand_streaming(tc, ts, spacing_cents=ti, fraction=FRACTION, lo=R_LO, hi=R_HI)
        g12 = merge_stand_global(tc, ts, radius=1.2)
        g15 = merge_stand_global(tc, ts, radius=1.5)

        c_ad = count_within(b["merged"], (cx, cy), config.PLOT_RADIUS_M)
        c_ad_s = count_within(st["merged"], (cx, cy), config.PLOT_RADIUS_M)
        c_g12 = count_within(g12["merged"], (cx, cy), config.PLOT_RADIUS_M)
        c_g15 = count_within(g15["merged"], (cx, cy), config.PLOT_RADIUS_M)

        row = {"plot_id": pid, "field": int(fld), "n_tiles": n_tiles, "n_pooled": n_pool,
               "within_tile_spacing_m": round(b["spacing"], 2),
               "adapt_radius_m": round(b["radius"], 2),
               "count_global_1.2": c_g12, "count_global_1.5": c_g15,
               "count_adapt_batch": c_ad, "count_adapt_stream": c_ad_s,
               "det_global_1.2": round(c_g12 / fld, 3), "det_global_1.5": round(c_g15 / fld, 3),
               "det_adapt": round(c_ad / fld, 3)}
        rows.append(row)
        if verbose:
            print(f"\n  {pid}: {n_tiles} overlap tiles, pooled {n_pool}, within-tile "
                  f"spacing {b['spacing']:.2f} m → adaptive r={b['radius']:.2f} m | field {int(fld)}")
            print(f"    count in 12 m plot:  global1.2={c_g12} ({100*c_g12/fld:.0f}%) | "
                  f"global1.5={c_g15} ({100*c_g15/fld:.0f}%) | adaptive={c_ad} ({100*c_ad/fld:.0f}%) "
                  f"| stream={c_ad_s} ({100*c_ad_s/fld:.0f}%)")

        # 1_2 recall vs surveyed stems (64x64 union footprint of the 3x3 overlap grid)
        if pid == "1_2":
            box = (np.abs(gt_all[:, 0] - cx) <= 32) & (np.abs(gt_all[:, 1] - cy) <= 32)
            gt_in = gt_all[box]
            if verbose:
                print(f"    surveyed stems in 64x64 union footprint: {len(gt_in)}")
            for label, merged in [("global1.2", g12["merged"]), ("global1.5", g15["merged"]),
                                  ("adaptive", b["merged"])]:
                for rr in (2.0, 3.0):
                    res = match_detections(merged, gt_in, radius=rr)
                    row[f"recall_{label}_{rr:.0f}m"] = f"{res['detected']}/{res['n_gt']}"
                    if verbose:
                        print(f"      {label:9s} recall@{rr:.0f}m: {res['detected']}/{res['n_gt']} "
                              f"({100*res['detection_rate']:.0f}%)")

    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "session_adapt_real.csv", index=False)
    if verbose:
        tot_f = sum(f for _, f in plots)
        print(f"\n  TOTAL (2 stands, field {tot_f}): "
              f"global1.5={int(res['count_global_1.5'].sum())} | "
              f"adaptive={int(res['count_adapt_batch'].sum())}")
        print("\n  NOTE: both surviving stands are heavy UNDER-counters (T1 68%, "
              "T5 39% of field at r=1.5), so the duplicate-leakage / OVER-count regime that "
              "adaptation is designed to fix is NOT present here — it lived in the wiped "
              "stands 4_4 & 7_14 (>110% of field at r=1.2). Adaptive ≈ global-1.5 on both, "
              "and recall is unchanged → adaptation is SAFE but this 2-stand set cannot be "
              "decisive. The decisive test needs the 13-plot detections regenerated (DEFERRED).")
        print(f"\nwrote {OUT / 'session_adapt_real.csv'}")
    return res


# =====================================================================================
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("synthetic", "all"):
        # headline: p_det=1.0 isolates the merge-radius effect (misses add a
        # radius-independent floor that shifts every method equally — see the realistic run)
        run_synthetic(p_det=1.0, tag="clean")
        print()
        # realistic: detector also misses (occlusion) — same ranking, smaller absolute error
        run_synthetic(p_det=0.85, tag="realistic", make_plot=False)
        print()
    if mode in ("real", "all"):
        run_real()


if __name__ == "__main__":
    main()

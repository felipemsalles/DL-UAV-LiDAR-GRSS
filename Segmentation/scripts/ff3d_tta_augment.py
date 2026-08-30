#!/usr/bin/env python3
"""Test-time augmentation (rotation/flip) for FF3D detection, on top of the 6b overlap scheme.

6b showed that spatial test-time augmentation works: a 3×3 grid of overlapping 32 m tiles (offset
views) plus NMS-merge lifts FF3D detection from 61% to 74% (r=1.5) by recovering view-dependent and
edge-cropped misses at no extra VRAM. This script generalises it: each tile is additionally augmented
with rotations about the vertical (z) axis (45°/90°/135°) and a horizontal flip, FF3D is run on each
augmented view, the detected instance centroids are un-transformed back to the original UTM frame,
and everything is pooled and NMS-merged across both the spatial offsets and the augmentations.

The question is whether overlap plus rotation/flip beats overlap alone. A modest or null effect is
the expected outcome: overlap already captures the multi-view benefit, a nadir plantation canopy is
approximately rotation-symmetric, and FF3D may degrade on off-axis input if it was trained
axis-aligned. Both outcomes are reported.

Scope: 2 plots (1_2, 5_9), ≤ ~30 tiles. Not intended to be auto-scaled to the 13 plots.

Transform (per tile, pivot c = tile centre (tcx,tcy)):
    S = diag(-1,1) if flip else I           # reflect x about a vertical plane through c
    R = [[cosθ,-sinθ],[sinθ,cosθ]]          # rotation about z
    M = R @ S ;  t = c - M @ c              # affine on (x,y); z unchanged
    forward  a = M @ p + t                  # what is written to the laz (still UTM-scale)
    inverse  p = M^{-1} @ (a - t)           # un-transform a recovered centroid back to UTM
Affine transforms commute with averaging, so a recovered instance centroid a = M·(true UTM centroid)+t
exactly, independent of FF3D's point assignment, and the un-transform is geometrically exact.

The laz→ply converter subtracts each tile's own mean(x,y); `instance_centroids` (imported from 6b)
recovers that shift from the backup tile's mean, giving centroids in the written (augmented) frame,
to which the inverse affine is then applied. Pool + NMS-merge (`nms_merge`, `in_plot_count`) are
reused from 6b.

Run:
  # 1) build augmented + overlap tiles for the spike plots (backs up to <tmp>/ff3d_tiles_tta)
  PYTHONPATH=. GREENVISTA_LAZ_DIR=/tmp/lazall python scripts/ff3d_tta_augment.py build
  # 2) run FF3D once over all tiles
  cd project/models/FF3D_inference/ff3d_forestsens && bash run_docker_locally.sh && cd -
  # 3) score the 3 conditions (baseline / overlap-only / overlap+aug) + checks
  PYTHONPATH=. GREENVISTA_LAZ_DIR=/tmp/lazall python scripts/ff3d_tta_augment.py score [ply_dir] [radius]
"""
import os, sys, shutil
from pathlib import Path
import numpy as np, pandas as pd, laspy, warnings
warnings.simplefilter("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OVERLAP_TALHOES", "1,5")  # for ff3d_make_tiles_overlap.load_plot_centres
from ff3d_detection_overlap import instance_centroids, nms_merge, in_plot_count  # noqa: E402
from ff3d_detection_numbers import find_plys  # noqa: E402
# ff3d_make_tiles_overlap parses sys.argv at import (SIZE/STRIDE) → shield it from our sub-command argv.
_argv = sys.argv
sys.argv = [_argv[0]]
import ff3d_make_tiles_overlap as ovl  # noqa: E402
sys.argv = _argv

from greenvista import config  # noqa: E402

# --- spike scope -----------------------------------------------------------------------------
SPIKE_PLOTS = os.environ.get("TTA_PLOTS", "1_2,5_9").split(",")
SIZE = 32.0
HALF = SIZE / 2.0
STRIDE = 16.0
BUCKET_IN = config.REPO / "project/models/FF3D_inference/FF3D_oracle/bucket_in_folder"
BACKUP = Path(config.LAZ_DIR).parent / "ff3d_tiles_tta"
BASELINE_CSV = config.OUT_DIR / "ff3d_detection.csv"
PLOT_R = config.PLOT_RADIUS_M
PLOT_AREA_HA = config.PLOT_AREA_HA

# 3×3 overlap grid (same as 6b): gi 0..8 = product([-16,0,16],[-16,0,16]); gi 4 = (0,0) centre.
import itertools
OFFS = [-STRIDE, 0.0, STRIDE]
GRID = list(itertools.product(OFFS, OFFS))

# Augmentations per plot: (gi, angle_deg, flip). g4 = centre (fully contains the 12 m plot), g0/g8 = corners.
#   45/135° = off-axis (breaks planting-grid and voxel-axis alignment)
#   90°      = grid-preserving rotation (near-symmetric baseline; also the un-rotation check)
#   flip     = horizontal reflection
AUG = [
    (4, 45.0, 0),
    (4, 90.0, 0),
    (4, 135.0, 0),
    (4, 0.0, 1),      # flip only
    (0, 45.0, 0),     # corner + off-axis
    (8, 45.0, 0),     # corner + off-axis
]


def affine(pivot, angle_deg, flip):
    """Return (M 2x2, t 2) with forward a = M@p + t. M = R(θ) @ S; t = c - M@c."""
    c = np.asarray(pivot, float)
    th = np.deg2rad(angle_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    S = np.diag([-1.0, 1.0]) if flip else np.eye(2)
    M = R @ S
    t = c - M @ c
    return M, t


def apply_fwd(xy, M, t):
    return xy @ M.T + t


def apply_inv(xy, M, t):
    return (xy - t) @ np.linalg.inv(M).T


def tile_name(t, p, gi, angle, flip):
    return f"t{t:03d}_p{p:03d}_g{gi}_r{int(round(angle)):03d}_f{flip}"


# =============================================================================================
def build():
    df = ovl.load_plot_centres()
    df = df[df.plot_id.isin(SPIKE_PLOTS)].sort_values("plot_id").reset_index(drop=True)
    if df.empty:
        print(f"no plot centres for {SPIKE_PLOTS}"); return
    BUCKET_IN.mkdir(parents=True, exist_ok=True)
    if BACKUP.exists():
        shutil.rmtree(BACKUP)
    BACKUP.mkdir(parents=True, exist_ok=True)

    cloud, rows, rt_err = {}, [], []
    for _, r in df.iterrows():
        t, p = int(r.talhao), int(r.parcela)
        if t not in cloud:
            cloud.clear()
            las = laspy.read(str(config.LAZ_DIR / f"SaoManuelTotal_{t:03d}.laz"))
            cloud[t] = (las, np.asarray(las.x), np.asarray(las.y))
        las, X, Y = cloud[t]
        offx, offy = float(las.header.offsets[0]), float(las.header.offsets[1])
        scx, scy = float(las.header.scales[0]), float(las.header.scales[1])

        # condition A: 9 overlap offset tiles at angle 0, flip 0 (the 6b overlap-only pool)
        specs = [(gi, GRID[gi][0], GRID[gi][1], 0.0, 0, "overlap") for gi in range(9)]
        # condition B: augmentations (rotation/flip) on selected positions
        for gi, ang, fl in AUG:
            specs.append((gi, GRID[gi][0], GRID[gi][1], ang, fl, "aug"))

        for gi, dx, dy, ang, fl, cond in specs:
            tcx, tcy = r.cx + dx, r.cy + dy
            m = (np.abs(X - tcx) <= HALF) & (np.abs(Y - tcy) <= HALF)
            n = int(m.sum())
            sub = laspy.LasData(las.header)
            sub.points = las.points[m].copy()
            pxy = np.column_stack([X[m], Y[m]])
            M, tvec = affine((tcx, tcy), ang, fl)
            a = apply_fwd(pxy, M, tvec)
            # write rotated coords as raw int32 X/Y (the copied record is not scale-aware, so set
            # the stored ints directly; z + all other dims are untouched).
            sub.X = np.round((a[:, 0] - offx) / scx).astype(np.int32)
            sub.Y = np.round((a[:, 1] - offy) / scy).astype(np.int32)
            # round-trip math check (independent of FF3D): decode written coords, un-transform → originals
            wxy = np.column_stack([sub.X * scx + offx, sub.Y * scy + offy])
            back = apply_inv(wxy, M, tvec)
            rt_err.append(float(np.abs(back - pxy).max()) if n else 0.0)
            name = tile_name(t, p, gi, ang, fl)
            sub.write(str(BUCKET_IN / f"{name}.laz"))
            shutil.copy(BUCKET_IN / f"{name}.laz", BACKUP / f"{name}.laz")
            sz = (BUCKET_IN / f"{name}.laz").stat().st_size / 1e6
            rows.append({"tile": name, "plot_id": r.plot_id, "talhao": t, "parcela": p,
                         "gi": gi, "dx": dx, "dy": dy, "angle_deg": ang, "flip": fl, "cond": cond,
                         "is_center": int(gi == 4 and ang == 0.0 and fl == 0),
                         "tcx": float(tcx), "tcy": float(tcy), "cx": float(r.cx), "cy": float(r.cy),
                         "field_n_arv": float(r.n_arv), "n_pts": n, "laz_MB": round(sz, 2)})
        nl = [x for x in rows if x["plot_id"] == r.plot_id]
        print(f"  {r.plot_id}: {len(nl)} tiles (9 overlap + {len(AUG)} aug), field {int(r.n_arv)} stems, "
              f"pts {min(x['n_pts'] for x in nl)}-{max(x['n_pts'] for x in nl)}, "
              f"MB {min(x['laz_MB'] for x in nl)}-{max(x['laz_MB'] for x in nl)}")

    man = pd.DataFrame(rows)
    man.to_csv(BACKUP / "tiles_manifest.csv", index=False)
    quantum = float(laspy.read(str(BUCKET_IN / (rows[0]["tile"] + ".laz"))).header.scales[0])
    print(f"\n{len(rows)} tiles → {BUCKET_IN}\n manifest → {BACKUP/'tiles_manifest.csv'}")
    print(f" round-trip (forward∘inverse) max error = {max(rt_err):.2e} m over {len(rt_err)} tiles "
          f"(laz quantum = {quantum:.0e} m — the exactness floor)")
    big = man[man.laz_MB > 10]
    if len(big):
        print(f" {len(big)} tiles > 10 MB (OOM watch): {big.tile.tolist()}")


# =============================================================================================
def _tile_centroids_utm(ply_path, backup_laz, M, tvec):
    """Recover instance centroids in the written (augmented) frame, then un-transform to UTM."""
    c_aug, sizes, disc = instance_centroids(ply_path, backup_laz)  # 6b recovery → augmented frame
    if len(c_aug):
        c_utm = apply_inv(c_aug, M, tvec)
    else:
        c_utm = c_aug
    return c_utm, sizes, disc


def score():
    ply_dir = sys.argv[2] if len(sys.argv) > 2 else \
        str(config.REPO / "project/models/FF3D_inference/FF3D_oracle/bucket_out_folder")
    radius = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5

    man = pd.read_csv(BACKUP / "tiles_manifest.csv")
    base = pd.read_csv(BASELINE_CSV).set_index("plot_id")
    plys = find_plys(ply_dir)
    print(f"found {len(plys)} PLY(s) under {ply_dir}   (merge radius = {radius} m)\n")
    if not plys:
        print("no FF3D output yet — run the pipeline first."); return

    # per-tile recovery cache: tile -> (c_utm, sizes, in_plot_count, disc, found)
    cache, discs = {}, []
    for _, tr in man.iterrows():
        cand = [pp for pp in plys if tr.tile in Path(pp).name]
        if not cand:
            cache[tr.tile] = (np.empty((0, 2)), np.empty((0,), int), 0, np.nan, False)
            continue
        M, tvec = affine((tr.tcx, tr.tcy), tr.angle_deg, tr.flip)
        c, s, disc = _tile_centroids_utm(cand[0], BACKUP / f"{tr.tile}.laz", M, tvec)
        if not np.isnan(disc):
            discs.append(disc)
        cache[tr.tile] = (c, s, in_plot_count(c, tr.cx, tr.cy) if len(c) else 0, disc, True)

    # ---- 3-condition comparison, per plot ----
    def pool(sub):
        cs = [cache[t][0] for t in sub.tile if cache[t][4] and len(cache[t][0])]
        ss = [cache[t][1] for t in sub.tile if cache[t][4] and len(cache[t][1])]
        if cs:
            return np.vstack(cs), np.concatenate(ss)
        return np.empty((0, 2)), np.empty((0,), int)

    rows = []
    for pid, g in man.groupby("plot_id"):
        cx, cy = float(g.cx.iloc[0]), float(g.cy.iloc[0])
        field = float(g.field_n_arv.iloc[0])
        ov_g = g[g.cond == "overlap"]
        # overlap-only
        c_ov, s_ov = pool(ov_g)
        ov_in = in_plot_count(nms_merge(c_ov, s_ov, radius), cx, cy)
        # overlap + aug
        c_all, s_all = pool(g)
        all_in = in_plot_count(nms_merge(c_all, s_all, radius), cx, cy)
        base_in = int(base.loc[pid, "ff3d_in_plot"]) if pid in base.index else np.nan
        # center-tile (baseline reproduction control)
        ctr = g[g.is_center == 1].tile
        ctr_in = cache[ctr.iloc[0]][2] if len(ctr) and cache[ctr.iloc[0]][4] else np.nan
        rows.append({"plot_id": pid, "field": field, "baseline_in": base_in,
                     "center_ctrl_in": ctr_in, "overlap_in": ov_in, "overlap_aug_in": all_in,
                     "base_det": round(base_in / field, 3), "overlap_det": round(ov_in / field, 3),
                     "aug_det": round(all_in / field, 3),
                     "aug_minus_overlap": all_in - ov_in,
                     "n_overlap_tiles": int(ov_g.tile.map(lambda t: cache[t][4]).sum()),
                     "n_aug_tiles": int(g[g.cond == "aug"].tile.map(lambda t: cache[t][4]).sum())})
        print(f"  {pid}: baseline {base_in} (center-ctrl {ctr_in}) | overlap {ov_in} | overlap+aug {all_in} "
              f"| field {int(field)}  →  {100*base_in/field:.0f}% / {100*ov_in/field:.0f}% / "
              f"{100*all_in/field:.0f}%   (aug−overlap = {all_in-ov_in:+d})")

    res = pd.DataFrame(rows)
    print("\n=== 3-condition detection (merge radius {:.1f} m) ===".format(radius))
    print(res[["plot_id", "field", "baseline_in", "overlap_in", "overlap_aug_in",
               "base_det", "overlap_det", "aug_det", "aug_minus_overlap"]].to_string(index=False))
    print(f"\n TOTAL  field {int(res.field.sum())} | baseline {int(res.baseline_in.sum())} "
          f"({100*res.baseline_in.sum()/res.field.sum():.0f}%) | overlap {int(res.overlap_in.sum())} "
          f"({100*res.overlap_in.sum()/res.field.sum():.0f}%) | overlap+aug {int(res.overlap_aug_in.sum())} "
          f"({100*res.overlap_aug_in.sum()/res.field.sum():.0f}%)")
    if discs:
        print(f"\n UTM shift cross-check (mean vs min recovery): max {max(discs):.3f} m, "
              f"median {np.median(discs):.3f} m over {len(discs)} tiles")

    # ---- rotation sensitivity: same grid tile, θ=0 vs rotated/flipped ----
    # For every position gi with an r000_f0 anchor and at least one rotated/flipped variant, compare
    # the FF3D in-plot detection count. If rotated views detect materially fewer stems, FF3D is
    # rotation-sensitive and rotation augmentation cannot help, since it only adds degraded views.
    print("\n=== rotation sensitivity — SAME grid tile, in-plot detection θ=0 vs rotated (un-rotated) ===")
    deg_rows = []
    for pid, g in man.groupby("plot_id"):
        for gi in sorted(g.gi.unique()):
            gg = g[g.gi == gi]
            anc = gg[(gg.angle_deg == 0.0) & (gg.flip == 0)]
            if not len(anc) or not cache[anc.tile.iloc[0]][4]:
                continue
            n0 = cache[anc.tile.iloc[0]][2]
            for _, tr in gg[(gg.angle_deg != 0.0) | (gg.flip != 0)].sort_values(["flip", "angle_deg"]).iterrows():
                if not cache[tr.tile][4]:
                    continue
                n = cache[tr.tile][2]
                deg_rows.append({"plot_id": pid, "pos": f"g{gi}", "n_rot0": n0,
                                 "view": ("flip" if tr.flip else f"r{int(tr.angle_deg)}"),
                                 "n_view": n, "delta": n - n0})
    deg = pd.DataFrame(deg_rows)
    if len(deg):
        print(deg.to_string(index=False))
        cen = deg[deg.pos == "g4"]
        if len(cen):
            print(f"  centre-tile (g4) mean delta vs θ=0: {cen.delta.mean():+.1f} stems "
                  f"(rotated views detect {'FEWER' if cen.delta.mean() < -0.5 else 'about the same / more'})")

    # ---- un-rotation correctness THROUGH the pipeline: every rotated tile vs its own θ=0 anchor ----
    # Combined test of transform correctness and FF3D rotation stability: un-transform the rotated
    # centroids and match them to the same-position θ=0 centroids. A tight NN means both hold.
    print("\n=== un-rotation correctness (un-transformed rotated centroids vs same-position θ=0) ===")
    for pid, g in man.groupby("plot_id"):
        for gi in sorted(g.gi.unique()):
            gg = g[g.gi == gi]
            anc = gg[(gg.angle_deg == 0.0) & (gg.flip == 0)]
            if not len(anc) or not cache[anc.tile.iloc[0]][4]:
                continue
            c0 = cache[anc.tile.iloc[0]][0]
            if not len(c0):
                continue
            for _, tr in gg[(gg.angle_deg != 0.0) | (gg.flip != 0)].sort_values(["flip", "angle_deg"]).iterrows():
                if not cache[tr.tile][4]:
                    continue
                cr = cache[tr.tile][0]
                if not len(cr):
                    continue
                d = np.min(np.hypot(cr[:, None, 0] - c0[None, :, 0], cr[:, None, 1] - c0[None, :, 1]), axis=1)
                view = "flip" if tr.flip else f"r{int(tr.angle_deg)}"
                print(f"  {pid} g{gi}@{view}: {len(cr)} vs {len(c0)} θ=0 centroids | "
                      f"median NN {np.median(d):.2f} m, <0.5 m {100*np.mean(d < 0.5):.0f}%, "
                      f"<1.0 m {100*np.mean(d < 1.0):.0f}%")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.OUT_DIR / "ff3d_tta_augment.csv", index=False)
    print(f"\nwrote {config.OUT_DIR/'ff3d_tta_augment.csv'}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    if mode == "build":
        build()
    elif mode == "score":
        score()
    else:
        print("usage: ff3d_tta_augment.py [build|score] ...")

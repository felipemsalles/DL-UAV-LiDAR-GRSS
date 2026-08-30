#!/usr/bin/env python3
"""Score the flight design study: tree count per acquisition condition.

Consumes the output of `scripts/run_flight_study.sh` (one folder of panoptic PLYs per condition) and
returns, for each condition, the counting metrics against the 717 trees tallied in the field, plus
the degradation curve.

The point of the study lies in the comparison between the two axes, not in the absolute number of
either:

* `dens_*` throws points away at random. It is the **control**.
* `ang_*` restricts the scan angle, which is what a higher, faster flight, or one with less strip
  overlap, actually looks like. It is the **realistic** axis.

If detection falls equally on both at the same final density, only density matters and geometry does
not. If it falls differently, geometry matters, and that is the finding. Condition `ang10_dens100`
exists to separate the two effects at the same point of the curve.

Deliberate scope: this study uses one tile per plot, not the 9 overlapping ones. The reference is
therefore the single-pass baseline (60.5%), and not the 72.8% of the aggregation. That keeps the cost
at 13 tiles per condition instead of 117, and the question here is about relative degradation, which
the aggregation does not change qualitatively.

Coordinates: the FF3D laz->ply converter subtracts the mean of x and y of each tile, so the PLY comes
out in local coordinates. To return to UTM the mean of the matching input tile has to be added back,
and it differs by condition because each condition has a different subset of points. Getting this
wrong shifts every instance and destroys the in-plot count without flagging any error.

Run: PYTHONPATH=. GREENVISTA_LAZ_DIR=<lazdir> python scripts/exp_flight_study_score.py [outdir]
"""
import re
import sys
import warnings
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

from greenvista import config
from greenvista.repro import seed_everything, write_manifest
from greenvista.segmentation.ff3d import load_panoptic_ply

warnings.simplefilter("ignore")

OUTDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else config.REPO / "work/ff3d_degraded_out"
TILES = config.REPO / "work/ff3d_degraded"
PLOT_R = config.PLOT_RADIUS_M


def condition_plys(d):
    """Canonical PLYs of one condition, one per tile, in the form {tile: path}.

    FF3D does not return `<tile>.ply`. It returns `<tile>_round2.ply` inside
    `round_2_after_remove_noise_200/`, plus a series of intermediates (`<tile>_1.ply`,
    `<tile>_2.ply`) and visualisation helpers (`<tile>_bluepoints_*.ply`) that are NOT panoptic
    maps. Matching on the exact tile name finds nothing, and taking any `*.ply` mixes
    intermediates with the final result. Same rule as
    `scripts/ff3d_detection_numbers.py::find_plys`.
    """
    out = {}
    prefer = d / "round_2_after_remove_noise_200"
    for root in (prefer, d):
        if not root.exists():
            continue
        for ply in sorted(root.rglob("*round2.ply")):
            if "bluepoints" in ply.name:
                continue
            out.setdefault(re.sub(r"_round\d+$", "", ply.stem), ply)
    return out


def plot_centres():
    import geopandas as gpd
    p = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for c in ("talhao", "parcela"):
        p[c] = pd.to_numeric(p[c], errors="coerce").astype("Int64")
    return {f"t{int(r.talhao):03d}_p{int(r.parcela):03d}": (r.geometry.x, r.geometry.y)
            for _, r in p.iterrows() if pd.notna(r.talhao) and pd.notna(r.parcela)}


def in_plot_count(ply_path, tile_laz, cx, cy):
    """Instances whose centroid falls inside the plot radius, in recovered UTM."""
    trees = load_panoptic_ply(ply_path)
    if not trees:
        return 0
    las = laspy.read(str(tile_laz))
    mx, my = float(np.asarray(las.x).mean()), float(np.asarray(las.y).mean())
    n = 0
    for t in trees.values():
        p = t["points"]
        x, y = p[:, 0].mean() + mx, p[:, 1].mean() + my
        if (x - cx) ** 2 + (y - cy) ** 2 <= PLOT_R ** 2:
            n += 1
    return n


def counting_metrics(field, pred, n_boot=4000, seed=0):
    field, pred = np.asarray(field, float), np.asarray(pred, float)
    err = pred - field
    mae = float(np.abs(err).mean())
    rng = np.random.default_rng(seed)
    boot = [np.abs(err[rng.integers(0, len(err), len(err))]).mean() for _ in range(n_boot)]
    return {"MAE": mae, "MAE_lo": float(np.percentile(boot, 2.5)),
            "MAE_hi": float(np.percentile(boot, 97.5)),
            "RMSE": float(np.sqrt((err ** 2).mean())),
            "rMAE_pct": float(100.0 * mae / field.mean()),
            "bias_pct": float(100.0 * err.mean() / field.mean()),
            "detection_ratio_pct": float(100.0 * pred.sum() / field.sum()),
            "n_plots": int(len(field))}


def main():
    seed_everything(0)
    if not OUTDIR.exists():
        sys.exit(f"output not found in {OUTDIR}. Run first: bash scripts/run_flight_study.sh")
    man = pd.read_csv(TILES / "degraded_manifest.csv")
    centres = plot_centres()

    rows, per_plot = [], []
    for d in sorted(x for x in OUTDIR.iterdir() if x.is_dir()):
        cond = d.name
        counts, fields = [], []
        for tile, ply in sorted(condition_plys(d).items()):
            tile_laz = TILES / cond / f"{tile}.laz"
            if tile not in centres or not tile_laz.exists():
                print(f"  [{cond}] {tile} has no matching input, skipping")
                continue
            cx, cy = centres[tile]
            n = in_plot_count(ply, tile_laz, cx, cy)
            mrow = man[(man.condition == cond) & (man.tile == tile)].iloc[0]
            fld = float(mrow.field_n_arv)
            counts.append(n); fields.append(fld)
            # Density is reported per tile, and not only as the condition mean, because the angle
            # restriction is not spatially uniform: a tile under a flight line keeps almost
            # everything, a tile between two lines loses almost everything. Measured on `ang_10`,
            # from 6 to 264 pts/m2, 44x, with a mean of 154 that describes neither end. This column
            # is what allows the analysis paired by effective density, which asks whether an
            # angle-restricted tile detects worse than a randomly thinned tile at the same density.
            per_plot.append({"condition": cond, "tile": tile, "detected": n, "field": fld,
                             "density_pts_m2": float(mrow.density_pts_m2),
                             "angle_limit_deg": mrow.angle_limit_deg})
        if not counts:
            print(f"  [{cond}] no readable PLY, skipping")
            continue
        sub = man[man.condition == cond]
        rows.append({"condition": cond, "density_pts_m2": round(float(sub.density_pts_m2.mean()), 1),
                     "density_min": round(float(sub.density_pts_m2.min()), 1),
                     "density_max": round(float(sub.density_pts_m2.max()), 1),
                     "angle_limit_deg": sub.angle_limit_deg.iloc[0],
                     "axis": "geometry" if cond.startswith("ang") else
                             ("control" if cond.startswith("dens") else "reference"),
                     **counting_metrics(fields, counts)})

    if not rows:
        sys.exit("no condition scored")
    res = pd.DataFrame(rows).sort_values("density_pts_m2", ascending=False)
    print(f"\n{'condition':16s} {'dens':>7} {'range across tiles':>19} {'axis':>11} "
          f"{'detection':>9} {'MAE':>6} {'rel MAE':>8}")
    for _, r in res.iterrows():
        faixa = f"{r.density_min:.0f} to {r.density_max:.0f}"
        print(f"{r.condition:16s} {r.density_pts_m2:7.0f} {faixa:>19} {r.axis:>11} "
              f"{r.detection_ratio_pct:8.1f}% {r.MAE:6.1f} {r.rMAE_pct:7.1f}%")
    if (res.density_max / res.density_min.clip(lower=1) > 5).any():
        print("\nSome condition varies more than 5x in density across tiles. The condition mean does"
              "\n    not describe it. Use manual_match/flight_study_per_plot.csv and analyse paired by"
              "\n    effective density, not by condition.")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.OUT_DIR / "flight_study.csv", index=False)
    pd.DataFrame(per_plot).to_csv(config.OUT_DIR / "flight_study_per_plot.csv", index=False)
    write_manifest(config.OUT_DIR / "flight_study_manifest.json", script="exp_flight_study_score",
                   config={"plot_radius_m": PLOT_R, "outdir": str(OUTDIR), "tiles": str(TILES),
                           "scope": "one tile per plot, reference = single-pass baseline"},
                   metrics={"per_condition": res.to_dict("records")})
    print(f"\nwritten to {config.OUT_DIR}")


if __name__ == "__main__":
    main()

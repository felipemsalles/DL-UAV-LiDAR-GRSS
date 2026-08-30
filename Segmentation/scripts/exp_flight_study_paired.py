"""Paired analysis by effective density for the flight design study.

The question that separates geometry from density: does a tile with a restricted scan angle detect
worse than a tile simply thinned down to the same point density?

Comparing by condition does not work: condition ang_10 ranges from 6 to 264 pts/m2 across tiles, 44x,
and the condition mean describes no tile at all. Random thinning, by contrast, delivers exactly the
target density on every tile. The pairing therefore has to be per tile.

Method: for each tile under an angle restriction, interpolate on the thinning curve of that same tile
the detection expected at that effective density, and report the difference. The thinning curve comes
from the 5 control points of that tile (dens_20, 50, 100, 200 and full).
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

CSV = Path("manual_match/flight_study_per_plot.csv")

rows = list(csv.DictReader(CSV.open()))
for r in rows:
    r["detected"] = float(r["detected"])
    r["field"] = float(r["field"])
    r["density_pts_m2"] = float(r["density_pts_m2"])
    r["angle_limit_deg"] = float(r["angle_limit_deg"]) if r["angle_limit_deg"] else np.nan

CONTROLS = {"full", "dens_200", "dens_100", "dens_50", "dens_20"}
ANGLE = {"ang_20": 20.0, "ang_10": 10.0, "ang10_dens100": 10.0}

by_tile = defaultdict(dict)
for r in rows:
    by_tile[r["tile"]][r["condition"]] = r

print(f"{'tile':<12} {'cond':<14} {'dens':>6} {'obs':>7} {'expected':>9} {'delta':>8}")
print("-" * 62)

deltas = defaultdict(list)
for tile, conds in sorted(by_tile.items()):
    ctrl = [conds[c] for c in CONTROLS if c in conds]
    if len(ctrl) < 3:
        continue
    ctrl.sort(key=lambda r: r["density_pts_m2"])
    xs = np.array([r["density_pts_m2"] for r in ctrl])
    ys = np.array([r["detected"] / r["field"] for r in ctrl])

    for cond in ANGLE:
        if cond not in conds:
            continue
        r = conds[cond]
        d, obs = r["density_pts_m2"], r["detected"] / r["field"]
        # interpolation in log density: thinning degrades approximately log-linearly
        exp = float(np.interp(np.log(max(d, 1e-9)), np.log(xs), ys))
        extrap = d < xs[0] or d > xs[-1]
        deltas[cond].append((obs - exp, extrap))
        flag = "  (out of range)" if extrap else ""
        print(f"{tile:<12} {cond:<14} {d:6.0f} {obs:6.1%} {exp:8.1%} "
              f"{(obs - exp) * 100:+6.1f}pp{flag}")

print()
print("summary, mean difference against the thinning of the SAME tile at the same density")
print("-" * 62)
for cond, vals in ANGLE.items():
    v = deltas.get(cond, [])
    if not v:
        continue
    inside = [d for d, ex in v if not ex]
    allv = [d for d, _ in v]
    m, s, n = np.mean(allv), np.std(allv, ddof=1), len(allv)
    se = s / np.sqrt(n)
    line = f"{cond:<14} {m*100:+6.1f}pp  ci95 [{100*(m-1.96*se):+.1f}, {100*(m+1.96*se):+.1f}]  n={n}"
    if len(inside) != n:
        line += f"  ({n - len(inside)} extrapolated)"
    print(line)
    neg = sum(1 for d in allv if d < 0)
    print(f"{'':14} worse than thinning on {neg} of {n} tiles")

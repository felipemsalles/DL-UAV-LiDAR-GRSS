#!/usr/bin/env python3
"""How precise is the reference ruler? Estimates the precision of the 26 surveyed tree GPS points by
testing how tightly they sit on the known planting lattice (2.87 x 1.88 m), orientation free, spacing
fixed.

If the surveyed trees lie tightly on a regular lattice, the residual ≈ (GPS error + planting jitter) is
small → the reference is precise → a better stem detector is both worthwhile and measurable. If the
residual is no better than random points fit the same way, the lattice is unresolvable from these points
→ the GPS is the accuracy limit and no detector can be shown to beat it.

Method: rotate points by candidate azimuth θ → (u,v); with spacing fixed, the optimal lattice phase is
the circular mean of (u mod s_within) and (v mod s_between); residual = wrapped distance to the nearest
node. Scan θ, keep the best. Baseline: the identical procedure on uniform-random point sets.

Run: PYTHONPATH=. python scripts/exp_gps_ruler.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from greenvista import config
from greenvista.io.geoloc import load_geoloc

S_WITHIN, S_BETWEEN = 1.88, 2.87
RNG = np.random.default_rng(0)


def wrap(x, s):
    m = np.mod(x, s)
    return np.minimum(m, s - m)


def lattice_residual(pts, theta):
    """Median wrapped residual of pts to a (S_WITHIN x S_BETWEEN) lattice at azimuth theta, optimal phase."""
    ct, st = np.cos(theta), np.sin(theta)
    u = pts[:, 0] * ct + pts[:, 1] * st
    v = -pts[:, 0] * st + pts[:, 1] * ct
    # optimal phase = circular mean → shift each coord so its modulo cloud is centred
    def center(coord, s):
        ang = 2 * np.pi * coord / s
        phase = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) * s / (2 * np.pi)
        return wrap(coord - phase, s)
    ru, rv = center(u, S_WITHIN), center(v, S_BETWEEN)
    r = np.hypot(ru, rv)
    return np.median(r), np.median(ru), np.median(rv)


def best_fit(pts, thetas):
    res = [lattice_residual(pts, th)[0] for th in thetas]
    i = int(np.argmin(res))
    return thetas[i], res[i]


def main():
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    geo = load_geoloc()
    pts = geo[["Easting", "Northing"]].to_numpy()
    pts = pts - pts.mean(0)
    n = len(pts)
    thetas = np.deg2rad(np.arange(0, 180, 0.5))

    th, med = best_fit(pts, thetas)
    _, ru, rv = lattice_residual(pts, th)
    az = np.rad2deg(th) % 180
    print(f"{n} surveyed trees | bbox {np.ptp(geo.Easting):.0f} x {np.ptp(geo.Northing):.0f} m")
    print(f"\nBest lattice fit: azimuth={az:.1f}°  (LiDAR grid-recovery found rows at ~4°)")
    print(f"  median residual to nearest planting node = {med:.2f} m")
    print(f"    within-row (1.88 m) residual = {ru:.2f} m | across-row (2.87 m) residual = {rv:.2f} m")

    # nearest-neighbour distances — should cluster near multiples of the spacing if a real grid
    d, _ = cKDTree(pts).query(pts, k=2)
    nn = d[:, 1]
    print(f"  nearest-neighbour distance: median={np.median(nn):.2f} m  min={nn.min():.2f} m "
          f"(planting min = {S_WITHIN} m)")

    # baseline: same fit on uniform-random point sets in the same bbox
    xr = (geo.Easting.min() - geo.Easting.mean(), geo.Easting.max() - geo.Easting.mean())
    yr = (geo.Northing.min() - geo.Northing.mean(), geo.Northing.max() - geo.Northing.mean())
    rand_meds = []
    for _ in range(500):
        rp = np.column_stack([RNG.uniform(*xr, n), RNG.uniform(*yr, n)])
        rand_meds.append(best_fit(rp, thetas)[1])
    rand_meds = np.array(rand_meds)
    pctile = 100 * (rand_meds < med).mean()
    print(f"\nRandom-point baseline (same fit): median residual {np.median(rand_meds):.2f} m "
          f"(5th pct {np.percentile(rand_meds,5):.2f})")
    print(f"  → real points are better than {100-pctile:.0f}% of random sets "
          f"({'structure is REAL' if pctile < 5 else 'NOT distinguishable from random'})")

    # figure: residual vs azimuth + the modulo scatter at best θ
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(np.rad2deg(thetas), [lattice_residual(pts, t)[0] for t in thetas], label="surveyed trees")
    ax[0].axhline(np.median(rand_meds), color="gray", ls="--", label="random baseline (median)")
    ax[0].axvline(az, color="r", ls=":", label=f"best az {az:.0f}°")
    ax[0].set_xlabel("lattice azimuth (°)"); ax[0].set_ylabel("median residual (m)")
    ax[0].set_title("Lattice residual vs orientation"); ax[0].legend(fontsize=8)
    ct, st = np.cos(th), np.sin(th)
    u = pts[:, 0] * ct + pts[:, 1] * st
    v = -pts[:, 0] * st + pts[:, 1] * ct
    ax[1].scatter(np.mod(u, S_WITHIN), np.mod(v, S_BETWEEN), s=40)
    ax[1].set_xlim(0, S_WITHIN); ax[1].set_ylim(0, S_BETWEEN)
    ax[1].set_xlabel("position within 1.88 m cell"); ax[1].set_ylabel("position within 2.87 m cell")
    ax[1].set_title("Trees folded into one grid cell\n(tight cluster = precise GPS on a real grid)")
    fig.tight_layout()
    figp = config.OUT_DIR / "gps_ruler_talhao001.png"
    fig.savefig(figp, dpi=110)

    print("\n" + "=" * 80)
    if pctile < 5 and med < 0.6:
        print(f"VERDICT: GPS is PRECISE (≈{med:.2f} m on a real lattice). A stem detector could beat the")
        print("  ~2 m crown-centroid floor AND we could measure it. Worth building.")
    elif pctile < 5:
        print(f"VERDICT: lattice is real but residual ≈{med:.2f} m — GPS is mapping-grade, ruler is coarse.")
        print("  A better detector would help only down to ~this residual; measurability is marginal.")
    else:
        print("VERDICT: the surveyed points do NOT resolve the planting lattice better than random →")
        print("  the GPS itself is the accuracy limit (~metre-scale). No detector can be *shown* to beat it")
        print("  with this reference. The unlock is better ground truth (RTK), not a better algorithm.")
    print("=" * 80)
    print(f"wrote {figp}")


if __name__ == "__main__":
    main()

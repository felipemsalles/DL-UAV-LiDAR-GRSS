#!/usr/bin/env python3
"""Feasibility spike: can per-tree coordinates for the ~700 census trees be reconstructed from the
planting-grid layout (row/tree index + spacing + plot centre), instead of surveyed?

Stand 001 is the validator — it has both the census tally and 26 surveyed tree GPS points. This
measures the two quantities that decide feasibility:

  1. Is the planting grid geometrically recoverable from the LiDAR canopy? Treetops are detected,
     then the row azimuth theta that makes the treetop pattern most periodic is found, and the
     recovered row spacing read off. If it lands near the nominal 2.87 m at a sharp theta, the grid
     is real and recoverable, so a reconstructed grid can be georeferenced.
  2. How tightly do the 26 surveyed trees sit on detected treetops? That is the localisation floor:
     the best per-tree accuracy any LiDAR-anchored reconstruction could inherit.

Run: PYTHONPATH=. python scripts/exp_grid_reconstruction.py
"""
import numpy as np
import laspy
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from greenvista import config, segmentation as seg

CLOUD = config.TILE_LAZ
SHP = config.DATA / "2-shapes/Cubagem_T001/arv_cubadas_001.shp"
NOMINAL_ROW = 2.87   # m between rows (spacing "2,87 x 1,88")
NOMINAL_TREE = 1.88  # m within row
OUT = config.OUT_DIR


def rowness(v, spacing, nbins=400):
    """Periodicity score of 1-D positions v at a given spacing: power at the fundamental
    frequency 1/spacing of the position histogram (higher = cleaner rows)."""
    lo, hi = v.min(), v.max()
    hist, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    f = 1.0 / spacing
    cc = np.sum(hist * np.cos(2 * np.pi * f * centers))
    ss = np.sum(hist * np.sin(2 * np.pi * f * centers))
    return np.hypot(cc, ss) / (hist.sum() + 1e-9)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # --- treetops from the LiDAR canopy ---
    las = laspy.read(CLOUD)
    keep = las.classification != 18            # drop high-noise class
    xy = np.column_stack([las.x[keep], las.y[keep]]).astype(np.float64)
    z = np.asarray(las.z[keep], np.float64)
    chm, x0, y0, res = seg.build_chm(xy, z, res=0.5)
    tops = seg.detect_treetops(chm, x0, y0, res, window=3, hmin=10.0, smooth=0.6)
    tt = tops[:, :2]
    area_ha = (np.ptp(xy[:, 0]) * np.ptp(xy[:, 1])) / 1e4
    print(f"cloud {len(las.x)/1e6:.1f}M pts over {area_ha*1e4:.0f} m^2 "
          f"({area_ha:.2f} ha) | detected {len(tt)} treetops "
          f"= {len(tt)/area_ha:.0f}/ha (nominal planting ~{1e4/(NOMINAL_ROW*NOMINAL_TREE):.0f}/ha)")

    # --- 1. recover row azimuth: theta maximising periodicity at the nominal row spacing ---
    c = tt.mean(0)
    p = tt - c
    thetas = np.deg2rad(np.arange(0, 180, 0.5))
    scores = []
    for th in thetas:
        v = -p[:, 0] * np.sin(th) + p[:, 1] * np.cos(th)   # coord perpendicular to candidate rows
        scores.append(rowness(v, NOMINAL_ROW))
    scores = np.array(scores)
    best = int(scores.argmax())
    theta = thetas[best]
    az_deg = np.rad2deg(theta) % 180
    v_best = -p[:, 0] * np.sin(theta) + p[:, 1] * np.cos(theta)

    # recovered spacing = median gap between adjacent row-histogram peaks
    hist, edges = np.histogram(v_best, bins=400)
    centers = 0.5 * (edges[:-1] + edges[1:])
    from scipy.signal import find_peaks
    sm = np.convolve(hist, np.ones(3) / 3, "same")
    pk, _ = find_peaks(sm, distance=int(NOMINAL_ROW / (centers[1] - centers[0]) * 0.6),
                       height=sm.max() * 0.25)
    gaps = np.diff(centers[pk])
    rec_row = float(np.median(gaps)) if len(gaps) else float("nan")
    contrast = scores[best] / (np.median(scores) + 1e-9)
    print(f"\n1) GRID RECOVERY  azimuth={az_deg:.1f} deg  "
          f"recovered row spacing={rec_row:.2f} m (nominal {NOMINAL_ROW})  "
          f"periodicity contrast={contrast:.1f}x median  (>~2 => clear rows)")

    # --- 2. localisation floor: surveyed trees -> nearest detected treetop ---
    g = gpd.read_file(SHP)
    gps = np.column_stack([g.geometry.x, g.geometry.y])
    from scipy.spatial import cKDTree
    kd = cKDTree(tt)
    dgps, _ = kd.query(gps, k=1)
    print(f"\n2) LOCALISATION FLOOR  26 surveyed trees -> nearest treetop: "
          f"median={np.median(dgps):.2f} m  mean={dgps.mean():.2f} m  "
          f"p90={np.percentile(dgps,90):.2f} m  (within {NOMINAL_TREE/2:.2f} m = same tree)")
    on_grid = (dgps <= NOMINAL_TREE / 2).mean() * 100
    print(f"   {on_grid:.0f}% of surveyed trees within half-spacing of a detected treetop")

    # --- reconstruction-error budget (propagate the recovered geometry) ---
    half_w = 0.5 * max(np.ptp(gps[:, 0]), np.ptp(gps[:, 1]))
    az_err = 2.0   # deg; conservative theta uncertainty from the periodicity peak width
    rot_edge = half_w * np.sin(np.deg2rad(az_err))
    print(f"\n3) RECONSTRUCTION-ERROR BUDGET (per tree, approximate)")
    print(f"   - plot-centre / origin uncertainty : ~1-3 m (plot GPS, unknown corner)")
    print(f"   - azimuth error {az_err:.0f} deg over {half_w:.0f} m half-plot : {rot_edge:.1f} m at edge")
    print(f"   - index->slot error from mortality/gaps (~20% F/M codes) : ~1 row = {NOMINAL_ROW:.1f} m per miscount")
    print(f"   - LiDAR treetop localisation floor (measured above)     : {np.median(dgps):.2f} m")

    # --- diagnostic figure ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].scatter(tt[:, 0], tt[:, 1], s=4, c="green", alpha=0.5, label=f"{len(tt)} LiDAR treetops")
    ax[0].scatter(gps[:, 0], gps[:, 1], s=60, c="red", marker="x", label="26 surveyed trees")
    d = np.array([np.cos(theta), np.sin(theta)]) * half_w
    ax[0].plot([c[0] - d[0], c[0] + d[0]], [c[1] - d[1], c[1] + d[1]], "b--", lw=1.5,
               label=f"recovered rows (az {az_deg:.0f} deg)")
    ax[0].set_aspect("equal"); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title("Stand 001: treetops vs surveyed trees + recovered row axis")
    ax[1].plot(np.rad2deg(thetas), scores)
    ax[1].axvline(az_deg, color="b", ls="--")
    ax[1].set_xlabel("candidate row azimuth (deg)"); ax[1].set_ylabel(f"periodicity @ {NOMINAL_ROW} m")
    ax[1].set_title(f"Row-azimuth recovery (peak {contrast:.1f}x median)")
    fig.tight_layout()
    figp = OUT / "grid_reconstruction_talhao001.png"
    fig.savefig(figp, dpi=110)
    print(f"\nwrote {figp}")

    # --- verdict ---
    grid_ok = contrast > 2.0 and 0.6 * NOMINAL_ROW < rec_row < 1.6 * NOMINAL_ROW
    floor = np.median(dgps)
    print("\n" + "=" * 78)
    if grid_ok and floor < NOMINAL_TREE:
        print("VERDICT: grid IS recoverable AND surveyed trees sit on treetops ->")
        print("  reconstruction is FEASIBLE if we get per-plot origin+azimuth (small Adimara ask).")
    elif grid_ok:
        print(f"VERDICT: grid recoverable BUT surveyed trees sit {floor:.1f} m off treetops ->")
        print("  reconstruction limited by GPS/detection mismatch; positions would be approximate.")
    else:
        print("VERDICT: grid NOT cleanly recoverable from LiDAR here -> reconstruction unreliable;")
        print("  a GPS campaign (not just orientation) would be needed for clean per-tree labels.")
    print("=" * 78)


if __name__ == "__main__":
    main()

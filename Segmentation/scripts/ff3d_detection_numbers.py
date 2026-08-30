#!/usr/bin/env python3
"""Parse FF3D panoptic output → detection numbers for all stands.

For each plot tile: recover the FF3D local→UTM shift from the backup tile bbox, convert instance
centroids to UTM, count instances whose centroid falls in the 12 m plot → compare to the field stem
count. Stand 001 tiles are additionally matched to the 26 known cubed-tree positions for a true
detection rate (dominant vs suppressed).

Run: PYTHONPATH=. GREENVISTA_LAZ_DIR=<lazdir> python scripts/ff3d_detection_numbers.py [ply_dir]
"""
import sys, glob, zipfile, tempfile
from pathlib import Path
import numpy as np, pandas as pd, laspy
from plyfile import PlyData

from greenvista import config
from greenvista.segmentation import match_detections

BACKUP = Path(config.LAZ_DIR).parent / "ff3d_tiles"
PLOT_R = config.PLOT_RADIUS_M
PLOT_AREA_HA = config.PLOT_AREA_HA


def find_plys(root):
    # Prefer the canonical final output (round_2_after_remove_noise_200); fall back to round_2.
    root = Path(root)
    final = glob.glob(str(root / "**/round_2_after_remove_noise_200/*round2.ply"), recursive=True)
    if final:
        return final
    plys = glob.glob(str(root / "**/round_2/*round2.ply"), recursive=True)
    if plys:
        return plys
    plys = glob.glob(str(root / "**/*.ply"), recursive=True)
    for z in glob.glob(str(root / "**/*.zip"), recursive=True):
        try:
            d = Path(tempfile.mkdtemp())
            with zipfile.ZipFile(z) as zf:
                zf.extractall(d)
            plys += glob.glob(str(d / "**/*round2.ply"), recursive=True)
        except Exception:
            pass
    return plys


def read_panoptic(path):
    ply = PlyData.read(str(path)); el = ply.elements[0]
    d = {p.name: np.array(el[p.name]) for p in el.properties}
    inst = d.get("instance_pred", d.get("instance"))
    return d["x"], d["y"], d["z"], inst


def centroids_utm(path, tile_laz):
    lx, ly, lz, inst = read_panoptic(path)
    las = laspy.read(str(tile_laz))
    ax0, ay0 = float(np.asarray(las.x).min()), float(np.asarray(las.y).min())
    sx, sy = ax0 - lx.min(), ay0 - ly.min()
    cents, htop = [], []
    for iid in np.unique(inst):
        if iid < 0:
            continue
        m = inst == iid
        cents.append([lx[m].mean() + sx, ly[m].mean() + sy]); htop.append(float(lz[m].max()))
    return np.array(cents).reshape(-1, 2), np.array(htop), int((inst < 0).sum()), len(inst)


def main():
    ply_dir = sys.argv[1] if len(sys.argv) > 1 else \
        str(config.REPO / "project/models/FF3D_inference/FF3D_oracle/bucket_out_folder")
    man = pd.read_csv(BACKUP / "tiles_manifest.csv")
    plys = find_plys(ply_dir)
    print(f"found {len(plys)} PLY(s) under {ply_dir}")
    if not plys:
        print("no FF3D output yet — run the pipeline first."); return

    rows = []
    for _, t in man.iterrows():
        cand = [p for p in plys if t.tile in Path(p).name]
        if not cand:
            print(f"  {t.tile}: no PLY"); continue
        cents, htop, n_unassigned, n_pts = centroids_utm(cand[0], BACKUP / f"{t.tile}.laz")
        if len(cents):
            d2 = (cents[:, 0] - t.cx) ** 2 + (cents[:, 1] - t.cy) ** 2
            in_plot = int((d2 <= PLOT_R ** 2).sum())
        else:
            in_plot = 0
        rows.append({"tile": t.tile, "talhao": int(t.talhao), "plot_id": t.plot_id,
                     "ff3d_instances_tile": len(cents), "ff3d_in_plot": in_plot,
                     "field_n_arv": t.field_n_arv, "det_ratio": in_plot / t.field_n_arv,
                     "ff3d_stems_ha": in_plot / PLOT_AREA_HA, "field_stems_ha": t.field_n_arv_ha,
                     "pct_unassigned": round(100 * n_unassigned / max(n_pts, 1), 1)})
        print(f"  {t.tile}: FF3D {len(cents)} inst in tile, {in_plot} in-plot vs {int(t.field_n_arv)} field "
              f"({100*in_plot/t.field_n_arv:.0f}%), {n_unassigned/max(n_pts,1)*100:.0f}% unassigned pts")

    res = pd.DataFrame(rows)
    if len(res):
        print("\n=== FF3D detection by stand ===")
        by = res.groupby("talhao").agg(plots=("tile", "count"), ff3d_in_plot=("ff3d_in_plot", "sum"),
                                       field=("field_n_arv", "sum")).reset_index()
        by["detection_rate_%"] = (100 * by.ff3d_in_plot / by.field).round(0)
        print(by.to_string(index=False))
        print(f"\nOVERALL detection rate: {100*res.ff3d_in_plot.sum()/res.field_n_arv.sum():.0f}% "
              f"({int(res.ff3d_in_plot.sum())}/{int(res.field_n_arv.sum())} stems) "
              f"vs CHM ~47% — 3D {'BEATS' if res.ff3d_in_plot.sum()/res.field_n_arv.sum()>0.47 else 'ties/below'} the CHM ceiling")

        # stand 001: position accuracy vs the known destructively scaled tree positions.
        # Restrict to trees inside the tiled footprints: most of the 25 known trees lie elsewhere in
        # the stand and were never tiled. Report several radii, since crown centroids sit a few
        # metres from stem GPS positions (lean/asymmetry) and 2 m is too tight.
        try:
            from greenvista.io.geoloc import load_geoloc
            geo = load_geoloc()
            gt = geo[["Easting", "Northing"]].to_numpy()
            allc, inside = [], np.zeros(len(gt), bool)
            for _, t in man[man.talhao == 1].iterrows():
                cand = [p for p in plys if t.tile in Path(p).name]
                if cand:
                    c, *_ = centroids_utm(cand[0], BACKUP / f"{t.tile}.laz")
                    allc.append(c)
                inside |= (np.abs(gt[:, 0] - t.cx) <= 16) & (np.abs(gt[:, 1] - t.cy) <= 16)
            if allc:
                allc = np.vstack(allc)
                gt_in = gt[inside]
                print(f"\nStand 001 position accuracy ({inside.sum()}/{len(gt)} known trees fall in the tiles):")
                for rad in (2.0, 3.0, 4.0):
                    r = match_detections(allc, gt_in, radius=rad)
                    print(f"  r={rad:.0f}m: {r['detected']}/{r['n_gt']} located ({r['detection_rate']*100:.0f}%)")
        except Exception as e:
            print(f"(stand-001 known-position match skipped: {e})")
        config.OUT_DIR.mkdir(parents=True, exist_ok=True)
        res.to_csv(config.OUT_DIR / "ff3d_detection.csv", index=False)
        print(f"\nwrote {config.OUT_DIR/'ff3d_detection.csv'}")


if __name__ == "__main__":
    main()

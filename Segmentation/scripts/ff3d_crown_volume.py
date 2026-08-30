#!/usr/bin/env python3
"""Does the disposition of FF3D crowns predict plot volume? (segmentation → volume)

For each plot, extract per-crown geometry from the FF3D segmentation (2-D crown area, 3-D crown-hull
volume, crown height, centroid), aggregate to stand features (crown count, size mean/CV/Gini, summed
crown-volume per ha, packing, nearest-neighbour spacing), and regress Vol_ha under LOPO /
leave-one-stand-out — against the LiDAR-PLS and Hdom+age baselines. This is the calibrated version of
"volume from canopy arrangement" (not the mechanistic ITD sum that failed on occlusion).

Run: PYTHONPATH=. GREENVISTA_LAZ_DIR=<lazdir> python scripts/ff3d_crown_volume.py
"""
import glob, zipfile, tempfile
from pathlib import Path
import numpy as np, pandas as pd, laspy
from scipy.spatial import ConvexHull, QhullError, cKDTree
from plyfile import PlyData

from greenvista import config
from greenvista.repro import seed_everything, write_manifest
from greenvista.tracking import log_experiment
from greenvista.area_based import data as abd, models as abm, validate as abv
from greenvista.eval import metrics, jackknife_plus_interval, coverage, permutation_test

BACKUP = Path(config.LAZ_DIR).parent / "ff3d_tiles"
PLOT_R = config.PLOT_RADIUS_M
PLOT_AREA_HA = config.PLOT_AREA_HA


def _hull(pts):
    try:
        return float(ConvexHull(pts).volume)     # area in 2D, volume in 3D
    except (QhullError, ValueError):
        return np.nan


def crowns_in_plot(ply_path, tile_laz, cx, cy):
    """Per-crown geometry for crowns whose centroid is within the 12 m plot."""
    ply = PlyData.read(str(ply_path)); el = ply.elements[0]
    D = {p.name: np.array(el[p.name]) for p in el.properties}
    las = laspy.read(str(tile_laz)); ax0, ay0 = float(np.asarray(las.x).min()), float(np.asarray(las.y).min())
    sx, sy = ax0 - D["x"].min(), ay0 - D["y"].min()
    inst = D["instance_pred"]
    rows = []
    for iid in np.unique(inst):
        if iid < 0:
            continue
        m = inst == iid
        x, y, z = D["x"][m] + sx, D["y"][m] + sy, D["z"][m]
        ux, uy = float(x.mean()), float(y.mean())
        if (ux - cx) ** 2 + (uy - cy) ** 2 > PLOT_R ** 2 or len(x) < 4:
            continue
        rows.append({"cx": ux, "cy": uy, "area": _hull(np.column_stack([x, y])),
                     "vol": _hull(np.column_stack([x, y, z])), "h": float(z.max()), "n": len(x)})
    return pd.DataFrame(rows)


def plot_crown_features(cr):
    """Aggregate per-crown geometry → stand-level crown-disposition features."""
    cr = cr.dropna(subset=["area", "vol"])
    if len(cr) < 2:
        return None
    a = cr.area.to_numpy(); sa = np.sort(a)
    gini = float((2 * np.arange(1, len(sa) + 1) - len(sa) - 1).dot(sa) / (len(sa) * sa.sum()))
    tree = cKDTree(cr[["cx", "cy"]].to_numpy())
    nn = tree.query(cr[["cx", "cy"]].to_numpy(), k=2)[0][:, 1]
    return {"n_crowns_ha": len(cr) / PLOT_AREA_HA, "crown_area_mean": float(a.mean()),
            "crown_area_cv": float(a.std() / a.mean()), "crown_area_gini": gini,
            "crown_vol_ha": float(cr.vol.sum()) / PLOT_AREA_HA, "crown_vol_mean": float(cr.vol.mean()),
            "crown_h_mean": float(cr.h.mean()), "crown_h_max": float(cr.h.max()),
            "packing": float(a.sum()) / (np.pi * PLOT_R ** 2), "nn_spacing": float(nn.mean())}


def main():
    seed_everything(0)
    man = pd.read_csv(BACKUP / "tiles_manifest.csv")
    zips = sorted(glob.glob(str(config.REPO / "project/models/FF3D_inference/FF3D_oracle/bucket_out_folder/*.zip")))
    if not zips:
        print("no FF3D output zip — run the full pipeline first."); return
    d = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(zips[-1]) as zf:
        zf.extractall(d)
    plys = glob.glob(str(d / "**/*round2.ply"), recursive=True)

    feats = []
    for _, t in man.iterrows():
        cand = [p for p in plys if t.tile in Path(p).name]
        if not cand:
            print(f"  {t.tile}: no PLY"); continue
        cr = crowns_in_plot(cand[0], BACKUP / f"{t.tile}.laz", t.cx, t.cy)
        f = plot_crown_features(cr)
        if f:
            feats.append({"plot_id": t.plot_id, "talhao": int(t.talhao), **f})
    fdf = pd.DataFrame(feats)
    df = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                        config.DATA / "5-dados_campo/inventario_est.csv").merge(fdf, on=["talhao", "plot_id"] if "plot_id" in fdf else "plot_id")
    y = df.Vol_ha.to_numpy()
    print(f"crown features for {len(df)} plots. corr with Vol_ha:")
    for c in ["crown_vol_ha", "crown_area_mean", "crown_h_mean", "n_crowns_ha", "packing", "crown_area_gini"]:
        print(f"  {c:16s} r={np.corrcoef(df[c], y)[0,1]:+.2f}")

    def ev(name, cols):
        make = lambda: abm.make_huber_allom(cols)
        lopo = abv._lopo(make, df)[0]; loto = abv._group_cv(make, df)
        a, b = metrics(y, lopo), metrics(y, loto)
        lo, hi = jackknife_plus_interval(y, lopo)
        print(f"  {name:34s} LOPO R2={a['R2']:+.2f} rRMSE={a['rRMSE_pct']:4.1f}%   LOTO R2={b['R2']:+.2f} rRMSE={b['rRMSE_pct']:4.1f}%")
        return {"model": name, "LOPO_R2": a["R2"], "LOPO_rRMSE": a["rRMSE_pct"],
                "LOTO_R2": b["R2"], "LOTO_rRMSE": b["rRMSE_pct"], "cov95": coverage(y, lo, hi),
                "perm_p": permutation_test(y, lopo, n_perm=3000)["p_value"]}

    print("\n=== crown DISPOSITION → Vol_ha (LiDAR segmentation only) ===")
    rows = [
        ev("crown volume/ha",           ("crown_vol_ha",)),
        ev("crown vol + size + packing", ("crown_vol_ha", "crown_area_mean", "packing")),
        ev("crown h_mean + count",      ("crown_h_mean", "n_crowns_ha")),
        ev("crown vol + Gini (heterog)", ("crown_vol_ha", "crown_area_gini")),
    ]
    print("  --- baselines ---")
    rows.append(ev("LiDAR PLS-equiv (zmean,zq90)", ("zmean", "zq90")))
    rows.append(ev("Hdom+age (deliverable)", ("zmax", "zq90", "idade_anos")))

    res = pd.DataFrame(rows)
    best_crown = res[~res.model.str.contains("baseline|PLS|Hdom")].sort_values("LOTO_rRMSE").iloc[0]
    verdict = ("crown disposition ADDS cross-stand signal" if best_crown.LOTO_R2 > 0.4
               else "crown disposition does NOT beat height+age at N=13 — arrangement can't rescue what age explains")
    print(f"\nVERDICT: best crown model LOTO R²={best_crown.LOTO_R2:+.2f} → {verdict}")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.OUT_DIR / "ff3d_crown_volume.csv", index=False)
    df.to_csv(config.OUT_DIR / "ff3d_crown_features.csv", index=False)
    write_manifest(config.OUT_DIR / "ff3d_crown_volume_manifest.json", script="ff3d_crown_volume.py",
                   config={"N": len(df), "source": "FF3D crowns"}, metrics=res.to_dict(orient="records"))
    log_experiment("ff3d-crown-volume", group="volume", tags=["segmentation-to-volume"],
                   config={"task": "Vol_ha from FF3D crown disposition", "N": int(len(df))},
                   metrics={"best_crown_LOTO_R2": float(best_crown.LOTO_R2),
                            "best_crown_LOTO_rRMSE": float(best_crown.LOTO_rRMSE), "verdict": verdict},
                   table_csv=str(config.OUT_DIR / "ff3d_crown_volume.csv"),
                   artifacts=[str(config.OUT_DIR / "ff3d_crown_features.csv")])
    print(f"wrote {config.OUT_DIR/'ff3d_crown_volume.csv'} + logged to W&B")


if __name__ == "__main__":
    main()

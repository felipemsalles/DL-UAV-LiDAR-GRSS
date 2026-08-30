#!/usr/bin/env python3
"""Spike S5 — segmentation-free per-tree volume from LiDAR height at KNOWN stem positions.

S2b showed FF3D crown geometry is unusable, while FF3D detection and the field-H signal (R²=0.78)
are not. Segmentation is therefore skipped: at each of the 23 geolocated cubed trees, take the LiDAR
height (local max-Z in a small XY cylinder) and regress V ~ LiDAR-height (LOOCV + bootstrap CI). Uses
all 23 trees (vs 7 in the tile). Predictor is LiDAR-only — field DBH is never used.

Diagnostic split by overtopping (lidar_Hmax - field_H): a suppressed tree's cylinder is dominated by the
taller neighbour above it, so its LiDAR height is biased. This is an occlusion limit, not an
implementation fault, and it drags the pooled R² down while dominant trees keep the signal.

Reference: field-H ceiling R²=0.78 ; crude-cylinder S1 R²=0.19 ; S2b FF3D crowns R²<0.
"""
import numpy as np, pandas as pd, laspy
from scipy.spatial import cKDTree
from greenvista.volume import CrownAllometry
from greenvista.eval import loocv_predict, metrics, bootstrap_ci

from greenvista import config
DATA = config.DATA
LAZ = config.TILE_LAZ


def loocv_line(label, X, y):
    if len(y) < 5 or not (X > 0).all():
        print(f"  {label:34s} N={len(y)}  (skipped: <5 pts or non-positive X)"); return
    yhat = loocv_predict(CrownAllometry, X, y)
    mm = metrics(y, yhat); ci = bootstrap_ci(y, yhat)
    print(f"  {label:34s} N={len(y)}  R²={mm['R2']:+.2f} "
          f"(CI {ci['R2_ci'][0]:+.2f}..{ci['R2_ci'][1]:+.2f})  rRMSE={mm['rRMSE_pct']:.0f}%")


def main():
    g = pd.read_csv(DATA/"2-shapes/Cubagem_T001/arvores_cubadas.csv")
    g = g[g.Name.astype(str).str.fullmatch(r"\d+")].copy(); g["arv"] = g.Name.astype(int)
    vol = pd.read_csv(DATA/"5-dados_campo/volumes.csv")
    lk = g.merge(vol[vol.talhao == 1][["arv", "V", "H_m"]], on="arv").reset_index(drop=True)

    las = laspy.read(LAZ)
    xy = np.column_stack([np.asarray(las.x), np.asarray(las.y)]); z = np.asarray(las.z)
    tree = cKDTree(xy)
    P = lk[["Easting", "Northing"]].values

    print(f"N={len(lk)} geolocated trees | cloud {len(z):,} pts\n")

    # --- LiDAR height per tree, several cylinder radii ---
    for r in (0.6, 1.0, 1.5):
        hmax, npts = [], []
        for (e, n) in P:
            idx = tree.query_ball_point([e, n], r)
            hmax.append(z[idx].max() if idx else np.nan); npts.append(len(idx))
        lk[f"hmax_{r}"] = hmax; lk[f"npts_{r}"] = npts
        ok = lk[f"hmax_{r}"].notna()
        c = np.corrcoef(lk.loc[ok, f"hmax_{r}"], lk.loc[ok, "H_m"])[0, 1]
        bias = (lk.loc[ok, f"hmax_{r}"] - lk.loc[ok, "H_m"]).mean()
        print(f"radius {r} m: corr(lidar_Hmax, field_H)={c:+.2f}  bias={bias:+.2f} m  "
              f"(median {int(lk[f'npts_{r}'].median())} pts/tree)")

    # --- pooled (all 23) LiDAR-height -> V ---
    R = 1.0
    sub = lk.dropna(subset=[f"hmax_{R}"]).copy()
    print(f"\nLOOCV  V ~ LiDAR height  (r={R} m, all trees):")
    loocv_line("V ~ lidar_Hmax", sub[[f"hmax_{R}"]].values, sub.V.values)

    # --- split by overtopping: suppressed = a taller neighbour sits over the stem ---
    sub["overtop"] = sub[f"hmax_{R}"] - sub.H_m
    dom = sub[sub.overtop <= 2.0]; sup = sub[sub.overtop > 2.0]
    print(f"\nOvertopping split (lidar_Hmax - field_H > 2 m => suppressed): "
          f"dominant N={len(dom)}, suppressed N={len(sup)}")
    loocv_line("V ~ lidar_Hmax  [DOMINANT only]", dom[[f"hmax_{R}"]].values, dom.V.values)
    if len(dom) >= 5:
        c = np.corrcoef(dom[f"hmax_{R}"], dom.H_m)[0, 1]
        print(f"     (dominant corr lidar_Hmax↔field_H = {c:+.2f})")

    # --- ceiling reference: field height (diagnostic, not a LiDAR predictor) ---
    print("\nReference (field H — diagnostic ceiling, not a LiDAR predictor):")
    loocv_line("V ~ field_H  [all 23]", lk[["H_m"]].values, lk.V.values)
    loocv_line("V ~ field_H  [dominant only]", dom[["H_m"]].values, dom.V.values)

    print("\nAnchors: field-H ceiling R²=0.78 | crude-cyl S1 R²=0.19 | S2b FF3D crowns R²<0.")


if __name__ == "__main__":
    main()

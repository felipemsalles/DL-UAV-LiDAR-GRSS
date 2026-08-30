#!/usr/bin/env python3
"""Spike S1 — does LiDAR crown geometry correlate with cubed volume on the 23 linked trees?

Go/no-go probe, without a segmentation model. For each of the 23 trees that are both geolocated
(arv_cubadas_001) and cubed (volumes.csv, talhao 1, Name==arv), extract a crude crown proxy from
cloud 001 (a vertical cylinder around the field XY) and regress it against ground-truth V.

Gate: if LiDAR-derived crown features reach LOOCV R² > ~0.4, signal exists → Tier A worth building.
"""
import numpy as np
import pandas as pd
import laspy
from scipy.spatial import ConvexHull, QhullError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

from greenvista import config
DATA = config.DATA
LAZ = config.TILE_LAZ

def load_linked():
    g = pd.read_csv(DATA / "2-shapes/Cubagem_T001/arvores_cubadas.csv")
    g = g[g.Name.astype(str).str.fullmatch(r"\d+")].copy()
    g["arv"] = g.Name.astype(int)
    vol = pd.read_csv(DATA / "5-dados_campo/volumes.csv")
    v1 = vol[vol.talhao == 1][["arv", "V", "D_cm", "H_m"]]
    m = g.merge(v1, on="arv", how="inner")  # Name==arv
    return m[["arv", "Easting", "Northing", "V", "D_cm", "H_m"]].reset_index(drop=True)

def crown_proxies(las_xy, las_z, e, n, radius):
    d2 = (las_xy[:, 0] - e) ** 2 + (las_xy[:, 1] - n) ** 2
    sel = d2 <= radius * radius
    z = las_z[sel]
    canopy = z[z > 2.0]
    if len(canopy) < 5:
        return None
    xy = las_xy[sel][z > 2.0]
    try:
        area = float(ConvexHull(xy).volume)  # 2D hull area
    except (QhullError, ValueError):
        area = np.nan
    return dict(H_max=float(z.max()), z99=float(np.percentile(z, 99)),
               n_canopy=int(len(canopy)), hull_area=area,
               crown_vol_proxy=float(len(canopy)) * float(np.percentile(canopy, 90)))

def loocv_r2(X, y):
    yp = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
    rmse = float(np.sqrt(mean_squared_error(y, yp)))
    return r2_score(y, yp), rmse, rmse / y.mean() * 100

def main():
    linked = load_linked()
    print(f"Linked trees (geolocated AND cubed, Name==arv): N={len(linked)}")
    las = laspy.read(str(LAZ))
    cls = np.asarray(las.classification)
    keep = (cls != 2) & (cls != 18)  # non-ground, non-noise = canopy/veg
    xy = np.column_stack([np.asarray(las.x)[keep], np.asarray(las.y)[keep]])
    z = np.asarray(las.z)[keep]
    print(f"Cloud 001 canopy points: {keep.sum():,}")

    for radius in (0.9, 1.25, 1.5):
        rows, kept = [], []
        for _, t in linked.iterrows():
            p = crown_proxies(xy, z, t.Easting, t.Northing, radius)
            if p is None:
                continue
            p.update(arv=int(t.arv), V=t.V)
            rows.append(p); kept.append(int(t.arv))
        df = pd.DataFrame(rows)
        if len(df) < 8:
            print(f"\n[r={radius}m] only {len(df)} trees had >=5 canopy points — skip")
            continue
        y = df.V.values
        print(f"\n=== radius={radius} m | trees used={len(df)} ===")
        # correlations
        for c in ["H_max", "z99", "n_canopy", "hull_area", "crown_vol_proxy"]:
            if df[c].notna().all():
                print(f"   r(V,{c}) = {np.corrcoef(y, df[c])[0,1]:+.3f}")
        # LOOCV OLS for a few feature sets
        sets = {
            "H_max": ["H_max"],
            "H_max+n_canopy": ["H_max", "n_canopy"],
            "H_max+hull_area+n_canopy": ["H_max", "hull_area", "n_canopy"],
            "crown_vol_proxy": ["crown_vol_proxy"],
        }
        for name, cols in sets.items():
            sub = df.dropna(subset=cols)
            if len(sub) < 8:
                continue
            r2, rmse, rrmse = loocv_r2(sub[cols].values, sub.V.values)
            print(f"   LOOCV  {name:28s}: R²={r2:+.3f}  RMSE={rmse:.3f} m³  rRMSE={rrmse:.0f}%")
    print("\nGATE: LOOCV R² > ~0.4 on any sane feature set → signal exists, Tier A worth building.")

if __name__ == "__main__":
    main()

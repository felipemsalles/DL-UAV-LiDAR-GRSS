#!/usr/bin/env python3
"""Spike S2b — Tier-A feasibility test on FF3D crowns.

Assign each of the 9 linked cubed trees to its FF3D instance (low-band stem-column vote), extract
per-instance crown features, and regress crown features -> V (LOOCV). Compare to: crude-cylinder S1
(R²=0.19) and the field-height ceiling (R²=0.78). N=9 here (tile only) — directional, wide CIs.
"""
import numpy as np, pandas as pd, laspy
from plyfile import PlyData
from greenvista.segmentation import assign_trees_to_instances
from greenvista.features import crown_features
from greenvista.volume import CrownAllometry
from greenvista.eval import loocv_predict, metrics, bootstrap_ci

from greenvista import config
DATA = config.DATA
PLY = str(config.FF3D_TILE_PLY)

def main():
    g = pd.read_csv(DATA/"2-shapes/Cubagem_T001/arvores_cubadas.csv")
    g = g[g.Name.astype(str).str.fullmatch(r"\d+")].copy(); g["arv"] = g.Name.astype(int)
    vol = pd.read_csv(DATA/"5-dados_campo/volumes.csv")
    lk = g.merge(vol[vol.talhao==1][["arv","V","H_m"]], on="arv")
    E, N = lk.Easting.values, lk.Northing.values
    e0, e1 = 748945, 748945+35; n0, n1 = N.min()-3, N.max()+3
    trees = lk[(E>=e0)&(E<=e1)&(N>=n0)&(N<=n1)].copy().reset_index(drop=True)

    las = laspy.read(config.TILE_LAZ)
    x, y = np.asarray(las.x), np.asarray(las.y); m = (x>=e0)&(x<=e1)&(y>=n0)&(y<=n1)
    ax0, ay0 = x[m].min(), y[m].min()
    ply = PlyData.read(PLY); el = ply.elements[0]
    d = {p.name: np.array(el[p.name]) for p in el.properties}
    sx, sy = ax0 - d['x'].min(), ay0 - d['y'].min()           # local -> abs shift
    axy = np.column_stack([d['x']+sx, d['y']+sy]); az = d['z']; inst = d['instance_pred']

    assigned = assign_trees_to_instances(axy, az, inst, trees[["Easting","Northing"]].values,
                                         radius=1.2, z_lo=2.0, z_hi=12.0)
    trees["inst"] = assigned
    n_unmatched = int((assigned < 0).sum())
    dup = pd.Series(assigned[assigned >= 0]).duplicated(keep=False).sum()
    print(f"trees={len(trees)} | unmatched={n_unmatched} | trees sharing an instance (collisions)={int(dup)}")

    rows = []
    for _, t in trees.iterrows():
        if t.inst < 0: continue
        pts = np.column_stack([d['x'], d['y'], d['z']])[inst == t.inst]  # local coords OK (hulls/Z invariant)
        f = crown_features(pts)
        f.update(arv=int(t.arv), V=t.V, field_H=t.H_m, inst=int(t.inst))
        rows.append(f)
    df = pd.DataFrame(rows)
    print(f"\nmatched trees with features: N={len(df)}")
    if len(df) >= 5:
        print("  corr(FF3D crown H_max, field_H) = %+.2f (bias %+.2f m)  [crude cylinder was r=0.58, bias +3.19]"
              % (np.corrcoef(df.H_max, df.field_H)[0,1], (df.H_max-df.field_H).mean()))
        for c in ["H_max", "z99", "crown_area", "crown_volume", "crown_diameter", "n_points"]:
            if df[c].notna().all():
                print(f"  r(V, {c}) = {np.corrcoef(df.V, df[c])[0,1]:+.2f}")
        for cols in (["crown_area","H_max"], ["crown_volume","H_max"], ["crown_area","crown_volume","H_max"]):
            sub = df.dropna(subset=cols)
            if len(sub) >= 6 and (sub[cols].values>0).all():
                yhat = loocv_predict(CrownAllometry, sub[cols].values, sub.V.values)
                mm = metrics(sub.V.values, yhat); ci = bootstrap_ci(sub.V.values, yhat, n_boot=2000)
                print(f"  LOOCV V ~ {'+'.join(cols):28s}: R²={mm['R2']:+.2f} (CI {ci['R2_ci'][0]:+.2f}..{ci['R2_ci'][1]:+.2f})"
                      f"  rRMSE={mm['rRMSE_pct']:.0f}%")
    print("\nReference: crude-cylinder S1 R²=0.19 ; field-H ceiling R²=0.78. N=9 -> directional only, wide CIs.")

if __name__ == "__main__":
    main()

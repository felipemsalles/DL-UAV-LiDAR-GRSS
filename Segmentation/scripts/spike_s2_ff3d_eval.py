#!/usr/bin/env python3
"""Spike S2 — evaluate FF3D instance segmentation on the tile vs the 9 linked cubed trees.

FF3D shifts coords to a local origin; recover the shift from the tile's absolute bbox (cloud 001, same
window), convert instance centroids back to UTM, then match to the known tree positions. The metric of
interest is detection of the suppressed trees that the CHM (S4) structurally misses.
"""
import numpy as np, pandas as pd, laspy
from plyfile import PlyData
from greenvista.segmentation import match_detections

from greenvista import config
DATA = config.DATA
PLY = str(config.FF3D_TILE_PLY)
# tile window used to create the input (scripts: e0=748945, +35; N over all linked ±3)
def linked_trees():
    g = pd.read_csv(DATA/"2-shapes/Cubagem_T001/arvores_cubadas.csv")
    g = g[g.Name.astype(str).str.fullmatch(r"\d+")].copy(); g["arv"] = g.Name.astype(int)
    vol = pd.read_csv(DATA/"5-dados_campo/volumes.csv")
    return g.merge(vol[vol.talhao==1][["arv","V","H_m"]], on="arv")

def main():
    lk = linked_trees()
    E, N = lk.Easting.values, lk.Northing.values
    e0, e1 = 748945, 748945+35; n0, n1 = N.min()-3, N.max()+3
    win = (E>=e0)&(E<=e1)&(N>=n0)&(N<=n1)
    trees = lk[win].copy()
    # absolute bbox of the tile points (recreate same spatial mask on cloud 001)
    las = laspy.read(config.TILE_LAZ)
    x, y = np.asarray(las.x), np.asarray(las.y)
    m = (x>=e0)&(x<=e1)&(y>=n0)&(y<=n1)
    ax0, ay0 = x[m].min(), y[m].min()
    # FF3D local coords
    ply = PlyData.read(PLY); el = ply.elements[0]
    d = {p.name: np.array(el[p.name]) for p in el.properties}
    lx, ly, lz = d['x'], d['y'], d['z']; inst = d['instance_pred']
    shift_x = ax0 - lx.min(); shift_y = ay0 - ly.min()  # abs = local + shift
    print(f"tile abs bbox min=({ax0:.1f},{ay0:.1f}) | FF3D local min=({lx.min():.1f},{ly.min():.1f}) "
          f"-> shift=({shift_x:.1f},{shift_y:.1f})")
    # instance centroids -> absolute
    cents, ihmax = [], []
    for iid in np.unique(inst):
        if iid < 0: continue
        mask = inst == iid
        cents.append([lx[mask].mean()+shift_x, ly[mask].mean()+shift_y]); ihmax.append(lz[mask].max())
    cents = np.array(cents); ihmax = np.array(ihmax)
    dom = trees.H_m >= lk.H_m.max()-5
    print(f"\nFF3D instances: {len(cents)} in tile (~{len(cents)/((e1-e0)*(n1-n0)/1e4):.0f}/ha) | "
          f"unassigned pts: {int((inst<0).sum())}/{len(inst)} ({100*(inst<0).mean():.0f}%)")
    print(f"linked trees in tile: {len(trees)} (dominant={int(dom.sum())}, suppressed={int((~dom).sum())})")
    for radius in (1.5, 2.5):
        r = match_detections(cents, trees[["Easting","Northing"]].values, radius=radius)
        hits = r["hits"]
        drd = np.mean([h for h,dd in zip(hits,dom) if dd])*100 if dom.sum() else 0
        drs = np.mean([h for h,dd in zip(hits,dom) if not dd])*100 if (~dom).sum() else 0
        # matched instance height vs field H
        mi = r["matched_idx"]; mh=[(ihmax[j], t.H_m) for j,(_,t) in zip(mi, trees.iterrows()) if j>=0]
        if mh:
            ih,fh=np.array(mh).T; corr=np.corrcoef(ih,fh)[0,1] if len(mh)>2 else np.nan; bias=(ih-fh).mean()
        else: corr,bias=np.nan,np.nan
        print(f"\n[match r={radius}m] detected {r['detected']}/{len(trees)} ({r['detection_rate']*100:.0f}%)"
              f" | dominant {drd:.0f}% | SUPPRESSED {drs:.0f}% | matched-H vs field: r={corr:+.2f} bias={bias:+.2f}m")
    print("\nCompare to S4/CHM (suppressed 19-31%). If FF3D suppressed-detection is higher, 3D beats the CHM ceiling.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Spike S4 — CHM local-maxima tree detection on full cloud 001, scored vs the 23 cubed trees.

A nadir CHM only sees the top canopy surface, so an overtopped tree has no local maximum and CHM-based
ITD cannot detect it. This quantifies that limit (dominant vs suppressed detection) and checks whether
detected-top height recovers field height better than the crude cylinder did. scipy-only (no skimage).
"""
import numpy as np, pandas as pd, laspy
from scipy.ndimage import maximum_filter, gaussian_filter

from greenvista import config
DATA = config.DATA
RES = 0.5          # CHM cell size (m)
HMIN = 5.0         # ignore maxima below this height (m)
MATCH_R = 1.5      # match a detection to a cubed tree within this radius (m)

def load_linked():
    g = pd.read_csv(DATA/"2-shapes/Cubagem_T001/arvores_cubadas.csv")
    g = g[g.Name.astype(str).str.fullmatch(r"\d+")].copy(); g["arv"] = g.Name.astype(int)
    vol = pd.read_csv(DATA/"5-dados_campo/volumes.csv")
    return g.merge(vol[vol.talhao==1][["arv","V","H_m"]], on="arv")[["arv","Easting","Northing","V","H_m"]]

def build_chm(x, y, z):
    x0, y0 = x.min(), y.min()
    nx = int((x.max()-x0)/RES)+1; ny = int((y.max()-y0)/RES)+1
    ix = ((x-x0)/RES).astype(int); iy = ((y-y0)/RES).astype(int)
    chm = np.zeros((nx, ny), np.float32)
    np.maximum.at(chm, (ix, iy), z.astype(np.float32))
    return chm, x0, y0

def detect(chm, win):
    sm = gaussian_filter(chm, 0.6)
    mx = maximum_filter(sm, size=win)
    peaks = (sm == mx) & (chm > HMIN)
    ix, iy = np.where(peaks)
    return ix, iy, chm[ix, iy]

def main():
    linked = load_linked()
    las = laspy.read(config.TILE_LAZ)
    cls = np.asarray(las.classification); keep = (cls!=2)&(cls!=18)
    x, y, z = np.asarray(las.x)[keep], np.asarray(las.y)[keep], np.asarray(las.z)[keep]
    chm, x0, y0 = build_chm(x, y, z)
    planted_ha = 0.74; expected = int(planted_ha*1500)
    dom = linked.H_m >= linked.H_m.max()-5
    print(f"Cloud 001 CHM {chm.shape} @ {RES} m | linked trees N={len(linked)} "
          f"(dominant={int(dom.sum())}, suppressed={int((~dom).sum())}) | expected ~{expected} stems")
    for win in (3, 5, 7):  # 1.5, 2.5, 3.5 m windows
        ix, iy, hh = detect(chm, win)
        det_x = x0 + ix*RES + RES/2; det_y = y0 + iy*RES + RES/2
        n_det = len(ix)
        # match each cubed tree to nearest detection
        det_h_match, hit = [], []
        for _, t in linked.iterrows():
            d2 = (det_x-t.Easting)**2 + (det_y-t.Northing)**2
            j = int(np.argmin(d2)); near = np.sqrt(d2[j]) <= MATCH_R
            hit.append(near); det_h_match.append(hh[j] if near else np.nan)
        linked["_hit"] = hit; linked["_dh"] = det_h_match
        dr = np.mean(hit)*100
        dr_dom = np.mean([h for h,d in zip(hit,dom) if d])*100
        dr_sup = np.mean([h for h,d in zip(hit,dom) if not d])*100
        matched = linked[linked._hit]
        if len(matched) > 2:
            r = np.corrcoef(matched._dh, matched.H_m)[0,1]; bias = (matched._dh-matched.H_m).mean()
        else:
            r, bias = np.nan, np.nan
        print(f"\n[win={win} ({win*RES:.1f} m)] detections={n_det} (~{n_det/planted_ha:.0f}/ha vs ~1500 expected)")
        print(f"   detection rate: overall {dr:.0f}%  | dominant {dr_dom:.0f}%  | SUPPRESSED {dr_sup:.0f}%")
        print(f"   matched-top height vs field H: r={r:+.2f}  bias={bias:+.2f} m  (n={len(matched)})")
    print("\nRead: if SUPPRESSED detection rate is low, CHM-based ITD structurally cannot see overtopped "
          "trees → need 3D (FF3D) or accept that ~16/23 labels are hard. Compare to FF3D (S2).")

if __name__ == "__main__":
    main()

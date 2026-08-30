#!/usr/bin/env python3
"""Per-tree DBH from the oblique band of the existing flight.

`exp_obliquo_do_voo_existente.py` split the cloud by scan-angle band, and the radial profile of the
20 to 30 degree band rises from 152 points/m2 at r = 1 cm to a peak of 277 at r = 11 cm, falling
afterwards. That is a ring, and the real stem sits at 9.5 cm of radius. The nadir bands (0 to 10 and
10 to 20 degrees) fall monotonically, the solid blur documented in
`2026-08-28-fuste-visivel-no-drone.md`. The stem surface does appear in the drone data, but diluted:
the 20 to 30 degree band has 79 thousand points in the stem section against 116 thousand for the
other bands combined, and mixing all four erases the ring.

The 30 to 90 degree band does not show the ring: there the range is longer, the incidence is grazing
and the position error grows. There is a useful window, measured at 20 to 30 degrees, and not a "the
more oblique the better".

The 1.0 to 6.0 m section blurs the ring: the taper measured on the TLS takes the radius from 9.5 to
about 8.8 cm over that interval, which adds some 0.7 cm of width to the aggregated ring. That is the
price of having five times more points.

Validation is against the TLS DBH, tree by tree, at the same positions of the stem map; no GPS
matching is involved.

Usage: PYTHONPATH=. python scripts/exp_dap_do_obliquo.py
Output: manual_match/dap_do_obliquo.csv
"""
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from greenvista import config  # noqa: E402

warnings.filterwarnings("ignore")
MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
DRONE = config.REPO / "work/lazall/SaoManuelTotal_001.laz"
SAIDA = config.OUT_DIR / "dap_do_obliquo.csv"

FAIXAS_ANG = [(0, 10), (10, 20), (15, 25), (20, 30), (18, 32), (20, 36), (25, 36), (0, 36)]
TRECHOS = [(1.0, 6.0), (1.0, 4.0), (1.0, 8.0), (2.0, 10.0)]
R_BUSCA = 0.45
# A 2.5 cm bin and not a 1 cm one: with tens of points per stem, a small-radius bin has a tiny area
# and two points near the axis generate a huge density, and the estimator sticks to the floor of the
# search (median DBH of 7 to 9 cm against 19 from TLS). Same failure mode described in the header of
# `exp_dap_tls_ransac.py`, aggravated here by the lower point count per stem.
BIN = 0.025
R_LIM = (0.050, 0.200)     # DBH from 10 to 40 cm, which is the physically possible range here
PTS_MIN = 25
CONTA_MIN = 6              # the winning bin must have real points inside
# The centre is not refined: refining by maximum count pulls the centre to where there are more
# points and, with few points, manufactures a dense core that destroys the ring. The aggregated
# profile that showed the ring at 11 cm uses the map position with no refinement.
REFINA_CENTRO = False
PASSO_C = 0.02
JANELA_C = 0.20


def raio_por_modo(pu, pv, cu=0.0, cv=0.0, voltas=1):
    """Radius from the peak of the radial density per unit area, kernel-estimated on a fine grid.

    A continuous kernel and not a histogram: with a 2.5 cm bin the radius takes only about six
    values, and the estimate returns 22.5 cm for almost every stem. That is not bias, it is
    resolution, because quantising in 2.5 cm on a DBH with a 4.9 cm standard deviation erases the
    ranking between trees, which is what one wants to measure. A 1 cm bin recovers the ranking but
    sticks to the floor of the search, since a small-radius bin has a tiny area and two points near
    the axis generate a huge density. The kernel smooths like a wide bin and returns a continuous
    value like a fine one.

    The normalisation is by 2*pi*r, the length of the ring, that is, density per unit area. Without
    it the outer ring wins merely by being larger, the failure mode described in the header of
    `exp_dap_tls_ransac.py`.

    The centre is not refined by default: refining by maximum count pulls the centre to where there
    are more points and, with few points, manufactures a dense core that destroys the ring.
    """
    d = np.hypot(pu - cu, pv - cv)
    grade = np.arange(R_LIM[0], R_LIM[1], 0.002)
    h = 0.02
    peso = np.exp(-0.5 * ((d[None, :] - grade[:, None]) / h) ** 2).sum(1)
    dens = peso / (2 * np.pi * grade)
    perto = (np.abs(d[None, :] - grade[:, None]) < h).sum(1)
    dens = np.where(perto >= CONTA_MIN, dens, 0.0)
    if not dens.any():
        return np.nan, np.nan, 0.0
    r = float(grade[int(dens.argmax())])
    miolo = dens[grade < 0.075].mean()
    casca = dens[np.abs(grade - r) < 0.02].mean()
    return r, casca / miolo if miolo else np.nan, float(np.hypot(cu, cv))


def centro_por_anel(px, py, r):
    """Centre that puts the most points in the band of radius r, with the radius pinned (it does not
    bias the radius)."""
    g = np.arange(-JANELA_C, JANELA_C + 1e-9, PASSO_C)
    du, dv = np.meshgrid(g, g, indexing="ij")
    du, dv = du.ravel(), dv.ravel()
    d = np.hypot(px[None, :] - du[:, None], py[None, :] - dv[:, None])
    n = (np.abs(d - r) < 0.03).sum(1)
    i = int(n.argmax())
    return float(du[i]), float(dv[i])


def avalia(x, y, z, ang, ref, dap_ref, a0, a1, h0, h1):
    m = (z >= h0) & (z <= h1) & (ang >= a0) & (ang < a1)
    if m.sum() < 5000:
        return None
    X, Y = x[m], y[m]
    T = cKDTree(np.column_stack([X, Y]))
    est, contr, npts = [], [], []
    for cx, cy in ref:
        idx = np.asarray(T.query_ball_point([cx, cy], R_BUSCA), dtype=int)
        if len(idx) < PTS_MIN:
            est.append(np.nan); contr.append(np.nan); npts.append(len(idx)); continue
        px, py = X[idx] - cx, Y[idx] - cy
        if REFINA_CENTRO:
            du, dv = centro_por_anel(px, py, 0.095)
            px, py = px - du, py - dv
        r, c, _ = raio_por_modo(px, py)
        est.append(r); contr.append(c); npts.append(len(idx))
    est = np.asarray(est, float)
    ok = np.isfinite(est) & np.isfinite(dap_ref)
    if ok.sum() < 60:
        return None
    r_p = float(np.corrcoef(200 * est[ok], dap_ref[ok])[0, 1])
    r_s = float(spearmanr(est[ok], dap_ref[ok]).statistic)
    return dict(a0=a0, a1=a1, h0=h0, h1=h1, n_pts=int(m.sum()),
                pts_por_fuste=float(m.sum() / len(ref)), n_medidos=int(ok.sum()),
                dap_mediano_cm=float(200 * np.nanmedian(est)),
                contraste=float(np.nanmedian(contr)), pearson=r_p, spearman=r_s)


def main():
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    ran = pd.read_csv(config.OUT_DIR / "dap_tls_ransac_talhao001.csv")
    dap_ref = np.where(ran.motivo.values == "ok", ran.dap_cm.values, np.nan)

    las = laspy.read(str(DRONE))
    cls = np.asarray(las.classification)
    veg = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    x, y, z = np.asarray(las.x)[veg], np.asarray(las.y)[veg], np.asarray(las.z)[veg]
    ang = np.abs(np.asarray(las.scan_angle_rank).astype(float))[veg]
    print(f"{len(x):,} vegetation points\n")
    print(f"reference DBH (TLS) on {np.isfinite(dap_ref).sum()} stems, "
          f"median {np.nanmedian(dap_ref):.1f} cm\n")

    print(f"{'angle':>10} {'height':>10} {'pts/stem':>10} {'measured':>8} "
          f"{'med DBH':>8} {'contrast':>10} {'pearson':>8} {'spearman':>9}")
    linhas = []
    for h0, h1 in TRECHOS:
        for a0, a1 in FAIXAS_ANG:
            r = avalia(x, y, z, ang, ref, dap_ref, a0, a1, h0, h1)
            if r is None:
                continue
            linhas.append(r)
            print(f"{a0:4.0f}-{a1:3.0f}  {h0:4.1f}-{h1:4.1f} {r['pts_por_fuste']:10.1f} "
                  f"{r['n_medidos']:8d} {r['dap_mediano_cm']:8.1f} {r['contraste']:10.2f} "
                  f"{r['pearson']:+8.3f} {r['spearman']:+9.3f}", flush=True)

    d = pd.DataFrame(linhas)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    d.to_csv(SAIDA, index=False)
    if len(d):
        b = d.loc[d.spearman.idxmax()]
        print(f"\nBEST: angle {b.a0:.0f} to {b.a1:.0f}, height {b.h0:.1f} to {b.h1:.1f} m")
        print(f"   spearman {b.spearman:+.3f}, pearson {b.pearson:+.3f}, "
              f"{b.n_medidos:.0f} stems measured, median DBH {b.dap_mediano_cm:.1f} cm "
              f"against {np.nanmedian(dap_ref):.1f} from TLS")
        print("   correlation is what matters here; scale can be corrected, ranking cannot.")
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

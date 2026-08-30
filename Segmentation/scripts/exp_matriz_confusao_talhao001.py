#!/usr/bin/env python3
"""Confusion matrix of stand 001 against the TLS map, and the adjudication of false positives.

The evaluable region comes from a physical measure: how much the terrestrial laser saw at
each point of the TLS/SLAM cloud. The convex hull of the map stems themselves would not do
as a boundary, because it is defined by the reference, so every detection outside it
becomes "not evaluable" by construction.

Detection has no true negative. There is no enumerable set of "places without a tree" to
count, so accuracy (TP+TN)/(total) is not definable. What does close is TP / FN / FP with
recall, precision and F1, and that is the form in which the matrix is printed.

Adjudication of unmatched detections (detector error or a stem missing from the map): the
criterion is the breast-height slab of the TLS cloud, 0.8 to 2.5 m above ground, where a
real stem leaves a dense column of points. Two controls calibrate the threshold: the 892
map stems (95% respond) and points drawn far from any stem. The reading is only valid where
the TLS covered well, because in a poorly scanned strip the absence of a response is
occlusion and not the absence of a tree. Hence the coverage subset.

Run: PYTHONPATH=. python scripts/exp_matriz_confusao_talhao001.py
Input: config.TLS_LAS, the terrestrial point cloud kept outside git
Output: manual_match/matriz_confusao_talhao001.csv
        manual_match/deteccoes_julgadas_talhao001.csv
"""
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenvista import config  # noqa: E402

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
DETEC = config.REPO / "data/detections/sat_w2w_arvores.csv"
TLS = config.TLS_LAS
SAIDA_M = config.OUT_DIR / "matriz_confusao_talhao001.csv"
SAIDA_D = config.OUT_DIR / "deteccoes_julgadas_talhao001.csv"

LIMIARES = (0.5, 1.0, 1.5, 2.0)
G_SOLO, G_PEITO = 2.0, 0.25          # grids of the terrain model and of the breast-height slab
H_PEITO = (0.8, 2.5)                 # height window of the slab, in metres above ground
R_FUSTE, R_COBERTURA = 1.0, 5.0      # reading radii on the slab
P_COBERTURA, P_FUSTE = 10, 5         # percentiles of the known stems that become thresholds


def grades_tls(x0, y0, nx, ny):
    """Coarse terrain model and breast-height slab, both read in chunks to fit in RAM."""
    zmin = np.full((nx, ny), 1e9, np.float32)
    with laspy.open(TLS) as fh:
        for p in fh.chunk_iterator(4_000_000):
            x, y, z = np.asarray(p.x), np.asarray(p.y), np.asarray(p.z, dtype=np.float32)
            ix = ((x - x0) / G_SOLO).astype(np.int32)
            iy = ((y - y0) / G_SOLO).astype(np.int32)
            m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            np.minimum.at(zmin, (ix[m], iy[m]), z[m])
    zmin[zmin > 1e8] = np.nan
    vazio = ~np.isfinite(zmin)
    if vazio.any():                  # unfilled, the slab vanishes at the edges of the flight
        _, (a, b) = distance_transform_edt(vazio, return_indices=True)
        zmin[vazio] = zmin[a[vazio], b[vazio]]
    zsolo = uniform_filter(zmin, 5)

    mx, my = int(nx * G_SOLO / G_PEITO), int(ny * G_SOLO / G_PEITO)
    peito = np.zeros((mx, my), np.int32)
    with laspy.open(TLS) as fh:
        for p in fh.chunk_iterator(4_000_000):
            x, y, z = np.asarray(p.x), np.asarray(p.y), np.asarray(p.z, dtype=np.float32)
            jx = np.clip(((x - x0) / G_SOLO).astype(np.int32), 0, nx - 1)
            jy = np.clip(((y - y0) / G_SOLO).astype(np.int32), 0, ny - 1)
            h = z - zsolo[jx, jy]
            k = (h > H_PEITO[0]) & (h < H_PEITO[1])
            ix = ((x[k] - x0) / G_PEITO).astype(np.int32)
            iy = ((y[k] - y0) / G_PEITO).astype(np.int32)
            m = (ix >= 0) & (ix < mx) & (iy >= 0) & (iy < my)
            np.add.at(peito, (ix[m], iy[m]), 1)
    return peito


def soma_disco(peito, x0, y0, P, raio):
    r = int(np.ceil(raio / G_PEITO))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disco = (xx ** 2 + yy ** 2) * G_PEITO ** 2 <= raio ** 2
    ix = ((P[:, 0] - x0) / G_PEITO).astype(int)
    iy = ((P[:, 1] - y0) / G_PEITO).astype(int)
    out = np.zeros(len(P), np.int64)
    for k in range(len(P)):
        a, b, c, e = ix[k] - r, ix[k] + r + 1, iy[k] - r, iy[k] + r + 1
        if a < 0 or c < 0 or b > peito.shape[0] or e > peito.shape[1]:
            continue
        out[k] = int((peito[a:b, c:e] * disco).sum())
    return out


def matriz(ref, pred, lim):
    D = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    li, ci = linear_sum_assignment(np.where(D <= lim, D, 1e6))
    ok = D[li, ci] <= lim
    vp = int(ok.sum())
    r, p = vp / len(ref), vp / len(pred)
    return {"VP": vp, "FN": len(ref) - vp, "FP": len(pred) - vp,
            "revocacao_pct": round(100 * r, 1), "precisao_pct": round(100 * p, 1),
            "f1_pct": round(200 * r * p / (r + p), 1)}, li[ok], ci[ok], D[li, ci][ok]


def main():
    if not TLS.exists():
        sys.exit(f"TLS point cloud not found at {TLS} (set GREENVISTA_TLS_LAS)")
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    d = pd.read_csv(DETEC)
    d1 = d[d.talhao == 1].reset_index(drop=True)
    pred = np.column_stack([d1.base_x.values, d1.base_y.values])

    x0 = np.floor(min(ref[:, 0].min(), pred[:, 0].min()) / 10) * 10 - 60
    y0 = np.floor(min(ref[:, 1].min(), pred[:, 1].min()) / 10) * 10 - 60
    nx = int((max(ref[:, 0].max(), pred[:, 0].max()) - x0) / G_SOLO) + 60
    ny = int((max(ref[:, 1].max(), pred[:, 1].max()) - y0) / G_SOLO) + 60
    print(f"{len(ref)} stems, {len(pred)} detections; reading {TLS.name}")
    peito = grades_tls(x0, y0, nx, ny)
    print(f"breast-height slab {H_PEITO[0]}–{H_PEITO[1]} m: {peito.sum():,} points")

    cob_ref = soma_disco(peito, x0, y0, ref, R_COBERTURA)
    cob_pred = soma_disco(peito, x0, y0, pred, R_COBERTURA)
    fus_ref = soma_disco(peito, x0, y0, ref, R_FUSTE)
    fus_pred = soma_disco(peito, x0, y0, pred, R_FUSTE)
    lim_cob = np.percentile(cob_ref, P_COBERTURA)
    lim_fus = np.percentile(fus_ref, P_FUSTE)
    bem_r, bem_p = cob_ref >= lim_cob, cob_pred >= lim_cob
    print(f"coverage threshold (p{P_COBERTURA} of the stems, radius {R_COBERTURA} m) = {lim_cob:,.0f} pts")
    print(f"stem threshold (p{P_FUSTE} of the stems, radius {R_FUSTE} m) = {lim_fus:,.0f} pts")
    print(f"well covered: {bem_r.sum()}/{len(ref)} stems, {bem_p.sum()}/{len(pred)} detections\n")

    linhas = []
    for lim in LIMIARES:
        for rot, rr, pp in (("tudo", ref, pred), ("TLS bem coberto", ref[bem_r], pred[bem_p])):
            m, *_ = matriz(rr, pp, lim)
            linhas.append({"limiar_m": lim, "recorte": rot, **m})
            print(f"{lim:.1f} m  {rot:16s} TP {m['VP']:5d} FN {m['FN']:4d} FP {m['FP']:4d}  "
                  f"R {m['revocacao_pct']:5.1f}%  P {m['precisao_pct']:5.1f}%  F1 {m['f1_pct']:5.1f}%")

    m2, li2, ci2, dist = matriz(ref, pred, 2.0)
    casou = np.zeros(len(pred), bool)
    casou[ci2] = True
    sem = ~casou
    tem_fuste = fus_pred >= lim_fus
    # Check the registration before blaming the detector: a datum offset between the drone
    # cloud and the terrestrial survey would break every tight match for a reason that is not
    # a detection error. The median of the pairwise vector measures that.
    desl = np.median(pred[ci2] - ref[li2], axis=0)
    print(f"\npair distance at 2 m: median {np.median(dist):.2f} m, "
          f"{100 * (dist < 1).mean():.0f}% below 1 m")
    print(f"systematic offset between the two surveys: "
          f"dx {desl[0]:+.3f} m, dy {desl[1]:+.3f} m, norm {np.hypot(*desl):.3f} m")
    print(f"\nof the {sem.sum()} unmatched detections:")
    print(f"  {int((sem & ~bem_p).sum())} in a strip the TLS barely scanned (no valid adjudication there)")
    print(f"  {int((sem & bem_p).sum())} in a well-scanned strip, of which "
          f"{int((sem & bem_p & tem_fuste).sum())} have a stem in the TLS cloud and "
          f"{int((sem & bem_p & ~tem_fuste).sum())} do not")
    print(f"  control, MATCHED detections that respond as a stem: "
          f"{100 * tem_fuste[casou].mean():.0f}%")

    # An operating curve, not a recommended filter. Height and instance size separate part
    # of the error, but the operating point depends on the use: counting prefers no filter,
    # a clean stem list prefers a filtered one.
    print("\nprecision-recall trade-off per filter (well-covered region, 2 m)")
    for zc, nc in ((0, 0), (5, 0), (10, 1000), (15, 1500), (20, 2000)):
        k = bem_p & (d1.z_max.values >= zc) & (d1.n_pts.values >= nc)
        m, *_ = matriz(ref[bem_r], pred[k], 2.0)
        print(f"  height>={zc:2d} m and points>={nc:5d} (n={int(k.sum()):4d})  "
              f"R {m['revocacao_pct']:5.1f}%  P {m['precisao_pct']:5.1f}%  F1 {m['f1_pct']:5.1f}%")
        linhas.append({"limiar_m": 2.0, "recorte": f"bem coberto, altura>={zc}m, pts>={nc}", **m})

    pd.DataFrame(linhas).to_csv(SAIDA_M, index=False)
    d1.assign(casou=casou, cobertura_tls=cob_pred, resposta_fuste=fus_pred,
              tls_bem_coberto=bem_p, tem_fuste_no_tls=tem_fuste).to_csv(SAIDA_D, index=False)
    print(f"\n{SAIDA_M}\n{SAIDA_D}")


if __name__ == "__main__":
    main()

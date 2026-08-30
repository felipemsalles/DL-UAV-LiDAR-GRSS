#!/usr/bin/env python3
"""How much of the R2 lost per tree is label noise rather than absence of signal?

`exp_por_arvore_com_809_rotulos.py` finds an R2 near zero for per-tree DBH from the drone
crown, even with oracle segmentation. That has two incompatible readings which that
experiment alone does not separate:
  (a) the crown seen from above does not carry stem diameter;
  (b) the DBH label taken from the TLS is noisy, and noise in the target depresses R2 even
      if the predictor were perfect.

Noise in the target imposes a hard ceiling: R2_max = 1 - var(noise) / var(target). If the TLS
DBH errs by 1.5 cm against a standard deviation of 4.86, the ceiling is 0.90 and reading (a)
stands; if it errs by 4 cm, the ceiling is 0.32 and reading (b) explains half the result.

What can be measured is repeatability, not accuracy: there is no per-tree field DBH paired to
these stems (the GPS of the scaled trees is metre-level, see `exp_gps_ruler.py`). The point
cloud is split into two independent halves, the same stem is measured in both, and the
disagreement is observed. This captures sampling and fitting noise; it does not capture
systematic bias, which is common to both halves, so the number that comes out is a floor on
the noise.

A half has half the points, so it is noisier than the full estimate. With noise independent
between halves, sd(difference) = sqrt(2) * sd(half) and sd(full) = sd(half) / sqrt(2), hence
sd(full) = sd(difference) / 2.

Run: PYTHONPATH=. python scripts/exp_ruido_do_rotulo_tls.py
Output: manual_match/ruido_do_rotulo_tls.csv
"""
import importlib.util
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from greenvista import config  # noqa: E402

_s = importlib.util.spec_from_file_location("ransac", R / "scripts/exp_dap_tls_ransac.py")
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
SAIDA = config.OUT_DIR / "ruido_do_rotulo_tls.csv"
SEED = 20260828


def mede(X, Y, H, ref):
    """The same estimator as `exp_dap_tls_ransac`, applied to a subset of points."""
    T = cKDTree(np.column_stack([X, Y]))
    out = np.full(len(ref), np.nan)
    for j, (cx, cy) in enumerate(ref):
        idx = np.asarray(T.query_ball_point([cx, cy], _m.R_BUSCA), dtype=int)
        if len(idx) < _m.INLIER_MIN:
            continue
        P = np.column_stack([X[idx] - cx, Y[idx] - cy, H[idx] - np.mean(_m.H_TRECHO)])
        d = _m.eixo(P)
        u, v = _m.base_ortogonal(d)
        pu, pv = P @ u, P @ v
        r, cu, cv, ins = _m.raio_por_modo(pu, pv)
        if not np.isfinite(r):
            continue
        bordas = np.linspace(P[:, 2].min(), P[:, 2].max(), _m.N_FATIAS + 1)
        raios = []
        for i in range(_m.N_FATIAS):
            k = (P[:, 2] >= bordas[i]) & (P[:, 2] < bordas[i + 1])
            if k.sum() < _m.PTS_FATIA:
                continue
            ri = _m.raio_por_modo(pu[k], pv[k], cu, cv, minimo=12)[0]
            if np.isfinite(ri):
                raios.append(ri)
        if len(raios) < _m.FATIAS_MIN:
            continue
        rr = np.asarray(raios)
        disp = (abs(rr[0] - rr[1]) / 2 if len(rr) == 2
                else float(np.median(np.abs(rr - np.median(rr)))))
        if disp <= _m.DISCORDA_MAX and ins.sum() >= _m.INLIER_MIN:
            out[j] = 200 * float(np.median(raios))
    return out


def main():
    rng = np.random.default_rng(SEED)
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])

    print(f"reading {config.TLS_LAS.name}", flush=True)
    xs, ys, zs = [], [], []
    with laspy.open(config.TLS_LAS) as fh:
        for p in fh.chunk_iterator(6_000_000):
            xs.append(np.asarray(p.x))
            ys.append(np.asarray(p.y))
            zs.append(np.asarray(p.z, dtype=np.float32))
    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    zsolo, x0, y0 = _m.terreno(x, y, z)
    h = z - zsolo[np.clip(((x - x0) / _m.G_SOLO).astype(int), 0, zsolo.shape[0] - 1),
                  np.clip(((y - y0) / _m.G_SOLO).astype(int), 0, zsolo.shape[1] - 1)]
    k = (h > _m.H_TRECHO[0]) & (h < _m.H_TRECHO[1])
    X, Y, H = x[k], y[k], h[k]
    print(f"{len(X):,} points in the breast-height slab", flush=True)

    metade = rng.random(len(X)) < 0.5
    print("measuring half A", flush=True)
    a = mede(X[metade], Y[metade], H[metade], ref)
    print("measuring half B", flush=True)
    b = mede(X[~metade], Y[~metade], H[~metade], ref)

    ok = np.isfinite(a) & np.isfinite(b)
    dif = a[ok] - b[ok]
    dp_dif = float(np.std(dif, ddof=1))
    dp_metade = dp_dif / np.sqrt(2)
    dp_cheia = dp_dif / 2.0

    ran = pd.read_csv(config.OUT_DIR / "dap_tls_ransac_talhao001.csv")
    cheio = np.where(ran.motivo.values == "ok", ran.dap_cm.values, np.nan)
    var_alvo = float(np.nanvar(cheio, ddof=1))
    teto = 1 - dp_cheia ** 2 / var_alvo

    print(f"\n{ok.sum()} stems measured in both halves")
    print(f"   difference between halves: mean {dif.mean():+.3f} cm, sd {dp_dif:.3f} cm, "
          f"median |error| {np.median(np.abs(dif)):.3f}")
    print(f"   noise of ONE half         {dp_metade:.3f} cm")
    print(f"   noise of the FULL estimate {dp_cheia:.3f} cm  (a floor, only what is "
          f"independent between the halves)")
    print(f"   between-tree DBH standard deviation {np.sqrt(var_alvo):.3f} cm")
    print(f"\nR2 CEILING imposed by label noise: {teto:.3f}")
    print("   and the per-tree R2 measured from the crown was 0.03 to 0.10.")
    if teto > 0.7:
        print("   => the label does NOT explain the result. The signal simply is not in the crown.")
    else:
        print("   the label explains a relevant share of the result and the conclusion needs "
              "to be rewritten.")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"fuste": np.arange(len(ref))[ok], "dap_A": a[ok], "dap_B": b[ok]}
                 ).to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

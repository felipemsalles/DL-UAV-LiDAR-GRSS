#!/usr/bin/env python3
"""The oblique point cloud that already exists inside the flight, and what it says about the stem.

`exp_densidade_e_fuste.py` measured that limiting the scan angle to 20 degrees keeps 59 % of the
cloud but only 26 % of the points at breast height: the little that reaches the trunk comes from
oblique beams. A dedicated oblique flight is out of the question, but the available flight
already scans out to 36 degrees and 33 % of the points lie beyond 30, so the oblique cloud exists
and only needs to be separated out.

This is not equivalent to a dedicated oblique flight. A scan-edge point comes from farther away,
with grazing incidence on the canopy and larger positional error, and its sampling is biased
towards the side of the swath. What is tested here is whether view geometry, on its own, already
changes what can be seen of the stem.

The breast-height slab has few points and cannot support a per-tree estimate on its own. That is
why the slab used here runs from 1.0 to 6.0 m, where the stem is still nearly cylindrical and the
taper measured on the TLS is a few centimetres, which yields about five times more points.

What comes out of this:
 1. the radial profile around the stem, from oblique returns only, against pure nadir;
 2. new per-tree features describing the stem seen from the side, not the crown;
 3. the model ladder with those features added to the old ones.

Run: PYTHONPATH=. python scripts/exp_obliquo_do_voo_existente.py
Output: manual_match/obliquo_perfil.csv and manual_match/obliquo_modelo.csv
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from greenvista import config  # noqa: E402


def _mod(nome, caminho):
    s = importlib.util.spec_from_file_location(nome, caminho)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_arv = _mod("porarv", R / "scripts/exp_por_arvore_com_809_rotulos.py")
warnings.filterwarnings("ignore")

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
DRONE = config.REPO / "work/lazall/SaoManuelTotal_001.laz"
SAIDA_PERFIL = config.OUT_DIR / "obliquo_perfil.csv"
SAIDA_MOD = config.OUT_DIR / "obliquo_modelo.csv"

TRECHO = (1.0, 6.0)        # nearly cylindrical stem slab
R_BUSCA = 0.60
BIN = 0.02
ANGULOS = [(0, 10), (10, 20), (20, 30), (30, 90)]
SEED = 20260828


def perfil(X, Y, ref, r_max=R_BUSCA):
    """Radial density per ring area around the stem position, pooled over stems."""
    if len(X) < 100:
        return None
    T = cKDTree(np.column_stack([X, Y]))
    bins = np.arange(0.0, r_max + BIN, BIN)
    area = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    acc = np.zeros(len(bins) - 1)
    n = 0
    for cx, cy in ref:
        idx = T.query_ball_point([cx, cy], r_max)
        n += 1
        if idx:
            d = np.hypot(X[np.asarray(idx, int)] - cx, Y[np.asarray(idx, int)] - cy)
            acc += np.histogram(d, bins=bins)[0]
    return 0.5 * (bins[1:] + bins[:-1]), acc / (area * n), acc


def feicoes_de_fuste(X, Y, Z, ref, rotulo):
    """Per-tree features describing the stem seen from the side, not the crown."""
    T = cKDTree(np.column_stack([X, Y]))
    linhas = []
    for j, (cx, cy) in enumerate(ref):
        idx = np.asarray(T.query_ball_point([cx, cy], R_BUSCA), dtype=int)
        f = {f"{rotulo}_n": float(len(idx))}
        if len(idx) >= 5:
            d = np.hypot(X[idx] - cx, Y[idx] - cy)
            z = Z[idx]
            f[f"{rotulo}_dmed"] = float(np.median(d))
            f[f"{rotulo}_dp10"] = float(np.quantile(d, 0.10))
            f[f"{rotulo}_dp25"] = float(np.quantile(d, 0.25))
            f[f"{rotulo}_dsd"] = float(np.std(d))
            f[f"{rotulo}_perto15"] = float(np.mean(d < 0.15))
            f[f"{rotulo}_perto25"] = float(np.mean(d < 0.25))
            f[f"{rotulo}_zmed"] = float(np.median(z))
            # Mode of the area-normalised radial density, the same estimator validated on the
            # TLS. With few points it is noisy; what matters here is the correlation with the
            # true DBH, not the accuracy of the absolute value.
            bins = np.arange(0.03, 0.40, 0.02)
            if len(d) >= 8:
                dens = np.histogram(d, bins=bins)[0] / (np.pi * (bins[1:] ** 2 - bins[:-1] ** 2))
                if dens.any():
                    f[f"{rotulo}_rmoda"] = float(0.5 * (bins[int(dens.argmax())]
                                                        + bins[int(dens.argmax()) + 1]))
        linhas.append(f)
    return pd.DataFrame(linhas)


def main():
    rng = np.random.default_rng(SEED)
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])

    las = laspy.read(str(DRONE))
    cls = np.asarray(las.classification)
    ang = np.abs(np.asarray(las.scan_angle_rank).astype(float))
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    veg = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    print(f"{veg.sum():,} vegetation points, scan angle up to {ang.max():.0f} degrees")

    tronco = veg & (z >= TRECHO[0]) & (z <= TRECHO[1])
    print(f"{tronco.sum():,} points in the stem slab from {TRECHO[0]} to {TRECHO[1]} m\n")

    # ---- 1. the radial profile by scan-angle band ---------------------------------------
    print("[1] radial profile around the stem, by scan-angle band")
    print(f"   {'band':12s} {'n points':>10} {'pts/stem':>10} {'peak (cm)':>10} "
          f"{'<15cm':>7} {'<25cm':>7}")
    linhas_perfil = []
    for a0, a1 in ANGULOS:
        m = tronco & (ang >= a0) & (ang < a1)
        r = perfil(x[m], y[m], ref)
        if r is None:
            continue
        cen, dens, acc = r
        pico = 100 * cen[int(np.argmax(dens))]
        p15 = acc[cen < 0.15].sum() / max(acc.sum(), 1)
        p25 = acc[cen < 0.25].sum() / max(acc.sum(), 1)
        print(f"   {a0:3d} to {a1:3d}  {m.sum():10,} {m.sum()/len(ref):10.1f} "
              f"{pico:10.1f} {100*p15:6.1f}% {100*p25:6.1f}%")
        for c, d in zip(cen, dens):
            linhas_perfil.append(dict(faixa=f"{a0}-{a1}", r_m=c, densidade=d))
        linhas_perfil.append(dict(faixa=f"{a0}-{a1}", r_m=np.nan, densidade=np.nan,
                                  n_pts=int(m.sum()), pico_cm=pico, frac15=p15, frac25=p25))
    # Reference null: with points spread uniformly out to 60 cm, the fraction within 15 cm is
    # (15/60)^2 = 6.25 % and within 25 cm it is 17.4 %. Without that floor, 20 % would be read
    # as signal when it is only area.
    print(f"   uniform null: <15cm = {100*(0.15/R_BUSCA)**2:.1f} %, "
          f"<25cm = {100*(0.25/R_BUSCA)**2:.1f} %")

    # ---- 2. per-tree stem features ------------------------------------------------------
    print("\n[2] per-tree stem features, oblique against nadir")
    F_nadir = feicoes_de_fuste(x[tronco & (ang < 15)], y[tronco & (ang < 15)],
                               z[tronco & (ang < 15)], ref, "nad")
    F_obl = feicoes_de_fuste(x[tronco & (ang >= 20)], y[tronco & (ang >= 20)],
                             z[tronco & (ang >= 20)], ref, "obl")
    NOVAS = pd.concat([F_nadir, F_obl], axis=1)

    base = pd.read_csv(config.OUT_DIR / "por_arvore_pred_voronoi.csv", index_col=0)
    NOVAS.index = np.arange(len(ref))
    D = base.join(NOVAS, how="inner")
    print(f"   {len(D)} trees with both a label and features")
    print("   raw correlation of each new feature with the TLS DBH:")
    novas_cols = [c for c in NOVAS.columns if D[c].notna().sum() > 200]
    for c in sorted(novas_cols):
        v = D[[c, "dap_cm"]].dropna()
        r = float(np.corrcoef(v[c], v.dap_cm)[0, 1]) if len(v) > 30 else np.nan
        print(f"      {c:14s} n={len(v):4d}  r = {r:+.3f}")

    # ---- 3. the ladder, with and without the new features -------------------------------
    print("\n[3] the model ladder, old features against old + stem features")
    D2 = D.copy()
    for c in novas_cols:
        D2[c] = D2[c].fillna(D2[c].median())
    antigas = _arv.VERT + _arv.GEOM + _arv.COMP
    _arv.N_EMB = 8
    _arv.ESCADA = [
        ("nulo, prediz a media", None, "media"),
        ("so feicoes de copa (antigas)", antigas, "rf"),
        ("so feicoes de FUSTE (novas)", novas_cols, "rf"),
        ("copa + fuste", antigas + novas_cols, "rf"),
        ("copa + fuste, ridge", antigas + novas_cols, "ridge"),
    ]
    saida = _arv.roda("obliquo do voo existente", D2,
                      [("dap_cm", "DBH (cm)"), ("vol_m3", "volume (m3)")], rng)

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas_perfil).to_csv(SAIDA_PERFIL, index=False)
    pd.DataFrame(saida).to_csv(SAIDA_MOD, index=False)
    print(f"\n{SAIDA_PERFIL}\n{SAIDA_MOD}")


if __name__ == "__main__":
    main()

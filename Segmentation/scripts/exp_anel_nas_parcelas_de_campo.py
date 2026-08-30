#!/usr/bin/env python3
"""The oblique ring on the 13 field plots, which span six stands and six years of age.

`exp_anel_agregado.py` shows that the peak of the aggregated oblique profile tracks DBH when the
groups are formed by DBH (r from +0.72 to +0.89, p-value from 0.383 to 0.067 as groups are gained).
The operational test inside stand 001 failed for lack of dynamic range: among 400 m2 plots of the
same stand the mean DBH varies by only 4.9 cm.

The 13 field plots cover six stands and ages from 4.8 to 10.6 years, hence a much wider DBH range,
and their DBH is measured with a tape in the field, independent of any estimate of ours.

The positions come from the segmentation, not from a stem map: outside stand 001 there is no TLS, so
the centres are the bases of the trees detected by SAT. Position error blurs the ring and can only
worsen the result, so a positive signal here is conservative and a negative one stays ambiguous.

The control is a permutation on the correlation, not on the amplitude. With 13 points the correlation
already has reasonable power, unlike the 3 to 5 groups of the sibling experiment.

Usage: PYTHONPATH=. python scripts/exp_anel_nas_parcelas_de_campo.py
Output: manual_match/anel_parcelas_campo.csv
"""
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

warnings.filterwarnings("ignore")
DET = config.REPO / "data/detections/sat_w2w_arvores.csv"
LAZ = config.REPO / "work/lazall"
SAIDA = config.OUT_DIR / "anel_parcelas_campo.csv"
FAIXAS = [(20, 30), (18, 32), (20, 36)]
TRECHO = (1.0, 6.0)
R_BUSCA = 0.45
GRADE = np.arange(0.03, 0.30, 0.002)
H_NUCLEO = 0.015
N_PERM = 500
SEED = 20260828


def perfil(T, X, Y, pos):
    d_all, n = [], 0
    for cx, cy in pos:
        idx = T.query_ball_point([cx, cy], R_BUSCA)
        n += 1
        if idx:
            i = np.asarray(idx, int)
            d_all.append(np.hypot(X[i] - cx, Y[i] - cy))
    if not d_all or n == 0:
        return None, 0
    d = np.concatenate(d_all)
    peso = np.exp(-0.5 * ((d[None, :] - GRADE[:, None]) / H_NUCLEO) ** 2).sum(1)
    return peso / (2 * np.pi * GRADE * n), len(d) / n


def pico(dens):
    if dens is None or not np.isfinite(dens).all():
        return np.nan
    i = int(np.argmax(dens))
    return np.nan if i == 0 else float(100 * GRADE[i])


def main():
    rng = np.random.default_rng(SEED)
    inv = pd.read_csv(config.DATA / "5-dados_campo/inventario_est.csv")
    par = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for c in ("talhao", "parcela"):
        par[c] = pd.to_numeric(par[c], errors="coerce")
    par = par.dropna(subset=["talhao", "parcela"]).merge(inv, on=["talhao", "parcela"])
    det = pd.read_csv(DET)
    print(f"{len(par)} field plots, DBH from {par.D_cm.min():.1f} to {par.D_cm.max():.1f} cm "
          f"(range {par.D_cm.max()-par.D_cm.min():.1f} cm), ages from "
          f"{par.idade_anos.min():.1f} to {par.idade_anos.max():.1f} years\n")

    linhas = []
    for a0, a1 in FAIXAS:
        reg = []
        for t in sorted(par.talhao.unique()):
            laz = LAZ / f"SaoManuelTotal_{int(t):03d}.laz"
            if not laz.exists():
                continue
            las = laspy.read(str(laz))
            cls = np.asarray(las.classification)
            veg = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
            x, y, z = np.asarray(las.x)[veg], np.asarray(las.y)[veg], np.asarray(las.z)[veg]
            ang = np.abs(np.asarray(las.scan_angle_rank).astype(float))[veg]
            m = (z >= TRECHO[0]) & (z <= TRECHO[1]) & (ang >= a0) & (ang < a1)
            if m.sum() < 2000:
                continue
            X, Y = x[m], y[m]
            T = cKDTree(np.column_stack([X, Y]))
            dt = det[det.talhao == t]
            for _, p in par[par.talhao == t].iterrows():
                d = np.hypot(dt.base_x - p.geometry.x, dt.base_y - p.geometry.y)
                pos = dt[d <= config.PLOT_RADIUS_M][["base_x", "base_y"]].to_numpy(float)
                if len(pos) < 15:
                    continue
                dens, ppf = perfil(T, X, Y, pos)
                reg.append(dict(talhao=int(t), parcela=int(p.parcela), n_arv=len(pos),
                                dap_campo=float(p.D_cm), idade=float(p.idade_anos),
                                pico_cm=pico(dens), pts_por_arv=ppf))
        P = pd.DataFrame(reg)
        v = P.dropna(subset=["pico_cm"])
        if len(v) < 6:
            print(f"band {a0}-{a1}: only {len(v)} plots with a peak, not run")
            continue
        r = float(np.corrcoef(v.dap_campo, v.pico_cm)[0, 1])
        rs = np.array([float(np.corrcoef(rng.permutation(v.dap_campo.values), v.pico_cm)[0, 1])
                       for _ in range(N_PERM)])
        pval = float((np.abs(rs) >= abs(r)).mean())
        print(f"band {a0}-{a1} degrees, {len(v)} plots")
        print(f"   {'stand':>7} {'plot':>5} {'n tree':>6} {'field DBH':>10} {'true radius':>10} "
              f"{'peak':>7} {'pts/tree':>8}")
        for _, q in v.sort_values("dap_campo").iterrows():
            print(f"   {q.talhao:7.0f} {q.parcela:5.0f} {q.n_arv:6.0f} {q.dap_campo:9.1f} cm "
                  f"{q.dap_campo/2:9.1f} cm {q.pico_cm:6.1f} cm {q.pts_por_arv:8.1f}")
        print(f"   correlation of peak against field DBH: r = {r:+.3f}   "
              f"p = {pval:.3f} ({N_PERM} permutations)")
        print(f"   median peak {v.pico_cm.median():.1f} cm against true radius "
              f"{v.dap_campo.median()/2:.1f} cm\n", flush=True)
        for _, q in v.iterrows():
            linhas.append(dict(faixa=f"{a0}-{a1}", **q.to_dict()))
        linhas.append(dict(faixa=f"{a0}-{a1}", correlacao=r, p_valor=pval, n=len(v)))

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(SAIDA, index=False)
    print(SAIDA)


if __name__ == "__main__":
    main()

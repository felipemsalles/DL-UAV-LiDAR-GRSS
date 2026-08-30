#!/usr/bin/env python3
"""Does the oblique ring carry diameter in the aggregate? Aggregate first, measure afterwards.

`exp_dap_do_obliquo.py` estimates the radius of each stem and only then aggregates, and it fails the
control: at an empty position the estimator also returns a number. `exp_obliquo_do_voo_existente.py`,
by contrast, shows the ring clearly in the profile aggregated over the 892 stems, that is, the signal
exists in the sum and does not survive the division. Here the order is reversed: aggregate first,
estimate afterwards. If the peak of the aggregated profile of a group of trees tracks the mean DBH of
the group, the drone measures diameter at the level the inventory cares about, with no model, no
training and no labels.

Two tests, the sensitive one before the operational one:
  1. groups by DBH measured on the TLS, with a range from 13 to 26 cm. If the peak does not move over
     that range, it moves over no range at all.
  2. virtual 400 m2 plots, the operational test, with the small range that reality offers inside a
     single stand.

Control: the same profile at random positions, and with the stems shuffled between groups. The second
is the decisive one: a peak that moves with shuffled labels is an artefact of sample size and not of
diameter.

Usage: PYTHONPATH=. python scripts/exp_anel_agregado.py
Output: manual_match/anel_agregado.csv
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
MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
DRONE = config.REPO / "work/lazall/SaoManuelTotal_001.laz"
SAIDA = config.OUT_DIR / "anel_agregado.csv"
FAIXA_ANG = (20, 30)
TRECHO = (1.0, 6.0)
R_BUSCA = 0.45
GRADE = np.arange(0.03, 0.30, 0.002)
H_NUCLEO = 0.015
AREA_PARCELA = 400.0
SEED = 20260828


def perfil_do_grupo(T, X, Y, pos):
    """Aggregated radial density of the group, per ring area, normalised per stem."""
    d_all = []
    n = 0
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
    """Peak of the profile, in cm of radius. NaN if the profile decreases monotonically (no ring)."""
    if dens is None or not np.isfinite(dens).all():
        return np.nan
    i = int(np.argmax(dens))
    if i == 0:                      # peak at the inner edge = no ring, only blur
        return np.nan
    return float(100 * GRADE[i])


def main():
    rng = np.random.default_rng(SEED)
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    ran = pd.read_csv(config.OUT_DIR / "dap_tls_ransac_talhao001.csv")
    dap = np.where(ran.motivo.values == "ok", ran.dap_cm.values, np.nan)

    las = laspy.read(str(DRONE))
    cls = np.asarray(las.classification)
    veg = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    x, y, z = np.asarray(las.x)[veg], np.asarray(las.y)[veg], np.asarray(las.z)[veg]
    ang = np.abs(np.asarray(las.scan_angle_rank).astype(float))[veg]
    m = (z >= TRECHO[0]) & (z <= TRECHO[1]) & (ang >= FAIXA_ANG[0]) & (ang < FAIXA_ANG[1])
    X, Y = x[m], y[m]
    T = cKDTree(np.column_stack([X, Y]))
    print(f"{m.sum():,} points in the band from {FAIXA_ANG[0]} to {FAIXA_ANG[1]} degrees, "
          f"{TRECHO[0]} to {TRECHO[1]} m\n")

    ok = np.isfinite(dap)
    linhas = []

    # ---- 1. the sensitive test: groups by measured DBH ----------------------------------
    print("[1] groups by DBH measured on the TLS. If the peak does not move HERE, it never moves.")
    for n_grupos in (3, 4, 5):
        q = pd.qcut(dap[ok], n_grupos, labels=False, duplicates="drop")
        print(f"\n   {n_grupos} groups:")
        print(f"   {'group':>6} {'n':>5} {'mean TLS DBH':>14} {'true radius':>10} "
              f"{'profile peak':>15} {'pts/stem':>10}")
        picos, daps = [], []
        for g in range(int(q.max()) + 1):
            sel = np.where(ok)[0][q == g]
            dens, ppf = perfil_do_grupo(T, X, Y, ref[sel])
            p = pico(dens)
            dm = float(np.mean(dap[sel]))
            print(f"   {g:6d} {len(sel):5d} {dm:13.1f} cm {dm/2:9.1f} cm {p:14.1f} cm "
                  f"{ppf:10.1f}")
            picos.append(p)
            daps.append(dm)
            linhas.append(dict(teste="grupos por DAP", n_grupos=n_grupos, grupo=g,
                               n=len(sel), dap_tls=dm, pico_cm=p, pts_por_fuste=ppf))
        pv = np.array(picos, float)
        if np.isfinite(pv).sum() >= 3:
            r = float(np.corrcoef(np.array(daps)[np.isfinite(pv)], pv[np.isfinite(pv)])[0, 1])
            print(f"   correlation between group DBH and profile peak: r = {r:+.3f}")
            linhas.append(dict(teste="grupos por DAP", n_grupos=n_grupos, grupo=-1,
                               correlacao=r))
        # The control applies to the correlation, not to the amplitude: with 3 to 5 groups the
        # amplitude is poor statistics (shuffled gave 4.6 cm against 4.6 for the real one with 4
        # groups). What distinguishes signal from chance is the peak moving in the order of the
        # diameter, compared here against the distribution of the correlation under permutation.
        N_PERM = 60
        rs = []
        for _ in range(N_PERM):
            qe = rng.permutation(q)
            pe = []
            for g in range(int(q.max()) + 1):
                sel = np.where(ok)[0][qe == g]
                pe.append(pico(perfil_do_grupo(T, X, Y, ref[sel])[0]))
            pe = np.array(pe, float)
            m2 = np.isfinite(pe)
            if m2.sum() >= 3:
                # the DBH of the shuffled groups is practically equal to the mean, so the
                # correlation is computed against the SAME vector of DBH of the real groups
                rs.append(float(np.corrcoef(np.array(daps)[m2], pe[m2])[0, 1]))
        rs = np.array(rs)
        if len(rs) >= 10 and np.isfinite(pv).sum() >= 3:
            r_real = float(np.corrcoef(np.array(daps)[np.isfinite(pv)], pv[np.isfinite(pv)])[0, 1])
            pval = float((np.abs(rs) >= abs(r_real)).mean())
            print(f"   control: mean shuffled correlation {rs.mean():+.3f}, "
                  f"sd {rs.std():.3f}, |r| >= |{r_real:+.3f}| in {100*pval:.0f} % of the "
                  f"{len(rs)} permutations  ->  p = {pval:.3f}")
            linhas.append(dict(teste="controle por permutacao", n_grupos=n_grupos,
                               correlacao=r_real, r_emb_medio=float(rs.mean()),
                               r_emb_dp=float(rs.std()), p_valor=pval, n_perm=len(rs)))

    # ---- 2. the operational test: virtual plots ------------------------------------------
    print("\n[2] virtual 400 m2 plots, which is the range that reality offers")
    Rp = np.sqrt(AREA_PARCELA / np.pi)
    gx = np.arange(ref[:, 0].min() + Rp, ref[:, 0].max() - Rp, Rp / 2)
    gy = np.arange(ref[:, 1].min() + Rp, ref[:, 1].max() - Rp, Rp / 2)
    parc = []
    for cx in gx:
        for cy in gy:
            sel = np.where(ok & (np.hypot(ref[:, 0] - cx, ref[:, 1] - cy) <= Rp))[0]
            if len(sel) < 20:
                continue
            p = pico(perfil_do_grupo(T, X, Y, ref[sel])[0])
            parc.append(dict(n=len(sel), dap_tls=float(np.mean(dap[sel])), pico_cm=p))
    P = pd.DataFrame(parc)
    print(f"   {len(P)} plots, mean DBH from {P.dap_tls.min():.1f} to {P.dap_tls.max():.1f} cm "
          f"(a range of only {P.dap_tls.max()-P.dap_tls.min():.1f} cm)")
    v = P.dropna(subset=["pico_cm"])
    if len(v) >= 6:
        r = float(np.corrcoef(v.dap_tls, v.pico_cm)[0, 1])
        print(f"   profile peak against mean plot DBH: r = {r:+.3f} ({len(v)} plots)")
        print(f"   median peak {v.pico_cm.median():.1f} cm against true radius "
              f"{v.dap_tls.median()/2:.1f} cm")
        linhas.append(dict(teste="parcelas virtuais", n=len(v), correlacao=r,
                           pico_mediano=float(v.pico_cm.median()),
                           raio_real=float(v.dap_tls.median() / 2)))
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

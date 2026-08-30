#!/usr/bin/env python3
"""At what density does the stem appear in the cloud? With a positive control on the TLS.

The radial density profile of the drone cloud around the mapped stems comes out as a solid blur
peaking at r = 1 cm, and not as a ring at the true radius, which suggests that at 676 pts/m2 nadir
does not resolve the stem surface. Two conditions are necessary for that reading to hold.

Positive control: a profile that never shows a ring on any data says more about the profile code
than about the drone. The TLS sees the trunk from the side and by construction has to show a ring;
if it does, the negative result for the drone is real.

Normalised aggregation: summing absolute radii of stems with DBH between 7 and 39 cm smears an
intrinsic ring of 1 to 3 cm over some 5 cm, purely from the variation in DBH. Here the radius of
each point is divided by the radius of its own stem before the histogram, so every ring falls at
r/R = 1. Both profiles are computed, the absolute one for comparison with absolute-radius
readings and the normalised one because it is the correct one.

Dividing by the ring area is mandatory: without dividing by 2*pi*r*dr the growth of the area mimics
a peak that is not there, and any radius read off that peak is wrong.

Subsampling the TLS is not equivalent to flying denser: thinning the TLS preserves the side-viewing
geometry and only removes points. If the TLS ring survives down to the drone density, what
separates the two is the viewing geometry and not the density, and flying denser does not solve it;
if the ring dies earlier, density is the limit.

Reference: Annals of Forest Science 2025 (10.1186/s13595-025-01291-w) measures DBH directly from a
UAV cloud in eucalypt, RMSE 2.18 cm, with a higher density than ours and HDBSCAN separating stem
from clutter before fitting.

Usage: PYTHONPATH=. python scripts/exp_densidade_e_fuste.py
Output: manual_match/densidade_e_fuste.csv and manual_match/densidade_e_fuste_perfis.csv
"""
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenvista import config  # noqa: E402

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
REF_DAP = config.OUT_DIR / "dap_tls_ransac_talhao001.csv"
DEG = config.REPO / "work/ff3d_degraded"
DRONE_T001 = config.REPO / "work/lazall/SaoManuelTotal_001.laz"
SAIDA = config.OUT_DIR / "densidade_e_fuste.csv"
SAIDA_PERFIS = config.OUT_DIR / "densidade_e_fuste_perfis.csv"

FAIXA = (1.0, 1.6)         # section of trunk, above the ground
R_MAX = 0.40               # how far the profile goes, in metres
BIN_ABS = 0.01             # ring of the absolute profile, in metres
BIN_REL = 0.10             # ring of the normalised profile, in units of R
REL_MAX = 4.0
G_SOLO = 0.5
SEED = 20260828


def terreno(x, y, z):
    """Minimum over a 0.5 m cell, holes filled from the neighbour and then smoothed."""
    x0, y0 = x.min() - 1, y.min() - 1
    nx = int((x.max() - x0) / G_SOLO) + 2
    ny = int((y.max() - y0) / G_SOLO) + 2
    zmin = np.full((nx, ny), 1e9, np.float32)
    np.minimum.at(zmin, (((x - x0) / G_SOLO).astype(int),
                         ((y - y0) / G_SOLO).astype(int)), z)
    zmin[zmin > 1e8] = np.nan
    vazio = ~np.isfinite(zmin)
    if vazio.any():
        _, (a, b) = distance_transform_edt(vazio, return_indices=True)
        zmin[vazio] = zmin[a[vazio], b[vazio]]
    return uniform_filter(zmin, 7), x0, y0


def le_tls():
    xs, ys, zs = [], [], []
    with laspy.open(config.TLS_LAS) as fh:
        for p in fh.chunk_iterator(6_000_000):
            xs.append(np.asarray(p.x))
            ys.append(np.asarray(p.y))
            zs.append(np.asarray(p.z, dtype=np.float32))
    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    zsolo, x0, y0 = terreno(x, y, z)
    h = z - zsolo[np.clip(((x - x0) / G_SOLO).astype(int), 0, zsolo.shape[0] - 1),
                  np.clip(((y - y0) / G_SOLO).astype(int), 0, zsolo.shape[1] - 1)]
    k = (h > FAIXA[0]) & (h < FAIXA[1])
    return x[k], y[k], float(np.ptp(x) * np.ptp(y))


def le_drone(caminho):
    """The drone cloud already comes height-normalised, so there is no terrain model here."""
    las = laspy.read(str(caminho))
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    cls = np.asarray(las.classification)
    area = float(np.ptp(x) * np.ptp(y))
    k = (z > FAIXA[0]) & (z < FAIXA[1]) & ~np.isin(cls, (18,))
    return x[k], y[k], area, len(x) / area, len(x), int(k.sum())


# The null is not zero: points spread evenly out to 4 R already put (1.2^2 - 0.8^2) / 4^2 = 5 % of
# the points in the 0.8 to 1.2 R band from area alone. Without that null, 15 % would read as signal.
NULO_CASCA = (1.2 ** 2 - 0.8 ** 2) / REL_MAX ** 2


def perfil(X, Y, ref_xy, ref_R, r_max, bins, normaliza):
    """Radial density per ring area, aggregated over the stems.

    Returns (ring centres, mean density per stem, number of stems used, number of points).
    With `normaliza`, the radius of each point is divided by the radius of its own stem, and the
    ring area comes out in units of R^2, so the density is in points per R^2.
    """
    if len(X) == 0:
        return bins[:-1] * np.nan, bins[:-1] * np.nan, np.zeros(len(bins) - 1), 0, 0
    T = cKDTree(np.column_stack([X, Y]))
    area = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    acc = np.zeros(len(bins) - 1)
    n_usados = n_pts = 0
    for (cx, cy), R in zip(ref_xy, ref_R):
        if not np.isfinite(R):
            continue
        alcance = r_max * R if normaliza else r_max
        idx = T.query_ball_point([cx, cy], alcance)
        if not idx:
            n_usados += 1
            continue
        idx = np.asarray(idx, dtype=int)
        d = np.hypot(X[idx] - cx, Y[idx] - cy)
        if normaliza:
            d = d / R
        hist, _ = np.histogram(d, bins=bins)
        acc += hist
        n_usados += 1
        n_pts += len(idx)
    if n_usados == 0:
        return bins[:-1] * np.nan, bins[:-1] * np.nan, np.zeros(len(bins) - 1), 0, 0
    centros = 0.5 * (bins[1:] + bins[:-1])
    return centros, acc / (area * n_usados), acc, n_usados, n_pts


def resume(centros, dens):
    """Ring contrast: density on the bark over density in the core, plus the background.

    Normalised profile, in which the true ring falls at r/R = 1. If the laser sees the surface, the
    density at 0.9 to 1.1 exceeds that of the core at 0 to 0.3 and the contrast goes above 1. If it
    sees a solid blur, the core wins and the contrast stays below 1.
    """
    def faixa(a, b):
        m = (centros >= a) & (centros < b)
        return float(np.nanmean(dens[m])) if m.any() else np.nan
    miolo, casca, fundo = faixa(0.0, 0.3), faixa(0.9, 1.1), faixa(2.5, 4.0)
    return dict(dens_miolo=miolo, dens_casca=casca, dens_fundo=fundo,
                contraste=casca / miolo if miolo else np.nan,
                casca_sobre_fundo=casca / fundo if fundo else np.nan)


def fracao_na_casca(centros, acc, a=0.8, b=1.2):
    """Fraction of the points that fall on the stem bark, between 0.8 and 1.2 of the true radius.

    This statistic tolerates low counts and the contrast does not: contrast is a ratio between two
    densities and explodes when the core happens to be left without points (the TLS thinned to 4
    points per stem scored 471 against 106 for the full TLS). The proportion has no denominator that
    goes to zero, and it compares conditions of very different counts without creating a trend.
    """
    tot = acc.sum()
    if tot <= 0:
        return np.nan
    m = (centros >= a) & (centros < b)
    return float(acc[m].sum() / tot)


def raio_medido(X, Y, ref_xy, ref_R, r_busca=0.30, n_min=10):
    """Radius of each stem from the peak of the radial density per area, and correlation with TLS."""
    if len(X) == 0:
        return np.nan, np.nan, 0
    T = cKDTree(np.column_stack([X, Y]))
    bins = np.arange(0.0, r_busca + 0.01, 0.01)
    area = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    est, ver = [], []
    for (cx, cy), R in zip(ref_xy, ref_R):
        if not np.isfinite(R):
            continue
        idx = T.query_ball_point([cx, cy], r_busca)
        if len(idx) < n_min:
            continue
        d = np.hypot(X[np.asarray(idx, int)] - cx, Y[np.asarray(idx, int)] - cy)
        hist, _ = np.histogram(d, bins=bins)
        dens = hist / area
        dens[bins[:-1] < 0.03] = 0.0
        if not dens.any():
            continue
        est.append(0.5 * (bins[int(dens.argmax())] + bins[int(dens.argmax()) + 1]))
        ver.append(R)
    if len(est) < 20:
        return np.nan, np.nan, len(est)
    est, ver = np.asarray(est), np.asarray(ver)
    return (float(spearmanr(est, ver).statistic), float(np.median(est) * 200), len(est))


def recorta(X, Y, ref_xy, ref_R, folga=1.0):
    """Keeps only the stems that fall inside the cloud of this condition.

    Without that clipping the per-stem count is wrong: the plot tiles cover a piece of the stand, and
    dividing their points by the 813 stems of the whole map includes stems the cloud does not reach,
    making the condition look four times sparser than it is.
    """
    if len(X) == 0:
        return ref_xy, ref_R
    m = ((ref_xy[:, 0] >= X.min() + folga) & (ref_xy[:, 0] <= X.max() - folga)
         & (ref_xy[:, 1] >= Y.min() + folga) & (ref_xy[:, 1] <= Y.max() - folga))
    return ref_xy[m], ref_R[m]


def avalia(nome, X, Y, ref_xy, ref_R, dens_nuvem, perfis, r_est=0.30):
    """One condition: absolute profile, normalised profile, contrast and measured radius."""
    ref_xy, ref_R = recorta(X, Y, ref_xy, ref_R)
    b_abs = np.arange(0.0, R_MAX + BIN_ABS, BIN_ABS)
    b_rel = np.arange(0.0, REL_MAX + BIN_REL, BIN_REL)
    c_abs, d_abs, _, n_f, n_p = perfil(X, Y, ref_xy, ref_R, R_MAX, b_abs, False)
    c_rel, d_rel, acc_rel, _, _ = perfil(X, Y, ref_xy, ref_R, REL_MAX, b_rel, True)
    r_sp, r_med, n_est = raio_medido(X, Y, ref_xy, ref_R, r_busca=r_est)
    for c, d in zip(c_rel, d_rel):
        perfis.append(dict(cond=nome, eixo="normalizado", r=c, densidade=d))
    for c, d in zip(c_abs, d_abs):
        perfis.append(dict(cond=nome, eixo="absoluto", r_m=c, densidade=d))
    ppf = n_p / n_f if n_f else np.nan
    linha = dict(cond=nome, dens_nuvem_pts_m2=dens_nuvem, n_fustes=n_f, pts_por_fuste=ppf,
                 spearman_raio=r_sp, raio_mediano_cm=r_med, n_com_raio=n_est,
                 frac_na_casca=fracao_na_casca(c_rel, acc_rel))
    linha.update(resume(c_rel, d_rel))
    # with fewer than 2 points per stem the contrast is noise, so it is not reported
    if not np.isfinite(ppf) or ppf < 2.0:
        linha["contraste"] = np.nan
    # peak of the absolute profile, for a direct comparison on the absolute radius
    if np.isfinite(d_abs).any():
        linha["pico_absoluto_cm"] = float(c_abs[np.nanargmax(d_abs)] * 100)
    print(f"  {nome:22s} dens {dens_nuvem:7.1f} pts/m2 | pts/stem {linha['pts_por_fuste']:7.1f} | "
          f"contrast {linha['contraste']:6.2f} | on the bark {100 * linha['frac_na_casca']:5.1f} % | "
          f"rho(radius,DBH) {linha['spearman_raio'] if np.isfinite(r_sp) else float('nan'):+.2f}",
          flush=True)
    return linha


def main():
    rng = np.random.default_rng(SEED)
    f = gpd.read_file(MAPA)
    ref_xy = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    dap = pd.read_csv(REF_DAP)
    ref_R = np.where(dap.motivo.values == "ok", dap.dap_cm.values / 200.0, np.nan)
    print(f"{np.isfinite(ref_R).sum()} stems with a reference radius, "
          f"median DBH {np.nanmedian(ref_R) * 200:.1f} cm\n")

    linhas, perfis = [], []

    # ---- 1. positive control, and without it nothing here holds -------------------------
    print("[1] TLS, positive control. Side viewing, it MUST show a ring.", flush=True)
    if not config.TLS_LAS.exists():
        sys.exit(f"TLS cloud not found at {config.TLS_LAS}")
    xt, yt, area_tls = le_tls()
    dens_tls = len(xt) / area_tls
    linhas.append(avalia("TLS integral", xt, yt, ref_xy, ref_R, dens_tls, perfis))

    # ---- 2. thinned TLS, to separate density from viewing geometry ----------------------
    print("\n[2] Thinned TLS. If the ring survives down to the drone density, what separates the "
          "two is the viewing geometry and not the density.", flush=True)
    for frac in (0.30, 0.10, 0.03, 0.01, 0.003):
        k = rng.random(len(xt)) < frac
        linhas.append(avalia(f"TLS {frac:.3f} do total", xt[k], yt[k], ref_xy, ref_R,
                             dens_tls * frac, perfis))

    # ---- 3. drone, the whole stand 001 --------------------------------------------------
    print("\n[3] drone, the whole stand 001", flush=True)
    xd, yd, area_d, dens_d, _, _ = le_drone(DRONE_T001)
    linhas.append(avalia("drone T001 integral", xd, yd, ref_xy, ref_R, dens_d, perfis))

    # ---- 4. degraded drone, the density ladder already built ----------------------------
    print("\n[4] degraded drone on the two plot tiles. ang_20 and ang_10 restrict the scan "
          "angle, that is, they bring the viewing geometry closer to pure nadir.", flush=True)
    man = pd.read_csv(DEG / "degraded_manifest.csv")
    man = man[man.tile.isin(["t001_p001", "t001_p002"])]
    for cond in ("full", "dens_200", "dens_100", "dens_50", "dens_20", "ang_20", "ang_10"):
        sub = man[man.condition == cond]
        if sub.empty:
            continue
        X = Y = None
        tot_pts = tot_peito = 0
        for t in sub.tile:
            a, b, _, _, np_all, np_peito = le_drone(DEG / cond / f"{t}.laz")
            tot_pts += np_all
            tot_peito += np_peito
            X = a if X is None else np.concatenate([X, a])
            Y = b if Y is None else np.concatenate([Y, b])
        linha = avalia(f"drone {cond}", X, Y, ref_xy, ref_R,
                       float(sub.density_pts_m2.mean()), perfis)
        linha["n_pts_nuvem"] = tot_pts
        linha["n_pts_peito"] = tot_peito
        linhas.append(linha)

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(linhas)
    d.to_csv(SAIDA, index=False)
    pd.DataFrame(perfis).to_csv(SAIDA_PERFIS, index=False)

    print("\n" + "=" * 78)
    tls = d[d.cond == "TLS integral"].iloc[0]
    dro = d[d.cond == "drone T001 integral"].iloc[0]
    print(f"POSITIVE CONTROL: the full TLS has contrast {tls.contraste:.2f} "
          f"(>1 is a ring, <1 is a solid blur)")
    print(f"DRONE           : contrast {dro.contraste:.2f}")
    if tls.contraste > 1.0 and dro.contraste < 1.0:
        print("The profile WORKS and the negative result for the drone is real.")
    elif tls.contraste <= 1.0:
        print("THE TLS DOES NOT SHOW A RING EITHER. The profile method is wrong, and the "
              "negative result for the drone falls with it.")
    raleado = d[d.cond.str.startswith("TLS ")]
    vivos = raleado[raleado.frac_na_casca > 0.5]
    if len(vivos):
        pior = vivos.iloc[-1]
        print(f"\nThe TLS thinned down to {pior.pts_por_fuste:.1f} points per stem STILL has "
              f"{100 * pior.frac_na_casca:.0f} % of its points on the stem bark.")
        print(f"The drone, with {dro.pts_por_fuste:.1f} points per stem, has "
              f"{100 * dro.frac_na_casca:.0f} %.")
        if pior.pts_por_fuste <= dro.pts_por_fuste:
            print("SAME COUNT PER STEM, OPPOSITE RESULT. What separates the two is not density, "
                  "it is where one looks from. Flying denser does not make the stem appear.")
    print(f"\n(the geometric null of the 0.8 to 1.2 R band is {100 * NULO_CASCA:.1f} %, "
          f"which is what evenly spread points already put there)")

    # A direct test of the viewing-geometry hypothesis: if the return at breast height exists only
    # because the beam came in from the side, restricting the scan angle must knock down the
    # breast-height return far more than the total return.
    base = d[d.cond == "drone full"]
    if len(base) and "n_pts_peito" in d.columns:
        b = base.iloc[0]
        print("\nenrichment of the breast-height return at wide scan angles:")
        for _, r in d[d.cond.str.contains("ang_")].iterrows():
            f_tot = r.n_pts_nuvem / b.n_pts_nuvem
            f_peito = r.n_pts_peito / b.n_pts_peito
            print(f"   {r.cond:18s} keeps {100 * f_tot:5.1f} % of the points of the cloud "
                  f"and only {100 * f_peito:5.1f} % of the breast-height ones  "
                  f"(ratio {f_peito / f_tot:.2f})")
        print("   a ratio below 1 means that the point at breast height comes "
              "disproportionately from the oblique beam, not from nadir.")
    print(f"\n{SAIDA}\n{SAIDA_PERFIS}")


if __name__ == "__main__":
    main()

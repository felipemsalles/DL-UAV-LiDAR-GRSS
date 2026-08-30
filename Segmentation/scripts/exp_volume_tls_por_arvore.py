#!/usr/bin/env python3
"""Per-tree volume straight from the terrestrial cloud, to go from 26 labels to hundreds.

The bottleneck is not the model, it is the label: per-tree volume exists only
for the 26 destructively scaled trees of stand 001, and destructive scaling is destructive. The
literature has already validated the substitute. TLS surface reconstruction in eucalyptus reports an
RMSE of 6.91 % against destructive scaling, QSM 15.31 %, and in teak QSM against destructive felling
showed no significant difference. The terrestrial laser measures stem volume well enough to serve as
a reference, and the terrestrial cloud of stand 001 with the 892 stem positions is available.

No QSM is fitted here. QSM reconstructs the whole tree with cylinders and needs branches; what
matters in eucalyptus grown for harvest is the stem, where the merchantable volume is, and the stem
is a stack of sections. What is done is measured taper: radius at several heights by the same radial
mode estimator already validated on DBH, and Smalian between sections. It is the digital analogue of
what destructive scaling does with a tape, and can therefore be compared with it section by section.

There is no tree-to-tree matching, by inherited constraint: the GPS of the scaled trees is of
metre-scale accuracy and the planting has 1.88 m between trees, so any point falls close to some stem
(see `exp_gps_ruler.py`). Every validation here is distributional or about shape, never a tree-against-
tree r2.

Relative taper cancels what matching does not allow. d(h) / d(1.3) against h / H is a form function:
it does not depend on which tree is which, only on the shape of the stem. If the mean TLS curve
matches the mean destructive-scaling curve, the measured taper is right, and so is the volume that
comes from integrating it.

The bias of the radius estimator enters the volume squared. Radial-mode DBH has about 8 % declared
bias, and sectional area goes with the square, so the volume inherits close to 17 %. The volume is
therefore produced in two versions, raw and with the scale corrected by the inventory, and both are
reported.

The height comes from the drone, and not from TLS. A handheld scan under the canopy misses the top of
a 33 m eucalyptus. The drone sees the top and not the stem, TLS sees the stem and not the top, so the
height comes from the drone and the taper from TLS. It is the only place in the project where both
clouds enter the same calculation.

Run: PYTHONPATH=. python scripts/exp_volume_tls_por_arvore.py
Out: manual_match/volume_tls_por_arvore.csv and manual_match/volume_tls_afilamento.csv
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

# reuses the already validated estimator instead of reimplementing it and diverging silently
_s = importlib.util.spec_from_file_location("ransac", R / "scripts/exp_dap_tls_ransac.py")
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)
terreno, eixo, base_ortogonal, raio_por_modo = (_m.terreno, _m.eixo, _m.base_ortogonal,
                                                _m.raio_por_modo)

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
DRONE_T001 = config.REPO / "work/lazall/SaoManuelTotal_001.laz"
SAIDA = config.OUT_DIR / "volume_tls_por_arvore.csv"
SAIDA_AFIL = config.OUT_DIR / "volume_tls_afilamento.csv"

G_SOLO = 0.5
R_CILINDRO = 1.20          # radius of the crop around the stem, generous because the tree leans
# The ladder goes up and down from 1.3 m, and not from the ground: the breast-height radius has
# already been measured and validated by `exp_dap_tls_ransac.py`, so it is the anchor and each
# following section only needs to stay close to the previous one.
H_ANCORA = 1.3
SOBE = np.array([1.7, 2.1, 2.6, 3.1, 3.7, 4.3, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 20.0, 22.0])
DESCE = np.array([1.0, 0.7, 0.4])
PTS_MIN = 40
INLIER_MIN = 25
R_LIM = (0.020, 0.300)     # plausible stem radius, in metres
# The radius window is per metre of ascent and almost symmetric. A per-section limit, with a fall of
# up to 30 % and a rise of only 10 %, acts as a ratchet: every sparse slice pulls the radius down and
# the next one has no way to recover, so the error accumulates section after section. With 15 steps
# between 1.3 and 13 m, the taper comes out 26 points below the destructive scaling at half the tree
# height. The limit is a physical prior, and not the destructive-scaling curve: a stem is a
# continuous solid and does not lose 10 % of its radius in one metre below the crown. The shape
# remains free to be measured, otherwise the validation against destructive scaling would be
# circular.
CRESCE_POR_M = 1.04
ENCOLHE_POR_M = 0.90
DESLOC_MAX = 0.10          # plus 0.15 m per metre of step: a stem does not jump sideways
FALHAS_SEGUIDAS = 3
SECOES_MIN = 5
TOL_ANEL = 0.020           # half-width of the inlier band, in metres
PASSO_CENTRO = 0.02
H_DAP = 1.3
DENS_BASICA = 0.5          # for reporting only; it enters no volume calculation


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
    k = (h > 0.10) & (h < SOBE.max() + 1.5)
    return x[k], y[k], h[k].astype(np.float64)


def alturas_do_drone(ref_xy, raio=1.2, q=0.995):
    """Total height of each stem, from the top the drone sees around its position."""
    las = laspy.read(str(DRONE_T001))
    cls = np.asarray(las.classification)
    m = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    xy = np.column_stack([np.asarray(las.x)[m], np.asarray(las.y)[m]])
    z = np.asarray(las.z)[m]
    T = cKDTree(xy)
    H = np.full(len(ref_xy), np.nan)
    for i, c in enumerate(ref_xy):
        idx = T.query_ball_point(c, raio)
        if len(idx) >= 20:
            H[i] = float(np.quantile(z[np.asarray(idx, int)], q))
    return H


def centro_do_anel(px, py, r, janela):
    """Centre that puts the most points in the band of radius r. With r fixed, this does not bias r.

    Centring by inlier count is only legitimate with the radius held fixed. If the radius varies
    along with it, the search always picks the largest, because a larger circle has a larger
    circumference and a band of the same thickness collects more points; under that condition the
    estimator sticks to the ceiling of the search and measures zero taper up to 15 m of height, which
    is physically impossible. Here the radius comes afterwards, from the mode of the per-area
    density, which is the estimator already validated on DBH.
    """
    g = np.arange(-janela, janela + 1e-9, PASSO_CENTRO)
    du, dv = np.meshgrid(g, g, indexing="ij")
    du, dv = du.ravel(), dv.ravel()
    d = np.hypot(px[None, :] - du[:, None], py[None, :] - dv[:, None])
    n = (np.abs(d - r) < TOL_ANEL).sum(1)
    i = int(n.argmax())
    return float(du[i]), float(dv[i]), int(n[i])


def raio_por_modo_limitado(pu, pv, rmin, rmax, minimo):
    """Mode of the PER-AREA radial density, with the radius held to a band around the expected one.

    Same mathematics as `raio_por_modo`, except that the search cannot escape to a tiny radius.
    Without the clamp, a sparse slice in which half a dozen points fall near the axis produces a peak
    at 3 cm, because the 3 cm ring has a minuscule area. The taper then collapses and the "cannot
    grow" clamp prevents recovery, i.e. the error becomes a ratchet.
    """
    bins = np.arange(rmin, rmax + 0.005, 0.005)
    if len(bins) < 3:
        return np.nan, 0
    area = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
    d = np.hypot(pu, pv)
    dens = np.histogram(d, bins=bins)[0] / area
    if not dens.any():
        return np.nan, 0
    i = int(dens.argmax())
    r = 0.5 * (bins[i] + bins[i + 1])
    n = int((np.abs(d - r) < TOL_ANEL).sum())
    return (r, n) if n >= minimo else (np.nan, n)


def _marcha(px, py, ph, alturas, r0, meia0, out):
    """Walks one branch of the ladder from the anchor, keeping the accepted sections."""
    cu = cv = 0.0
    r_ant = r0
    h_ant = H_ANCORA
    falhas = 0
    for h_alvo in alturas:
        meia = max(meia0, 0.10 + 0.012 * h_alvo)
        m = np.abs(ph - h_alvo) <= meia
        if m.sum() < PTS_MIN:
            falhas += 1
            if falhas >= FALHAS_SEGUIDAS:
                break
            continue
        qx, qy = px[m] - cu, py[m] - cv
        salto = abs(h_alvo - h_ant)
        # the centre search window grows with the height step, because the tree leans
        janela = min(0.40, 0.06 + 0.05 * salto)
        du, dv, n_band = centro_do_anel(qx, qy, r_ant, janela)
        if n_band < INLIER_MIN or np.hypot(du, dv) > DESLOC_MAX + 0.15 * salto:
            falhas += 1
            if falhas >= FALHAS_SEGUIDAS:
                break
            continue
        lo, hi = ENCOLHE_POR_M ** salto, CRESCE_POR_M ** salto
        r, n = raio_por_modo_limitado(qx - du, qy - dv, max(R_LIM[0], lo * r_ant),
                                      min(R_LIM[1], hi * r_ant), INLIER_MIN)
        if not np.isfinite(r):
            falhas += 1
            if falhas >= FALHAS_SEGUIDAS:
                break
            continue
        falhas = 0
        cu, cv = cu + du, cv + dv
        r_ant, h_ant = r, h_alvo
        out.append((h_alvo, r, n))


def afilamento_de_um_fuste(px, py, ph, r_dap):
    """Taper anchored on the measured DBH, marching upwards and downwards."""
    if not np.isfinite(r_dap):
        return []
    m = np.abs(ph - H_ANCORA) <= 0.15
    if m.sum() < PTS_MIN:
        return []
    du, dv, n = centro_do_anel(px[m], py[m], r_dap, 0.12)
    if n < INLIER_MIN:
        return []
    out = [(H_ANCORA, r_dap, n)]
    _marcha(px - du, py - dv, ph, SOBE, r_dap, 0.15, out)
    _marcha(px - du, py - dv, ph, DESCE, r_dap, 0.12, out)
    out = sorted(out)
    # Isotonic at the end: it imposes no shape at all, it only requires that the radius not
    # increase going up, which is the one stem fact that holds without exception below the crown.
    # Without it a dirty slice becomes a step and Smalian integrates the step as if it were wood.
    if len(out) >= 3:
        h = np.array([a for a, _, _ in out])
        r = np.array([b for _, b, _ in out])
        n = [c for _, _, c in out]
        r = -_isotonica(-r)
        out = list(zip(h, r, n))
    return out


def _isotonica(y):
    """Non-decreasing isotonic regression, pool adjacent violators algorithm."""
    y = np.asarray(y, float).copy()
    w = np.ones_like(y)
    i = 0
    while i < len(y) - 1:
        if y[i] <= y[i + 1] + 1e-12:
            i += 1
            continue
        novo = (w[i] * y[i] + w[i + 1] * y[i + 1]) / (w[i] + w[i + 1])
        y[i] = novo
        w[i] = w[i] + w[i + 1]
        y = np.delete(y, i + 1)
        w = np.delete(w, i + 1)
        # the merged block may violate the previous one, so step back
        while i > 0 and y[i - 1] > y[i]:
            novo = (w[i - 1] * y[i - 1] + w[i] * y[i]) / (w[i - 1] + w[i])
            y[i - 1] = novo
            w[i - 1] = w[i - 1] + w[i]
            y = np.delete(y, i)
            w = np.delete(w, i)
            i -= 1
    return np.repeat(y, w.astype(int))


def volume_smalian(h, r):
    """Smalian over the measured sections, plus the stump as a cylinder up to the first section."""
    a = np.pi * np.asarray(r) ** 2
    h = np.asarray(h)
    v = float(np.sum(0.5 * (a[:-1] + a[1:]) * np.diff(h)))
    return v + float(a[0] * h[0]), v


def curva_cubagem():
    """Relative taper measured on the destructively scaled trees: d/d(1.3) against h/H."""
    c = pd.read_csv(config.DATA / "5-dados_campo/Cubagem_SaoManuel.csv", sep=";")
    linhas = []
    for (t, a), g in c.groupby(["talhao", "arv"]):
        g = g.sort_values("h_m")
        H = float(g.H_m.iloc[0])
        if not np.isfinite(H) or H <= 2:
            continue
        d13 = float(np.interp(H_DAP, g.h_m, g.d_cm))
        if d13 <= 0:
            continue
        for _, s in g.iterrows():
            linhas.append(dict(talhao=t, arv=a, x=s.h_m / H, y=s.d_cm / d13,
                               h_m=s.h_m, d_cm=s.d_cm, H=H))
    return pd.DataFrame(linhas)


def q_de_x(cub, nx=25):
    """Median form function of the destructive scaling, non-parametric, by h/H band."""
    bordas = np.linspace(0.0, 1.0, nx + 1)
    i = np.clip(np.digitize(cub.x, bordas) - 1, 0, nx - 1)
    med = pd.Series(cub.y.values).groupby(i).median()
    xs = 0.5 * (bordas[1:] + bordas[:-1])
    ys = np.full(nx, np.nan)
    ys[med.index.values] = med.values
    ok = np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    return np.concatenate([xs, [1.0]]), np.concatenate([ys, [0.0]])


def volume_acima(h_ult, r_ult, H, xq, yq, passo=0.25):
    """Stem volume above the last measured section, anchored on it and with the scaling form."""
    if not np.isfinite(H) or H <= h_ult + passo:
        return 0.0
    hs = np.arange(h_ult, H, passo)
    q = np.interp(hs / H, xq, yq)
    q0 = np.interp(h_ult / H, xq, yq)
    if q0 <= 0:
        return 0.0
    r = r_ult * q / q0
    a = np.pi * r ** 2
    return float(np.sum(0.5 * (a[:-1] + a[1:]) * np.diff(hs)))


def main():
    if not config.TLS_LAS.exists():
        sys.exit(f"TLS cloud not found at {config.TLS_LAS}")
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    ran = pd.read_csv(config.OUT_DIR / "dap_tls_ransac_talhao001.csv")
    r_dap = np.where(ran.motivo.values == "ok", ran.dap_cm.values / 200.0, np.nan)
    print(f"{np.isfinite(r_dap).sum()} stems with a measured anchor DBH\n")

    print("height of each stem from the top the drone sees", flush=True)
    H_drone = alturas_do_drone(ref)
    print(f"   {np.isfinite(H_drone).sum()} of {len(ref)} with a height, "
          f"median {np.nanmedian(H_drone):.1f} m\n", flush=True)

    print(f"reading {config.TLS_LAS.name}", flush=True)
    x, y, h = le_tls()
    print(f"   {len(x):,} points between 0.1 and {SOBE.max() + 1.5:.0f} m", flush=True)
    T = cKDTree(np.column_stack([x, y]))

    cub = curva_cubagem()
    xq, yq = q_de_x(cub)

    linhas, afil = [], []
    for j, (cx, cy) in enumerate(ref):
        idx = np.asarray(T.query_ball_point([cx, cy], R_CILINDRO), dtype=int)
        reg = dict(fuste=j, x=cx, y=cy, H_m=H_drone[j], n_secoes=0, dap_cm=np.nan,
                   h_ultima_m=np.nan, vol_medido_m3=np.nan, vol_extrap_m3=np.nan,
                   vol_m3=np.nan)
        if len(idx) >= PTS_MIN:
            sec = afilamento_de_um_fuste(x[idx] - cx, y[idx] - cy, h[idx], r_dap[j])
            if len(sec) >= SECOES_MIN:
                hs = np.array([s[0] for s in sec])
                rs = np.array([s[1] for s in sec])
                v_tot, _ = volume_smalian(hs, rs)
                v_up = volume_acima(hs[-1], rs[-1], H_drone[j], xq, yq)
                d13 = 200 * r_dap[j]
                reg.update(n_secoes=len(sec), dap_cm=d13, h_ultima_m=float(hs[-1]),
                           vol_medido_m3=v_tot, vol_extrap_m3=v_up, vol_m3=v_tot + v_up)
                if np.isfinite(H_drone[j]) and np.isfinite(d13):
                    for hh, rr, ni in sec:
                        afil.append(dict(fuste=j, h_m=hh, d_cm=200 * rr, n_inliers=ni,
                                         x=hh / H_drone[j], y=200 * rr / d13))
        linhas.append(reg)
        if (j + 1) % 100 == 0:
            print(f"   {j + 1}/{len(ref)}", flush=True)

    d = pd.DataFrame(linhas)
    A = pd.DataFrame(afil)
    ok = d.vol_m3.notna() & d.H_m.notna() & d.dap_cm.notna()
    print(f"\nvolume obtained for {ok.sum()} of the {len(d)} stems ({100 * ok.mean():.0f} %)")
    print(f"   sections per stem, median {d.n_secoes[ok].median():.0f}, "
          f"last section median at {d.h_ultima_m[ok].median():.1f} m")
    print(f"   fraction of the volume coming from the extrapolation above the last section: "
          f"{100 * (d.vol_extrap_m3[ok] / d.vol_m3[ok]).median():.1f} % (median)")
    print(f"   DBH of the taper, median {d.dap_cm[ok].median():.1f} cm")
    print(f"   volume per tree, median {d.vol_m3[ok].median():.4f} m3, "
          f"mean {d.vol_m3[ok].mean():.4f}")

    # ---- 1. the shape. It is the only test that does not need tree-to-tree matching ------
    print("\n[1] relative taper, TLS against destructive scaling")
    if len(A):
        xt, yt = q_de_x(A.rename(columns={"x": "x", "y": "y"}))
        xs = np.linspace(0.05, 0.6, 12)
        a_tls = np.interp(xs, xt, yt)
        a_cub = np.interp(xs, xq, yq)
        for u, v, w in zip(xs, a_tls, a_cub):
            print(f"   h/H {u:.2f}   TLS {v:.3f}   scaling {w:.3f}   "
                  f"diff {100 * (v - w):+5.1f} points")
        print(f"   RMSE between the two form curves: "
              f"{100 * np.sqrt(np.mean((a_tls - a_cub) ** 2)):.1f} percentage points")

    # ---- 2. distribution, against the scaling and against the equation ------------------
    print("\n[2] volume distribution")
    v = pd.read_csv(config.VOLUMES_CSV)
    v1 = v[v.talhao == 1]
    print(f"   destructive scaling of stand 001 (n={len(v1)}): median {v1.V.median():.4f} m3, "
          f"mean {v1.V.mean():.4f}, mean DBH {v1.D_cm.mean():.1f} cm, mean H {v1.H_m.mean():.1f} m")
    print(f"   TLS      (n={ok.sum()}): median {d.vol_m3[ok].median():.4f} m3, "
          f"mean {d.vol_m3[ok].mean():.4f}, mean DBH {d.dap_cm[ok].mean():.1f} cm, "
          f"mean H {d.H_m[ok].mean():.1f} m")
    print("   the scaled trees are not a random sample of the stand, they are chosen by diameter "
          "class, so their mean does not have to match that of the stand.")

    # Schumacher-Hall equation fitted on the scaled trees, applied to the TLS DBH and H
    m = np.isfinite(v.V) & (v.V > 0)
    Xc = np.column_stack([np.ones(m.sum()), np.log(v.D_cm[m]), np.log(v.H_m[m])])
    beta = np.linalg.lstsq(Xc, np.log(v.V[m]), rcond=None)[0]
    resid = np.log(v.V[m]) - Xc @ beta
    corr = np.exp(np.var(resid, ddof=3) / 2)          # Baskerville correction
    v_eq = corr * np.exp(beta[0] + beta[1] * np.log(d.dap_cm) + beta[2] * np.log(d.H_m))
    d["vol_equacao_m3"] = v_eq
    razao = (d.vol_m3[ok] / v_eq[ok])
    print(f"\n   equation ln V = {beta[0]:.3f} + {beta[1]:.3f} ln D + {beta[2]:.3f} ln H "
          f"(n={m.sum()}, R2 {1 - np.var(resid) / np.var(np.log(v.V[m])):.3f})")
    print(f"   TLS / equation, on the SAME stems: median {razao.median():.3f}, "
          f"mean {razao.mean():.3f}, sd {razao.std():.3f}")
    print(f"   dispersion of the ratio = {100 * razao.std() / razao.mean():.1f} %, and this is the "
          f"shape the equation does not see")

    # ---- 3. the stand, which is where there is a field number to compare against --------
    print("\n[3] volume per hectare, against the inventory")
    area_ha = float(np.ptp(ref[:, 0]) * np.ptp(ref[:, 1]) / 1e4)
    esc = len(d) / max(ok.sum(), 1)          # unmeasured stems enter at the mean of the measured
    vha = d.vol_m3[ok].sum() * esc / area_ha
    vha_eq = v_eq[ok].sum() * esc / area_ha
    print(f"   map area {area_ha:.3f} ha, {len(d)} stems, "
          f"{len(d) / area_ha:.0f} stems/ha")
    print(f"   TLS      {vha:7.1f} m3/ha")
    print(f"   equation {vha_eq:7.1f} m3/ha  (same DBH and H)")
    cf = pd.read_csv(config.OUT_DIR / "ff3d_crown_features.csv")
    campo = cf[(cf.talhao == 1)][["plot_id", "Vol_ha"]]
    print(f"   field    {campo.Vol_ha.mean():7.1f} m3/ha  "
          f"(plots {', '.join(campo.plot_id)}: "
          f"{', '.join(f'{u:.0f}' for u in campo.Vol_ha)})")
    k = campo.Vol_ha.mean() / vha
    print(f"   TLS -> field scale: {k:.3f}  "
          f"(the raw volume is {100 * (1 / k - 1):+.0f} % relative to the field)")
    d["vol_corrigido_m3"] = d.vol_m3 * k

    # ---- 4. where the difference comes from, split into three pieces -------------------
    # The form factor is the clean validation of the taper. V = g * H * f, so f = V / (g H) takes
    # DBH and height out of the calculation and leaves only the shape of the stem, which is exactly
    # what the taper measures. Comparing raw volume with the field mixes three errors and does not
    # separate their origin.
    print("\n[4] where the volume difference comes from")
    inv = pd.read_csv(config.DATA / "5-dados_campo/inv_euc.csv")
    i1 = inv[inv.talhao == 1]
    d_campo = float(i1.D_cm.mean())
    h_campo = float(pd.to_numeric(i1.H_m, errors="coerce").mean())
    g_tls = np.pi * (d.dap_cm / 200) ** 2
    f_tls = (d.vol_m3 / (g_tls * d.H_m))[ok]
    f_cub = v.V / (np.pi * (v.D_cm / 200) ** 2 * v.H_m)
    rd = d.dap_cm[ok].mean() / d_campo
    rh = d.H_m[ok].mean() / h_campo
    rf = float(f_tls.median()) / float(f_cub.median())
    print(f"   DBH     TLS {d.dap_cm[ok].mean():5.2f} cm against {d_campo:5.2f} from the inventory "
          f"(n={len(i1)}) -> x{rd:.3f}, and in the volume x{rd ** 2:.3f}")
    print(f"   height  drone {d.H_m[ok].mean():5.2f} m against {h_campo:5.2f} measured "
          f"-> x{rh:.3f}")
    print(f"   shape   TLS {f_tls.median():.3f} against {f_cub.median():.3f} from the scaling "
          f"(n={len(v)}) -> x{rf:.3f}")
    print(f"   product x{rd ** 2 * rh * rf:.3f}, against x{vha / campo.Vol_ha.mean():.3f} "
          f"observed in the volume per hectare")
    print(f"   only the SHAPE piece belongs to the taper. The other two were already known and "
          f"are inputs, not products of this script.")
    print(f"   dispersion of the form factor: TLS sd {f_tls.std():.3f} against {f_cub.std():.3f} "
          f"from the scaling, i.e. TLS spreads {f_tls.std() / f_cub.std():.1f} times more")
    print("   part of that dispersion is real shape and part is measurement noise, and without "
          "tree-to-tree matching the two cannot be separated.")
    d["fator_forma"] = d.vol_m3 / (g_tls * d.H_m)

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    d.to_csv(SAIDA, index=False)
    A.to_csv(SAIDA_AFIL, index=False)
    print(f"\n{SAIDA}\n{SAIDA_AFIL}")


if __name__ == "__main__":
    main()

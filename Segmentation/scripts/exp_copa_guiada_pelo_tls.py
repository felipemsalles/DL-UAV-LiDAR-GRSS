#!/usr/bin/env python3
"""The drone crown assigned by the 3D geometry the TLS sees, and not by the 2D neighbour.

`exp_por_arvore_com_809_rotulos.py` tested three crown definitions, and the best of them, the
oracle Voronoi, assigns each drone point to the nearest mapped stem in plan view. That assumes
the crown sits on top of the stem base, which does not hold: `exp_volume_tls_por_arvore.py`
measured a median displacement of 16 cm at 7 m and 26 cm at 15 m of height, so at 30 m the crown
may be half a metre off the plumb line of the base, in a plantation with 2.4 m spacing and a
crown radius of 1.8 m.

This is an even stronger oracle: if the drone crown does not tell the diameter even when the
real position and axis of each stem are known, the negative conclusion stops depending on the
quality of the segmentation. It is the ceiling of what any segmentation could deliver.

Two corrections, measured separately so it is clear which one pays:
  axis     : assigns by the stem axis at the height of the point, with the lean measured on the
             TLS and extrapolated to the canopy. It corrects where the crown is.
  envelope : divides the distance by the crown radius the TLS sees at that height, so that a
             tree with a wide crown reaches further. It corrects whose crown it is.

The envelope may have no content: a 10-year-old eucalypt self-prunes, so between 3 and 15 m, the
band the TLS sees, there may be no crown at all, only a clean stem. In that case the radius
measured there is that of the trunk. That is why the r(z) profile is printed before any model and
must be read before the conclusion.

Usage: PYTHONPATH=. python scripts/exp_copa_guiada_pelo_tls.py
Output: manual_match/copa_guiada_tls.csv and manual_match/copa_guiada_tls_envelope.csv
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

warnings.filterwarnings("ignore")


def _mod(nome, caminho):
    s = importlib.util.spec_from_file_location(nome, caminho)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_vol = _mod("volarv", R / "scripts/exp_volume_tls_por_arvore.py")
_arv = _mod("porarv", R / "scripts/exp_por_arvore_com_809_rotulos.py")

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
DRONE = config.REPO / "work/lazall/SaoManuelTotal_001.laz"
SAIDA = config.OUT_DIR / "copa_guiada_tls.csv"
SAIDA_ENV = config.OUT_DIR / "copa_guiada_tls_envelope.csv"

H_EIXO = np.array([1.3, 2.1, 3.1, 4.3, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
FAIXAS_ENV = np.arange(3.0, 20.1, 1.0)     # where the TLS can still see
R_ENV_MAX = 2.5
Z_MIN, Z_MAX = 2.0, 45.0
R_ATRIB = 3.0                              # how far a drone point can be disputed
SEED = 20260828


def eixo_dos_fustes(x, y, h, ref, r_dap):
    """Axis of each stem, from a line fitted to the ring centres measured at several heights."""
    T = cKDTree(np.column_stack([x, y]))
    eixos = np.full((len(ref), 4), np.nan)     # a0x, a1x, a0y, a1y
    desl = []
    for j, (cx, cy) in enumerate(ref):
        if not np.isfinite(r_dap[j]):
            continue
        idx = np.asarray(T.query_ball_point([cx, cy], 1.2), dtype=int)
        if len(idx) < 200:
            continue
        px, py, ph = x[idx] - cx, y[idx] - cy, h[idx]
        cu = cv = 0.0
        r_ant = r_dap[j]
        hs, us, vs = [], [], []
        for k, ha in enumerate(H_EIXO):
            meia = 0.15 + 0.012 * ha
            m = np.abs(ph - ha) <= meia
            if m.sum() < 40:
                continue
            qx, qy = px[m] - cu, py[m] - cv
            salto = ha - (hs[-1] if hs else 1.3)
            du, dv, n = _vol.centro_do_anel(qx, qy, r_ant, min(0.40, 0.08 + 0.05 * salto))
            if n < 25:
                continue
            rr, _ = _vol.raio_por_modo_limitado(
                qx - du, qy - dv, max(0.02, _vol.ENCOLHE_POR_M ** max(salto, 0.1) * r_ant),
                min(0.30, _vol.CRESCE_POR_M ** max(salto, 0.1) * r_ant), 25)
            if not np.isfinite(rr):
                continue
            cu, cv, r_ant = cu + du, cv + dv, rr
            hs.append(ha); us.append(cu); vs.append(cv)
        if len(hs) < 4:
            continue
        hs = np.asarray(hs)
        A = np.column_stack([np.ones(len(hs)), hs])
        eixos[j, :2] = np.linalg.lstsq(A, np.asarray(us), rcond=None)[0]
        eixos[j, 2:] = np.linalg.lstsq(A, np.asarray(vs), rcond=None)[0]
        desl.append((hs[-1], np.hypot(us[-1], vs[-1])))
    d = np.asarray(desl)
    if len(d):
        print(f"   axis obtained on {np.isfinite(eixos[:, 0]).sum()} stems; "
              f"median displacement of {np.median(d[:, 1]) * 100:.1f} cm "
              f"at {np.median(d[:, 0]):.0f} m", flush=True)
    return eixos


def posicao_do_eixo(eixos, z):
    """Where the axis of each stem passes at height z, with the lean extrapolated."""
    return (eixos[:, 0] + eixos[:, 1] * z, eixos[:, 2] + eixos[:, 3] * z)


def envelope_do_tls(x, y, h, ref, eixos, r_dap):
    """Radius the TLS sees around each axis, per height band.

    It may be the radius of the trunk and not of the crown; the printed profile distinguishes the
    two. If the high quantile of the radius stays in the order of centimetres, the TLS is seeing a
    clean stem and there is no crown there to inform anything.
    """
    ok = np.isfinite(eixos[:, 0]) & np.isfinite(r_dap)
    env = np.full((len(ref), len(FAIXAS_ENV) - 1), np.nan)
    linhas = []
    for i in range(len(FAIXAS_ENV) - 1):
        z0, z1 = FAIXAS_ENV[i], FAIXAS_ENV[i + 1]
        m = (h >= z0) & (h < z1)
        if m.sum() < 1000:
            continue
        zc = 0.5 * (z0 + z1)
        ex, ey = posicao_do_eixo(eixos, zc)
        val = np.isfinite(ex) & ok
        # The band is sliced once only: indexing `x[m]` inside the per-stem loop copies the whole
        # band at every iteration (800 stems x 17 bands of millions of points), which raises no
        # error but does not finish in any useful time.
        xm, ym = x[m], y[m]
        Tp = cKDTree(np.column_stack([xm, ym]))
        for j in np.where(val)[0]:
            cx, cy = ref[j, 0] + ex[j], ref[j, 1] + ey[j]
            idx = Tp.query_ball_point([cx, cy], R_ENV_MAX)
            if len(idx) < 15:
                continue
            idx = np.asarray(idx, int)
            d = np.hypot(xm[idx] - cx, ym[idx] - cy)
            env[j, i] = float(np.quantile(d, 0.90))
        v = env[:, i]
        linhas.append(dict(z=zc, n_fustes=int(np.isfinite(v).sum()),
                           raio_p90_m=float(np.nanmedian(v)),
                           n_pts_faixa=int(m.sum())))
    e = pd.DataFrame(linhas)
    print("\n   profile of what the TLS sees around the axis:")
    for _, r in e.iterrows():
        print(f"      z {r.z:5.1f} m   median p90 radius {100 * r.raio_p90_m:6.1f} cm   "
              f"({r.n_fustes:3.0f} stems)")
    return env, e


def copas_guiadas(xyz, ref, eixos, env, usa_envelope):
    """Assigns each drone point to the stem with the smallest distance to the axis at that height.

    With `usa_envelope`, the distance is divided by the radius the TLS sees at that height, so a
    tree with a wide crown reaches further. Without it, only the axis corrects.
    """
    val = np.isfinite(eixos[:, 0])
    idx_val = np.where(val)[0]
    out = {k: [] for k in idx_val}
    z = xyz[:, 2]
    bordas = np.arange(Z_MIN, Z_MAX + 1.0, 1.0)
    env_med = np.nanmedian(env, axis=1)
    env_med = np.where(np.isfinite(env_med), env_med, np.nanmedian(env_med))
    # Vectorised, and not a loop per point: walking the 3.2 M points in Python inside a
    # `query_ball_point` does not finish in any useful time. Each height band makes one k-nearest
    # neighbour query and the score comes out by broadcasting. k = 8 is enough because in a 2.4 m
    # plantation no point is disputed by more than that within 3 m.
    K = min(8, len(idx_val))
    for i in range(len(bordas) - 1):
        m = (z >= bordas[i]) & (z < bordas[i + 1])
        if not m.any():
            continue
        zc = 0.5 * (bordas[i] + bordas[i + 1])
        ex, ey = posicao_do_eixo(eixos, zc)
        cen = np.column_stack([ref[idx_val, 0] + ex[idx_val], ref[idx_val, 1] + ey[idx_val]])
        T = cKDTree(cen)
        P = xyz[m]
        d, j = T.query(P[:, :2], k=K)
        d = np.atleast_2d(d.T).T if K > 1 else d.reshape(-1, 1)
        j = np.atleast_2d(j.T).T if K > 1 else j.reshape(-1, 1)
        escore = d / np.maximum(env_med[idx_val[j]], 0.15) if usa_envelope else d
        escore = np.where(d <= R_ATRIB, escore, np.inf)
        melhor = escore.argmin(1)
        linha = np.arange(len(P))
        vencedor = j[linha, melhor]
        valido = np.isfinite(escore[linha, melhor])
        pos = np.where(m)[0]
        for k_local in np.unique(vencedor[valido]):
            sel = valido & (vencedor == k_local)
            out[idx_val[k_local]].extend(pos[sel].tolist())
    return {k: xyz[np.asarray(v, int)] for k, v in out.items() if len(v) >= _arv.PTS_MIN}


def main():
    rng = np.random.default_rng(SEED)
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    ran = pd.read_csv(config.OUT_DIR / "dap_tls_ransac_talhao001.csv")
    r_dap = np.where(ran.motivo.values == "ok", ran.dap_cm.values / 200.0, np.nan)
    volt = pd.read_csv(config.OUT_DIR / "volume_tls_por_arvore.csv")
    rot = pd.DataFrame({"x": ref[:, 0], "y": ref[:, 1], "dap_cm": r_dap * 200,
                        "vol_m3": volt.vol_m3.values, "H_m": volt.H_m.values})

    print("reading the TLS and tracking the axis of each stem", flush=True)
    x, y, h = _vol.le_tls()
    eixos = eixo_dos_fustes(x, y, h, ref, r_dap)
    env, perfil = envelope_do_tls(x, y, h, ref, eixos, r_dap)
    del x, y, h

    las = laspy.read(str(DRONE))
    cls = np.asarray(las.classification)
    m = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    xyz = np.column_stack([np.asarray(las.x)[m], np.asarray(las.y)[m], np.asarray(las.z)[m]])
    xyz = xyz[(xyz[:, 2] >= Z_MIN) & (xyz[:, 2] <= Z_MAX)]
    print(f"\n{len(xyz):,} canopy points from the drone", flush=True)

    # Fewer shufflings here by scope: the control is already established in the sibling experiment
    # for the same features and the same harness, and what this script decides is the comparison
    # between crown definitions. Eight permutations are enough to separate zero from signal.
    _arv.N_EMB = 8
    alvos = [("dap_cm", "DBH (cm), TLS only"), ("vol_m3", "volume (m3)"),
             ("controle", "positive control")]
    conjuntos = {
        "2D Voronoi at the base": enumerate(_arv.copas_voronoi(xyz, ref, _arv.R_VORONOI)),
        "3D TLS axis": iter(copas_guiadas(xyz, ref, eixos, env, False).items()),
        "3D axis + TLS envelope": iter(copas_guiadas(xyz, ref, eixos, env, True).items()),
    }
    saida, tabelas = [], {}
    for nome, ger in conjuntos.items():
        F = _arv.tabela(ger, ref, len(ref))
        F = F.join(rot).dropna(subset=["dap_cm", "vol_m3"])
        z = (F.zmax - F.zmax.mean()) / F.zmax.std()
        a = (F.area_copa - F.area_copa.mean()) / F.area_copa.std()
        F["controle"] = 10 + 2.0 * z + 1.5 * a + rng.normal(0, 1.0, len(F))
        F = _arv.competicao(F, ref)
        tabelas[nome] = F
        saida += _arv.roda(nome, F, alvos, rng)

    comum = sorted(set.intersection(*(set(F.index) for F in tabelas.values())))
    print(f"\n{'=' * 78}\nPAIRED on the {len(comum)} stems common to the three definitions")
    for nome, F in tabelas.items():
        saida += _arv.roda(f"{nome} [paired]", F.loc[comum], alvos[:2], rng)

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(saida).to_csv(SAIDA, index=False)
    perfil.to_csv(SAIDA_ENV, index=False)
    print(f"\n{SAIDA}\n{SAIDA_ENV}")


if __name__ == "__main__":
    main()

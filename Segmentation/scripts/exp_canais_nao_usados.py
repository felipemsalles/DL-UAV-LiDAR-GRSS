#!/usr/bin/env python3
"""Is the limit having used geometry alone? Tests intensity, return structure and RGB.

The per-tree experiments in this repository use exclusively x, y and z. The delivered cloud
carries other channels that they leave untouched:

  intensity            reflectance of the target. Bark, new leaves and old leaves reflect
                       differently, and vigour is linked to growth and to diameter.
  return number        how many echoes the pulse generated and which one this is. A direct
                       measure of penetration, that is, of how empty the crown is, a structure
                       that point geometry alone expresses poorly.
  red, green, blue     a yellowish or pale crown signals a stressed or suppressed tree, and a
                       suppressed tree is thin.

The hypothesis is that, if the bottleneck is saturation of crown size, colour and intensity may
still separate a vigorous tree from a suppressed one among crowns of the same size.

A ladder of feature sets, so any gain can be attributed to a channel: geometry alone (the baseline
already measured), each channel in isolation, and everything together.

Intensity varies with range, angle and flight strip, so the absolute value carries acquisition
geometry and not a property of the tree. That is why the versions relative to the local
neighbourhood, which cancel the strip, also enter.

Validation by spatial block, with a shuffling control and a positive control.

Usage: PYTHONPATH=. python scripts/exp_canais_nao_usados.py
Output: manual_match/canais_nao_usados.csv
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from greenvista import config  # noqa: E402


def _mod(n, c):
    s = importlib.util.spec_from_file_location(n, c)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_rel = _mod("rel", R / "scripts/exp_relacional_vizinhanca.py")
warnings.filterwarnings("ignore")

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
DRONE = config.REPO / "work/lazall/SaoManuelTotal_001.laz"
SAIDA = config.OUT_DIR / "canais_nao_usados.csv"
R_VOR = 2.5
Z_MIN, Z_MAX = 2.0, 45.0
PTS_MIN = 40
N_JOBS = 8
N_EMB = 8
SEED = 20260828


def por_copa(ref, D):
    """Intensity, return-structure and colour features, per crown (oracle Voronoi)."""
    las = laspy.read(str(DRONE))
    cls = np.asarray(las.classification)
    m = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    xyz = np.column_stack([np.asarray(las.x)[m], np.asarray(las.y)[m], np.asarray(las.z)[m]])
    inten = np.asarray(las.intensity)[m].astype(float)
    rn = np.asarray(las.return_number)[m].astype(float)
    nr = np.asarray(las.number_of_returns)[m].astype(float)
    tem_cor = all(c in las.point_format.dimension_names for c in ("red", "green", "blue"))
    if tem_cor:
        rgb = np.column_stack([np.asarray(las.red)[m], np.asarray(las.green)[m],
                               np.asarray(las.blue)[m]]).astype(float)
    k = (xyz[:, 2] >= Z_MIN) & (xyz[:, 2] <= Z_MAX)
    xyz, inten, rn, nr = xyz[k], inten[k], rn[k], nr[k]
    if tem_cor:
        rgb = rgb[k]
    print(f"   {len(xyz):,} points | colour available: {tem_cor} | "
          f"intensity {inten.min():.0f} to {inten.max():.0f} | "
          f"returns per pulse up to {nr.max():.0f}")

    _, j = cKDTree(ref).query(xyz[:, :2], distance_upper_bound=R_VOR)
    o = np.argsort(j, kind="stable")
    j_o = j[o]
    corte = np.searchsorted(j_o, np.arange(len(ref) + 1))
    inten_o, rn_o, nr_o, z_o = inten[o], rn[o], nr[o], xyz[o, 2]
    rgb_o = rgb[o] if tem_cor else None

    linhas, idx = [], []
    for kk in range(len(ref)):
        a, b = corte[kk], corte[kk + 1]
        if b - a < PTS_MIN:
            continue
        I, Rn, Nr, Z = inten_o[a:b], rn_o[a:b], nr_o[a:b], z_o[a:b]
        f = {
            # intensity
            "int_med": I.mean(), "int_sd": I.std(), "int_p10": np.quantile(I, .1),
            "int_p90": np.quantile(I, .9),
            # intensity of the crown top only, the part the pulse reaches first and where
            # reflectance is least contaminated by the path inside the canopy
            "int_topo": I[Z > np.quantile(Z, .8)].mean(),
            # return structure: penetration
            "ret_prim": float(np.mean(Rn == 1)), "ret_ult": float(np.mean(Rn == Nr)),
            "ret_unico": float(np.mean(Nr == 1)), "ret_nmed": Nr.mean(),
            "ret_prof": float(np.mean(Rn >= 3)),
        }
        if tem_cor:
            C = rgb_o[a:b]
            soma = C.sum(1) + 1e-9
            f.update(vermelho=C[:, 0].mean(), verde=C[:, 1].mean(), azul=C[:, 2].mean(),
                     # normalised colour indices, which cancel overall illumination
                     gcc=float(np.mean(C[:, 1] / soma)), rcc=float(np.mean(C[:, 0] / soma)),
                     exg=float(np.mean(2 * C[:, 1] - C[:, 0] - C[:, 2])),
                     brilho=float(np.mean(soma)))
        linhas.append(f)
        idx.append(kk)
    F = pd.DataFrame(linhas, index=idx)
    return F.loc[[i for i in F.index if i in D.index]]


def relativas(F, D, k=6):
    """Versions RELATIVE to the neighbourhood, which cancel flight strip and illumination."""
    xy = D.loc[F.index, ["x", "y"]].to_numpy(float)
    _, j = cKDTree(xy).query(xy, k=min(k + 1, len(xy)))
    j = j[:, 1:]
    out = {}
    for c in F.columns:
        v = F[c].to_numpy(float)
        out[c + "_rel"] = v - v[j].mean(1)
    return pd.DataFrame(out, index=F.index)


def main():
    rng = np.random.default_rng(SEED)
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    D = pd.read_csv(config.OUT_DIR / "por_arvore_pred_voronoi.csv", index_col=0)
    print("extracting the unused channels")
    F = por_copa(ref, D)
    REL = relativas(F, D)
    D = D.loc[F.index]
    PROP, AGREG, _, _ = _rel.blocos_de_feicoes(D)
    blocos = _rel.blocos_de(D.x.to_numpy(float))
    print(f"   {len(D)} trees, {F.shape[1]} new channels (+{REL.shape[1]} relative)\n")

    z = (D.zmax - D.zmax.mean()) / D.zmax.std()
    a = (D.area_copa - D.area_copa.mean()) / D.area_copa.std()
    D["controle"] = 10 + 2.0 * z + 1.5 * a + rng.normal(0, 1.0, len(D))

    inten = [c for c in F.columns if c.startswith("int_")]
    ret = [c for c in F.columns if c.startswith("ret_")]
    cor = [c for c in F.columns if c not in inten + ret]
    print("raw correlation of each channel with the TLS DBH:")
    for c in list(F.columns):
        r = float(np.corrcoef(F[c], D.dap_cm)[0, 1])
        rr = float(np.corrcoef(REL[c + "_rel"], D.dap_cm)[0, 1])
        print(f"   {c:12s} absolute {r:+.3f}   relative to neighbourhood {rr:+.3f}")

    conj = {
        "geometry (baseline)": PROP,
        "geometry + intensity": pd.concat([PROP, F[inten], REL[[c + "_rel" for c in inten]]], axis=1),
        "geometry + return structure": pd.concat([PROP, F[ret]], axis=1),
        "geometry + colour": pd.concat([PROP, F[cor], REL[[c + "_rel" for c in cor]]], axis=1),
        "geometry + ALL the channels": pd.concat([PROP, F, REL], axis=1),
        "geometry + neighbourhood + ALL": pd.concat([PROP, AGREG, F, REL], axis=1),
        "ONLY the new channels": pd.concat([F, REL], axis=1),
    }
    mods = {"forest": lambda s: RandomForestRegressor(
                500, min_samples_leaf=3, max_features=0.4, random_state=s, n_jobs=N_JOBS),
            "ridge": lambda s: make_pipeline(StandardScaler(),
                                             RidgeCV(alphas=np.logspace(-3, 4, 20)))}
    saida = []
    for alvo, rot in (("dap_cm", "DBH (cm)"), ("vol_m3", "volume (m3)"),
                      ("controle", "positive control")):
        y = D[alvo].to_numpy(float)
        print(f"\n=== {rot}   mean {y.mean():.3f}  sd {y.std(ddof=1):.3f}")
        for nome, X in conj.items():
            for mn, mk in mods.items():
                pred = np.full(len(y), np.nan)
                for tr, te in blocos:
                    m = mk(0)
                    m.fit(X.iloc[tr], y[tr])
                    pred[te] = np.asarray(m.predict(X.iloc[te])).ravel()
                r2, rr = _rel.marca(y, pred)
                embs = []
                for s in range(N_EMB if mn == "ridge" else 3):
                    ys = y[np.random.default_rng(s).permutation(len(y))]
                    pe = np.full(len(y), np.nan)
                    for tr, te in blocos:
                        m = mk(s)
                        m.fit(X.iloc[tr], ys[tr])
                        pe[te] = np.asarray(m.predict(X.iloc[te])).ravel()
                    embs.append(_rel.marca(ys, pe)[0])
                print(f"   {nome:34s} {mn:9s} R2 {r2:+.3f}  rRMSE {rr:5.1f} %"
                      f"   (shuffled {np.mean(embs):+.3f})", flush=True)
                saida.append(dict(alvo=alvo, conjunto=nome, modelo=mn, R2=r2, rRMSE=rr,
                                  R2_embaralhado=float(np.mean(embs)), n=len(D),
                                  n_colunas=X.shape[1]))
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(saida).to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

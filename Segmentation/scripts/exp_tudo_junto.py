#!/usr/bin/env python3
"""All the remaining feature blocks together, to find the ceiling of the data we have.

After exhausting the routes one by one, each yielded little on its own: crown +0.098, plus
neighbourhood +0.118, plus oblique stem +0.110. What remains is to find out whether these are the
same gains counted three times or whether they add up. This script puts everything together and
measures the ceiling.

More features is not more signal: with 808 trees and a hundred or so columns, a random forest has
plenty of room to memorise. The shuffling control is what separates a real gain from a fit to noise,
and it is on every line.

Run: PYTHONPATH=. python scripts/exp_tudo_junto.py
Out: manual_match/tudo_junto.csv
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
SAIDA = config.OUT_DIR / "tudo_junto.csv"
MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
DRONE = config.REPO / "work/lazall/SaoManuelTotal_001.laz"
SEED = 20260828
N_EMB = 8


def feicoes_de_peito(ref):
    """Return count and return concentration at breast height, by scan-angle band."""
    las = laspy.read(str(DRONE))
    cls = np.asarray(las.classification)
    veg = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    x, y, z = np.asarray(las.x)[veg], np.asarray(las.y)[veg], np.asarray(las.z)[veg]
    ang = np.abs(np.asarray(las.scan_angle_rank).astype(float))[veg]
    out = {}
    for a0, a1, tag in [(0, 15, "nad"), (20, 36, "obl")]:
        m = (z >= 1.0) & (z <= 6.0) & (ang >= a0) & (ang < a1)
        T = cKDTree(np.column_stack([x[m], y[m]]))
        for raio in (0.30, 0.60):
            n = np.array([len(T.query_ball_point(p, raio)) for p in ref], float)
            out[f"{tag}_n{int(raio * 100)}"] = n
        Xm, Ym = x[m], y[m]
        conc = np.full(len(ref), np.nan)
        for i, (cx, cy) in enumerate(ref):
            idx = T.query_ball_point([cx, cy], 0.60)
            if len(idx) >= 5:
                d = np.hypot(Xm[np.asarray(idx, int)] - cx, Ym[np.asarray(idx, int)] - cy)
                conc[i] = np.mean(d < 0.20) / ((0.20 / 0.60) ** 2)
        out[f"{tag}_conc"] = conc
    return pd.DataFrame(out)


def main():
    rng = np.random.default_rng(SEED)
    D = pd.read_csv(config.OUT_DIR / "por_arvore_pred_voronoi.csv", index_col=0)
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])

    PROP, AGREG, DESD, _ = _rel.blocos_de_feicoes(D)
    PEITO = feicoes_de_peito(ref)
    PEITO.index = np.arange(len(ref))
    PEITO = PEITO.loc[D.index]
    for c in PEITO.columns:
        PEITO[c] = PEITO[c].fillna(PEITO[c].median())

    blocos = _rel.blocos_de(D.x.to_numpy(float))
    z = (D.zmax - D.zmax.mean()) / D.zmax.std()
    a = (D.area_copa - D.area_copa.mean()) / D.area_copa.std()
    D["controle"] = 10 + 2.0 * z + 1.5 * a + rng.normal(0, 1.0, len(D))
    print(f"{len(D)} trees | crown {PROP.shape[1]} | neighbourhood {AGREG.shape[1]} | "
          f"breast height {PEITO.shape[1]} columns")

    conj = {
        "copa": PROP,
        "copa + peito": pd.concat([PROP, PEITO], axis=1),
        "copa + vizinhanca": pd.concat([PROP, AGREG], axis=1),
        "copa + vizinhanca + peito": pd.concat([PROP, AGREG, PEITO], axis=1),
        "TUDO (com desdobrado)": pd.concat([PROP, AGREG, DESD, PEITO], axis=1),
    }
    mods = {"floresta": lambda s: RandomForestRegressor(
                600, min_samples_leaf=3, max_features=0.35, random_state=s, n_jobs=-1),
            "ridge": lambda s: make_pipeline(StandardScaler(),
                                             RidgeCV(alphas=np.logspace(-3, 4, 25)))}
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
                print(f"   {nome:28s} {mn:9s} R2 {r2:+.3f}  rRMSE {rr:5.1f} %"
                      f"   (shuffled {np.mean(embs):+.3f})", flush=True)
                saida.append(dict(alvo=alvo, conjunto=nome, modelo=mn, R2=r2, rRMSE=rr,
                                  R2_embaralhado=float(np.mean(embs)), n=len(D),
                                  n_colunas=X.shape[1]))
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(saida).to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

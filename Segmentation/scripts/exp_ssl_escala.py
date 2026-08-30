#!/usr/bin/env python3
"""How much more does pre-training yield with more unlabelled crowns? And does it add to the features?

`exp_ssl_e_destilacao.py` found the only route that worked: self-supervised pre-training on 12,000
unlabelled crowns takes the per-tree R2 from 0.066 to 0.149, and the gain survives removing the test
stand from the corpus. Two questions with direct practical consequences remain.

1. The scaling curve. 12,000 of the 13,946 detected crowns were used, against the 500,000 of the
Forestry 2026 paper. If R2 is still rising with corpus size, that supports flying more area, which is
cheap because it involves no fieldwork at all; if it has already saturated, the recommendation
changes.

2. Is the representation complementary to the features? Pre-training gives 0.149 and the hand-made
features with all channels give 0.138. If the two carry the same thing, combining them does not pass
0.15; if they carry different things, the sum passes both.

The corpus never includes stand 001, at any size. It is already measured that including it changes
little (0.154 against 0.149), but varying two things at once would spoil the curve.

Each point of the curve is three seeds, because the difference between neighbouring corpus sizes is
smaller than the variation between seeds at a single point.

Run: PYTHONPATH=. python scripts/exp_ssl_escala.py
Out: manual_match/ssl_escala.csv
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from greenvista import config  # noqa: E402


def _mod(n, c):
    s = importlib.util.spec_from_file_location(n, c)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_rel = _mod("rel", R / "scripts/exp_relacional_vizinhanca.py")
_ssl = _mod("ssl", R / "scripts/exp_ssl_e_destilacao.py")
warnings.filterwarnings("ignore")

SAIDA = config.OUT_DIR / "ssl_escala.csv"
MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
LAZ = config.REPO / "work/lazall"
DET = config.REPO / "data/detections/sat_w2w_arvores.csv"
TAMANHOS = (500, 1500, 4000, 8000, 12000)
SEEDS = (0, 1, 2)
SEED = 20260829


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    D = pd.read_csv(config.OUT_DIR / "por_arvore_pred_voronoi.csv", index_col=0)

    # labelled crowns of stand 001
    las = laspy.read(str(LAZ / "SaoManuelTotal_001.laz"))
    cls = np.asarray(las.classification)
    m = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    xyz = np.column_stack([np.asarray(las.x)[m], np.asarray(las.y)[m], np.asarray(las.z)[m]])
    xyz = xyz[(xyz[:, 2] >= _ssl.Z_MIN) & (xyz[:, 2] <= _ssl.Z_MAX)]
    _, j = cKDTree(ref).query(xyz[:, :2], distance_upper_bound=_ssl.R_VOR)
    o = np.argsort(j, kind="stable")
    j_o, xyz_o = j[o], xyz[o]
    corte = np.searchsorted(j_o, np.arange(len(ref) + 1))
    P_lab, ks = {}, []
    for k in range(len(ref)):
        p = xyz_o[corte[k]:corte[k + 1]]
        if len(p) >= _ssl.PTS_MIN and k in D.index:
            q = p.copy()
            q[:, :2] -= ref[k]
            P_lab[k] = q.astype(np.float32)
            ks.append(k)
    P = [P_lab[k] for k in ks]
    Dk = D.loc[ks]
    y = Dk.dap_cm.to_numpy(float)
    blocos = _rel.blocos_de(Dk.x.to_numpy(float))
    print(f"device {dev} | {len(ks)} labelled trees\n")

    # corpus without stand 001
    det = pd.read_csv(DET)
    corpus = []
    for t in sorted(det.talhao.unique()):
        if int(t) == 1:
            continue
        laz = LAZ / f"SaoManuelTotal_{int(t):03d}.laz"
        if not laz.exists():
            continue
        s = det[det.talhao == t][["base_x", "base_y"]].to_numpy(float)
        corpus += _ssl.copas_do_talhao(laz, s)
    print(f"corpus available without stand 001: {len(corpus)} crowns\n", flush=True)

    saida, embs_por_tam = [], {}
    print("[1] corpus scaling curve")
    for n in TAMANHOS:
        if n > len(corpus):
            continue
        sub = [corpus[i] for i in rng.choice(len(corpus), n, replace=False)]
        r2s = []
        for s in SEEDS:
            tronco = _ssl.pre_treina(sub, s, dev, epocas=40)
            pred = np.full(len(y), np.nan)
            for tr, te in blocos:
                pred[te] = _ssl.ajusta(P, y, tr, te, s, dev, tronco)
            r2s.append(_rel.marca(y, pred)[0])
            if s == 0:
                embs_por_tam[n] = tronco
        print(f"   corpus of {n:6d} crowns   R2 {np.mean(r2s):+.3f} "
              f"(sd across seeds {np.std(r2s):.3f})", flush=True)
        saida.append(dict(teste="escala", n_corpus=n, R2=float(np.mean(r2s)),
                          R2_dp=float(np.std(r2s))))

    # ---- 2. does the representation add to the features? ---------------------------------
    print("\n[2] does the learned representation add to the hand-made features?")
    tronco = embs_por_tam[max(embs_por_tam)]
    net = _ssl.Rede(1).to(dev)
    net.mlp.load_state_dict({k: v.to(dev) for k, v in tronco.items()})
    net.eval()
    rr = np.random.default_rng(0)
    with torch.no_grad():
        E = np.zeros((len(P), 256), np.float32)
        for _ in range(8):
            X = torch.tensor(np.stack([_ssl.amostra(p, rr) for p in P]), device=dev)
            E += net.emb(X).cpu().numpy()
    E = E / 8
    from sklearn.decomposition import PCA
    EMB = pd.DataFrame(PCA(32, random_state=0).fit_transform(E),
                       index=Dk.index, columns=[f"emb{i}" for i in range(32)])
    PROP, AGREG, _, _ = _rel.blocos_de_feicoes(Dk)
    conj = {"so feicoes (copa + vizinhanca)": pd.concat([PROP, AGREG], axis=1),
            "so a representacao pre-treinada": EMB,
            "feicoes + representacao": pd.concat([PROP, AGREG, EMB], axis=1)}
    for nome, X in conj.items():
        pred = np.full(len(y), np.nan)
        for tr, te in blocos:
            mm = RandomForestRegressor(500, min_samples_leaf=3, max_features=0.4,
                                       random_state=0, n_jobs=8)
            mm.fit(X.iloc[tr], y[tr])
            pred[te] = mm.predict(X.iloc[te])
        r2, rrm = _rel.marca(y, pred)
        embs = []
        for s in range(3):
            ys = y[np.random.default_rng(s).permutation(len(y))]
            pe = np.full(len(y), np.nan)
            for tr, te in blocos:
                mm = RandomForestRegressor(500, min_samples_leaf=3, max_features=0.4,
                                           random_state=s, n_jobs=8)
                mm.fit(X.iloc[tr], ys[tr])
                pe[te] = mm.predict(X.iloc[te])
            embs.append(_rel.marca(ys, pe)[0])
        print(f"   {nome:34s} R2 {r2:+.3f}  rRMSE {rrm:5.1f} %  "
              f"(shuffled {np.mean(embs):+.3f})", flush=True)
        saida.append(dict(teste="soma", conjunto=nome, R2=r2, rRMSE=rrm,
                          R2_embaralhado=float(np.mean(embs))))

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(saida).to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

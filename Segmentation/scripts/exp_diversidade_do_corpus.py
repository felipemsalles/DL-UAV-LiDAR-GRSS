#!/usr/bin/env python3
"""Diversity or size of the pre-training corpus, with enough statistical power.

`exp_professor_melhor_e_diversidade.py` ran this test with three seeds and did not decide: a corpus
concentrated in one stand gave R2 +0.112 and one spread over five gave +0.128, a difference of +0.016
with a standard error of 0.011 (1.5 standard errors, 95 % interval from -0.006 to +0.038, containing
zero).

The hypothesis under test is that the scaling curve saturates at 8,000 crowns for lack of variety and
not of size, since the crowns all come from the same clone and the same management.

Ten seeds per condition: with ten the standard error of the difference drops to about 0.006, and the
same difference of +0.016, if real, reaches almost three standard errors.

The corpus size is identical in both arms, 4,000 crowns; only the origin changes, 4,000 from one stand
against 800 from each of five. Varying the size as well would invalidate the test.

Usage: PYTHONPATH=. python scripts/exp_diversidade_do_corpus.py
Output: manual_match/diversidade_do_corpus.csv
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

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
LAZ = config.REPO / "work/lazall"
DET = config.REPO / "data/detections/sat_w2w_arvores.csv"
SAIDA = config.OUT_DIR / "diversidade_do_corpus.csv"
N_CORPUS = 4000
N_SEEDS = 10
SEED = 20260829


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    D = pd.read_csv(config.OUT_DIR / "por_arvore_pred_voronoi.csv", index_col=0)

    las = laspy.read(str(LAZ / "SaoManuelTotal_001.laz"))
    cls = np.asarray(las.classification)
    m = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    xyz = np.column_stack([np.asarray(las.x)[m], np.asarray(las.y)[m], np.asarray(las.z)[m]])
    xyz = xyz[(xyz[:, 2] >= _ssl.Z_MIN) & (xyz[:, 2] <= _ssl.Z_MAX)]
    _, j = cKDTree(ref).query(xyz[:, :2], distance_upper_bound=_ssl.R_VOR)
    o = np.argsort(j, kind="stable")
    j_o, xyz_o = j[o], xyz[o]
    corte = np.searchsorted(j_o, np.arange(len(ref) + 1))
    ks, P = [], []
    for k in range(len(ref)):
        p = xyz_o[corte[k]:corte[k + 1]]
        if len(p) >= _ssl.PTS_MIN and k in D.index:
            q = p.copy()
            q[:, :2] -= ref[k]
            P.append(q.astype(np.float32))
            ks.append(k)
    Dk = D.loc[ks]
    y = Dk.dap_cm.to_numpy(float)
    blocos = _rel.blocos_de(Dk.x.to_numpy(float))
    print(f"device {dev} | {len(ks)} labelled trees\n")

    det = pd.read_csv(DET)
    por_talhao = {}
    for t in sorted(det.talhao.unique()):
        if int(t) == 1:
            continue
        laz = LAZ / f"SaoManuelTotal_{int(t):03d}.laz"
        if laz.exists():
            s = det[det.talhao == t][["base_x", "base_y"]].to_numpy(float)
            por_talhao[int(t)] = _ssl.copas_do_talhao(laz, s)
    maior = max(por_talhao, key=lambda t: len(por_talhao[t]))
    print(f"the concentrated corpus comes from stand {maior} ({len(por_talhao[maior])} crowns)")
    print(f"the spread corpus comes from {len(por_talhao)} stands\n", flush=True)

    linhas = []
    res = {}
    # The two arm labels stay in Portuguese: `nome.startswith("concentrado")` below selects on them.
    for nome in ("concentrado (1 talhao)", f"espalhado ({len(por_talhao)} talhoes)"):
        r2s = []
        for s in range(N_SEEDS):
            rs = np.random.default_rng(1000 + s)
            # The corpus draw changes with the seed, and not only the network initialisation: with a
            # fixed corpus the ten seeds would measure only the training variation and the interval
            # would come out far too narrow, ignoring the variation of which crowns entered.
            if nome.startswith("concentrado"):
                c = por_talhao[maior]
                corpus = [c[i] for i in rs.choice(len(c), min(N_CORPUS, len(c)), False)]
            else:
                corpus, por_t = [], N_CORPUS // len(por_talhao)
                for c in por_talhao.values():
                    corpus += [c[i] for i in rs.choice(len(c), min(por_t, len(c)), False)]
            tronco = _ssl.pre_treina(corpus, s, dev, epocas=40)
            pred = np.full(len(y), np.nan)
            for tr, te in blocos:
                pred[te] = _ssl.ajusta(P, y, tr, te, s, dev, tronco)
            r2 = _rel.marca(y, pred)[0]
            r2s.append(r2)
            print(f"   {nome:28s} seed {s}  R2 {r2:+.3f}", flush=True)
            linhas.append(dict(corpus=nome, semente=s, R2=r2, n_corpus=len(corpus)))
        res[nome] = np.array(r2s)
        print(f"   {nome:28s} MEAN {np.mean(r2s):+.3f} (sd {np.std(r2s, ddof=1):.3f})\n",
              flush=True)

    a, b = list(res.values())
    d = b.mean() - a.mean()
    ep = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    from scipy import stats
    t, pv = stats.ttest_ind(b, a, equal_var=False)
    print("=" * 70)
    print(f"difference (spread minus concentrated): {d:+.4f}")
    print(f"standard error {ep:.4f}, that is {d/ep:.1f} standard errors")
    print(f"95 % interval: {d - 1.96*ep:+.4f} to {d + 1.96*ep:+.4f}")
    print(f"Welch t test: t = {t:.2f}, p = {pv:.4f}")
    if pv < 0.05:
        print("\nDIVERSITY MATTERS. The claim made without measuring is supported.")
    else:
        print("\nNO CLAIM CAN BE MADE. The saturation was said to come from lack of variety,")
        print("   and with ten seeds per arm that remains undemonstrated.")
    linhas.append(dict(corpus="RESUMO", diferenca=float(d), erro_padrao=float(ep),
                       t=float(t), p_valor=float(pv), n_sementes=N_SEEDS))
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

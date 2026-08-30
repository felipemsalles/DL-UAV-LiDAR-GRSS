#!/usr/bin/env python3
"""Gaussian-process head over the point cloud, in the design of Xi and Hopkinson 2024.

Xi and Hopkinson 2024 (ISPRS) calibrate TLS to ALS in a supervised way, with a transformer to
segment and a deep network with a Gaussian-process layer to estimate DBH, and report an RMSE of
18.9 %. The design is the same as here, TLS as label and airborne as predictor, and ours stands
at 24.1 %. The architectural difference they highlight is the Gaussian-process head, which
replaces a point output by a posterior and regularises with little data.

This does not reproduce their paper: the data are different and their forest is montane and
mixed. What is tested is whether the central idea, learning a representation and regressing with
a Gaussian process on top of it, yields more than the linear head already in use.

Three levels, to locate the origin of any gain:
  1. Gaussian process directly on the hand-made features;
  2. PointNet with a linear head, the baseline already measured (R2 +0.072);
  3. PointNet as an extractor and a Gaussian process over the learned representation, analogous
     to their design.
If step 3 does not beat step 2, the head is not the separating factor.

The Gaussian process is fitted on the training part of each fold only, together with the network;
fitting the kernel on all the data is leakage.

Usage: PYTHONPATH=. python scripts/exp_cabeca_gp.py
Output: manual_match/cabeca_gp.csv
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
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from torch import nn

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from greenvista import config  # noqa: E402
from greenvista.per_tree_dap.models import PointNetReg  # noqa: E402


def _mod(n, c):
    s = importlib.util.spec_from_file_location(n, c)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_pn = _mod("pn", R / "scripts/exp_por_arvore_pointnet.py")
_rel = _mod("rel", R / "scripts/exp_relacional_vizinhanca.py")
warnings.filterwarnings("ignore")

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
DRONE = config.REPO / "work/lazall/SaoManuelTotal_001.laz"
SAIDA = config.OUT_DIR / "cabeca_gp.csv"
SEEDS = (0, 1, 2)
N_COMP = 24                 # dimensions that enter the Gaussian process


def faz_gp(n_dim):
    # The anisotropic kernel must have the effective dimension of the data, and not the constant
    # N_COMP: with the hand-made features the PCA returns 21 components and not 24, and sklearn
    # raises ValueError.
    k = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_dim)) + WhiteKernel(0.1)
    return GaussianProcessRegressor(kernel=k, normalize_y=True, alpha=1e-6,
                                    n_restarts_optimizer=1, random_state=0)


def gp_em(Xtr, ytr, Xte):
    """Gaussian process with dimensionality reduction fitted on the training set only."""
    sc = StandardScaler().fit(Xtr)
    nc = min(N_COMP, Xtr.shape[1], Xtr.shape[0] - 1)
    pca = PCA(n_components=nc, random_state=0).fit(sc.transform(Xtr))
    Ztr = pca.transform(sc.transform(Xtr))
    g = faz_gp(Ztr.shape[1]).fit(Ztr, ytr)
    return g.predict(pca.transform(sc.transform(Xte)))


class ComEmb(nn.Module):
    """PointNet that also returns the global representation, so the GP can act on top of it."""

    def __init__(self):
        super().__init__()
        self.net = PointNetReg()

    def emb(self, x):
        f = self.net.mlp(x.transpose(1, 2))
        return f.max(dim=2).values

    def forward(self, x):
        return self.net.head(self.emb(x)).squeeze(-1)


def treina(tr_p, tr_c, tr_y, te_p, te_c, seed, dev, epocas=120, lote=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = ComEmb().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    mu, sd = float(np.mean(tr_y)), float(np.std(tr_y) + 1e-9)
    yz = torch.tensor((tr_y - mu) / sd, dtype=torch.float32, device=dev)
    perda = nn.SmoothL1Loss()
    net.train()
    for _ in range(epocas):
        ordem = rng.permutation(len(tr_p))
        for i in range(0, len(ordem) - lote + 1, lote):
            b = ordem[i:i + lote]
            X = torch.tensor(np.stack([_pn.amostra(tr_p[k], tr_c[k], rng, True) for k in b]),
                             device=dev)
            opt.zero_grad()
            perda(net(X), yz[b]).backward()
            opt.step()
    net.eval()

    def saida(P, C, quantas=6):
        pred = np.zeros(len(P))
        emb = np.zeros((len(P), 256))
        with torch.no_grad():
            for _ in range(quantas):
                X = torch.tensor(np.stack([_pn.amostra(p, c, rng, True)
                                           for p, c in zip(P, C)]), device=dev)
                pred += net(X).cpu().numpy()
                emb += net.emb(X).cpu().numpy()
        return pred / quantas * sd + mu, emb / quantas

    p_tr, e_tr = saida(tr_p, tr_c)
    p_te, e_te = saida(te_p, te_c)
    return p_te, e_tr, e_te


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    D = pd.read_csv(config.OUT_DIR / "por_arvore_pred_voronoi.csv", index_col=0)

    las = laspy.read(str(DRONE))
    cls = np.asarray(las.classification)
    m = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    xyz = np.column_stack([np.asarray(las.x)[m], np.asarray(las.y)[m], np.asarray(las.z)[m]])
    C = _pn.copas(xyz, ref, _pn.R_VORONOI)
    ks = [k for k in C if k in D.index]
    P = [C[k] for k in ks]
    Cn = [ref[k] for k in ks]
    Dk = D.loc[ks]
    blocos = _rel.blocos_de(Dk.x.to_numpy(float))
    print(f"device {dev} | {len(ks)} trees | blocks {[len(t) for _, t in blocos]}")

    FEITAS = _rel.PROPRIO
    saida = []
    for alvo, rot in (("dap_cm", "DBH (cm)"), ("vol_m3", "volume (m3)")):
        y = Dk[alvo].to_numpy(float)
        print(f"\n=== {rot}   mean {y.mean():.3f}  sd {y.std(ddof=1):.3f}")

        # 1. Gaussian process on the hand-made features
        pred = np.full(len(y), np.nan)
        for tr, te in blocos:
            pred[te] = gp_em(Dk[FEITAS].values[tr], y[tr], Dk[FEITAS].values[te])
        r2, rr = _rel.marca(y, pred)
        print(f"   {'GP on the hand-made features':38s} R2 {r2:+.3f}  rRMSE {rr:5.1f} %")
        saida.append(dict(alvo=alvo, modelo="GP on the features", R2=r2, rRMSE=rr))

        # 2 and 3. PointNet with a linear head and with a GP head over the representation
        for embaralha in (False, True):
            yy = y if not embaralha else np.random.default_rng(7).permutation(y)
            r2_lin, r2_gp = [], []
            for s in SEEDS:
                p_lin = np.full(len(yy), np.nan)
                p_gp = np.full(len(yy), np.nan)
                for tr, te in blocos:
                    pt, e_tr, e_te = treina([P[i] for i in tr], [Cn[i] for i in tr], yy[tr],
                                            [P[i] for i in te], [Cn[i] for i in te], s, dev)
                    p_lin[te] = pt
                    p_gp[te] = gp_em(e_tr, yy[tr], e_te)
                r2_lin.append(_rel.marca(yy, p_lin)[0])
                r2_gp.append(_rel.marca(yy, p_gp)[0])
            tag = "shuffled" if embaralha else "real"
            print(f"   {'PointNet, linear head':30s} {tag:8s} R2 {np.mean(r2_lin):+.3f} "
                  f"(sd {np.std(r2_lin):.3f})")
            print(f"   {'PointNet + GP head':30s} {tag:8s} R2 {np.mean(r2_gp):+.3f} "
                  f"(sd {np.std(r2_gp):.3f})", flush=True)
            saida.append(dict(alvo=alvo, modelo=f"PointNet linear [{tag}]",
                              R2=float(np.mean(r2_lin)), rRMSE=np.nan))
            saida.append(dict(alvo=alvo, modelo=f"PointNet + GP [{tag}]",
                              R2=float(np.mean(r2_gp)), rRMSE=np.nan))

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(saida).to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

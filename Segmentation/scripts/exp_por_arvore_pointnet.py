#!/usr/bin/env python3
"""PointNet on the crown, with 809 labels and spatial-block validation.

`exp_por_arvore_com_809_rotulos.py` finds an R2 near zero for per-tree DBH with hand-crafted
features, even with oracle segmentation and a competition index. Against that result one can
object that it is the features that are poor; a network that reads the raw point cloud does not
depend on them, so if it also finds nothing the objection falls.

The network is the same as in `exp_per_tree_dap.py`, on purpose. That script ran PointNet on 23
destructively scaled trees and found an R2 of 0.111 with self-supervised pretraining, a number
that with 23 samples does not separate method from chance. Here it is the same architecture with
35 times more labels; changing the architecture at the same time would make it impossible to
know what changed.

Block validation rather than LOOCV: with 809 trees, LOOCV would be 809 trainings, and a
neighbouring tree shares crown and terrain, so leaving one out is not leaving information out.
Five blocks in x, one held out at a time.

The shuffling control is necessary here. A network with dropout and data augmentation can
memorise the mean of the training block and produce a slightly positive R2 with no signal at
all, and the shuffled run must collapse.

Z is not normalised. DBH depends on tree height, so centring z would destroy the only signal
known to exist; only x and y are centred on the stem. This is the same choice as in
`per_tree_dap/data.py`.

Run: PYTHONPATH=. python scripts/exp_por_arvore_pointnet.py
Output: manual_match/por_arvore_pointnet.csv
"""
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch import nn

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from greenvista import config  # noqa: E402
from greenvista.per_tree_dap.models import PointNetReg  # noqa: E402

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
DRONE = config.REPO / "work/lazall/SaoManuelTotal_001.laz"
SAIDA = config.OUT_DIR / "por_arvore_pointnet.csv"

Z_MIN, Z_MAX = 2.0, 45.0
R_VORONOI = 2.5
N_PTS = 512
PTS_MIN = 64
N_BLOCOS = 5
EPOCAS = 120
LOTE = 64
LR = 2e-3
ESCALA = 10.0
SEEDS = (0, 1, 2)


def copas(xyz, ref, raio):
    """Each point goes to the nearest mapped stem. The same oracle crown as in the sibling script."""
    T = cKDTree(ref)
    _, j = T.query(xyz[:, :2], distance_upper_bound=raio)
    ordem = np.argsort(j, kind="stable")
    j_ord, xyz_ord = j[ordem], xyz[ordem]
    corte = np.searchsorted(j_ord, np.arange(len(ref) + 1))
    out = {}
    for k in range(len(ref)):
        p = xyz_ord[corte[k]:corte[k + 1]]
        p = p[(p[:, 2] >= Z_MIN) & (p[:, 2] <= Z_MAX)]
        if len(p) >= PTS_MIN:
            out[k] = p.astype(np.float32)
    return out


def amostra(p, centro, rng, aumenta):
    q = p.copy()
    q[:, 0] -= centro[0]
    q[:, 1] -= centro[1]
    idx = rng.choice(len(q), N_PTS, replace=len(q) < N_PTS)
    q = q[idx]
    if aumenta:
        th = rng.uniform(0, 2 * np.pi)
        ct, st = np.cos(th), np.sin(th)
        x, y = q[:, 0].copy(), q[:, 1].copy()
        q[:, 0] = x * ct - y * st
        q[:, 1] = x * st + y * ct
        q += rng.normal(0, 0.02, q.shape).astype(np.float32)
    return (q / ESCALA).astype(np.float32)


def treina_e_prediz(tr_p, tr_y, te_p, te_c, tr_c, seed, dev):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = PointNetReg().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCAS *
                                                max(1, len(tr_p) // LOTE))
    mu, sd = float(np.mean(tr_y)), float(np.std(tr_y) + 1e-9)
    yz = torch.tensor((tr_y - mu) / sd, dtype=torch.float32, device=dev)
    perda = nn.SmoothL1Loss()
    net.train()
    for _ in range(EPOCAS):
        ordem = rng.permutation(len(tr_p))
        for i in range(0, len(ordem) - LOTE + 1, LOTE):
            b = ordem[i:i + LOTE]
            X = torch.tensor(np.stack([amostra(tr_p[k], tr_c[k], rng, True) for k in b]),
                             device=dev)
            opt.zero_grad()
            perda(net(X), yz[b]).backward()
            opt.step()
            sched.step()
    net.eval()
    with torch.no_grad():
        # mean of 8 rotations, because DBH does not depend on azimuth
        acc = np.zeros(len(te_p))
        for _ in range(8):
            X = torch.tensor(np.stack([amostra(p, c, rng, True)
                                       for p, c in zip(te_p, te_c)]), device=dev)
            acc += net(X).cpu().numpy()
        return acc / 8 * sd + mu


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {dev}")
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    ran = pd.read_csv(config.OUT_DIR / "dap_tls_ransac_talhao001.csv")
    dap = np.where(ran.motivo.values == "ok", ran.dap_cm.values, np.nan)

    las = laspy.read(str(DRONE))
    cls = np.asarray(las.classification)
    m = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    xyz = np.column_stack([np.asarray(las.x)[m], np.asarray(las.y)[m], np.asarray(las.z)[m]])
    C = copas(xyz, ref, R_VORONOI)
    ks = [k for k in C if np.isfinite(dap[k])]
    P = [C[k] for k in ks]
    Cn = [ref[k] for k in ks]
    y = dap[ks].astype(float)
    xs = ref[ks, 0]
    bloco = np.digitize(xs, np.quantile(xs, np.linspace(0, 1, N_BLOCOS + 1)[1:-1]))
    print(f"{len(ks)} labelled crowns, blocks {np.bincount(bloco)}, "
          f"{np.median([len(p) for p in P]):.0f} points per crown (median)")

    linhas = []
    for embaralha in (False, True):
        r2s, rrs = [], []
        for seed in SEEDS:
            alvo = y if not embaralha else np.random.default_rng(seed).permutation(y)
            pred = np.full(len(y), np.nan)
            for b in range(N_BLOCOS):
                te = bloco == b
                idx_tr = np.where(~te)[0]
                idx_te = np.where(te)[0]
                pred[idx_te] = treina_e_prediz(
                    [P[i] for i in idx_tr], alvo[idx_tr],
                    [P[i] for i in idx_te], [Cn[i] for i in idx_te],
                    # A list aligned with tr_p, not a dictionary keyed by the original index:
                    # the batch draws positions within the training set, so a dictionary keyed
                    # by the global index would raise KeyError or give the wrong tree centre.
                    [Cn[i] for i in idx_tr], seed, dev)
            e = pred - alvo
            r2 = 1 - np.sum(e ** 2) / np.sum((alvo - alvo.mean()) ** 2)
            rr = 100 * float(np.sqrt(np.mean(e ** 2))) / alvo.mean()
            r2s.append(r2)
            rrs.append(rr)
            print(f"   {'shuffled' if embaralha else 'real':12s} seed {seed}  "
                  f"R2 {r2:+.3f}  rRMSE {rr:5.1f} %", flush=True)
        linhas.append(dict(cond="embaralhado" if embaralha else "real", n=len(y),
                           R2=float(np.mean(r2s)), R2_dp=float(np.std(r2s)),
                           rRMSE=float(np.mean(rrs))))

    d = pd.DataFrame(linhas)
    print("\n" + "=" * 70)
    real, emb = d.iloc[0], d.iloc[1]
    print(f"PointNet, {len(y)} trees, spatial blocks: R2 {real.R2:+.3f} "
          f"(sd across seeds {real.R2_dp:.3f}), rRMSE {real.rRMSE:.1f} %")
    print(f"shuffled                                    R2 {emb.R2:+.3f}")
    print(f"null (predict the mean)                     R2  0.000, rRMSE "
          f"{100 * y.std() / y.mean():.1f} %")
    print("\nfor comparison, with 23 destructively scaled trees `exp_per_tree_dap` found R2 "
          "0.037 raw and 0.111 with pretraining.")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    d.to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

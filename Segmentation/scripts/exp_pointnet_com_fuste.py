#!/usr/bin/env python3
"""PointNet seeing the trunk and the scan angle, and not only the crown.

An alternative to point-cloud completion. The state-of-the-art survey identified two completion
lines (TreeDBH 2025, InceptionFormer 2026) that attack the loss of structure in the lower third.
Training a completion network from scratch requires a FOR-instance-style labelled set and days
of GPU, and completion invents points from a learned prior: if the information is not in the
data, the metric improves and the measurement does not. Before that, it is worth checking
whether the network can use what already exists.

Two things the previous version hid from the network and that now enter as channels:
 1. the points below 2 m, cut by the height filter inherited from the ABA, which are exactly
    where the SHAP `zq25` points to the signal;
 2. the scan angle of each point. `exp_obliquo_do_voo_existente.py` showed that the 20 to 30
    degree band is the only one that resolves the stem ring. Giving the network the angle of
    each point lets it separate oblique from nadir views without a hand-picked band.

A ladder, to see which of the two pays off: crown only (the baseline already measured), crown
plus trunk, and crown plus trunk plus angle.

Z is not normalised, because DBH depends on tree height.

Run: PYTHONPATH=. python scripts/exp_pointnet_com_fuste.py
Output: manual_match/pointnet_com_fuste.csv
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
from torch import nn

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
SAIDA = config.OUT_DIR / "pointnet_com_fuste.csv"
R_VORONOI = 2.5
N_PTS = 768
PTS_MIN = 64
EPOCAS = 120
LOTE = 64
LR = 2e-3
ESCALA = 10.0
SEEDS = (0, 1, 2)
SEED = 20260828


class PointNetN(nn.Module):
    """PointNet with a configurable number of input channels."""

    def __init__(self, n_in=3, p_drop=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(n_in, 64, 1), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(p_drop), nn.Linear(128, 64),
                                  nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.mlp(x.transpose(1, 2)).max(dim=2).values).squeeze(-1)


def copas(xyz, ang, ref, z_min):
    """Oracle Voronoi, also keeping the scan angle of each point."""
    T = cKDTree(ref)
    _, j = T.query(xyz[:, :2], distance_upper_bound=R_VORONOI)
    ordem = np.argsort(j, kind="stable")
    j_o, xyz_o, ang_o = j[ordem], xyz[ordem], ang[ordem]
    corte = np.searchsorted(j_o, np.arange(len(ref) + 1))
    out = {}
    for k in range(len(ref)):
        p, a = xyz_o[corte[k]:corte[k + 1]], ang_o[corte[k]:corte[k + 1]]
        m = (p[:, 2] >= z_min) & (p[:, 2] <= 45.0)
        if m.sum() >= PTS_MIN:
            out[k] = (p[m].astype(np.float32), a[m].astype(np.float32))
    return out


def amostra(p, a, centro, rng, canais, aumenta=True):
    q = p.copy()
    q[:, 0] -= centro[0]
    q[:, 1] -= centro[1]
    idx = rng.choice(len(q), N_PTS, replace=len(q) < N_PTS)
    q, aa = q[idx], a[idx]
    if aumenta:
        th = rng.uniform(0, 2 * np.pi)
        ct, st = np.cos(th), np.sin(th)
        x0, y0 = q[:, 0].copy(), q[:, 1].copy()
        q[:, 0] = x0 * ct - y0 * st
        q[:, 1] = x0 * st + y0 * ct
        q = q + rng.normal(0, 0.02, q.shape).astype(np.float32)
    q = q / ESCALA
    if canais == 3:
        return q.astype(np.float32)
    # The angle enters normalised and in absolute value: its sign says which side of the swath
    # the point came from, which is flight geometry and not tree geometry, and would give the
    # network a positional cue.
    return np.column_stack([q, (np.abs(aa) / 36.0).astype(np.float32)]).astype(np.float32)


def treina(P, A, C, y, tr, te, seed, dev, canais):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = PointNetN(canais).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR,
                                                total_steps=EPOCAS * max(1, len(tr) // LOTE))
    mu, sd = float(y[tr].mean()), float(y[tr].std() + 1e-9)
    yz = torch.tensor((y[tr] - mu) / sd, dtype=torch.float32, device=dev)
    perda = nn.SmoothL1Loss()
    net.train()
    for _ in range(EPOCAS):
        ordem = rng.permutation(len(tr))
        for i in range(0, len(ordem) - LOTE + 1, LOTE):
            b = ordem[i:i + LOTE]
            X = torch.tensor(np.stack([amostra(P[tr[k]], A[tr[k]], C[tr[k]], rng, canais)
                                       for k in b]), device=dev)
            opt.zero_grad()
            perda(net(X), yz[b]).backward()
            opt.step()
            sched.step()
    net.eval()
    acc = np.zeros(len(te))
    with torch.no_grad():
        for _ in range(8):
            X = torch.tensor(np.stack([amostra(P[k], A[k], C[k], rng, canais) for k in te]),
                             device=dev)
            acc += net(X).cpu().numpy()
    return acc / 8 * sd + mu


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    D = pd.read_csv(config.OUT_DIR / "por_arvore_pred_voronoi.csv", index_col=0)

    las = laspy.read(str(DRONE))
    cls = np.asarray(las.classification)
    m = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    xyz = np.column_stack([np.asarray(las.x)[m], np.asarray(las.y)[m], np.asarray(las.z)[m]])
    ang = np.abs(np.asarray(las.scan_angle_rank).astype(float))[m]

    saida = []
    for z_min, canais, nome in ((2.0, 3, "so copa (z acima de 2 m)"),
                                (0.8, 3, "copa + tronco (z acima de 0,8 m)"),
                                (0.8, 4, "copa + tronco + angulo de varredura")):
        C_all = copas(xyz, ang, ref, z_min)
        ks = [k for k in C_all if k in D.index]
        P = {k: C_all[k][0] for k in ks}
        A = {k: C_all[k][1] for k in ks}
        Cn = {k: ref[k] for k in ks}
        Dk = D.loc[ks]
        y = Dk.dap_cm.to_numpy(float)
        blocos = _rel.blocos_de(Dk.x.to_numpy(float))
        ks_arr = np.array(ks)
        print(f"\n=== {nome}  |  {len(ks)} trees, "
              f"{np.median([len(P[k]) for k in ks]):.0f} points per crown", flush=True)
        for embaralha in (False, True):
            yy = y if not embaralha else np.random.default_rng(3).permutation(y)
            r2s = []
            for s in SEEDS:
                pred = np.full(len(yy), np.nan)
                for tr_i, te_i in [(np.where(np.isin(np.arange(len(ks)), tr))[0],
                                    np.where(np.isin(np.arange(len(ks)), te))[0])
                                   for tr, te in blocos]:
                    pred[te_i] = treina(
                        {i: P[ks_arr[i]] for i in range(len(ks))},
                        {i: A[ks_arr[i]] for i in range(len(ks))},
                        {i: Cn[ks_arr[i]] for i in range(len(ks))},
                        yy, tr_i, te_i, s, dev, canais)
                r2s.append(_rel.marca(yy, pred)[0])
            tag = "embaralhado" if embaralha else "real"
            print(f"   {tag:12s} R2 {np.mean(r2s):+.3f} (sd across seeds {np.std(r2s):.3f})",
                  flush=True)
            saida.append(dict(conjunto=nome, cond=tag, n=len(ks),
                              R2=float(np.mean(r2s)), R2_dp=float(np.std(r2s))))

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(saida).to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

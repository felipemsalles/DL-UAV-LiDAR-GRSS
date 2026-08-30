#!/usr/bin/env python3
"""The last two state-of-the-art routes: self-supervised pre-training and distillation from TLS.

TLS has already entered as a label, as an oracle delineation and as 3D geometry, and none of these
took the per-tree R2 out of the 0.12 band. Two forms remain untested.

1. Self-supervised pre-training at scale. There are 13,946 crowns detected across the six stands,
with no label at all. The Forestry 99(2) 2026 paper pre-trained a network on 500 thousand unlabelled
clouds and showed a gain precisely where labels are scarce. This repository's `per_tree_dap/ssl.py`
gives R2 0.111 on 23 trees; here the same pretext runs on 809. The pretext is the same: the
network predicts the structural metrics of the crown itself, with no external label.

2. Cross-modal distillation. A network trained on the TLS cloud sees the stem, and the idea is to
transfer its internal representation to the network that only sees the drone, instead of transferring
a single number. It differs from ordinary supervised regression because the teacher delivers a
256-dimensional target instead of a scalar.

Expectation recorded before running: both routes attack extraction, and not information. It is
already measured that the label is not the bottleneck, with a label-noise ceiling at R2 0.967 against
the 0.12 obtained. If the information is not in the drone cloud, no teacher puts it there. What may
pay off is the learning curve not having saturated, i.e. weak signal poorly extracted with 809
labels; the expectation is 0.12 moving to something between 0.15 and 0.20, not to 0.4.

The unlabelled corpus includes stand 001, where the test trees are. In self-supervised learning that
is standard and is not label leakage, because the pretext uses no DBH at all, but it is transduction:
the network saw the test inputs without their labels. The `sem_t001` variant runs the pretext without
stand 001 to measure whether it makes a difference.

Run: PYTHONPATH=. python scripts/exp_ssl_e_destilacao.py
Out: manual_match/ssl_e_destilacao.csv
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
_vol = _mod("vol", R / "scripts/exp_volume_tls_por_arvore.py")
warnings.filterwarnings("ignore")

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
LAZ = config.REPO / "work/lazall"
DET = config.REPO / "data/detections/sat_w2w_arvores.csv"
SAIDA = config.OUT_DIR / "ssl_e_destilacao.csv"

N_PTS = 512
PTS_MIN = 64
R_VOR = 2.5
Z_MIN, Z_MAX = 2.0, 45.0
ESCALA = 10.0
EP_PRE, EP_FIN = 40, 120
LOTE = 96
SEEDS = (0, 1, 2)
MAX_CORPUS = 12000
PRETEXTO = 8               # structural metrics predicted in the pretext task


class Rede(nn.Module):
    def __init__(self, n_saidas=1, p_drop=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(True), nn.Conv1d(64, 128, 1), nn.ReLU(True),
            nn.Conv1d(128, 256, 1), nn.ReLU(True))
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(p_drop),
                                  nn.Linear(128, 64), nn.ReLU(True), nn.Linear(64, n_saidas))

    def emb(self, x):
        return self.mlp(x.transpose(1, 2)).max(dim=2).values

    def forward(self, x):
        return self.head(self.emb(x)).squeeze(-1)


def amostra(p, rng, aumenta=True):
    q = p[rng.choice(len(p), N_PTS, replace=len(p) < N_PTS)].copy()
    if aumenta:
        th = rng.uniform(0, 2 * np.pi)
        ct, st = np.cos(th), np.sin(th)
        x0, y0 = q[:, 0].copy(), q[:, 1].copy()
        q[:, 0] = x0 * ct - y0 * st
        q[:, 1] = x0 * st + y0 * ct
        q = q + rng.normal(0, 0.02, q.shape).astype(np.float32)
    return (q / ESCALA).astype(np.float32)


def metricas(p):
    """The structural metrics of the pretext task. Taken from the crown itself, no external label."""
    z = p[:, 2]
    d = np.hypot(p[:, 0], p[:, 1])
    return np.array([np.quantile(z, .25), np.quantile(z, .5), np.quantile(z, .75),
                     np.quantile(z, .95), z.max() - z.min(), np.quantile(d, .9),
                     np.log(len(p)), z.std()], dtype=np.float32)


def copas_do_talhao(laz, seeds_xy):
    las = laspy.read(str(laz))
    cls = np.asarray(las.classification)
    m = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    xyz = np.column_stack([np.asarray(las.x)[m], np.asarray(las.y)[m], np.asarray(las.z)[m]])
    xyz = xyz[(xyz[:, 2] >= Z_MIN) & (xyz[:, 2] <= Z_MAX)]
    _, j = cKDTree(seeds_xy).query(xyz[:, :2], distance_upper_bound=R_VOR)
    o = np.argsort(j, kind="stable")
    j_o, xyz_o = j[o], xyz[o]
    corte = np.searchsorted(j_o, np.arange(len(seeds_xy) + 1))
    out = []
    for k in range(len(seeds_xy)):
        p = xyz_o[corte[k]:corte[k + 1]]
        if len(p) >= PTS_MIN:
            q = p.copy()
            q[:, :2] -= seeds_xy[k]
            out.append(q.astype(np.float32))
    return out


def pre_treina(corpus, seed, dev, epocas=EP_PRE):
    """Pretext task: predict the structural metrics of the crown itself. Returns the trained trunk."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Rede(PRETEXTO).to(dev)
    Y = np.stack([metricas(p) for p in corpus])
    mu, sd = Y.mean(0), Y.std(0) + 1e-9
    Yz = torch.tensor((Y - mu) / sd, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    perda = nn.SmoothL1Loss()
    net.train()
    for ep in range(epocas):
        ordem = rng.permutation(len(corpus))
        tot = 0.0
        for i in range(0, len(ordem) - LOTE + 1, LOTE):
            b = ordem[i:i + LOTE]
            X = torch.tensor(np.stack([amostra(corpus[k], rng) for k in b]), device=dev)
            opt.zero_grad()
            l = perda(net(X), Yz[b])
            l.backward()
            opt.step()
            tot += float(l)
        if ep % 10 == 0:
            print(f"      pretext epoch {ep:3d}  loss {tot / max(1, len(ordem)//LOTE):.4f}",
                  flush=True)
    return {k: v.cpu().clone() for k, v in net.mlp.state_dict().items()}


def ajusta(P, y, tr, te, seed, dev, tronco=None, prof=None, peso_dest=0.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Rede(1).to(dev)
    if tronco is not None:
        net.mlp.load_state_dict({k: v.to(dev) for k, v in tronco.items()})
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3,
                                              total_steps=EP_FIN * max(1, len(tr) // LOTE))
    mu, sd = float(y[tr].mean()), float(y[tr].std() + 1e-9)
    yz = torch.tensor((y[tr] - mu) / sd, dtype=torch.float32, device=dev)
    perda, mse = nn.SmoothL1Loss(), nn.MSELoss()
    net.train()
    for _ in range(EP_FIN):
        ordem = rng.permutation(len(tr))
        for i in range(0, len(ordem) - LOTE + 1, LOTE):
            b = ordem[i:i + LOTE]
            X = torch.tensor(np.stack([amostra(P[tr[k]], rng) for k in b]), device=dev)
            opt.zero_grad()
            l = perda(net(X), yz[b])
            if peso_dest > 0 and prof is not None:
                # Distillation matches the representation, not the output: matching the teacher
                # output would swap one good label for another, and the label is not the
                # bottleneck. The possible gain lies in the 256-dimensional target per sample.
                alvo = torch.tensor(prof[tr[b]], device=dev)
                l = l + peso_dest * mse(net.emb(X), alvo)
            l.backward()
            opt.step()
            sch.step()
    net.eval()
    acc = np.zeros(len(te))
    with torch.no_grad():
        for _ in range(8):
            X = torch.tensor(np.stack([amostra(P[k], rng) for k in te]), device=dev)
            acc += net(X).cpu().numpy()
    return acc / 8 * sd + mu


def professor_tls(ref, dap, dev, seed=0):
    """Network trained on the TLS cloud, which sees the stem. Returns its representation."""
    x, y, h = _vol.le_tls()
    T = cKDTree(np.column_stack([x, y]))
    P, idx_ok = [], []
    for j, (cx, cy) in enumerate(ref):
        if not np.isfinite(dap[j]):
            continue
        k = np.asarray(T.query_ball_point([cx, cy], 0.6), dtype=int)
        k = k[(h[k] >= 1.0) & (h[k] <= 6.0)]
        if len(k) >= PTS_MIN:
            P.append(np.column_stack([x[k] - cx, y[k] - cy, h[k]]).astype(np.float32))
            idx_ok.append(j)
    print(f"   teacher: {len(P)} stems with a usable TLS cloud", flush=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Rede(1).to(dev)
    yv = dap[np.array(idx_ok)]
    mu, sd = yv.mean(), yv.std() + 1e-9
    yz = torch.tensor((yv - mu) / sd, dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    perda = nn.SmoothL1Loss()
    net.train()
    for _ in range(EP_FIN):
        ordem = rng.permutation(len(P))
        for i in range(0, len(ordem) - LOTE + 1, LOTE):
            b = ordem[i:i + LOTE]
            X = torch.tensor(np.stack([amostra(P[k], rng) for k in b]), device=dev)
            opt.zero_grad()
            perda(net(X), yz[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = np.zeros(len(P))
        emb = np.zeros((len(P), 256), np.float32)
        for _ in range(6):
            X = torch.tensor(np.stack([amostra(p, rng) for p in P]), device=dev)
            pred += net(X).cpu().numpy()
            emb += net.emb(X).cpu().numpy()
    pred = pred / 6 * sd + mu
    r2 = _rel.marca(yv, pred)[0]
    print(f"   teacher on TLS, R2 on its own training set: {r2:+.3f} "
          f"(only to confirm that it LEARNED something)", flush=True)
    return dict(zip(idx_ok, emb / 6)), r2


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    D = pd.read_csv(config.OUT_DIR / "por_arvore_pred_voronoi.csv", index_col=0)
    ran = pd.read_csv(config.OUT_DIR / "dap_tls_ransac_talhao001.csv")
    dap = np.where(ran.motivo.values == "ok", ran.dap_cm.values, np.nan)
    print(f"device {dev}")

    # ---- labelled crowns (stand 001, positions from the TLS map) -------------------------
    lab = copas_do_talhao(LAZ / "SaoManuelTotal_001.laz", ref)
    # copas_do_talhao drops the sparse ones, so the index is rebuilt
    P_lab = {}
    # recomputed with an explicit index so that nothing goes out of alignment
    las = laspy.read(str(LAZ / "SaoManuelTotal_001.laz"))
    cls = np.asarray(las.classification)
    m = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    xyz = np.column_stack([np.asarray(las.x)[m], np.asarray(las.y)[m], np.asarray(las.z)[m]])
    xyz = xyz[(xyz[:, 2] >= Z_MIN) & (xyz[:, 2] <= Z_MAX)]
    _, jj = cKDTree(ref).query(xyz[:, :2], distance_upper_bound=R_VOR)
    o = np.argsort(jj, kind="stable")
    j_o, xyz_o = jj[o], xyz[o]
    corte = np.searchsorted(j_o, np.arange(len(ref) + 1))
    for k in range(len(ref)):
        p = xyz_o[corte[k]:corte[k + 1]]
        if len(p) >= PTS_MIN and k in D.index:
            q = p.copy()
            q[:, :2] -= ref[k]
            P_lab[k] = q.astype(np.float32)
    ks = sorted(P_lab)
    P = [P_lab[k] for k in ks]
    Dk = D.loc[ks]
    y = Dk.dap_cm.to_numpy(float)
    blocos = _rel.blocos_de(Dk.x.to_numpy(float))
    print(f"{len(ks)} labelled trees, blocks {[len(t) for _, t in blocos]}\n")

    # ---- unlabelled corpus, the 13,946 detected across the six stands --------------------
    det = pd.read_csv(DET)
    corpus, corpus_sem = [], []
    for t in sorted(det.talhao.unique()):
        laz = LAZ / f"SaoManuelTotal_{int(t):03d}.laz"
        if not laz.exists():
            continue
        s = det[det.talhao == t][["base_x", "base_y"]].to_numpy(float)
        c = copas_do_talhao(laz, s)
        corpus += c
        if int(t) != 1:
            corpus_sem += c
        print(f"   stand {int(t):03d}: {len(c)} unlabelled crowns", flush=True)
    rng = np.random.default_rng(0)
    if len(corpus) > MAX_CORPUS:
        corpus = [corpus[i] for i in rng.choice(len(corpus), MAX_CORPUS, replace=False)]
    if len(corpus_sem) > MAX_CORPUS:
        corpus_sem = [corpus_sem[i] for i in rng.choice(len(corpus_sem), MAX_CORPUS, replace=False)]
    print(f"unlabelled corpus: {len(corpus)} crowns (without stand 001: {len(corpus_sem)})\n")

    saida = []

    def roda(nome, tronco=None, prof=None, peso=0.0):
        for emb in (False, True):
            yy = y if not emb else np.random.default_rng(5).permutation(y)
            r2s = []
            for s in SEEDS:
                pred = np.full(len(yy), np.nan)
                for tr, te in blocos:
                    pred[te] = ajusta(P, yy, tr, te, s, dev, tronco, prof, peso)
                r2s.append(_rel.marca(yy, pred)[0])
            tag = "embaralhado" if emb else "real"
            print(f"   {nome:34s} {tag:12s} R2 {np.mean(r2s):+.3f} "
                  f"(sd across seeds {np.std(r2s):.3f})", flush=True)
            saida.append(dict(rota=nome, cond=tag, R2=float(np.mean(r2s)),
                              R2_dp=float(np.std(r2s)), n=len(ks)))

    print("[1] baseline, network from scratch")
    roda("do zero")

    print("\n[2] self-supervised pre-training, full corpus")
    tr_all = pre_treina(corpus, 0, dev)
    roda("pre-treinado (corpus inteiro)", tronco=tr_all)

    print("\n[3] pre-training WITHOUT stand 001, to measure transduction")
    tr_sem = pre_treina(corpus_sem, 0, dev)
    roda("pre-treinado (sem o talhao 001)", tronco=tr_sem)

    print("\n[4] distillation of the representation of the teacher trained on TLS")
    emb_prof, r2_prof = professor_tls(ref, dap, dev)
    prof = np.zeros((len(ks), 256), np.float32)
    tem = np.array([k in emb_prof for k in ks])
    for i, k in enumerate(ks):
        if k in emb_prof:
            prof[i] = emb_prof[k]
    print(f"   {tem.sum()} of the {len(ks)} trees have a teacher representation")
    for peso in (0.1, 0.5):
        roda(f"destilado do TLS (peso {peso})", prof=prof, peso_dest=peso)
    print("\n[5] pre-training + distillation together")
    roda("pre-treinado + destilado", tronco=tr_all, prof=prof, peso_dest=0.1)

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(saida).to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

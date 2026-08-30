#!/usr/bin/env python3
"""Hierarchical teacher against PointNet, and diversity against quantity in the corpus.

Part 1. The geometric estimator on the same TLS cloud measures DBH well, so much so that it
produced the 809 labels, but the teacher network seeing the same points stalled at an R2 of 0.43.
That weakens the distillation test, because the teacher may not know enough to teach. The
diagnosis is architectural: PointNet aggregates by global maximum and the published criticism is
that it does not exploit geometric context in per-point learning. Measuring a radius requires a
local relation between neighbouring points, which the global maximum discards. Here a
hierarchical teacher enters, with local grouping at two scales before the maximum, the standard
fix (PointNet++). If the better teacher also fails to teach, the negative distillation result
becomes firmer.

Part 2. The self-supervised corpus saturated, and the hypothesis that it saturated for lack of
variety rather than size has not yet been measured. The test holds corpus size fixed and compares
crowns taken from a single stand against crowns spread over five.

Run: PYTHONPATH=. python scripts/exp_professor_melhor_e_diversidade.py
Output: manual_match/professor_melhor_e_diversidade.csv
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
_ssl = _mod("ssl", R / "scripts/exp_ssl_e_destilacao.py")
_des = _mod("des", R / "scripts/exp_destilacao_tls.py")
_vol = _mod("vol", R / "scripts/exp_volume_tls_por_arvore.py")
warnings.filterwarnings("ignore")

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
LAZ = config.REPO / "work/lazall"
DET = config.REPO / "data/detections/sat_w2w_arvores.csv"
SAIDA = config.OUT_DIR / "professor_melhor_e_diversidade.csv"
N_PTS = 512
# Ten seeds and not three: with three, the difference between a concentrated and a spread corpus
# was +0.016 with a standard error of 0.011, and the 95 % interval ran from -0.006 to +0.038,
# including zero. With ten the standard error drops to about 0.006, and the same difference, if
# real, becomes about 2.7 standard errors.
SEEDS = tuple(range(10))
SEED = 20260829
N_DIV = 4000                 # fixed corpus size for the diversity test


class Hierarquica(nn.Module):
    """Slim PointNet++: two levels of local grouping before the global maximum.

    In plain PointNet each point becomes a feature on its own and the global maximum picks the
    winner, so the relation between neighbouring points is never computed, and a trunk radius is
    exactly a relation between neighbours. Here each centre aggregates its K neighbours with
    coordinates relative to that centre.
    """

    def __init__(self, n_in=3, dim=128, p_drop=0.3):
        super().__init__()
        self.n1 = nn.Sequential(nn.Conv2d(n_in, 64, 1), nn.ReLU(True),
                                nn.Conv2d(64, 64, 1), nn.ReLU(True))
        self.n2 = nn.Sequential(nn.Conv2d(64 + 3, dim, 1), nn.ReLU(True),
                                nn.Conv2d(dim, dim, 1), nn.ReLU(True))
        self.n3 = nn.Sequential(nn.Conv1d(dim, 256, 1), nn.ReLU(True))
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(p_drop),
                                  nn.Linear(128, 64), nn.ReLU(True), nn.Linear(64, 1))

    def _agrupa(self, xyz, feat, n_cen, k):
        B, N, _ = xyz.shape
        idx_c = torch.randperm(N, device=xyz.device)[:n_cen]
        cen = xyz[:, idx_c]                                   # (B, M, 3)
        d = torch.cdist(cen, xyz)                             # (B, M, N)
        idx = d.topk(k, largest=False).indices                # (B, M, k)
        viz = torch.gather(xyz.unsqueeze(1).expand(-1, n_cen, -1, -1), 2,
                           idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        rel = viz - cen.unsqueeze(2)                          # coordinate relative to the centre
        if feat is None:
            g = rel
        else:
            fv = torch.gather(feat.unsqueeze(1).expand(-1, n_cen, -1, -1), 2,
                              idx.unsqueeze(-1).expand(-1, -1, -1, feat.shape[-1]))
            g = torch.cat([rel, fv], -1)
        return cen, g

    def forward(self, x):
        cen1, g1 = self._agrupa(x, None, 128, 16)
        f1 = self.n1(g1.permute(0, 3, 1, 2)).max(dim=3).values.permute(0, 2, 1)
        cen2, g2 = self._agrupa(cen1, f1, 32, 16)
        f2 = self.n2(g2.permute(0, 3, 1, 2)).max(dim=3).values
        return self.head(self.n3(f2).max(dim=2).values).squeeze(-1)

    def emb(self, x):
        cen1, g1 = self._agrupa(x, None, 128, 16)
        f1 = self.n1(g1.permute(0, 3, 1, 2)).max(dim=3).values.permute(0, 2, 1)
        cen2, g2 = self._agrupa(cen1, f1, 32, 16)
        f2 = self.n2(g2.permute(0, 3, 1, 2)).max(dim=3).values
        return self.n3(f2).max(dim=2).values


def treina(rede_fn, P, y, tr, te, seed, dev, esc, jit, epocas=120, lote=48):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = rede_fn().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3,
                                              total_steps=epocas * max(1, len(tr) // lote))
    mu, sd = float(y[tr].mean()), float(y[tr].std() + 1e-9)
    yz = torch.tensor((y[tr] - mu) / sd, dtype=torch.float32, device=dev)
    perda = nn.SmoothL1Loss()
    net.train()
    for _ in range(epocas):
        ordem = rng.permutation(len(tr))
        for i in range(0, len(ordem) - lote + 1, lote):
            b = ordem[i:i + lote]
            X = torch.tensor(np.stack([_des.amostra(P[tr[k]], rng, esc, jit) for k in b]),
                             device=dev)
            opt.zero_grad()
            perda(net(X), yz[b]).backward()
            opt.step()
            sch.step()
    net.eval()
    with torch.no_grad():
        acc = np.zeros(len(te))
        emb = np.zeros((len(te), 256), np.float32)
        for _ in range(6):
            X = torch.tensor(np.stack([_des.amostra(P[k], rng, esc, jit) for k in te]),
                             device=dev)
            acc += net(X).cpu().numpy()
            emb += net.emb(X).cpu().numpy()
    return acc / 6 * sd + mu, emb / 6


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    D = pd.read_csv(config.OUT_DIR / "por_arvore_pred_voronoi.csv", index_col=0)
    ran = pd.read_csv(config.OUT_DIR / "dap_tls_ransac_talhao001.csv")
    dap = np.where(ran.motivo.values == "ok", ran.dap_cm.values, np.nan)
    saida = []

    # ---- PART 1: a better teacher -------------------------------------------------------
    print("[1] hierarchical teacher against global-maximum teacher\n")
    xt, yt, ht = _vol.le_tls()
    T = cKDTree(np.column_stack([xt, yt]))
    ks, P_prof = [], []
    for k in range(len(ref)):
        if k not in D.index or not np.isfinite(dap[k]):
            continue
        idx = np.asarray(T.query_ball_point(ref[k], _des.R_FUSTE), dtype=int)
        idx = idx[(ht[idx] >= _des.TRECHO[0]) & (ht[idx] <= _des.TRECHO[1])]
        if len(idx) >= _des.PTS_MIN:
            P_prof.append(np.column_stack([xt[idx] - ref[k][0], yt[idx] - ref[k][1],
                                           ht[idx] - np.mean(_des.TRECHO)]).astype(np.float32))
            ks.append(k)
    yp = dap[np.array(ks)]
    bl = _rel.blocos_de(D.loc[ks].x.to_numpy(float))
    print(f"   {len(ks)} stems with a TLS point cloud\n")
    embs_prof = None
    for nome, fn in (("maximo global (PointNet)", _des.Rede),
                     ("hierarquica (agrupamento local)", Hierarquica)):
        pred = np.full(len(yp), np.nan)
        E = np.zeros((len(yp), 256), np.float32)
        for tr, te in bl:
            p_, e_ = treina(fn, P_prof, yp, tr, te, 0, dev, _des.ESC_FUSTE, _des.JIT_FUSTE)
            pred[te], E[te] = p_, e_
        r2, rr = _rel.marca(yp, pred)
        print(f"   teacher {nome:34s} R2 {r2:+.3f}  rRMSE {rr:5.1f} %", flush=True)
        saida.append(dict(parte="professor", modelo=nome, R2=r2, rRMSE=rr, n=len(yp)))
        if "hierarquica" in nome:
            embs_prof = E

    # distillation from the hierarchical teacher
    las = laspy.read(str(LAZ / "SaoManuelTotal_001.laz"))
    cls = np.asarray(las.classification)
    m = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    xyz = np.column_stack([np.asarray(las.x)[m], np.asarray(las.y)[m], np.asarray(las.z)[m]])
    xyz = xyz[(xyz[:, 2] >= _ssl.Z_MIN) & (xyz[:, 2] <= _ssl.Z_MAX)]
    _, j = cKDTree(ref).query(xyz[:, :2], distance_upper_bound=_ssl.R_VOR)
    o = np.argsort(j, kind="stable")
    j_o, xyz_o = j[o], xyz[o]
    corte = np.searchsorted(j_o, np.arange(len(ref) + 1))
    P_al = []
    for k in ks:
        p = xyz_o[corte[k]:corte[k + 1]].copy()
        p[:, :2] -= ref[k]
        P_al.append(p.astype(np.float32))
    print("\n   distilling from the hierarchical teacher to the student that only sees the drone")
    for peso in (0.0, 0.1):
        nome = "aluno sozinho" if peso == 0 else "aluno destilado do professor melhor"
        r2s = []
        for s in SEEDS:
            pr = np.full(len(yp), np.nan)
            for tr, te in bl:
                torch.manual_seed(s)
                rng = np.random.default_rng(s)
                net = _des.Rede().to(dev)
                opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
                mu, sd = float(yp[tr].mean()), float(yp[tr].std() + 1e-9)
                yz = torch.tensor((yp[tr] - mu) / sd, dtype=torch.float32, device=dev)
                perda, mse = nn.SmoothL1Loss(), nn.MSELoss()
                net.train()
                for _ in range(120):
                    ordem = rng.permutation(len(tr))
                    for i in range(0, len(ordem) - 48 + 1, 48):
                        b = ordem[i:i + 48]
                        X = torch.tensor(np.stack([_des.amostra(P_al[tr[k]], rng,
                                                                _des.ESC_COPA, _des.JIT_COPA)
                                                   for k in b]), device=dev)
                        opt.zero_grad()
                        l = perda(net(X), yz[b])
                        if peso > 0:
                            l = l + peso * mse(net.emb(X),
                                               torch.tensor(embs_prof[tr[b]], device=dev))
                        l.backward()
                        opt.step()
                net.eval()
                with torch.no_grad():
                    acc = np.zeros(len(te))
                    for _ in range(6):
                        X = torch.tensor(np.stack([_des.amostra(P_al[k], rng, _des.ESC_COPA,
                                                                _des.JIT_COPA) for k in te]),
                                         device=dev)
                        acc += net(X).cpu().numpy()
                pr[te] = acc / 6 * sd + mu
            r2s.append(_rel.marca(yp, pr)[0])
        print(f"   {nome:38s} R2 {np.mean(r2s):+.3f} (sd {np.std(r2s):.3f})", flush=True)
        saida.append(dict(parte="destilacao com professor melhor", modelo=nome,
                          R2=float(np.mean(r2s)), rRMSE=np.nan, n=len(yp)))

    # ---- PART 2: diversity against quantity ---------------------------------------------
    print(f"\n[2] diversity against quantity, fixed corpus of {N_DIV} crowns")
    det = pd.read_csv(DET)
    por_talhao = {}
    for t in sorted(det.talhao.unique()):
        if int(t) == 1:
            continue
        laz = LAZ / f"SaoManuelTotal_{int(t):03d}.laz"
        if laz.exists():
            s = det[det.talhao == t][["base_x", "base_y"]].to_numpy(float)
            por_talhao[int(t)] = _ssl.copas_do_talhao(laz, s)
            print(f"   stand {int(t):03d}: {len(por_talhao[int(t)])} crowns", flush=True)
    rng = np.random.default_rng(SEED)
    maior = max(por_talhao, key=lambda t: len(por_talhao[t]))
    conc = [por_talhao[maior][i] for i in
            rng.choice(len(por_talhao[maior]), min(N_DIV, len(por_talhao[maior])), False)]
    espalhado = []
    por_t = N_DIV // len(por_talhao)
    for t, c in por_talhao.items():
        espalhado += [c[i] for i in rng.choice(len(c), min(por_t, len(c)), False)]
    print(f"   concentrated: {len(conc)} crowns from stand {maior}")
    print(f"   spread:       {len(espalhado)} crowns from {len(por_talhao)} stands\n")

    ks2 = [k for k in range(len(ref)) if k in D.index]
    P2 = []
    for k in ks2:
        p = xyz_o[corte[k]:corte[k + 1]].copy()
        p[:, :2] -= ref[k]
        P2.append(p.astype(np.float32))
    Dk = D.loc[ks2]
    y2 = Dk.dap_cm.to_numpy(float)
    bl2 = _rel.blocos_de(Dk.x.to_numpy(float))
    for nome, corpus in (("concentrado (1 talhao)", conc),
                         (f"espalhado ({len(por_talhao)} talhoes)", espalhado)):
        r2s = []
        for s in SEEDS:
            tronco = _ssl.pre_treina(corpus, s, dev, epocas=40)
            pred = np.full(len(y2), np.nan)
            for tr, te in bl2:
                pred[te] = _ssl.ajusta(P2, y2, tr, te, s, dev, tronco)
            r2s.append(_rel.marca(y2, pred)[0])
        print(f"   {nome:28s} R2 {np.mean(r2s):+.3f} (sd across seeds {np.std(r2s):.3f})",
              flush=True)
        saida.append(dict(parte="diversidade", modelo=nome, R2=float(np.mean(r2s)),
                          rRMSE=np.nan, n=len(corpus)))

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(saida).to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Distillation from the TLS to the drone, with a teacher at a scale compatible with the stem.

In `exp_ssl_e_destilacao.py` the teacher trained on the TLS gave R2 +0.013 on its own training
set, that is, it learned nothing. The cause is the normalisation scale: the coordinates were
divided by 10 and the data augmentation added 2 cm of noise. For a task whose signal is the trunk
radius, of 9.5 cm, that fails along two paths:
  - dividing by 10 takes the radius to 0.0095 in the units of the network, and global max pooling
    does not resolve differences at that scale;
  - 2 cm of noise is 20 % of the signal, so the augmentation erases the variable.
The same normalisation works for the crown, a structure of metres, and destroys the stem, a
structure of centimetres. Here the scale is 0.3 and the noise is 2 mm.

The teacher is validated before teaching, both in and out of the fold. If it does not learn, the
distillation stage does not run, because distilling from a bad teacher only transfers its noise.

The distillation matches the representation and not the output: matching the output would swap one
good label for another, and it is already measured that the label is not the bottleneck (ceiling of
0.967). What may pay off is the 256-dimensional target per sample instead of a scalar.

Usage: PYTHONPATH=. python scripts/exp_destilacao_tls.py
Output: manual_match/destilacao_tls.csv
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
DRONE = config.REPO / "work/lazall/SaoManuelTotal_001.laz"
SAIDA = config.OUT_DIR / "destilacao_tls.csv"
N_PTS = 512
PTS_MIN = 64
R_VOR, R_FUSTE = 2.5, 0.60
Z_MIN, Z_MAX = 2.0, 45.0
TRECHO = (1.0, 6.0)
EP = 120
LOTE = 64
SEEDS = (0, 1, 2)
# two normalisation scales, one per type of structure
ESC_COPA, JIT_COPA = 10.0, 0.02      # the crown is a structure of metres
ESC_FUSTE, JIT_FUSTE = 0.30, 0.002   # the stem is a structure of centimetres


class Rede(nn.Module):
    def __init__(self, p_drop=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(True), nn.Conv1d(64, 128, 1), nn.ReLU(True),
            nn.Conv1d(128, 256, 1), nn.ReLU(True))
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(p_drop),
                                  nn.Linear(128, 64), nn.ReLU(True), nn.Linear(64, 1))

    def emb(self, x):
        return self.mlp(x.transpose(1, 2)).max(dim=2).values

    def forward(self, x):
        return self.head(self.emb(x)).squeeze(-1)


def amostra(p, rng, esc, jit, aumenta=True):
    q = p[rng.choice(len(p), N_PTS, replace=len(p) < N_PTS)].copy()
    if aumenta:
        th = rng.uniform(0, 2 * np.pi)
        ct, st = np.cos(th), np.sin(th)
        x0, y0 = q[:, 0].copy(), q[:, 1].copy()
        q[:, 0] = x0 * ct - y0 * st
        q[:, 1] = x0 * st + y0 * ct
        q = q + rng.normal(0, jit, q.shape).astype(np.float32)
    return (q / esc).astype(np.float32)


def treina(P, y, tr, te, seed, dev, esc, jit, prof=None, peso=0.0, epocas=EP):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Rede().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3,
                                              total_steps=epocas * max(1, len(tr) // LOTE))
    mu, sd = float(y[tr].mean()), float(y[tr].std() + 1e-9)
    yz = torch.tensor((y[tr] - mu) / sd, dtype=torch.float32, device=dev)
    perda, mse = nn.SmoothL1Loss(), nn.MSELoss()
    net.train()
    for _ in range(epocas):
        ordem = rng.permutation(len(tr))
        for i in range(0, len(ordem) - LOTE + 1, LOTE):
            b = ordem[i:i + LOTE]
            X = torch.tensor(np.stack([amostra(P[tr[k]], rng, esc, jit) for k in b]), device=dev)
            opt.zero_grad()
            l = perda(net(X), yz[b])
            if peso > 0 and prof is not None:
                l = l + peso * mse(net.emb(X), torch.tensor(prof[tr[b]], device=dev))
            l.backward()
            opt.step()
            sch.step()
    net.eval()
    with torch.no_grad():
        acc = np.zeros(len(te))
        emb = np.zeros((len(te), 256), np.float32)
        for _ in range(8):
            X = torch.tensor(np.stack([amostra(P[k], rng, esc, jit) for k in te]), device=dev)
            acc += net(X).cpu().numpy()
            emb += net.emb(X).cpu().numpy()
    return acc / 8 * sd + mu, emb / 8


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    D = pd.read_csv(config.OUT_DIR / "por_arvore_pred_voronoi.csv", index_col=0)
    ran = pd.read_csv(config.OUT_DIR / "dap_tls_ransac_talhao001.csv")
    dap = np.where(ran.motivo.values == "ok", ran.dap_cm.values, np.nan)

    # ---- drone crowns (student) ----------------------------------------------------------
    las = laspy.read(str(DRONE))
    cls = np.asarray(las.classification)
    m = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    xyz = np.column_stack([np.asarray(las.x)[m], np.asarray(las.y)[m], np.asarray(las.z)[m]])
    xyz = xyz[(xyz[:, 2] >= Z_MIN) & (xyz[:, 2] <= Z_MAX)]
    _, j = cKDTree(ref).query(xyz[:, :2], distance_upper_bound=R_VOR)
    o = np.argsort(j, kind="stable")
    j_o, xyz_o = j[o], xyz[o]
    corte = np.searchsorted(j_o, np.arange(len(ref) + 1))
    P_al, ks = {}, []
    for k in range(len(ref)):
        p = xyz_o[corte[k]:corte[k + 1]]
        if len(p) >= PTS_MIN and k in D.index and np.isfinite(dap[k]):
            q = p.copy()
            q[:, :2] -= ref[k]
            P_al[k] = q.astype(np.float32)
            ks.append(k)
    Dk = D.loc[ks]
    y = dap[np.array(ks)]
    blocos = _rel.blocos_de(Dk.x.to_numpy(float))
    P = [P_al[k] for k in ks]
    print(f"device {dev} | {len(ks)} trees | blocks {[len(t) for _, t in blocos]}\n")

    # ---- TLS stem cloud (teacher) -------------------------------------------------------
    xt, yt, ht = _vol.le_tls()
    T = cKDTree(np.column_stack([xt, yt]))
    P_prof = []
    for k in ks:
        idx = np.asarray(T.query_ball_point(ref[k], R_FUSTE), dtype=int)
        idx = idx[(ht[idx] >= TRECHO[0]) & (ht[idx] <= TRECHO[1])]
        if len(idx) >= PTS_MIN:
            P_prof.append(np.column_stack([xt[idx] - ref[k][0], yt[idx] - ref[k][1],
                                           ht[idx] - np.mean(TRECHO)]).astype(np.float32))
        else:
            P_prof.append(None)
    tem = np.array([p is not None for p in P_prof])
    print(f"teacher: {tem.sum()} of {len(ks)} stems with a usable TLS cloud")

    # the teacher is validated before teaching, and out of the fold
    idx_ok = np.where(tem)[0]
    Pp = [P_prof[i] for i in idx_ok]
    yp = y[idx_ok]
    bl_p = _rel.blocos_de(Dk.x.to_numpy(float)[idx_ok])
    pred = np.full(len(yp), np.nan)
    embs = np.zeros((len(yp), 256), np.float32)
    for tr, te in bl_p:
        p_, e_ = treina(Pp, yp, tr, te, 0, dev, ESC_FUSTE, JIT_FUSTE)
        pred[te], embs[te] = p_, e_
    r2p, rrp = _rel.marca(yp, pred)
    print(f"teacher on the TLS, OUT OF FOLD: R2 {r2p:+.3f}, rRMSE {rrp:.1f} %")
    print(f"   (for comparison, the version with the wrong scale gave +0.013 on its own training set)\n",
          flush=True)
    saida = [dict(rota="professor no TLS (fora da dobra)", cond="real", R2=r2p, rRMSE=rrp,
                  n=len(yp))]
    if r2p < 0.25:
        print("THE TEACHER DID NOT LEARN ENOUGH. Distilling from it would only transfer noise,")
        print("   so the distillation stage does NOT run. This is a result, not an execution failure:")
        print("   it means that not even the TERRESTRIAL cloud, in this architecture, resolves the stem.")
        pd.DataFrame(saida).to_csv(SAIDA, index=False)
        print(f"\n{SAIDA}")
        return

    prof = np.zeros((len(ks), 256), np.float32)
    prof[idx_ok] = embs
    print("teacher approved. Distilling.\n")
    # The `rota` labels stay in Portuguese: fig_noite_das_hipoteses.py selects rows of the output
    # CSV with `rota.str.startswith("professor")` and `rota == "aluno sozinho"`.
    for peso in (0.0, 0.1, 0.5):
        nome = "aluno sozinho" if peso == 0 else f"aluno destilado (peso {peso})"
        for emb_ctl in (False, True):
            yy = y if not emb_ctl else np.random.default_rng(9).permutation(y)
            r2s = []
            for s in SEEDS:
                pr = np.full(len(yy), np.nan)
                for tr, te in blocos:
                    pr[te] = treina(P, yy, tr, te, s, dev, ESC_COPA, JIT_COPA,
                                    prof=prof, peso=peso)[0]
                r2s.append(_rel.marca(yy, pr)[0])
            tag = "shuffled" if emb_ctl else "real"
            print(f"   {nome:30s} {tag:12s} R2 {np.mean(r2s):+.3f} "
                  f"(sd {np.std(r2s):.3f})", flush=True)
            saida.append(dict(rota=nome, cond=tag, R2=float(np.mean(r2s)),
                              rRMSE=np.nan, n=len(ks)))
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(saida).to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

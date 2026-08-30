#!/usr/bin/env python3
"""Competition is relational, so the model should see the neighbourhood and not a scalar.

The SHAP analysis in `exp_por_que_a_literatura_acerta.py` puts `zq25` first, the lower quartile
of height inside the crown, followed by `zmax_rel`, height relative to the six neighbours, both
indicators of suppression. The little signal that exists is not "this crown is large" but "this
tree is losing to its neighbours". An aggregate scalar of the Hegyi-index kind collapses the
whole neighbourhood into one number and loses track of who is shading whom.

Three levels, so that the gain is attributable:
  own          : only the tree's own features, which is the baseline already measured;
  aggregate    : plus mean, maximum and dispersion of the neighbours' features;
  unrolled     : plus the features of each neighbour, ordered by distance, with the relative
                 geometry (delta x, delta y, distance, azimuth);
  attention    : a small network that reads the set of neighbours and learns whom to weight.
If the unrolled level does not beat the aggregate one, neighbour identity does not matter and
the attention network will not change the result either.

Position enters only as relative geometry, never absolute. Giving absolute x and y to a model
validated by spatial block does not leak, but it would leak in a randomly validated one, and the
comparison with the literature would be contaminated. Only differences enter here.

Positive and shuffling controls, the same as in the sibling experiment: a whole ladder near zero
does not distinguish absence of signal from a broken harness.

Run: PYTHONPATH=. python scripts/exp_relacional_vizinhanca.py
Output: manual_match/relacional_vizinhanca.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from greenvista import config  # noqa: E402

warnings.filterwarnings("ignore")
SAIDA = config.OUT_DIR / "relacional_vizinhanca.csv"
SEED = 20260828
N_BLOCOS = 5
K = 8                      # neighbours considered
N_EMB = 8

PROPRIO = ["n_pts", "log_n", "zmax", "zmean", "zsd", "zskew", "pzabovezmean", "raio_med",
           "raio_p90", "area_copa", "prof_copa", "zq10", "zq25", "zq50", "zq75", "zq90",
           "zq95", "zq99", "vol_copa", "dens_copa", "esbeltez"]
DO_VIZINHO = ["zmax", "area_copa", "prof_copa", "zq25", "n_pts", "raio_p90"]


def blocos_de(x):
    b = np.digitize(x, np.quantile(x, np.linspace(0, 1, N_BLOCOS + 1)[1:-1]))
    return [(np.where(b != k)[0], np.where(b == k)[0]) for k in np.unique(b)]


def marca(y, p):
    res = np.asarray(p) - y
    return (1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2),
            100 * float(np.sqrt(np.mean(res ** 2))) / y.mean())


def vizinhanca(D):
    """Indices, distances and relative geometry of the K nearest neighbours."""
    xy = D[["x", "y"]].to_numpy(float)
    d, j = cKDTree(xy).query(xy, k=K + 1)
    d, j = d[:, 1:], j[:, 1:]
    dx = xy[j, 0] - xy[:, [0]]
    dy = xy[j, 1] - xy[:, [1]]
    return j, d, dx, dy


def blocos_de_feicoes(D):
    """Assemble the three levels of feature set."""
    j, d, dx, dy = vizinhanca(D)
    V = D[DO_VIZINHO].to_numpy(float)          # (n, f)
    biz = V[j]                                  # (n, K, f)
    prop = D[DO_VIZINHO].to_numpy(float)[:, None, :]
    delta = biz - prop                          # what matters is the difference
    out = D[PROPRIO].copy()

    agreg = {}
    for i, c in enumerate(DO_VIZINHO):
        agreg[f"viz_med_{c}"] = biz[:, :, i].mean(1)
        agreg[f"viz_max_{c}"] = biz[:, :, i].max(1)
        agreg[f"viz_sd_{c}"] = biz[:, :, i].std(1)
        agreg[f"dif_med_{c}"] = delta[:, :, i].mean(1)
        agreg[f"dif_max_{c}"] = delta[:, :, i].max(1)
        # how many neighbours are larger: the most direct form of "I am being suppressed"
        agreg[f"n_maior_{c}"] = (delta[:, :, i] > 0).sum(1).astype(float)
    agreg["d_nn1"] = d[:, 0]
    agreg["d_med"] = d.mean(1)
    agreg["d_sd"] = d.std(1)
    AGREG = pd.DataFrame(agreg, index=D.index)

    desd = {}
    for k in range(K):
        desd[f"v{k}_d"] = d[:, k]
        desd[f"v{k}_dx"] = dx[:, k]
        desd[f"v{k}_dy"] = dy[:, k]
        for i, c in enumerate(DO_VIZINHO):
            desd[f"v{k}_dif_{c}"] = delta[:, k, i]
    DESD = pd.DataFrame(desd, index=D.index)
    return out, AGREG, DESD, (j, d, dx, dy, delta)


class Atencao(nn.Module):
    """Reads the set of neighbours and learns who weighs, instead of getting a ready-made mean."""

    def __init__(self, n_prop, n_viz, dim=64):
        super().__init__()
        self.emb_p = nn.Sequential(nn.Linear(n_prop, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.emb_v = nn.Sequential(nn.Linear(n_viz, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.att = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.cab = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(dim, 1))

    def forward(self, p, v):
        ep = self.emb_p(p).unsqueeze(1)
        ev = self.emb_v(v)
        a, _ = self.att(ep, ev, ev)
        return self.cab(torch.cat([ep.squeeze(1), a.squeeze(1)], 1)).squeeze(-1)


def treina_atencao(P, V, y, tr, te, seed, dev, epocas=250):
    torch.manual_seed(seed)
    net = Atencao(P.shape[1], V.shape[2]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
    mu, sd = y[tr].mean(), y[tr].std() + 1e-9
    Pt = torch.tensor(P[tr], dtype=torch.float32, device=dev)
    Vt = torch.tensor(V[tr], dtype=torch.float32, device=dev)
    yt = torch.tensor((y[tr] - mu) / sd, dtype=torch.float32, device=dev)
    perda = nn.SmoothL1Loss()
    rng = np.random.default_rng(seed)
    for _ in range(epocas):
        b = rng.choice(len(tr), min(128, len(tr)), replace=False)
        opt.zero_grad()
        perda(net(Pt[b], Vt[b]), yt[b]).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        p = net(torch.tensor(P[te], dtype=torch.float32, device=dev),
                torch.tensor(V[te], dtype=torch.float32, device=dev)).cpu().numpy()
    return p * sd + mu


def main():
    rng = np.random.default_rng(SEED)
    D = pd.read_csv(config.OUT_DIR / "por_arvore_pred_voronoi.csv", index_col=0)
    PROP, AGREG, DESD, (j, d, dx, dy, delta) = blocos_de_feicoes(D)
    blocos = blocos_de(D.x.to_numpy(float))
    print(f"{len(D)} trees, {K} neighbours, blocks {[len(te) for _, te in blocos]}")

    z = (D.zmax - D.zmax.mean()) / D.zmax.std()
    a = (D.area_copa - D.area_copa.mean()) / D.area_copa.std()
    D["controle"] = 10 + 2.0 * z + 1.5 * a + rng.normal(0, 1.0, len(D))

    conjuntos = {
        "proprio (linha de base)": PROP,
        "proprio + agregado dos vizinhos": pd.concat([PROP, AGREG], axis=1),
        "proprio + agregado + desdobrado": pd.concat([PROP, AGREG, DESD], axis=1),
        "so o agregado dos vizinhos": AGREG,
    }
    modelos = {"floresta": lambda s: RandomForestRegressor(
                   n_estimators=500, min_samples_leaf=3, max_features=0.4,
                   random_state=s, n_jobs=-1),
               "ridge": lambda s: make_pipeline(StandardScaler(),
                                                RidgeCV(alphas=np.logspace(-3, 4, 20)))}
    saida = []
    for alvo, rot in (("dap_cm", "DBH (cm)"), ("vol_m3", "volume (m3)"),
                      ("controle", "positive control")):
        y = D[alvo].to_numpy(float)
        print(f"\n=== target {rot}  mean {y.mean():.3f}  sd {y.std(ddof=1):.3f}")
        for nome, X in conjuntos.items():
            for mn, mk in modelos.items():
                pred = np.full(len(y), np.nan)
                for tr, te in blocos:
                    m = mk(0)
                    m.fit(X.iloc[tr], y[tr])
                    pred[te] = np.asarray(m.predict(X.iloc[te])).ravel()
                r2, rr = marca(y, pred)
                embs = []
                for s in range(N_EMB if mn == "ridge" else 3):
                    ys = y[np.random.default_rng(s).permutation(len(y))]
                    pe = np.full(len(y), np.nan)
                    for tr, te in blocos:
                        m = mk(s)
                        m.fit(X.iloc[tr], ys[tr])
                        pe[te] = np.asarray(m.predict(X.iloc[te])).ravel()
                    embs.append(marca(ys, pe)[0])
                print(f"   {nome:34s} {mn:9s} R2 {r2:+.3f}  rRMSE {rr:5.1f} %"
                      f"   (shuffled {np.mean(embs):+.3f})", flush=True)
                saida.append(dict(alvo=alvo, conjunto=nome, modelo=mn, R2=r2, rRMSE=rr,
                                  R2_embaralhado=float(np.mean(embs)), n=len(D)))

    # ---- the attention network over the set of neighbours -------------------------------
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== attention network over the neighbourhood ({dev})")
    Pn = StandardScaler().fit_transform(PROP.to_numpy(float))
    Vn = np.concatenate([delta, d[:, :, None], dx[:, :, None], dy[:, :, None]], axis=2)
    Vn = (Vn - Vn.reshape(-1, Vn.shape[2]).mean(0)) / (Vn.reshape(-1, Vn.shape[2]).std(0) + 1e-9)
    for alvo, rot in (("dap_cm", "DBH (cm)"), ("controle", "positive control")):
        y = D[alvo].to_numpy(float)
        for embaralha in (False, True):
            yy = y if not embaralha else np.random.default_rng(1).permutation(y)
            r2s = []
            for s in range(3):
                pred = np.full(len(yy), np.nan)
                for tr, te in blocos:
                    pred[te] = treina_atencao(Pn, Vn, yy, tr, te, s, dev)
                r2s.append(marca(yy, pred)[0])
            tag = "embaralhado" if embaralha else "real"
            print(f"   {rot:20s} {tag:12s} R2 {np.mean(r2s):+.3f} "
                  f"(sd across seeds {np.std(r2s):.3f})")
            saida.append(dict(alvo=alvo, conjunto="atencao na vizinhanca", modelo=tag,
                              R2=float(np.mean(r2s)), rRMSE=np.nan,
                              R2_embaralhado=np.nan, n=len(D)))

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(saida).to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

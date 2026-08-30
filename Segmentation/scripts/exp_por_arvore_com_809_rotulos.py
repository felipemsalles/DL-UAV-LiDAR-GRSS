#!/usr/bin/env python3
"""Per-tree DBH and volume with 809 labels, not with 23.

`exp_per_tree_dap.py` in this repository trained per-tree networks on 23 georeferenced
destructively scaled trees and reached an R2 of 0.086 with allometry and 0.111 with a
pretrained PointNet. With 23 labels in a single stand, that number does not separate method
from chance. `exp_volume_tls_por_arvore.py` produced 809 stems with DBH and volume measured on
the terrestrial laser, which makes it possible to measure what the crown seen from above says
about the stem.

The volume target carries the drone height. The label is V = g(DBH_TLS) * H_drone * f_TLS, so
it contains H_drone, which is roughly the `zmax` of the crown itself, and a model that reads
zmax already wins that share. Hence:
  1. the main target is DBH, measured only on the TLS, with no drone inside it at all;
  2. volume comes second, always alongside the "zmax only" row, which is the free floor;
  3. there is a ladder of feature sets, and what matters is the increment of each rung.

Three definitions of crown:
  disc     : everything within a radius of the stem position, needs no segmentation at all;
  voronoi  : each point goes to the nearest mapped stem, i.e. oracle segmentation, because it
             uses the true stem map that no method has in production;
  ff3d     : the instance the network actually segmented, matched to the stem.
If the oracle does not beat the disc, better segmentation does not help volume.

Spatial-block validation. The 892 stems lie in a 147 x 44 m strip and a neighbouring tree
shares crown, terrain and genetic material; random splitting puts a neighbour in both train and
test and inflates the result. Here there are five blocks in x, one held out at a time, plus a
shuffling control.

Run: PYTHONPATH=. python scripts/exp_por_arvore_com_809_rotulos.py
Output: manual_match/por_arvore_809.csv
"""
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree, ConvexHull, QhullError
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from greenvista import config  # noqa: E402
from greenvista.segmentation.ff3d import load_panoptic_ply  # noqa: E402

warnings.filterwarnings("ignore")

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
DRONE = config.REPO / "work/lazall/SaoManuelTotal_001.laz"
# The panoptic PLY comes out in local coordinates, centred on the mean of the input tile, not
# in UTM. Without undoing that offset, the matching against the stem map finds nothing and the
# table comes out empty. The recovery is the same as in
# `ff3d_detection_overlap.instance_centroids`: add the tile mean back.
FF3D_TILES = [(config.REPO / "work/ff3d_degraded_out/full/t001_p001_round2.ply",
               config.REPO / "work/ff3d_degraded/full/t001_p001.laz"),
              (config.REPO / "work/ff3d_degraded_out/full/t001_p002_round2.ply",
               config.REPO / "work/ff3d_degraded/full/t001_p002.laz")]
SAIDA = config.OUT_DIR / "por_arvore_809.csv"

Z_MIN, Z_MAX = 2.0, 45.0
R_DISCO = 1.5
R_VORONOI = 2.5
TOL_PAREIA = 2.0           # the same matching tolerance as in the paper
PTS_MIN = 40
N_BLOCOS = 5
N_EMB = 20
SEED = 20260828

QS = (10, 25, 50, 75, 90, 95, 99)
MELHOR = "tudo + competicao, floresta"     # the rung that paid off most, used in the aggregation
AREA_PARCELA = 400.0


def feicoes(p):
    """Metrics of one crown. None of them depends on the stem position in the world."""
    z = p[:, 2]
    d = np.hypot(p[:, 0] - p[:, 0].mean(), p[:, 1] - p[:, 1].mean())
    area = np.nan
    if len(p) >= 4:
        try:
            area = float(ConvexHull(p[:, :2]).volume)
        except (QhullError, ValueError):
            area = np.nan
    f = {"n_pts": float(len(p)), "log_n": float(np.log(len(p))),
         "zmax": float(z.max()), "zmean": float(z.mean()), "zsd": float(z.std(ddof=1)),
         "zskew": float(((z - z.mean()) ** 3).mean() / (z.std() ** 3 + 1e-9)),
         "pzabovezmean": float(100 * np.mean(z > z.mean())),
         "raio_med": float(np.median(d)), "raio_p90": float(np.quantile(d, 0.90)),
         "area_copa": area, "prof_copa": float(z.max() - np.quantile(z, 0.05))}
    for q in QS:
        f[f"zq{q}"] = float(np.quantile(z, q / 100))
    f["vol_copa"] = f["area_copa"] * f["prof_copa"]
    f["dens_copa"] = f["n_pts"] / max(f["area_copa"], 1e-6)
    f["esbeltez"] = f["prof_copa"] / max(f["raio_p90"], 1e-6)
    return f


def competicao(F, ref, k=6, raio=5.0):
    """Neighbourhood indices, what silviculture uses when the crown alone does not explain.

    In an even-aged plantation what makes one tree larger than its neighbour is competition, and
    the Hegyi-type index (neighbour size over distance, summed) has been the standard instrument
    for that since the 1970s. The crown seen from above may not show stem diameter and yet still
    show which tree is being suppressed, and a suppressed tree is thin.

    The position comes from the true map, i.e. it is an oracle; in production it would come from
    the segmentation. That is consistent with the rest of the experiment, which already
    conditions on knowing where the tree is.
    """
    xy = ref[F.index.values]
    T = cKDTree(xy)
    d, j = T.query(xy, k=min(k + 1, len(xy)))
    d, j = d[:, 1:], j[:, 1:]                      # drop the tree itself
    z = F.zmax.values
    a = F.area_copa.values
    out = {"d_nn1": d[:, 0], "d_nn_med": d.mean(1),
           "zmax_rel": z - z[j].mean(1), "area_rel": a / np.maximum(a[j].mean(1), 1e-6)}
    viz = T.query_ball_point(xy, raio)
    n_viz, hegyi = np.zeros(len(xy)), np.zeros(len(xy))
    for i, vs in enumerate(viz):
        vs = [v for v in vs if v != i]
        n_viz[i] = len(vs)
        if vs:
            dd = np.hypot(xy[vs, 0] - xy[i, 0], xy[vs, 1] - xy[i, 1])
            hegyi[i] = float(np.sum((a[vs] / max(a[i], 1e-6)) / np.maximum(dd, 0.3)))
    out["n_viz_5m"] = n_viz
    out["hegyi"] = hegyi
    for c, v in out.items():
        F[c] = v
    return F


# Each rung of the ladder adds one kind of information, and what one reads is the increment.
# Reporting only the full model hides that height alone was already doing almost everything.
COMP = ["d_nn1", "d_nn_med", "zmax_rel", "area_rel", "n_viz_5m", "hegyi"]
GEOM = ["area_copa", "raio_med", "raio_p90", "log_n", "dens_copa"]
VERT = ["zmax", "zmean", "zsd", "zskew", "pzabovezmean", "prof_copa", "esbeltez"] + \
       [f"zq{q}" for q in QS]
ESCADA = [("nulo, prediz a media", None, "media"),
          ("so a altura (zmax)", ["zmax"], "ols"),
          ("so a geometria da copa", GEOM, "ridge"),
          ("altura + geometria", ["zmax"] + GEOM, "ridge"),
          ("perfil vertical inteiro", VERT, "ridge"),
          ("tudo, ridge", VERT + GEOM, "ridge"),
          ("tudo, floresta aleatoria", VERT + GEOM, "rf"),
          ("alometria log(zmax, area)", ["zmax", "area_copa"], "alom"),
          ("so competicao", COMP, "ridge"),
          ("tudo + competicao, ridge", VERT + GEOM + COMP, "ridge"),
          ("tudo + competicao, floresta", VERT + GEOM + COMP, "rf")]


def faz_modelo(tipo, seed=0):
    if tipo == "ols":
        return LinearRegression()
    if tipo == "ridge":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 20)))
    if tipo == "rf":
        return RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                     max_features=0.5, random_state=seed, n_jobs=-1)
    raise ValueError(tipo)


class Alometria:
    """ln y = a + b ln x1 + c ln x2, with the Baskerville correction on the way back."""

    def fit(self, X, y):
        L = np.log(np.clip(np.asarray(X, float), 1e-6, None))
        ly = np.log(np.clip(y, 1e-9, None))
        self.m = LinearRegression().fit(L, ly)
        self.c = float(np.exp(np.var(ly - self.m.predict(L), ddof=L.shape[1] + 1) / 2))
        return self

    def predict(self, X):
        L = np.log(np.clip(np.asarray(X, float), 1e-6, None))
        return self.c * np.exp(self.m.predict(L))


def cv_por_bloco(F, y, bloco, cols, tipo, seed=0):
    """Spatial block held out, model fitted on the training part only."""
    pred = np.full(len(y), np.nan)
    for b in np.unique(bloco):
        te = bloco == b
        if tipo == "media":
            pred[te] = y[~te].mean()
            continue
        m = Alometria() if tipo == "alom" else faz_modelo(tipo, seed)
        m.fit(F.loc[~te, cols], y[~te])
        pred[te] = np.asarray(m.predict(F.loc[te, cols])).ravel()
    return pred


def marca(y, p):
    res = p - y
    return dict(R2=1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2),
                RMSE=float(np.sqrt(np.mean(res ** 2))),
                rRMSE=100 * float(np.sqrt(np.mean(res ** 2))) / y.mean())


def copas_disco(T, xyz, ref, raio):
    for cx, cy in ref:
        idx = T.query_ball_point([cx, cy], raio)
        yield (xyz[np.asarray(idx, int)] if idx else np.empty((0, 3)))


def copas_voronoi(xyz, ref, raio):
    """Each point goes to the nearest mapped stem. Oracle segmentation."""
    Tr = cKDTree(ref)
    d, j = Tr.query(xyz[:, :2], distance_upper_bound=raio)
    ordem = np.argsort(j, kind="stable")
    j_ord, xyz_ord = j[ordem], xyz[ordem]
    corte = np.searchsorted(j_ord, np.arange(len(ref) + 1))
    for k in range(len(ref)):
        yield xyz_ord[corte[k]:corte[k + 1]]


def copas_ff3d(ref):
    """Instance segmented by the network, matched to the stem nearest its base."""
    out = {}
    for t, backup in FF3D_TILES:
        if not t.exists() or not backup.exists():
            continue
        las = laspy.read(str(backup))
        desl = np.array([float(np.asarray(las.x).mean()), float(np.asarray(las.y).mean()), 0.0])
        for iid, tr in load_panoptic_ply(t).items():
            p = tr["points"] + desl
            if len(p) < PTS_MIN:
                continue
            baixo = p[p[:, 2] <= np.quantile(p[:, 2], 0.10)]
            base = baixo[:, :2].mean(0) if len(baixo) else p[:, :2].mean(0)
            d = np.hypot(ref[:, 0] - base[0], ref[:, 1] - base[1])
            k = int(d.argmin())
            if d[k] <= TOL_PAREIA and (k not in out or len(p) > len(out[k])):
                out[k] = p
    return out


def tabela(gerador, ref, n):
    linhas, quais = [], []
    for k, p in gerador:
        if len(p) < PTS_MIN:
            continue
        p = p[(p[:, 2] >= Z_MIN) & (p[:, 2] <= Z_MAX)]
        if len(p) < PTS_MIN:
            continue
        linhas.append(feicoes(p))
        quais.append(k)
    F = pd.DataFrame(linhas, index=quais)
    return F[F.notna().all(1)]


def roda(nome, F, alvos, rng):
    saida = []
    if len(F) < 40:
        print(f"\n{nome}: only {len(F)} stems, not run")
        return saida
    bloco = np.digitize(F.x.values, np.quantile(F.x.values,
                                                np.linspace(0, 1, N_BLOCOS + 1)[1:-1]))
    print(f"\n{'=' * 78}\n{nome}  |  n = {len(F)}  |  blocks {np.bincount(bloco)}")
    for alvo, rotulo in alvos:
        y = F[alvo].values.astype(float)
        print(f"\n  target: {rotulo}   mean {y.mean():.3f}  sd {y.std(ddof=1):.3f}")
        for desc, cols, tipo in ESCADA:
            cols = cols or ["zmax"]
            if tipo == "alom" and y.min() <= 0:
                continue          # the log of a non-positive target does not exist
            p = cv_por_bloco(F, y, bloco, cols, tipo)
            m = marca(y, p)
            emb = np.nan
            if tipo not in ("media",):
                r2s = [marca(y, cv_por_bloco(F, y[rng.permutation(len(y))], bloco, cols, tipo))["R2"]
                       for _ in range(N_EMB if tipo != "rf" else 5)]
                emb = float(np.mean(r2s))
            print(f"    {desc:28s} R2 {m['R2']:+.3f}  rRMSE {m['rRMSE']:5.1f} %"
                  + (f"   (shuffled {emb:+.3f})" if np.isfinite(emb) else ""), flush=True)
            saida.append(dict(copa=nome, alvo=alvo, modelo=desc, n=len(F),
                              R2_embaralhado=emb, **m))
            if desc == MELHOR:
                F[f"pred_{alvo}"] = p
    return saida


def main():
    rng = np.random.default_rng(SEED)
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    vol = pd.read_csv(config.OUT_DIR / "volume_tls_por_arvore.csv")
    ran = pd.read_csv(config.OUT_DIR / "dap_tls_ransac_talhao001.csv")
    rot = pd.DataFrame({"x": ref[:, 0], "y": ref[:, 1],
                        "dap_cm": np.where(ran.motivo.values == "ok", ran.dap_cm.values, np.nan),
                        "vol_m3": vol.vol_m3.values, "H_m": vol.H_m.values})
    rot["g_m2"] = np.pi * (rot.dap_cm / 200) ** 2
    print(f"{rot.dap_cm.notna().sum()} stems with DBH, {rot.vol_m3.notna().sum()} with volume")

    las = laspy.read(str(DRONE))
    cls = np.asarray(las.classification)
    m = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    xyz = np.column_stack([np.asarray(las.x)[m], np.asarray(las.y)[m], np.asarray(las.z)[m]])
    print(f"{len(xyz):,} vegetation points in stand 001")

    T = cKDTree(xyz[:, :2])
    alvos = [("dap_cm", "DBH (cm), TLS only, with no drone inside it"),
             ("vol_m3", "volume (m3), contains the drone height"),
             # Positive control. A column of near-zero R2 from top to bottom can be absence of
             # signal or a broken harness, and the two readings are indistinguishable without
             # this. This target is a function of the crown features plus noise, so the ladder
             # must rise; if it does not, the problem is in the code and not in the data.
             ("controle", "positive control, a function of the crown itself plus noise")]

    conjuntos = {
        f"disco de {R_DISCO} m": enumerate(copas_disco(T, xyz, ref, R_DISCO)),
        "voronoi no mapa (oraculo)": enumerate(copas_voronoi(xyz, ref, R_VORONOI)),
        "instancia do FF3D": iter(copas_ff3d(ref).items()),
    }
    saida = []
    tabelas = {}
    for nome, ger in conjuntos.items():
        F = tabela(ger, ref, len(ref))
        F = F.join(rot).dropna(subset=["dap_cm", "vol_m3"])
        z = (F.zmax - F.zmax.mean()) / F.zmax.std()
        a = (F.area_copa - F.area_copa.mean()) / F.area_copa.std()
        F["controle"] = 10 + 2.0 * z + 1.5 * a + rng.normal(0, 1.0, len(F))
        F = competicao(F, ref)
        # the raw correlation, which is the classic crown allometry and which is absent here
        for c in ("area_copa", "zmax", "raio_p90", "n_pts"):
            r = float(np.corrcoef(F[c], F.dap_cm)[0, 1])
            print(f"   r({c:10s}, DBH) = {r:+.3f}", end="   ")
        print()
        tabelas[nome] = F
        saida += roda(nome, F, alvos, rng)

    # Paired comparison. The three crown definitions do not cover the same stems, so comparing
    # the R2 of the full tables compares different sets of trees. The question "does segmentation
    # help?" only has an answer on the stems all three definitions reach.
    comum = set.intersection(*(set(F.index) for F in tabelas.values()))
    print(f"\n{'=' * 78}\nPAIRED on the {len(comum)} stems all three definitions reach")
    for nome, F in tabelas.items():
        Fc = F.loc[sorted(comum)]
        saida += roda(f"{nome} [pareado]", Fc, alvos, rng)

    # ---- why the plot works and the tree does not --------------------------------------
    # If the per-tree error is largely independent between neighbouring trees, it shrinks with
    # the square root of the number of trees when summed over the plot. A per-tree R2 of zero is
    # therefore compatible with a useful per-plot result, and how much it shrinks is measurable
    # here.
    print(f"\n{'=' * 78}\nAGGREGATION: the same model, summed over a {AREA_PARCELA:.0f} m2 plot")
    Rp = np.sqrt(AREA_PARCELA / np.pi)
    for nome, F in tabelas.items():
        if f"pred_dap_cm" not in F.columns:
            continue
        xy = ref[F.index.values]
        gx = np.arange(xy[:, 0].min() + Rp, xy[:, 0].max() - Rp, Rp)
        gy = np.arange(xy[:, 1].min() + Rp, xy[:, 1].max() - Rp, Rp)
        linhas = []
        for cx in gx:
            for cy in gy:
                k = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) <= Rp
                if k.sum() < 15:
                    continue
                linhas.append(dict(
                    dap_ver=F.dap_cm.values[k].mean(), dap_pre=F.pred_dap_cm.values[k].mean(),
                    vol_ver=F.vol_m3.values[k].sum(), vol_pre=F.pred_vol_m3.values[k].sum(),
                    # The plot total is dominated by the count: the predicted total is about n
                    # times the predicted mean, and if the model predicts close to the mean the
                    # R2 of the total measures the variation of n, i.e. the count and not the
                    # sizing. The per-tree mean removes n from the equation and shows what is
                    # left.
                    volm_ver=F.vol_m3.values[k].mean(), volm_pre=F.pred_vol_m3.values[k].mean(),
                    n=int(k.sum())))
        if len(linhas) < 5:
            continue
        P = pd.DataFrame(linhas)
        for alvo, ver, pre in (("DAP medio", "dap_ver", "dap_pre"),
                               ("volume total", "vol_ver", "vol_pre"),
                               ("volume medio", "volm_ver", "volm_pre")):
            e = P[pre] - P[ver]
            rr = 100 * float(np.sqrt(np.mean(e ** 2))) / P[ver].mean()
            r2 = 1 - np.sum(e ** 2) / np.sum((P[ver] - P[ver].mean()) ** 2)
            arv = [d for d in saida if d["copa"] == nome and d["modelo"] == MELHOR
                   and d["alvo"] == ("dap_cm" if alvo.startswith("DAP") else "vol_m3")]
            # correlation between the true total and the tree count
            if alvo == "volume total":
                rc = float(np.corrcoef(P[ver], P.n)[0, 1])
                print(f"   {nome:26s} {'(count)':13s} r(plot total volume, "
                      f"n of trees) = {rc:+.3f}")
            base = arv[0]["rRMSE"] if arv else float("nan")
            print(f"   {nome:26s} {alvo:13s} {len(P):3d} plots of {P.n.mean():.0f} trees"
                  f"   per tree {base:5.1f} %  ->  per plot {rr:5.1f} %   R2 {r2:+.3f}")
            saida.append(dict(copa=nome, alvo=alvo, modelo=f"{MELHOR} [agregado]",
                              n=len(P), R2=r2, RMSE=float(np.sqrt(np.mean(e ** 2))), rRMSE=rr,
                              R2_embaralhado=np.nan))
    print("   the virtual plots overlap and come from a single stand, so this shows the SCALE "
          "of the shrinkage, not a production number.")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    # store the per-tree predictions, so as not to rerun the pipeline for every new figure
    for nome, F in tabelas.items():
        if "pred_dap_cm" in F.columns:
            F.assign(copa=nome).to_csv(
                config.OUT_DIR / f"por_arvore_pred_{nome.split()[0]}.csv", index=True)
    pd.DataFrame(saida).to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

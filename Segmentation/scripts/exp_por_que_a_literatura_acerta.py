#!/usr/bin/env python3
"""Why does the literature report a per-tree R2 of 0.41 while we find 0.10?

`exp_por_arvore_com_809_rotulos.py` found an R2 of 0.05 to 0.10 for per-tree DBH, and concluded
that the nadir crown does not carry stem diameter, which collides with the literature. Dalla
Corte et al. 2020, on eucalyptus, with UAV-LiDAR at 1500 to 2500 pts/m2 and 370 trees, report an
R2 of 0.41 and an rRMSE of 13.3 % for per-tree DBH with random forest. Their DBH standard
deviation, recovered from the identity rRMSE = CV * sqrt(1 - R2), is 4.95 cm, almost identical to
our 4.89, so the difference is not explained by population variability.

Their methods section describes: "the data set was divided into training (70%) and testing
(30%) using stratified sampling based on tree diameter", i.e. a random split of individual
trees, with two problems stacked on top of each other:
 1. a neighbouring tree falls on both sides. In a plantation with 2.4 m spacing, a neighbour
    shares terrain, genetic material, competition history and even overlapping crown points. The
    model does not need to learn the crown-stem relation, it is enough to recognise the
    neighbourhood.
 2. the stratification is done on the target itself, which guarantees that the test set covers
    the same DBH distribution as the training set and inflates R2.

Their protocol is the standard across much of the field. This script runs both protocols on the
same data and measures how much the number changes. If the random split takes our R2 from 0.10 to
near 0.4, the two results are compatible and what separates them is the validation design; if it
does not, the earlier conclusion needs revisiting.

Three further questions answered by measurement:

 B. Are 809 trees too few? Learning curve. If R2 is still rising at 800, more labels help. If it
    has been flat since 200, they do not.
 C. Where is the little signal that exists? Permutation importance and SHAP.
 D. Can dominant be separated from suppressed, even without getting the value right? Exact
    regression is one question; ranking a tree into a size class is another, easier and useful
    for operational decisions.

Run: PYTHONPATH=. python scripts/exp_por_que_a_literatura_acerta.py
Output: manual_match/por_que_a_literatura_acerta.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from greenvista import config  # noqa: E402

warnings.filterwarnings("ignore")
SAIDA = config.OUT_DIR / "por_que_a_literatura_acerta.csv"
SEED = 20260828
N_BLOCOS = 5

FEATS = ["n_pts", "log_n", "zmax", "zmean", "zsd", "zskew", "pzabovezmean", "raio_med",
         "raio_p90", "area_copa", "prof_copa", "zq10", "zq25", "zq50", "zq75", "zq90",
         "zq95", "zq99", "vol_copa", "dens_copa", "esbeltez",
         "d_nn1", "d_nn_med", "zmax_rel", "area_rel", "n_viz_5m", "hegyi"]


def rf(seed=0):
    return RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.5,
                                 random_state=seed, n_jobs=-1)


def marca(y, p):
    res = np.asarray(p) - y
    return (1 - np.sum(res ** 2) / np.sum((y - y.mean()) ** 2),
            100 * float(np.sqrt(np.mean(res ** 2))) / y.mean())


def cv_generico(F, y, folds, seed=0):
    pred = np.full(len(y), np.nan)
    for tr, te in folds:
        m = rf(seed)
        m.fit(F.iloc[tr], y[tr])
        pred[te] = m.predict(F.iloc[te])
    return pred


def folds_bloco(x):
    b = np.digitize(x, np.quantile(x, np.linspace(0, 1, N_BLOCOS + 1)[1:-1]))
    return [(np.where(b != k)[0], np.where(b == k)[0]) for k in np.unique(b)]


def main():
    rng = np.random.default_rng(SEED)
    F = pd.read_csv(config.OUT_DIR / "por_arvore_pred_voronoi.csv", index_col=0)
    y = F.dap_cm.to_numpy(float)
    yv = F.vol_m3.to_numpy(float)
    X = F[FEATS]
    print(f"{len(F)} trees, mean DBH {y.mean():.2f} cm, sd {y.std(ddof=1):.2f}, "
          f"CV {100 * y.std(ddof=1) / y.mean():.1f} %")
    print(f"for comparison, Dalla Corte 2020: implied sd 4.95 cm, R2 0.41, rRMSE 13.3 %\n")
    linhas = []

    # ---- A. the validation protocol, which is the main hypothesis -----------------------
    print("[A] the same data, both validation protocols")
    protocolos = {
        "bloco espacial (o nosso)": folds_bloco(F.x.to_numpy(float)),
        "aleatorio 5 dobras": list(KFold(5, shuffle=True, random_state=SEED).split(X)),
        "aleatorio estratificado no ALVO (o deles)":
            list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(
                X, pd.qcut(y, 5, labels=False, duplicates="drop"))),
    }
    for nome, folds in protocolos.items():
        r2, rr = marca(y, cv_generico(X, y, folds))
        r2v, rrv = marca(yv, cv_generico(X, yv, folds))
        print(f"   {nome:42s} DBH R2 {r2:+.3f} rRMSE {rr:5.1f} %   "
              f"volume R2 {r2v:+.3f} rRMSE {rrv:5.1f} %")
        linhas.append(dict(parte="A protocolo", cond=nome, alvo="dap_cm", R2=r2, rRMSE=rr))
        linhas.append(dict(parte="A protocolo", cond=nome, alvo="vol_m3", R2=r2v, rRMSE=rrv))

    # Leakage test: if the gain of the random split comes from the neighbourhood, giving the
    # model only the position (x, y) must reproduce much of it. Position carries no information
    # at all about the stem; if it "predicts" DBH, what is there is spatial leakage.
    print("\n   leakage control: ONLY the position (x, y) as predictor")
    XY = F[["x", "y"]]
    for nome, folds in protocolos.items():
        r2, rr = marca(y, cv_generico(XY, y, folds))
        print(f"   {nome:42s} DBH R2 {r2:+.3f}")
        linhas.append(dict(parte="A vazamento", cond=nome, alvo="dap_cm (so x,y)", R2=r2,
                           rRMSE=rr))

    # ---- B. are 809 too few? ------------------------------------------------------------
    print("\n[B] learning curve, spatial blocks")
    blocos = folds_bloco(F.x.to_numpy(float))
    for n in (50, 100, 200, 400, 600, len(F)):
        r2s = []
        for s in range(3):
            r = np.random.default_rng(SEED + s)
            pred = np.full(len(y), np.nan)
            for tr, te in blocos:
                tr_s = r.choice(tr, min(n, len(tr)), replace=False)
                m = rf(s)
                m.fit(X.iloc[tr_s], y[tr_s])
                pred[te] = m.predict(X.iloc[te])
            r2s.append(marca(y, pred)[0])
        print(f"   {n:4d} training trees per fold   R2 {np.mean(r2s):+.3f} "
              f"(sd across seeds {np.std(r2s):.3f})")
        linhas.append(dict(parte="B curva", cond=f"n={n}", alvo="dap_cm",
                           R2=float(np.mean(r2s)), rRMSE=np.nan))

    # ---- C. where the little signal is --------------------------------------------------
    print("\n[C] where the little signal that exists comes from")
    m = rf(0).fit(X, y)
    imp = pd.Series(m.feature_importances_, index=FEATS).sort_values(ascending=False)
    print("   random-forest importance (fitted on everything, only to rank):")
    for k, v in imp.head(8).items():
        print(f"      {k:14s} {v:.3f}")
    try:
        import shap
        sv = shap.TreeExplainer(m).shap_values(X.sample(min(300, len(X)), random_state=0))
        s = pd.Series(np.abs(sv).mean(0), index=FEATS).sort_values(ascending=False)
        print("   SHAP, mean absolute value (in cm of DBH):")
        for k, v in s.head(8).items():
            print(f"      {k:14s} {v:.3f}")
        for k, v in s.items():
            linhas.append(dict(parte="C shap", cond=k, alvo="dap_cm", R2=np.nan, rRMSE=np.nan,
                               shap=float(v), importancia=float(imp[k])))
    except Exception as e:
        print(f"   shap unavailable: {e}")

    # permutation importance, but out of fold, which is the one that counts
    print("\n   OUT-OF-FOLD permutation importance (drop in R2 when the column is shuffled):")
    base = marca(y, cv_generico(X, y, blocos))[0]
    quedas = {}
    for c in FEATS:
        Xp = X.copy()
        Xp[c] = rng.permutation(Xp[c].values)
        quedas[c] = base - marca(y, cv_generico(Xp, y, blocos))[0]
    for k, v in sorted(quedas.items(), key=lambda kv: -kv[1])[:8]:
        print(f"      {k:14s} {v:+.4f}")
        linhas.append(dict(parte="C permuta", cond=k, alvo="dap_cm", R2=base - v, rRMSE=np.nan))

    # ---- D. can dominant be told from suppressed? ---------------------------------------
    # A different and easier question: getting DBH right in centimetres is one thing, saying
    # whether the tree is in the bottom, middle or top tercile is another. The second is enough
    # for many operational decisions, and a model can get the order right without the value.
    print("\n[D] can the tree at least be RANKED into a size class?")
    cls = pd.qcut(y, 3, labels=[0, 1, 2]).astype(int)
    pred = np.full(len(y), -1)
    for tr, te in blocos:
        c = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.5,
                                   random_state=0, n_jobs=-1)
        c.fit(X.iloc[tr], cls[tr])
        pred[te] = c.predict(X.iloc[te])
    acc = float((pred == cls).mean())
    from scipy.stats import spearmanr
    rho = float(spearmanr(cv_generico(X, y, blocos), y).statistic)
    extremos = (cls != 1)
    acc_ex = float((pred[extremos] == cls[extremos]).mean())
    print(f"   tercile accuracy: {100 * acc:.1f} %   (chance = 33.3 %)")
    print(f"   accuracy on the extremes only (terciles 1 and 3): {100 * acc_ex:.1f} %")
    print(f"   rank correlation between predicted and measured: rho = {rho:+.3f}")
    linhas.append(dict(parte="D classe", cond="acerto do tercil", alvo="dap_cm", R2=acc,
                       rRMSE=np.nan))
    linhas.append(dict(parte="D classe", cond="spearman", alvo="dap_cm", R2=rho, rRMSE=np.nan))

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

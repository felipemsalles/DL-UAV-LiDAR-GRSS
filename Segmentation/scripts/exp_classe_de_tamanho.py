#!/usr/bin/env python3
"""Rank the tree into a size class, the formulation that showed a signal.

`exp_por_que_a_literatura_acerta.py` measured 48.6 % accuracy on the size tercile against
33.3 % by chance, with a rank correlation of +0.353, while regression of the value stays at
R2 0.10. Estimating DBH in centimetres is one question; saying whether the tree is large,
medium or small is another, and it suffices for harvest, assortment and mapping decisions.

Ordinary classification ignores that the classes are ordered: mistaking "small" for "medium" is
less serious than mistaking "small" for "large". Hence three elements enter: the quadratic
weighted kappa, which penalises by the square of the distance between classes; the rank
correlation; and an ordinal classifier, built as cumulative binary classifiers (greater than
class 1? greater than class 2?), which respects the order without inventing a distance between
classes.

In the end the predicted classes become volume through the class mean, compared against
predicting the overall mean for everyone, which is the null model.

Validation by spatial block and a shuffling control.

Usage: PYTHONPATH=. python scripts/exp_classe_de_tamanho.py
Output: manual_match/classe_de_tamanho.csv
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, f1_score
from scipy.stats import spearmanr

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
SAIDA = config.OUT_DIR / "classe_de_tamanho.csv"
SEED = 20260828
N_CLASSES = (2, 3, 4, 5)


def rf(seed=0):
    return RandomForestClassifier(500, min_samples_leaf=3, max_features=0.4,
                                  random_state=seed, n_jobs=-1)


def ordinal(Xtr, ytr, Xte, n_cls, seed=0):
    """Ordinal classifier from cumulative binaries: P(y > k) for each threshold k."""
    p = np.zeros((len(Xte), n_cls))
    acum = np.ones(len(Xte))
    for k in range(n_cls - 1):
        alvo = (ytr > k).astype(int)
        if alvo.min() == alvo.max():
            continue
        pk = rf(seed).fit(Xtr, alvo).predict_proba(Xte)[:, 1]
        p[:, k] = acum - pk
        acum = pk
    p[:, -1] = acum
    return p.argmax(1)


def avalia(y_cls, pred, y_val, n_cls):
    return dict(acerto=float((pred == y_cls).mean()), acaso=1.0 / n_cls,
                kappa_quad=float(cohen_kappa_score(y_cls, pred, weights="quadratic")),
                f1_macro=float(f1_score(y_cls, pred, average="macro")),
                spearman=float(spearmanr(pred, y_val).statistic))


def main():
    D = pd.read_csv(config.OUT_DIR / "por_arvore_pred_voronoi.csv", index_col=0)
    PROP, AGREG, DESD, _ = _rel.blocos_de_feicoes(D)
    X = pd.concat([PROP, AGREG], axis=1)
    blocos = _rel.blocos_de(D.x.to_numpy(float))
    print(f"{len(D)} trees, {X.shape[1]} features (crown + neighbourhood)\n")

    linhas = []
    for alvo, rot in (("dap_cm", "DBH"), ("vol_m3", "volume")):
        yv = D[alvo].to_numpy(float)
        print(f"=== target {rot}")
        print(f"   {'classes':>8} {'model':>10} {'accuracy':>8} {'chance':>7} "
              f"{'quad kappa':>11} {'macro F1':>9} {'rank':>7} {'shuffled':>12}")
        for n_cls in N_CLASSES:
            ycls = pd.qcut(yv, n_cls, labels=False, duplicates="drop")
            n_real = int(ycls.max()) + 1
            for nome, fn in (("plain", None), ("ordinal", ordinal)):
                pred = np.full(len(yv), -1)
                for tr, te in blocos:
                    if fn is None:
                        pred[te] = rf().fit(X.iloc[tr], ycls[tr]).predict(X.iloc[te])
                    else:
                        pred[te] = fn(X.iloc[tr], ycls[tr], X.iloc[te], n_real)
                m = avalia(ycls, pred, yv, n_real)
                # shuffling control
                ke = []
                for s in range(3):
                    ys = ycls[np.random.default_rng(s).permutation(len(ycls))]
                    pe = np.full(len(yv), -1)
                    for tr, te in blocos:
                        pe[te] = (rf(s).fit(X.iloc[tr], ys[tr]).predict(X.iloc[te])
                                  if fn is None else fn(X.iloc[tr], ys[tr], X.iloc[te], n_real, s))
                    ke.append(float(cohen_kappa_score(ys, pe, weights="quadratic")))
                print(f"   {n_real:8d} {nome:>10} {100*m['acerto']:7.1f}% "
                      f"{100*m['acaso']:6.1f}% {m['kappa_quad']:11.3f} {m['f1_macro']:9.3f} "
                      f"{m['spearman']:+7.3f} {np.mean(ke):+12.3f}", flush=True)
                linhas.append(dict(alvo=alvo, n_classes=n_real, modelo=nome,
                                   kappa_embaralhado=float(np.mean(ke)), **m))

    # ---- what the class costs in the volume account ---------------------------------------
    print("\n=== what classifying costs in the volume account")
    v = D.vol_m3.to_numpy(float)
    for n_cls in N_CLASSES:
        ycls = pd.qcut(v, n_cls, labels=False, duplicates="drop")
        n_real = int(ycls.max()) + 1
        pred_v = np.full(len(v), np.nan)
        for tr, te in blocos:
            c = rf().fit(X.iloc[tr], ycls[tr]).predict(X.iloc[te])
            medias = {k: v[tr][ycls[tr] == k].mean() for k in range(n_real)}
            pred_v[te] = [medias[k] for k in c]
        r2, rr = _rel.marca(v, pred_v)
        print(f"   {n_real} classes, volume from the class mean: R2 {r2:+.3f}  rRMSE {rr:5.1f} %")
        linhas.append(dict(alvo="vol_por_classe", n_classes=n_real, modelo="class mean",
                           acerto=np.nan, acaso=np.nan, kappa_quad=np.nan, f1_macro=np.nan,
                           spearman=r2, kappa_embaralhado=rr))
    r2n, rrn = _rel.marca(v, np.full(len(v), np.nan))if False else (0.0, 100*v.std()/v.mean())
    print(f"   null (overall mean for everyone):              R2  0.000  rRMSE {rrn:5.1f} %")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

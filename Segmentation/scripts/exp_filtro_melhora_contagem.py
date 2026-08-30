#!/usr/bin/env python3
"""Does a better AUC turn into a better counting F1?

`exp_obliquo_filtra_deteccao.py` showed that the evidence of a return at breast height adds +0.031
of AUC to what the detector already knows, from 0.868 to 0.899. AUC measures ranking and does not
say, on its own, whether a threshold exists that improves the count: there can be a gain in ranking
without an operational gain, if the false positives identified are few or if removing them costs too
much recall.

The F1 used here is the F1 of the paper, and not classification accuracy. Filtering detections acts
asymmetrically on the three terms: it removes false positives (precision rises), it may remove true
positives (recall falls) and it never recovers an undetected tree. That is why the threshold is swept
and the gain is read on the paired F1, on the same set.

The threshold is chosen outside the fold, since maximising F1 on the same data on which it is
measured produces a fictitious gain. The probability comes from spatial-block validation, and the
optimal threshold is also reported with the cut chosen on the training folds.

Usage: PYTHONPATH=. python scripts/exp_filtro_melhora_contagem.py
Output: manual_match/filtro_melhora_contagem.csv
"""
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from greenvista import config  # noqa: E402

warnings.filterwarnings("ignore")
DET = config.OUT_DIR / "deteccoes_julgadas_talhao001.csv"
MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
DRONE = config.REPO / "work/lazall/SaoManuelTotal_001.laz"
SAIDA = config.OUT_DIR / "filtro_melhora_contagem.csv"
N_BLOCOS = 5
N_JOBS = 8
SEED = 20260828


def f1_de(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def prob_fora_da_dobra(X, y, blocos, seed=0):
    p = np.zeros(len(y))
    for tr, te in blocos:
        m = RandomForestClassifier(500, min_samples_leaf=5, random_state=seed, n_jobs=N_JOBS)
        m.fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


def main():
    d = pd.read_csv(DET)
    d = d[d.talhao == 1].copy().reset_index(drop=True)
    pos = d[["base_x", "base_y"]].to_numpy(float)
    casou = d.casou.astype(int).to_numpy()
    n_ref = len(gpd.read_file(MAPA))
    print(f"{len(d)} detections, {casou.sum()} matched, {n_ref} stems in the reference map")

    las = laspy.read(str(DRONE))
    cls = np.asarray(las.classification)
    veg = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    x, y, z = np.asarray(las.x)[veg], np.asarray(las.y)[veg], np.asarray(las.z)[veg]
    ang = np.abs(np.asarray(las.scan_angle_rank).astype(float))[veg]
    for a0, a1, tag in [(0, 15, "nad"), (20, 36, "obl")]:
        m = (z >= 1.0) & (z <= 6.0) & (ang >= a0) & (ang < a1)
        T = cKDTree(np.column_stack([x[m], y[m]]))
        for raio in (0.30, 0.60):
            d[f"{tag}_n{int(raio * 100)}"] = [len(T.query_ball_point(p, raio)) for p in pos]
    for c in [c for c in d.columns if c.startswith(("nad_n", "obl_n"))]:
        d[c + "_rel"] = d[c] / np.maximum(d.n_pts, 1)

    BASE = ["n_pts", "z_max", "dist_divisa"]
    PEITO = [c for c in d.columns if c.startswith(("nad_n", "obl_n"))]
    b = np.digitize(pos[:, 0], np.quantile(pos[:, 0], np.linspace(0, 1, N_BLOCOS + 1)[1:-1]))
    blocos = [(np.where(b != k)[0], np.where(b == k)[0]) for k in np.unique(b)]

    # starting point: no filter at all
    tp0, fp0 = int(casou.sum()), int((1 - casou).sum())
    fn0 = n_ref - tp0
    p0, r0, f0 = f1_de(tp0, fp0, fn0)
    print(f"\nwith no filter at all: P {p0:.3f}  R {r0:.3f}  F1 {f0:.3f} "
          f"(TP {tp0}, FP {fp0}, FN {fn0})\n")

    linhas = []
    probs = {}
    # The two set labels stay in Portuguese: fig_noite_das_hipoteses.py selects rows of the output
    # CSV with `conjunto.str.contains("detector")` and `("peito")`.
    for nome, cols in (("so o que o detector sabe", BASE),
                       ("com a evidencia de peito", BASE + PEITO)):
        pr = prob_fora_da_dobra(d[cols], casou, blocos)
        probs[nome] = pr
        auc = roc_auc_score(casou, pr)
        # threshold sweep
        melhor = (f0, 0.0, tp0, fp0)
        curva = []
        for t in np.arange(0.05, 0.96, 0.01):
            keep = pr >= t
            tp = int(casou[keep].sum())
            fp = int((1 - casou[keep]).sum())
            fn = n_ref - tp
            p, r, f1 = f1_de(tp, fp, fn)
            curva.append((t, p, r, f1, tp, fp))
            if f1 > melhor[0]:
                melhor = (f1, t, tp, fp)
        # threshold chosen outside the fold: for each test block it comes from the optimal F1
        # computed on the training blocks alone
        pred_keep = np.zeros(len(casou), bool)
        for tr, te in blocos:
            melhor_tr, t_tr = -1, 0.5
            for t in np.arange(0.05, 0.96, 0.01):
                k = pr[tr] >= t
                tp = int(casou[tr][k].sum())
                fp = int((1 - casou[tr][k]).sum())
                fn = int(casou[tr].sum()) - tp + int(round((n_ref - casou.sum()) * len(tr) / len(casou)))
                _, _, f1 = f1_de(tp, fp, max(fn, 0))
                if f1 > melhor_tr:
                    melhor_tr, t_tr = f1, t
            pred_keep[te] = pr[te] >= t_tr
        tp = int(casou[pred_keep].sum())
        fp = int((1 - casou[pred_keep]).sum())
        fn = n_ref - tp
        p_h, r_h, f1_h = f1_de(tp, fp, fn)
        print(f"{nome}")
        print(f"   AUC {auc:.3f}")
        print(f"   best F1 achievable (threshold seen): {melhor[0]:.3f} at threshold {melhor[1]:.2f} "
              f"(TP {melhor[2]}, FP {melhor[3]})")
        print(f"   F1 with the threshold CHOSEN OUTSIDE THE FOLD: {f1_h:.3f}  "
              f"(P {p_h:.3f} R {r_h:.3f}, TP {tp}, FP {fp})")
        print(f"   count: {int(pred_keep.sum())} trees against {n_ref} real ones "
              f"({100 * (pred_keep.sum() / n_ref - 1):+.1f} %)\n")
        linhas.append(dict(conjunto=nome, auc=auc, f1_sem_filtro=f0,
                           f1_limiar_visto=melhor[0], limiar=melhor[1],
                           f1_fora_da_dobra=f1_h, precisao=p_h, revocacao=r_h,
                           n_contadas=int(pred_keep.sum()), n_ref=n_ref))
    ganho = linhas[1]["f1_fora_da_dobra"] - linhas[0]["f1_fora_da_dobra"]
    print(f"GAIN of the breast-height evidence on the out-of-fold F1: {ganho:+.3f}")
    print(f"and against not filtering at all: {linhas[1]['f1_fora_da_dobra'] - f0:+.3f}")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The area-based literature menu applied to the mean DBH per plot.

`exp_dap_da_segmentacao.py` obtained a single-feature ridge at the level of the three networks of
the paper. A result like that has to be confronted with what the field has already validated,
otherwise it is just one more hand-picked model. Here the zoo implemented in this repository from
the ABA literature enters (Naesset 2002, White 2013/2017, Silva 2016 PCA->MLR, Cosenza 2021
OLS/kNN/RF), run against the mean DBH instead of volume.

A route that exists only for DBH and that requires the count also enters, the dendrometric identity:

    DQ = sqrt( 40000 * G / (pi * N) )       G in m2/ha, N in stems/ha, DQ in cm

The quadratic mean diameter follows from basal area and density, with no fitting. Basal area from
LiDAR is a classical ABA target and density is what the segmentation delivers, so this is the link
between the two halves of the paper, by identity and not by model. It is the analogue of the
"semi-ITD" of Shinzato et al. 2017.

DQ is not the arithmetic mean: DQ^2 = Dbar^2 + var(D) within the plot, hence DQ >= Dbar always. The
paper reports the arithmetic mean; both are measured here and the conversion is stated.

Validation by stand left out and a shuffling control, for the same reasons as the sibling
experiment: with 13 plots and one young stand, LOPO inflates and a positive R2 by chance is easy.

Usage: PYTHONPATH=. python scripts/exp_dap_literatura.py
Output: manual_match/dap_literatura.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_dap_da_segmentacao import tabela  # noqa: E402

from greenvista import config  # noqa: E402
from greenvista.area_based import models as Z  # noqa: E402

warnings.filterwarnings("ignore")
# 50 shufflings and not 200: there are 12 models x 6 folds per shuffling, and random forest,
# GPR and nested best-subset are expensive. 50 is enough to separate p<0.05 from p~0.5, which is the
# decision this control makes.
SEED, N_EMB = 20260827, 50
AREA_HA = 0.04


def campo_basal_e_dq():
    """Measured basal area and quadratic mean diameter, per plot."""
    inv = pd.read_csv(config.DATA / "5-dados_campo/inv_euc.csv")
    v = inv[inv.D_cm.notna()].copy()
    v["g_m2"] = np.pi * (v.D_cm / 200.0) ** 2          # cross-sectional area of each stem
    g = (v.groupby(["talhao", "parcela"])
         .agg(G_ha=("g_m2", lambda s: s.sum() / AREA_HA),
              DQ=("D_cm", lambda s: float(np.sqrt((s ** 2).mean())))).reset_index())
    g["plot_id"] = g.talhao.astype(str) + "_" + g.parcela.astype(str)
    return g[["plot_id", "G_ha", "DQ"]]


def cv_loto(df, y, faz_modelo, embaralha=None):
    """Stand left out, with the zoo model fitted on the training set only."""
    alvo = y if embaralha is None else y[embaralha]
    pred = np.empty(len(df))
    for t in df.talhao.unique():
        te = (df.talhao == t).to_numpy()
        mod = faz_modelo()
        mod.fit(df[~te], alvo[~te])
        pred[te] = np.asarray(mod.predict(df[te])).ravel()
    return alvo, pred


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - y) ** 2)))


def main():
    df = tabela().merge(campo_basal_e_dq(), on="plot_id")
    y = df.Dbar.to_numpy(float)
    dq = df.DQ.to_numpy(float)
    rng = np.random.default_rng(SEED)

    print(f"{len(df)} plots")
    print(f"  arithmetic mean DBH   {y.mean():.2f} cm, sd {y.std(ddof=1):.2f}")
    print(f"  quadratic diameter    {dq.mean():.2f} cm, sd {dq.std(ddof=1):.2f}  "
          f"(DQ - Dbar = {np.mean(dq - y):+.2f} cm, sd of the difference {np.std(dq - y, ddof=1):.2f})")
    print(f"  basal area            {df.G_ha.mean():.1f} m2/ha\n")

    # Validate the identity before using it: if it does not reproduce the measured DQ from the
    # measured G and N, any number derived from it is wrong by algebra.
    dq_id = np.sqrt(40000 * df.G_ha.to_numpy() / (np.pi * df.n_field.to_numpy() / AREA_HA))
    print(f"identity DQ = f(G, N) against the measured DQ: maximum error {np.abs(dq_id - dq).max():.6f} cm\n")

    # The repository models expect the target column in Vol_ha. Since the target here is another
    # one, the column is rewritten before any fitting, never in the middle of a fold.
    d2 = df.copy()
    d2["Vol_ha"] = y
    MENU = {
        "log allometry (zmean, zq90)":      Z.make_log_allometric,
        "best subset, stand-wise selection": lambda: Z.make_bestsubset_allom(select_by="group"),
        "PCA -> OLS, full suite":           Z.make_pca_ols,
        "PLS, full suite":                  Z.make_pls,
        "ridge, 6 metrics":                 Z.make_ridge,
        "elastic net":                      Z.make_elasticnet,
        "random forest":                    Z.make_rf,
        "kNN (Nordic imputation)":          Z.make_knn,
        "GPR":                              Z.make_gpr,
        "simple linear regression":         Z.make_baseline_lm,
        "log allometry on zmean only":      lambda: Z.make_log_allometric(cols=("zmean",)),
        "log allometry on the crowns":      lambda: Z.make_log_allometric(cols=("seg_z_med",)),
    }
    linhas = []
    print(f"{'model':>38} {'RMSE':>6} {'R2':>7}   {'shuffled':>11}")
    p0 = np.array([y[(df.talhao != t).to_numpy()].mean() for t in df.talhao])
    print(f"{'null, global mean':>38} {rmse(y, p0):6.2f} {'   —':>7}")
    for nome, fab in MENU.items():
        try:
            _, p = cv_loto(d2, y, fab)
            r = rmse(y, p)
            emb = [rmse(*cv_loto(d2, y, fab, rng.permutation(len(y)))) for _ in range(N_EMB)]
            pv = float(np.mean(np.array(emb) <= r))
            r2 = 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)
            print(f"{nome:>38} {r:6.2f} {r2:+7.3f}   {np.mean(emb):6.2f}  p={pv:.3f}")
            linhas.append(dict(rota="zoo", modelo=nome, RMSE=r, R2=r2,
                               RMSE_emb=float(np.mean(emb)), p=pv))
        except Exception as e:
            print(f"{nome:>38}    failed: {type(e).__name__}: {e}")

    # ---------------- the identity, which is the route that uses the count
    print(f"\n{'identity DQ = sqrt(40000 G / pi N)':>38}   target = quadratic diameter")
    d3 = df.copy()
    d3["Vol_ha"] = df.G_ha.to_numpy(float)       # basal area as the ABA target
    _, g_lidar = cv_loto(d3, df.G_ha.to_numpy(float), Z.make_pca_ols)
    for nome, G, N in (("field G + field N", df.G_ha.to_numpy(float), df.n_field.to_numpy(float)),
                       ("field G + SAT N",   df.G_ha.to_numpy(float), df.n_det.to_numpy(float)),
                       ("LiDAR G + SAT N",  g_lidar,                 df.n_det.to_numpy(float)),
                       ("LiDAR G + field N", g_lidar,                df.n_field.to_numpy(float))):
        est = np.sqrt(40000 * np.clip(G, 1e-6, None) / (np.pi * N / AREA_HA))
        print(f"{nome:>38} {rmse(dq, est):6.2f} {1 - np.sum((est-dq)**2)/np.sum((dq-dq.mean())**2):+7.3f}"
              f"   against DQ | and against Dbar {rmse(y, est):.2f}")
        linhas.append(dict(rota="identidade", modelo=nome, RMSE=rmse(dq, est),
                           R2=1 - np.sum((est - dq) ** 2) / np.sum((dq - dq.mean()) ** 2),
                           RMSE_emb=np.nan, p=np.nan, RMSE_contra_Dbar=rmse(y, est)))
    print(f"\n{'basal area from LiDAR (PCA->OLS)':>38} {rmse(df.G_ha.to_numpy(float), g_lidar):6.2f} m2/ha"
          f"  over a mean of {df.G_ha.mean():.1f}")
    print(f"{'reference of the paper (Table IV, S1)':>38} {2.02:6.2f}")

    out = config.OUT_DIR / "dap_literatura.csv"
    pd.DataFrame(linhas).to_csv(out, index=False)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

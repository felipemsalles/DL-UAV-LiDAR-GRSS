#!/usr/bin/env python3
"""Does the segmentation add anything to the estimate of mean DBH per plot?

In the paper, individual-tree counting and DBH estimation are isolated approaches: the
DBH section clips the clouds by field plot centre and does not use the output of the
segmentation. This script tests the missing link, predicting the mean DBH from what the
segmentation delivers (how many trees, and what the crowns look like).

The question is comparative and not absolute: a model that uses the segmentation and
arrives at the same place as the mean LiDAR height demonstrates no link at all. Hence
the ladder, in which step 2 is the comparator that decides:

  1. null            predict the global mean for every plot
  2. LiDAR only      height metrics of the flight, without segmentation   <- the comparator
  3. segmentation only  count and crown statistics of the detections
  4. both together

Validation by stand left out, and not by plot: LOPO inflates here, because stand 4 is
young (mean DBH 12.9 to 13.8 cm) and the other five are mature (~17 cm), so any metric
that separates young from mature succeeds by stand identity and collapses when the whole
stand leaves the training set.

A shuffling control is mandatory: with 13 plots and 6 groups, a positive R2 occurs by
chance easily. Every model also runs with the target shuffled inside the same validation,
and what does not separate from the shuffled version does not count.

Usage: PYTHONPATH=. python scripts/exp_dap_da_segmentacao.py
Output: manual_match/dap_da_segmentacao.csv
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_sat_13parcelas import centros_e_censo  # noqa: E402

from greenvista import config  # noqa: E402
# nms_malha returns the indices of what is kept; nms_merge returns merged coordinates. Here
# the indices are needed, because the features (z_max, n_pts) travel along with the row.
from exp_w2w_contagem import nms_malha  # noqa: E402

PR400 = math.sqrt(400 / math.pi)     # 11.28 m, the 400 m2 the field measured
RAIO = 1.5                            # SegmentAnyTree merge radius
SEED = 20260827
N_EMB = 400                           # shuffling repetitions


def tabela():
    """One row per plot: target, flight metrics and segmentation features."""
    inv = pd.read_csv(config.DATA / "5-dados_campo/inv_euc.csv")
    viv = inv[inv.D_cm.notna()]
    alvo = (viv.groupby(["talhao", "parcela"])
            .agg(Dbar=("D_cm", "mean"), n_field=("D_cm", "size")).reset_index())
    alvo["plot_id"] = alvo.talhao.astype(str) + "_" + alvo.parcela.astype(str)

    from greenvista.area_based import data as abd
    lid = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                         config.DATA / "5-dados_campo/inventario_est.csv")

    d = pd.read_csv(config.REPO / "data/detections/sat_13parcelas_instancias.csv")
    d = d[d.condicao == "adabn"]
    centros = centros_e_censo()
    linhas = []
    for pid, (px, py, _campo) in centros.items():
        g = d[d.plot_id == pid]
        if g.empty:
            continue
        xy = g[["cx", "cy"]].to_numpy(float)
        keep = nms_malha(xy, g.n_pts.to_numpy(float), RAIO)
        z = g.z_max.to_numpy(float)[keep]
        npts = g.n_pts.to_numpy(float)[keep]
        m = xy[keep]
        dentro = np.hypot(m[:, 0] - px, m[:, 1] - py) <= PR400
        z, npts = z[dentro], npts[dentro]
        # Only what the segmentation delivers: a flight metric in here would make step 3
        # stop testing the segmentation and go back to testing height.
        linhas.append(dict(plot_id=pid, n_det=int(dentro.sum()),
                           seg_z_med=z.mean(), seg_z_dp=z.std(ddof=1), seg_z_q90=np.quantile(z, 0.90),
                           seg_pts_med=npts.mean(), seg_pts_dp=npts.std(ddof=1),
                           seg_dens=dentro.sum() / 0.04))
    seg = pd.DataFrame(linhas)
    return alvo.merge(lid.drop(columns=["talhao", "parcela"]), on="plot_id").merge(seg, on="plot_id")


def loto(df, feats, y, embaralha=None):
    """Stand left out. Standardised ridge, fitted on the training set only."""
    X = df[feats].to_numpy(float)
    alvo = y.copy()
    if embaralha is not None:
        alvo = y[embaralha]
    pred = np.empty(len(df))
    for t in df.talhao.unique():
        te = (df.talhao == t).to_numpy()
        mod = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        mod.fit(X[~te], alvo[~te])
        pred[te] = mod.predict(X[te])
    return alvo, pred


def metricas(y, p):
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    return rmse, 1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    df = tabela()
    y = df.Dbar.to_numpy(float)
    rng = np.random.default_rng(SEED)
    print(f"{len(df)} plots | mean DBH {y.mean():.2f} cm, sd {y.std(ddof=1):.2f} cm")
    print(f"SAT detections inside the circle: {int(df.n_det.sum())} against {int(df.n_field.sum())} in the census\n")

    # the null model: predict the global mean, under the same validation
    p0 = np.array([y[(df.talhao != t).to_numpy()].mean() for t in df.talhao])
    r0, _ = metricas(y, p0)
    print(f"{'step':>34}  {'RMSE':>6} {'R2':>7}   {'shuffled RMSE':>17}")
    print(f"{'0. null, global mean':>34}  {r0:6.2f} {'   —':>7}   {'—':>17}")

    ESCADA = {
        "1. flight, mean height (zmean)":        ["zmean"],
        "2. flight, height + dispersion":        ["zmean", "zsd"],
        "3. flight, short suite":                ["zmean", "zsd", "zq90", "pzabovezmean"],
        "4. segmentation, density":              ["seg_dens"],
        "5. segmentation, mean crown":           ["seg_pts_med"],
        "6. segmentation, crown height":         ["seg_z_med"],
        "7. segmentation, density + crown":      ["seg_dens", "seg_pts_med"],
        "8. segmentation, four features":        ["seg_dens", "seg_pts_med", "seg_z_med", "seg_z_dp"],
        "9. flight + segmentation":              ["zmean", "seg_dens", "seg_pts_med"],
    }
    linhas = []
    for nome, feats in ESCADA.items():
        _, p = loto(df, feats, y)
        rmse, r2 = metricas(y, p)
        emb = [metricas(*loto(df, feats, y, rng.permutation(len(y))))[0] for _ in range(N_EMB)]
        # p is the fraction of shufflings that come out better (lower RMSE) than the real one
        pval = float(np.mean(np.array(emb) <= rmse))
        linhas.append(dict(degrau=nome, n_feats=len(feats), RMSE=rmse, R2=r2,
                           RMSE_embaralhado=float(np.mean(emb)), p=pval))
        print(f"{nome:>34}  {rmse:6.2f} {r2:+7.3f}   {np.mean(emb):6.2f}   p={pval:.3f}")

    print(f"\n{'reference of the paper (Table IV, S1)':>34}  {2.02:6.2f} {0.004:+7.3f}")

    # ---------------- what this does to the volume chain
    # The prediction travels with the plot_id, never as a loose vector: the two tables have
    # different row orders and comparing by position silently swaps plot for plot. The columns carry
    # a prefix because the volume table already has a `campo` column.
    from exp_cadeia_volume import AREA_HA, loto as loto_vol, tabela as tab_vol
    from greenvista.mechanistic_volume import campos_height, g2_volume

    preds = pd.DataFrame({"plot_id": df.plot_id, "dap_campo": y})
    for rot, feats in (("dap_voo", ["zmean"]), ("dap_copas", ["seg_z_med"])):
        preds[rot] = loto(df, feats, y)[1]
    m = tab_vol().merge(preds, on="plot_id")
    assert len(m) == len(df), "the merge by plot_id lost a plot"
    vy = m.Vol_ha.to_numpy(float)

    print(f"\n{'the volume chain, with the DBH of each route':>44}")
    print(f"{'DBH coming from':>26} {'DBH error':>9} {'count':>9} {'R2':>8} {'rRMSE':>8}")
    for col, rot in (("dap_campo", "field (ceiling)"), ("dap_voo", "flight height"),
                     ("dap_copas", "crown height")):
        d = m[col].to_numpy(float)
        erro = float(np.sqrt(np.mean((d - m.Dbar.to_numpy(float)) ** 2)))
        for cont, ncol in (("field", "n_field"), ("SAT", "n_sat")):
            vp = loto_vol(m, m[ncol].to_numpy(float) / AREA_HA
                          * g2_volume(d, campos_height(d, m.Hdom.to_numpy())))
            rmse = float(np.sqrt(np.mean((vp - vy) ** 2)))
            r2 = 1 - np.sum((vp - vy) ** 2) / np.sum((vy - vy.mean()) ** 2)
            print(f"{rot:>26} {erro:8.2f}cm {cont:>9} {r2:+8.3f} {100*rmse/vy.mean():7.1f}%")
            linhas.append(dict(degrau=f"volume, {rot}, count {cont}", n_feats=np.nan,
                               RMSE=rmse, R2=r2, RMSE_embaralhado=np.nan, p=np.nan,
                               erro_dap=erro, rRMSE_vol=100 * rmse / vy.mean()))

    out = config.OUT_DIR / "dap_da_segmentacao.csv"
    pd.DataFrame(linhas).to_csv(out, index=False)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

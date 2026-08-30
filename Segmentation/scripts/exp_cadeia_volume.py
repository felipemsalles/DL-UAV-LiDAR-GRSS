#!/usr/bin/env python3
"""Does the count x mean volume chain close? And how much DBH accuracy does it need?

The detection section and the DBH section of the paper share only the dataset: the DBH one
uses neither SAT nor FF3D and clips the clouds by field plot centre. This script measures
whether the two can be joined into a single chain:

    V_ha = (N / area) x v(D, H),  H = Campos(D),  v = G2(D, H)

DBH enters directly and the hypsometric relation runs forward, in the direction for which it
was fitted; LiDAR height leaves the chain. The alternative topology, taking DBH from inverting
the hypsometric relation on the LiDAR height, is the broken link: nadir measures dominant
height, the calculation asks for the height of the mean tree, and the inversion inflates DBH
up to 30.7 cm against 13 to 20 cm in the field.

Instead of the predictions of a DBH model, what enters is the field-measured DBH plus noise the
size of the published error (MAE 1.78 cm, RMSE 2.05). This answers two questions: with a perfect
DBH the chain closes, and how much error it tolerates, which gives an accuracy target.

The floor is Jensen's inequality, and it is not constant: volume is convex in D, so
v(mean D) falls below the mean of the v(D_i). Measured here, -9.4% on average with a spread of
4.6 points across plots. A multiplicative factor corrects the mean, not the spread, so 4.6%
of error enters the chain before any model.

The calibration factor is fitted on the training folds only; fitting it on the whole set would be
leakage and Jensen's bias would vanish for free.

Usage: PYTHONPATH=. python scripts/exp_cadeia_volume.py
Output: manual_match/cadeia_volume.csv
"""
import numpy as np
import pandas as pd

from greenvista import config
from greenvista.area_based import data as abd
from greenvista.mechanistic_volume import campos_height, g2_volume

AREA_HA = 0.04            # 400 m2 plot, area_parc of the inventory
SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.05, 3.0]   # error on mean DBH, cm. 2.05 = Felipe's RMSE
N_DRAWS = 300
SEED = 20260827


def tabela():
    """One row per plot: reference, counts and measured DBH."""
    inv = pd.read_csv(config.DATA / "5-dados_campo/inv_euc.csv")
    viv = inv[inv.D_cm.notna()]
    g = (viv.groupby(["talhao", "parcela"])
         .agg(n_field=("D_cm", "size"), Dbar=("D_cm", "mean"),
              vbar_real=("V_est", "mean"), Hdom=("H_dom", "first")).reset_index())
    g["plot_id"] = g.talhao.astype(str) + "_" + g.parcela.astype(str)

    ref = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                         config.DATA / "5-dados_campo/inventario_est.csv")[["plot_id", "talhao", "Vol_ha"]]
    sat = pd.read_csv(config.REPO / "manual_match/sat_adabn_13parcelas.csv")
    sat = sat.rename(columns={"parcela": "plot_id", "adabn_r400": "n_sat"})[["plot_id", "n_sat", "campo"]]

    df = ref.merge(g.drop(columns=["talhao"]), on="plot_id").merge(sat, on="plot_id")
    assert len(df) == 13, f"expected 13 plots, got {len(df)}"
    # The two census sources disagree on one tree, in two plots: inv_euc gives 67 in 2_7
    # and 50 in 5_9, the SAT CSV gives 66 and 51, in opposite directions, and the total closes
    # at 717 in both. It is a plot-assignment disagreement over one tree, irrelevant here. The
    # check stays because an equal total with different rows is the typical signature of a merge
    # error: it requires identical totals and at most one tree of difference per plot.
    dif = (df.campo - df.n_field).abs()
    assert df.campo.sum() == df.n_field.sum(), "the two censuses do not add up to the same total"
    assert dif.max() <= 1, f"census diverges by up to {dif.max()} trees in one plot"
    return df


def prediz(df, n_col, dbar):
    """The whole chain, from a count and a mean DBH to m3/ha."""
    h = campos_height(dbar, df.Hdom.to_numpy())
    v = g2_volume(dbar, h)
    return df[n_col].to_numpy(float) / AREA_HA * v


def loto(df, pred):
    """Stand left out, with the multiplicative factor fitted on the training set only."""
    y = df.Vol_ha.to_numpy(float)
    out = np.empty_like(y)
    for t in df.talhao.unique():
        te = (df.talhao == t).to_numpy()
        k = y[~te].mean() / pred[~te].mean()      # Jensen bias calibration, training only
        out[te] = pred[te] * k
    return out


def metricas(y, p):
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    return dict(R2=1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2),
                rRMSE=100 * rmse / y.mean(), vies=100 * (p - y).mean() / y.mean())


def main():
    df = tabela()
    y = df.Vol_ha.to_numpy(float)
    rng = np.random.default_rng(SEED)
    linhas = []

    print(f"13 plots, {int(df.n_field.sum())} census stems, "
          f"{int(df.n_sat.sum())} detected by SAT ({100*df.n_sat.sum()/df.n_field.sum():.1f}%)\n")

    # ---- ceiling: perfect count and DBH, only Jensen and G2 err
    for nome, n_col in (("field count", "n_field"), ("SAT count", "n_sat")):
        p = loto(df, prediz(df, n_col, df.Dbar.to_numpy()))
        m = metricas(y, p)
        linhas.append(dict(condicao=f"perfect DBH + {nome}", sigma_dap=0.0, **m, dp_rRMSE=0.0))
        print(f"  perfect DBH + {nome:18s}  R2 {m['R2']:+.3f}  rRMSE {m['rRMSE']:5.1f}%  bias {m['vies']:+5.1f}%")

    # ---- sweep of the DBH error
    print()
    for sig in SIGMAS:
        for nome, n_col in (("field count", "n_field"), ("SAT count", "n_sat")):
            res = []
            for _ in range(N_DRAWS if sig > 0 else 1):
                d = df.Dbar.to_numpy() + (rng.normal(0, sig, len(df)) if sig > 0 else 0)
                res.append(metricas(y, loto(df, prediz(df, n_col, d))))
            r2 = np.array([r["R2"] for r in res]); rr = np.array([r["rRMSE"] for r in res])
            linhas.append(dict(condicao=f"DBH with error + {nome}", sigma_dap=sig,
                               R2=r2.mean(), rRMSE=rr.mean(), vies=np.mean([r["vies"] for r in res]),
                               dp_rRMSE=rr.std()))
            print(f"  sigma {sig:4.2f} cm + {nome:18s}  R2 {r2.mean():+.3f}  "
                  f"rRMSE {rr.mean():5.1f}% +- {rr.std():4.1f}")

    out = config.OUT_DIR / "cadeia_volume.csv"
    pd.DataFrame(linhas).to_csv(out, index=False)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

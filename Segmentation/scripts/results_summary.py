#!/usr/bin/env python3
"""Consolidated Phase-1/2 results: every model under the identical LOPO, with conformal coverage.

Reads the saved CNN predictions (S7 scratch, S8 sim-pretrained), recomputes the classical models and
their per-plot LOPO predictions, adds a GPR+CNN ensemble, and prints + saves the final comparison
(R², rRMSE, R² bootstrap CI, conformal-95 coverage). No training here — pure aggregation.

Run (repo root, greenvista env): PYTHONPATH=. python scripts/results_summary.py
"""
import numpy as np, pandas as pd
from greenvista import config
from greenvista.eval import metrics, bootstrap_ci, coverage, permutation_test
from greenvista.image_cnn import dataset as ds, train as tr
from greenvista.area_based import data as abd, validate as abv, models as abm
from greenvista.repro import write_manifest, seed_everything


def row(name, y, p, perm=2000):
    lo, hi = tr.conformal_interval(y, p)                       # finite-sample jackknife+
    mm = metrics(y, p)
    return {"model": name, "R2": mm["R2"], "rRMSE_pct": mm["rRMSE_pct"],
            "R2_ci": bootstrap_ci(y, p)["R2_ci"], "cov95_conformal": coverage(y, lo, hi),
            "perm_p": permutation_test(y, p, n_perm=perm)["p_value"]}


def main():
    seed_everything(config.ABA.seed)
    df = ds.build_plot_table().set_index("plot_id")
    y = df["Vol_ha"].to_numpy(np.float32)
    ids = df.index.tolist()

    # CNN predictions (saved), aligned by plot_id
    s7 = pd.read_csv(config.OUT_DIR / "s7_lopo_predictions.csv").set_index("plot_id").loc[ids]
    cnn_scratch = s7["Vol_pred_cnn"].to_numpy(np.float32)
    rows = [row("CNN scratch (S7)", y, cnn_scratch)]
    try:
        s8 = pd.read_csv(config.OUT_DIR / "s8_lopo_predictions.csv").set_index("plot_id").loc[ids]
        cnn_sim = s8["Vol_pred_s8"].to_numpy(np.float32)
        rows.append(row("CNN sim-pretrained (S8)", y, cnn_sim))
    except FileNotFoundError:
        cnn_sim = None

    # classical models + their per-plot LOPO predictions (same split)
    dfa = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                         config.DATA / "5-dados_campo/inventario_est.csv").copy()
    dfa["plot_id"] = dfa.talhao.astype(str) + "_" + dfa.parcela.astype(str)
    dfa = dfa.set_index("plot_id").loc[ids]
    ya = dfa["Vol_ha"].to_numpy(np.float32)
    makers = {"GPR": abm.make_gpr, "ElasticNet": abm.make_elasticnet,
              "log-allometric": abm.make_log_allometric, "baseline_lm": abm.make_baseline_lm}
    gpr_pred = None
    for name, mk in makers.items():
        p, _, _ = abv._lopo(mk, dfa.reset_index())
        rows.append(row(name, ya, np.asarray(p, np.float32)))
        if name == "GPR":
            gpr_pred = np.asarray(p, np.float32)

    # ensembles (GPR ⊕ CNN) — canopy texture + metric regression
    if gpr_pred is not None:
        rows.append(row("Ensemble GPR+CNN(S7)", y, 0.5 * (gpr_pred + cnn_scratch)))
        if cnn_sim is not None:
            rows.append(row("Ensemble GPR+CNN(S8)", y, 0.5 * (gpr_pred + cnn_sim)))

    res = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    print("\n=== Plot-level Vol_ha — all models under identical LOPO (N=13); cov95=conformal, p=permutation ===")
    for _, r in res.iterrows():
        print(f"  {r['model']:24s} R2={r['R2']:+.2f} (CI {r['R2_ci'][0]:+.2f}..{r['R2_ci'][1]:+.2f})  "
              f"rRMSE={r['rRMSE_pct']:4.0f}%  conformal-cov95={r['cov95_conformal']:.2f}  p={r['perm_p']:.3f}")
    res.to_csv(config.OUT_DIR / "results_summary.csv", index=False)
    write_manifest(config.OUT_DIR / "results_summary_manifest.json", script="results_summary.py",
                   config={"models": list(makers), "seed": config.ABA.seed},
                   metrics=res.drop(columns=["R2_ci"]).to_dict(orient="records"))
    print(f"\nwrote {config.OUT_DIR / 'results_summary.csv'} + results_summary_manifest.json")


if __name__ == "__main__":
    main()

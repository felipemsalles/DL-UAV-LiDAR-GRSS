#!/usr/bin/env python3
"""Iterate established ABA literature ideas for plot-Vol_ha at N=13, judged cross-stand.

Models are judged on leave-one-stand-out (LOTO, cross-stand generalization), not only on LOPO:
`gpr+age` reaches LOPO R²=0.74 but collapses to LOTO R²≈−0.22, its RBF memorizing stand identity.
Iterates the log-allometric / dimension-reduction / robust families (Silva 2016, Cosenza 2021,
Moeur & Stage MSN) and keeps only models that generalise across stands with calibrated conformal
coverage and a significant permutation p.

Run: PYTHONPATH=. python scripts/exp_better_numbers.py
"""
import numpy as np, pandas as pd
from greenvista import config
from greenvista.repro import seed_everything, write_manifest
from greenvista.area_based import data as abd, models as abm, validate as abv
from greenvista.eval import (metrics, jackknife_plus_interval, coverage, interval_score,
                             permutation_test)

AGE = abd.AGE
F, FA = abd.FEATURES, abd.FEATURES_AGE


def evaluate(name, lopo, loto, y):
    """Metric bundle for one model's out-of-fold predictions."""
    m = metrics(y, lopo)
    lo, hi = jackknife_plus_interval(y, lopo)
    row = {"model": name, "LOPO_R2": m["R2"], "LOPO_rRMSE": m["rRMSE_pct"], "bias_pct": m["bias_pct"],
           "cov95": coverage(y, lo, hi), "int_score": interval_score(y, lo, hi),
           "perm_p": permutation_test(y, lopo, n_perm=3000)["p_value"]}
    if loto is not None:
        gm = metrics(y, loto)
        row["LOTO_R2"] = gm["R2"]; row["LOTO_rRMSE"] = gm["rRMSE_pct"]
    return row


def main():
    seed_everything(config.ABA.seed)
    df = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                        config.DATA / "5-dados_campo/inventario_est.csv")
    # merge structural features (rumple/cover/penetration) if built (scripts/build_structural_features.py)
    struct_csv = config.OUT_DIR / "plot_structural.csv"
    has_struct = struct_csv.exists()
    if has_struct:
        s = pd.read_csv(struct_csv)[["plot_id", "rumple", "cover", "penetration"]]
        df = df.merge(s, on="plot_id", how="left")
    y = df.Vol_ha.to_numpy()
    print(f"N={len(df)} plots, {df.talhao.nunique()} stands | Vol_ha {y.min():.0f}–{y.max():.0f}"
          f" | structural features: {'YES' if has_struct else 'no'}\n")
    STRUCT_CANDS = (*abm.ALLOM_CANDIDATES, "rumple", "cover", "penetration")

    # --- the model zoo: baselines (old) + literature iterations (new) --------------------------
    zoo = {
        "gpr+age (old headline)":   lambda: abm.make_gpr(FA),
        "log_allom+age (v1)":lambda: abm.make_log_allometric(("zmean", "zq90", AGE)),
        "pca_ols":                  abm.make_pca_ols,
        # --- new literature iterations ---
        "bss_allom (LOPO-select)":  lambda: abm.make_bestsubset_allom(select_by="loo"),   # the age-only trap
        "bss_allom (LOTO-select)":  lambda: abm.make_bestsubset_allom(select_by="group"), # honest selection
        "bss_allom (LOTO+robust)":  lambda: abm.make_bestsubset_allom(select_by="group", robust=True),
        "huber_allom+age":          abm.make_huber_allom,                            # robust to leverage
        "pls":                      abm.make_pls,                                    # Cosenza PLS
        "loggpr+age":               lambda: abm.make_loggpr(FA),                     # calibrated + generalizes
    }
    if has_struct:      # structural iterations (rumple/cover/penetration; group-selected → honest)
        zoo["bss_struct (LOTO-select)"] = lambda: abm.make_bestsubset_allom(STRUCT_CANDS, select_by="group")
        zoo["bss_struct+robust"] = lambda: abm.make_bestsubset_allom(STRUCT_CANDS, select_by="group", robust=True)
        zoo["huber_allom+rumple"] = lambda: abm.make_huber_allom(("zmean", "zq90", AGE, "rumple"))

    # out-of-fold predictions (both schemes) for every model
    oof = {name: abv.oof_predictions(mk, df) for name, mk in zoo.items()}
    rows = [evaluate(name, lopo, loto, y) for name, (lopo, loto) in oof.items()]

    # --- ensembles: average OOF preds of the cross-stand generalizers -------------------
    def ens(names):
        lopo = np.mean([oof[n][0] for n in names], axis=0)
        loto = np.mean([oof[n][1] for n in names], axis=0)
        return lopo, loto

    ens_specs = {
        "ENS huber+bss(LOTO)":      ["huber_allom+age", "bss_allom (LOTO-select)"],
        "ENS huber+logallom":       ["huber_allom+age", "log_allom+age (v1)"],
        "ENS huber+bss+logallom":   ["huber_allom+age", "bss_allom (LOTO+robust)", "log_allom+age (v1)"],
    }
    for name, comp in ens_specs.items():
        lopo, loto = ens(comp)
        rows.append(evaluate(name, lopo, loto, y))

    res = pd.DataFrame(rows)
    # ranking: prefer positive LOTO R², then lowest LOTO rRMSE, then LOPO rRMSE
    res["honest_ok"] = (res.get("LOTO_R2", -1) > 0) & (res["perm_p"] < 0.05) & (res["cov95"] >= 0.85)
    res = res.sort_values(["LOTO_rRMSE", "LOPO_rRMSE"]).reset_index(drop=True)

    print("=== Model comparison (sorted by leave-one-stand-out rRMSE) ===")
    print(f"  {'model':28s} {'LOPO_R2':>7} {'LOPO%':>6} {'LOTO_R2':>7} {'LOTO%':>6} {'cov95':>6} {'perm_p':>7} ok")
    for _, r in res.iterrows():
        print(f"  {r['model']:28s} {r['LOPO_R2']:+7.2f} {r['LOPO_rRMSE']:6.1f} "
              f"{r.get('LOTO_R2', float('nan')):+7.2f} {r.get('LOTO_rRMSE', float('nan')):6.1f} "
              f"{r['cov95']:6.2f} {r['perm_p']:7.3f} {'✓' if r['honest_ok'] else ' '}")

    ok = res[res.honest_ok]
    best = ok.iloc[0] if len(ok) else res.iloc[0]
    print(f"\nWINNER: {best['model']} — LOTO R²={best.get('LOTO_R2', float('nan')):+.2f} "
          f"(rRMSE {best.get('LOTO_rRMSE', float('nan')):.1f}%), LOPO R²={best['LOPO_R2']:+.2f} "
          f"(rRMSE {best['LOPO_rRMSE']:.1f}%), cov95={best['cov95']:.2f}, p={best['perm_p']:.3f}")

    # show what best-subset selected under each criterion (interpretability + the trap)
    for sb in ("loo", "group"):
        b = abm.make_bestsubset_allom(select_by=sb).fit(df, y)
        print(f"  best-subset ({sb}-select) chose: ln(V) ~ {' + '.join('ln('+c+')' for c in b.sel_cols_)}")
    if has_struct:
        bs = abm.make_bestsubset_allom(STRUCT_CANDS, select_by="group").fit(df, y)
        print(f"  best-subset (+struct, group-select) chose: ln(V) ~ {' + '.join('ln('+c+')' for c in bs.sel_cols_)}")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.OUT_DIR / "exp_better_numbers.csv", index=False)
    write_manifest(config.OUT_DIR / "exp_better_numbers_manifest.json", script="exp_better_numbers.py",
                   config=config.ABA, seed=config.ABA.seed,
                   metrics=res.to_dict(orient="records"),
                   extra={"selection": "leave-one-stand-out rRMSE, honest_ok gate"})
    print(f"\nwrote {config.OUT_DIR / 'exp_better_numbers.csv'} + manifest")
    return res


if __name__ == "__main__":
    main()

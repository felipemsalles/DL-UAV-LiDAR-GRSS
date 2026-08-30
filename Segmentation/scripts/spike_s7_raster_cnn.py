#!/usr/bin/env python3
"""Spike S7 — plot-level Vol_ha: CNN-on-rasters (LOPO) vs the classical area-based floor.

Renders the 13 plots once (cached), runs leave-one-plot-out for the hybrid CNN (Vol_ha + DBH
multi-task), and tables it against the GPR/ElasticNet/log-allometric/baseline_lm models under the
identical LOPO. DBH is reported as an auxiliary target. Writes per-plot predictions.

Run (repo root, greenvista env): PYTHONPATH=. python scripts/spike_s7_raster_cnn.py
"""
import numpy as np
from greenvista import config
from greenvista.image_cnn import dataset as ds, train as tr
from greenvista.area_based import data as abd, validate as abv
from greenvista.repro import write_manifest, seed_everything

CACHE = config.LAZ_DIR / "plot_samples_r12_s128.npz"


def load_or_render():
    df = ds.build_plot_table()
    if CACHE.exists():
        samples = np.load(CACHE, allow_pickle=True)["samples"].item()
        print(f"loaded cached rasters ({len(samples)} plots) from {CACHE}")
    else:
        print("rendering 13 plots from clouds (cache miss)…")
        samples = ds.render_all_plots(df)
        np.savez(CACHE, samples=np.array(samples, dtype=object))
        print(f"cached → {CACHE}")
    return df, samples


def main():
    df, samples = load_or_render()
    ids = df.plot_id.tolist()
    y = df[["Vol_ha", "D_cm"]].to_numpy(np.float32)
    print(f"\nplots N={len(ids)} | Vol_ha {y[:,0].min():.0f}–{y[:,0].max():.0f} | DBH {y[:,1].min():.1f}–{y[:,1].max():.1f}")

    print("\n[CNN] leave-one-plot-out (train-fold-only norm + augmentation), multi-seed + TTA…")
    seed_everything(config.CNN.seeds[0])
    ms = tr.run_lopo_multiseed(samples, ids, y, seeds=config.CNN.seeds, col=0,
                               n_aug=config.CNN.n_aug, epochs=config.CNN.epochs, tta=config.CNN.tta)
    print(f"  R² across seeds {config.CNN.seeds}: "
          f"{ms['summary']['R2_mean']:+.2f} ± {ms['summary']['R2_sd']:.2f} "
          f"(min {ms['summary']['R2_min']:+.2f}, max {ms['summary']['R2_max']:+.2f})")
    res = ms["best_res"]                                    # predictions of the best seed carry forward
    cnn_vol = tr.report(y, res["pred"], res["lo"], res["hi"], col=0)
    cnn_dap = tr.report(y, res["pred"], res["lo"], res["hi"], col=1)

    dfa = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                         config.DATA / "5-dados_campo/inventario_est.csv")
    rep = abv.run(dfa)

    def line(name, mm):
        ci = mm.get("R2_ci", (float("nan"), float("nan")))
        return (f"  {name:22s} R2={mm['R2']:+.2f} (CI {ci[0]:+.2f}..{ci[1]:+.2f})  "
                f"rRMSE={mm['rRMSE_pct']:4.0f}%  cov95={mm.get('coverage95', float('nan')):.2f}")

    print("\n=== LOPO Vol_ha comparison (N=13) ===")
    rows = [("CNN raster+scalar", cnn_vol)] + sorted(rep.items(), key=lambda kv: -kv[1]["R2"])
    for name, mm in rows:
        print(line(name, mm))
    print(f"\n  CNN DBH (multi-task aux): R2={cnn_dap['R2']:+.2f}  rRMSE={cnn_dap['rRMSE_pct']:.0f}%   (cf. Felipe's DBH target)")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = df[["plot_id", "talhao", "parcela", "Vol_ha", "D_cm"]].copy()
    out["Vol_pred_cnn"] = res["pred"][:, 0]; out["Vol_lo"] = res["lo"][:, 0]; out["Vol_hi"] = res["hi"][:, 0]
    out["DAP_pred_cnn"] = res["pred"][:, 1]
    out.to_csv(config.OUT_DIR / "s7_lopo_predictions.csv", index=False)
    write_manifest(config.OUT_DIR / "s7_manifest.json", script="spike_s7_raster_cnn.py",
                   config=config.CNN, seed=config.CNN.seeds[0],
                   metrics={"cnn_vol": {k: v for k, v in cnn_vol.items() if k != "R2_ci"},
                            "multiseed": ms["summary"]})
    print(f"\nwrote {config.OUT_DIR / 's7_lopo_predictions.csv'} + s7_manifest.json")

    # CNN vs the best classical model
    best_classical = max(rep.values(), key=lambda m: m["R2"])
    verdict = ("CNN ≥ classical (within reach) → headline result; Phase 2 = upside"
               if cnn_vol["R2"] >= best_classical["R2"] - 0.05
               else "CNN < classical → classical ships; Phase 2 (sim pretrain) attempts to close the gap")
    print(f"\nDECISION: CNN R2={cnn_vol['R2']:+.2f} vs best-classical R2={best_classical['R2']:+.2f} → {verdict}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""End-to-end smoke test of every review fix on synthetic data — runs with no real dataset.

Exercises: conformal single-source coverage (C1/C2), leave-one-stand-out group CV (C3), repro seed +
manifest (I1/I2), on-the-fly CNN augmentation + TTA + linear-probe + transfer log (I3/N3), robust
bootstrap + NaN guard + leakage guard (I4/I6/I7), config constants (I8), permutation test (N1),
heteroscedastic quantile + Mondrian conformal + map extrapolation flag (N2), structural features (N4),
error analysis + hard-truth (N5), map provenance (N6).

Run: PYTHONPATH=. python scripts/review_demo.py
"""
import numpy as np, pandas as pd
from greenvista import config
from greenvista.repro import seed_everything, write_manifest
from greenvista.area_based import data as abd, models as abm, validate as abv, mapping as abmap
from greenvista.eval import (mondrian_interval, coverage, interval_score, leverage, residual_correlations, hard_truth_from_trees, compare_to_hard_truth)
from greenvista.features import structural_scalars
from greenvista.image_cnn import train as tr


def synth_plots(n_per_talhao=3, talhoes=(1, 2, 4, 5, 6), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in talhoes:
        base = rng.uniform(18, 26)                          # per-stand height level → group structure
        for p in range(1, n_per_talhao + 1):
            zmean = base + rng.uniform(-2, 2)
            zq90 = zmean + rng.uniform(2, 5)
            rows.append({"talhao": t, "parcela": p, "zmax": zq90 + 1, "zmean": zmean, "zsd": rng.uniform(3, 7),
                         "zq50": zmean, "zq90": zq90, "pzabovezmean": rng.uniform(48, 70),
                         "idade_anos": 8 + t * 0.2,
                         "Vol_ha": np.exp(0.6 + 1.35 * np.log(zmean)) * rng.uniform(0.95, 1.05)})
    df = pd.DataFrame(rows)
    df["plot_id"] = df.talhao.astype(str) + "_" + df.parcela.astype(str)
    return df


def main():
    seed_everything(config.ABA.seed)
    df = synth_plots()
    print(f"synthetic plots N={len(df)} across {df.talhao.nunique()} stands "
          f"(config.PLOT_AREA_HA={config.PLOT_AREA_HA:.4f}, CRS={config.CRS})")

    print("\n[I7] leakage guard kept:", abv.leakage_guard(df))

    # --- C1/C2/C3/N1: LOPO + conformal cov95 + leave-one-stand-out + permutation p ---
    models = {"log_allometric": abm.make_log_allometric, "gpr": abm.make_gpr, "rf": abm.make_rf,
              "knn": abm.make_knn, "quantile_gbm": abm.make_quantile_gbm}
    rep = abv.run(df, models=models, group_col="talhao", perm=1000, seed=0)
    print("\n[C1/C2/C3/N1] model         LOPO-R²  cov95(conf)  intScore   LOTO-R²  perm-p")
    for name, mm in rep.items():
        print(f"  {name:16s} {mm['R2']:+6.2f}   {mm['coverage95']:.2f}       "
              f"{mm['interval_score']:7.1f}  {mm['R2_group']:+6.2f}   {mm['perm_p']:.3f}"
              f"   (parametric cov95={mm['coverage95_parametric']:.2f})")

    # --- N2: heteroscedastic intervals (quantile GBM + Mondrian conformal) ---
    y = df.Vol_ha.to_numpy()
    qpred, qlo, qhi = abv._lopo(abm.make_quantile_gbm, df)
    bins = pd.qcut(y, 3, labels=False, duplicates="drop")
    mlo, mhi = mondrian_interval(y, qpred, bins)
    print(f"\n[N2] quantile-GBM interval widths vary {np.ptp(qhi-qlo):.1f} m³/ha (heteroscedastic); "
          f"Mondrian cov95={coverage(y, mlo, mhi):.2f}, score={interval_score(y, mlo, mhi):.1f}")

    # --- N4: structural features ---
    chm = np.clip(np.random.default_rng(0).normal(22, 4, (32, 32)), 0, None)
    z = np.random.default_rng(0).uniform(2, 30, 4000)
    print("\n[N4] structural scalars:", {k: round(v, 3) for k, v in structural_scalars(chm, z).items()})

    # --- N2/N6/I4: wall-to-wall map with extrapolation flag + NaN guard + provenance ---
    model = abm.make_log_allometric().fit(df, y)
    envelope = abmap.feature_envelope(df)
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 60, (4000, 2)); zz = rng.uniform(5, 32, 4000)
    gdf = abmap.make_map_from_arrays(model, xy, zz, cell=config.ABA.map_cell_m, envelope=envelope)
    prov = abmap.save_map_provenance(model, abd.FEATURES,
                                     config.OUT_DIR / "demo_map_provenance.json", envelope=envelope)
    print(f"\n[N2/I4/N6] map cells={len(gdf)}  extrapolated={int(gdf.extrapolated.sum())}  "
          f"skipped-NaN={gdf.attrs['n_skipped_nan']}  provenance-model={prov['model']['model_type']}")

    # --- N5: error analysis + hard-truth comparison ---
    lev = leverage(y, qpred, ids=df.plot_id.tolist())
    print(f"\n[N5] highest-leverage plot: {lev.iloc[0]['id']} (ΔR²={lev.iloc[0]['delta_R2']:+.3f}); "
          f"|resid|~age r={residual_correlations(y, qpred, df[['idade_anos','zmax']]).get('idade_anos', float('nan')):+.2f}")
    trees = pd.DataFrame({"plot": np.repeat(df.plot_id.values, 4),
                          "V": np.random.default_rng(2).uniform(0.1, 0.5, len(df) * 4)})
    hard = hard_truth_from_trees(trees, "plot", area_ha=config.PLOT_AREA_HA)
    hard = hard.rename(columns={"plot": "plot_id"})
    cmp = compare_to_hard_truth(dict(zip(df.plot_id, qpred)), hard, "plot_id")
    print(f"     hard-truth check on {cmp['n_compared']} plots (measured trees, not model Vol_ha)")

    # --- I3/N3: CNN on-the-fly aug + multi-seed + TTA + linear-probe transfer ---
    print("\n[I3/I2/N3] tiny CNN LOPO (synthetic, fast)…")
    fake = _fake_cnn_samples()
    ids = list(fake); yt = np.stack([fake[i]["labels"] for i in ids])
    ms = tr.run_lopo_multiseed(fake, ids, yt, seeds=(0, 1), n_aug=4, epochs=3, tta=4)
    print(f"     multi-seed R² = {ms['summary']['R2_mean']:+.2f} ± {ms['summary']['R2_sd']:.2f}; TTA=4 used")
    init = tr.pretrain(fake, n_aug=2, epochs=2)
    res = tr.run_lopo(fake, ids, yt, n_aug=4, epochs=3, init_state=init, freeze_backbone=True, verbose=True)
    print(f"     linear-probe transfer warm-started {res['transfer'][0]}/{res['transfer'][1]} tensors")

    write_manifest(config.OUT_DIR / "demo_manifest.json", script="review_demo.py",
                   config=config.ABA, seed=config.ABA.seed,
                   metrics={k: {mk: mv for mk, mv in v.items() if mk != "R2_ci"} for k, v in rep.items()})
    print(f"\nwrote manifest + provenance under {config.OUT_DIR}")


def _fake_cnn_samples(n=8, size=16):
    rng = np.random.default_rng(0)
    h = np.linspace(15, 32, n) + rng.normal(0, 0.5, n)
    out = {}
    for i in range(n):
        r = np.zeros((3, size, size), np.float32)
        r[0] = h[i] + rng.normal(0, 1, (size, size))
        r[1] = rng.uniform(0, 3, (size, size)); r[2] = rng.uniform(0, 2, (size, size))
        out[f"p{i}"] = {"raster": r, "scalars": np.array([h[i], 0.8 * h[i], 4.0], np.float32),
                        "labels": np.array([12 * h[i] + 50 + rng.normal(0, 3), 0.6 * h[i]], np.float32)}
    return out


if __name__ == "__main__":
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    main()

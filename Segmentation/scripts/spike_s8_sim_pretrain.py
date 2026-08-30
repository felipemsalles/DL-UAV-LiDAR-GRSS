#!/usr/bin/env python3
"""Spike S8 — transfer learning: pretrain the CNN on simulated plots, fine-tune per LOPO fold.

Compares to the from-scratch CNN (S7) under the identical leave-one-plot-out split. The simulator is
generic (never conditioned on any real plot), so warm-starting every fold from one pretrained net does
not leak across folds. Reuses S7's saved predictions for the scratch baseline.

Run (repo root, greenvista env): PYTHONPATH=. python scripts/spike_s8_sim_pretrain.py
"""
import numpy as np, pandas as pd
from greenvista import config
from greenvista.eval import metrics, bootstrap_ci
from greenvista.image_cnn import dataset as ds, train as tr, simulate

CACHE = config.LAZ_DIR / "plot_samples_r12_s128.npz"
S7_PRED = config.OUT_DIR / "s7_lopo_predictions.csv"
N_SIM = 3000


def _rep_from_cols(y, p, lo=None, hi=None):
    mm = metrics(y, p); mm["R2_ci"] = bootstrap_ci(y, p)["R2_ci"]
    if lo is not None:
        mm["coverage95"] = float(((y >= lo) & (y <= hi)).mean())
    return mm


def main():
    df = ds.build_plot_table()
    samples = np.load(CACHE, allow_pickle=True)["samples"].item()
    ids = df.plot_id.tolist()
    y = df[["Vol_ha", "D_cm"]].to_numpy(np.float32)

    # scratch baseline from S7's saved predictions (same plots, same split)
    s7 = pd.read_csv(S7_PRED).set_index("plot_id").loc[ids]
    rep0 = _rep_from_cols(s7.Vol_ha.to_numpy(), s7.Vol_pred_cnn.to_numpy(),
                          s7.Vol_lo.to_numpy(), s7.Vol_hi.to_numpy())

    print(f"[sim] generating {N_SIM} synthetic plots…")
    sim = simulate.make_dataset(N_SIM, seed=0)
    sy = np.array([sim[k]["labels"][0] for k in sim])
    print(f"  sim Vol_ha range {sy.min():.0f}–{sy.max():.0f} (real 170–523)")
    print("[pretrain] CNN on sim…")
    state = tr.pretrain(sim, n_aug=1, epochs=15, lr=1e-3, seed=0)  # sim already orientation-diverse
    print("[finetune] LOPO warm-started from pretrained…")
    res1 = tr.run_lopo(samples, ids, y, n_aug=64, epochs=40, lr=5e-4, seed=0, init_state=state, verbose=True)
    rep1 = tr.report(y, res1["pred"], res1["lo"], res1["hi"], col=0)

    def line(name, mm):
        ci = mm["R2_ci"]
        return (f"  {name:26s} R2={mm['R2']:+.2f} (CI {ci[0]:+.2f}..{ci[1]:+.2f})  "
                f"rRMSE={mm['rRMSE_pct']:4.0f}%  cov95={mm.get('coverage95', float('nan')):.2f}")

    print("\n=== Vol_ha LOPO: scratch (S7) vs sim-pretrained (S8) ===")
    print(line("CNN scratch", rep0))
    print(line("CNN sim-pretrained", rep1))
    gain = rep1["R2"] - rep0["R2"]
    print(f"\n  transfer effect: ΔR2={gain:+.2f}, rRMSE {rep0['rRMSE_pct']:.0f}%→{rep1['rRMSE_pct']:.0f}%"
          f"  → {'sim pretraining HELPS' if gain > 0.02 else 'no clear lift (sim-to-real gap)'}")

    out = df[["plot_id", "Vol_ha"]].copy()
    out["Vol_pred_s8"] = res1["pred"][:, 0]; out["Vol_lo"] = res1["lo"][:, 0]; out["Vol_hi"] = res1["hi"][:, 0]
    out.to_csv(config.OUT_DIR / "s8_lopo_predictions.csv", index=False)
    print(f"\nwrote {config.OUT_DIR / 's8_lopo_predictions.csv'}")


if __name__ == "__main__":
    main()

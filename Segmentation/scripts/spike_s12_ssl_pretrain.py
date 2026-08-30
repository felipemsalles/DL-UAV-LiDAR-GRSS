#!/usr/bin/env python3
"""Spike S12 — self-supervised pretraining on real wall-to-wall clouds, fine-tune, vs scratch & sim.

Pretrains the CNN backbone on thousands of unlabelled 12 m windows (pretext = predict canopy metrics),
then fine-tunes on the 13 plots under LOPO. Compares to the from-scratch CNN (S7) and the
synthetic pretraining (S8), to test whether real-cloud SSL helps at N=13.

Run (repo root, greenvista env): PYTHONPATH=. python scripts/spike_s12_ssl_pretrain.py
"""
import numpy as np, pandas as pd
from greenvista import config
from greenvista.eval import metrics, bootstrap_ci
from greenvista.image_cnn import dataset as ds, train as tr, ssl_pretrain as ssl

CACHE = config.LAZ_DIR / "plot_samples_r12_s128.npz"
N_WINDOWS = 3000


def rep_from(y, p):
    mm = metrics(y, p); mm["R2_ci"] = bootstrap_ci(y, p)["R2_ci"]; return mm


def main():
    df = ds.build_plot_table(); ids = df.plot_id.tolist()
    y = df[["Vol_ha", "D_cm"]].to_numpy(np.float32)
    samples = np.load(CACHE, allow_pickle=True)["samples"].item()

    s7 = pd.read_csv(config.OUT_DIR / "s7_lopo_predictions.csv").set_index("plot_id").loc[ids]
    rep0 = rep_from(y[:, 0], s7.Vol_pred_cnn.to_numpy())

    print(f"[ssl] sampling {N_WINDOWS} unlabelled windows from 6 clouds (excluding the 13 plots)…")
    ssl_samples = ssl.build_ssl_samples(N_WINDOWS, seed=0)
    print(f"  built {len(ssl_samples)} windows; pretext metrics = {ssl.PRETEXT_METRICS}")
    print("[ssl] pretraining backbone on raster→metrics…")
    state = ssl.pretrain_backbone(ssl_samples, epochs=20, seed=0)
    print("[finetune] LOPO warm-started from SSL backbone…")
    res = tr.run_lopo(samples, ids, y, n_aug=64, epochs=40, lr=5e-4, seed=0, init_state=state, verbose=True)
    rep1 = rep_from(y[:, 0], res["pred"][:, 0])

    print("\n=== Vol_ha LOPO: scratch vs sim-pretrained vs SSL-pretrained ===")
    rows = [("CNN scratch (S7)", rep0), ("CNN SSL-real (S12)", rep1)]
    try:
        s8 = pd.read_csv(config.OUT_DIR / "s8_lopo_predictions.csv").set_index("plot_id").loc[ids]
        rows.insert(1, ("CNN sim-pretrained (S8)", rep_from(y[:, 0], s8.Vol_pred_s8.to_numpy())))
    except FileNotFoundError:
        pass
    for name, mm in rows:
        print(f"  {name:26s} R2={mm['R2']:+.2f} (CI {mm['R2_ci'][0]:+.2f}..{mm['R2_ci'][1]:+.2f})  "
              f"rRMSE={mm['rRMSE_pct']:4.0f}%  bias={mm['bias_pct']:+5.0f}%")
    d = rep1["R2"] - rep0["R2"]
    print(f"\n  GPR floor: R²=0.74, rRMSE 13%.  SSL effect vs scratch: ΔR²={d:+.2f} "
          f"→ {'SSL HELPS' if d > 0.02 else 'no clear lift at N=13'}")

    out = df[["plot_id", "Vol_ha"]].copy(); out["Vol_pred_s12"] = res["pred"][:, 0]
    out.to_csv(config.OUT_DIR / "s12_lopo_predictions.csv", index=False)
    print(f"  wrote {config.OUT_DIR / 's12_lopo_predictions.csv'}")


if __name__ == "__main__":
    main()

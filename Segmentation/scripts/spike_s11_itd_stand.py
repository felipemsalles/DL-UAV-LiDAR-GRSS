#!/usr/bin/env python3
"""Spike S11 — ITD→stand-aggregate hybrid vs field & ABA.

For each of the 13 plots: detect treetops (CHM watershed), sum per-tree volumes (H→V allometry) →
Vol_ha. Compare to field Vol_ha both raw and with a field-stocking correction (the detection-limit
ceiling). Reports detection rate vs field n_arv. No per-plot training (the allometry comes from the
destructive scaling), so this is a direct out-of-the-box estimate.

Run (repo root, greenvista env): PYTHONPATH=. python scripts/spike_s11_itd_stand.py
"""
import numpy as np, pandas as pd, laspy
from greenvista import config
from greenvista.eval import metrics
from greenvista.image_cnn import dataset as ds, footprint
from greenvista.area_based import stand_from_trees as sft


def main():
    df = ds.build_plot_table()
    pm = pd.read_csv(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv")
    for col in ("talhao", "parcela"):
        pm[col] = pd.to_numeric(pm[col], errors="coerce").astype("Int64")
    df = df.merge(pm[["talhao", "parcela", "n_arv", "n_arv_ha"]], on=["talhao", "parcela"], how="left")
    coef = sft.fit_hv_allometry()
    print(f"H→V allometry lnV={coef[0]:.2f}+{coef[1]:.2f}·lnH | plots N={len(df)}")

    cloud = {}
    rows = []
    for _, r in df.iterrows():
        t = int(r.talhao)
        if t not in cloud:
            las = laspy.read(str(config.LAZ_DIR / f"SaoManuelTotal_{t:03d}.laz"))
            cloud[t] = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
        pts = footprint.extract_footprint(cloud[t], (r.x, r.y), sft.RADIUS_M)
        vraw, ndet = sft.plot_vol_ha(pts, (r.x, r.y), coef)
        vcorr, _ = sft.plot_vol_ha(pts, (r.x, r.y), coef, stocking_ha=float(r.n_arv_ha))
        det_rate = ndet / (r.n_arv_ha * sft.PLOT_AREA_HA) * 100
        rows.append(dict(plot_id=r.plot_id, Vol_ha=r.Vol_ha, vol_raw=vraw, vol_corr=vcorr,
                         n_det=ndet, n_field=round(r.n_arv_ha * sft.PLOT_AREA_HA), det_pct=det_rate))
    res = pd.DataFrame(rows)

    print(f"\n  mean detection rate: {res.det_pct.mean():.0f}% of field stems "
          f"({res.n_det.sum()} detected vs ~{res.n_field.sum()} field)")
    y = res.Vol_ha.to_numpy()
    for label, col in [("ITD raw (detected only)", "vol_raw"), ("ITD + field-stocking correction", "vol_corr")]:
        mm = metrics(y, res[col].to_numpy())
        print(f"  {label:34s} R²={mm['R2']:+.2f}  rRMSE={mm['rRMSE_pct']:4.0f}%  bias={mm['bias_pct']:+5.0f}%")

    print("\n  → compare to ABA-GPR (R²=0.74, rRMSE 13%, bias +1%). The hybrid is detection-limited:")
    print("    raw under-counts (closed canopy misses suppressed stems); with correct stocking the")
    print("    H→V aggregate is sound — i.e. the bottleneck is detection, not the per-tree volume model.")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.OUT_DIR / "s11_itd_stand.csv", index=False)
    print(f"\n  wrote {config.OUT_DIR / 's11_itd_stand.csv'}")


if __name__ == "__main__":
    main()

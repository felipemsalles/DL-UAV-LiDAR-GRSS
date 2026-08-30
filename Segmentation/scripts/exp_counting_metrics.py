#!/usr/bin/env python3
"""Re-express the FF3D detection results as a tree-counting task.

Detection is reported elsewhere in this repo as a ratio (detected / field count), which reads as a
proxy for segmentation quality but is not one: São Manuel has no per-point instance ground truth,
no RTK survey is possible and manual annotation is out of scope, so segmentation quality is not
measurable on this data.

Counting is a different task with its own metrics — MAE, RMSE, relative MAE, bias — evaluated
against ground-truth counts, which the 717 field-tallied stems across 13 plots supply. This script
computes those metrics.

Three methods, identical plots and footprint:
  * baseline    — one 32 m tile per plot, single FF3D pass (`manual_match/ff3d_detection.csv`)
  * overlap     — 3x3 offset tiles pooled and NMS-merged, recomputed here from the durable centroid
                  dump so the merge radius is a parameter rather than a frozen choice
  * chm         — classical CHM local maxima (`manual_match/density_count.csv::chm_in_plot`).
                  Labelled "CHM watershed" in older material, but it is not watershed; the real
                  watershed baseline is scripts/exp_watershed_baseline.py.

Both merge radii are reported: r=1.2 m is duplicate-padded (two plots exceed 100% of the stems that
exist in the field) and r=1.5 m is the defensible one. Recomputing from centroids is what allows
per-plot counting metrics at r=1.5, since the committed `ff3d_detection_overlap.csv` holds only the
r=1.2 run.

Run: PYTHONPATH=. python scripts/exp_counting_metrics.py
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from greenvista import config
from greenvista.repro import seed_everything, write_manifest

warnings.simplefilter("ignore")

CENTROIDS = config.REPO / "data/detections/ff3d_overlap_centroids_13plot.csv"
PLOT_R = config.PLOT_RADIUS_M
RADII = (1.2, 1.5)


def nms_merge(cents, sizes, radius):
    """Greedy NMS keeping the largest instance per cluster — same rule as ff3d_detection_overlap."""
    if len(cents) == 0:
        return cents
    kept = []
    for i in np.argsort(-sizes):
        p = cents[i]
        if not kept or np.min(np.hypot(*(np.asarray(kept) - p).T)) > radius:
            kept.append(p)
    return np.asarray(kept).reshape(-1, 2)


def overlap_counts(radius):
    """Per-plot merged in-plot instance count at a given merge radius, from the centroid dump."""
    c = pd.read_csv(CENTROIDS)
    parc = _plot_centres()
    out = {}
    for pid, g in c.groupby("plot_id"):
        merged = nms_merge(g[["cx", "cy"]].to_numpy(float), g["n_pts"].to_numpy(float), radius)
        px, py = parc[pid]
        d2 = (merged[:, 0] - px) ** 2 + (merged[:, 1] - py) ** 2
        out[pid] = int((d2 <= PLOT_R ** 2).sum())
    return out


def _plot_centres():
    import geopandas as gpd
    p = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for c in ("talhao", "parcela"):
        p[c] = pd.to_numeric(p[c], errors="coerce").astype("Int64")
    return {f"{int(r.talhao)}_{int(r.parcela)}": (r.geometry.x, r.geometry.y)
            for _, r in p.iterrows() if pd.notna(r.talhao) and pd.notna(r.parcela)}


def counting_metrics(field, pred, n_boot=4000, seed=0):
    """MAE / RMSE / relative MAE / bias, with a bootstrap interval on MAE.

    Relative MAE is normalised by the mean field count, which is the convention in the object
    counting literature and keeps the number comparable across plots of different stocking.
    """
    field = np.asarray(field, float)
    pred = np.asarray(pred, float)
    err = pred - field
    mae = float(np.abs(err).mean())
    rng = np.random.default_rng(seed)
    boot = [np.abs(err[rng.integers(0, len(err), len(err))]).mean() for _ in range(n_boot)]
    return {
        "MAE": mae,
        "MAE_lo": float(np.percentile(boot, 2.5)),
        "MAE_hi": float(np.percentile(boot, 97.5)),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "rMAE_pct": float(100.0 * mae / field.mean()),
        "bias": float(err.mean()),
        "bias_pct": float(100.0 * err.mean() / field.mean()),
        "r": float(np.corrcoef(field, pred)[0, 1]) if np.std(pred) > 0 else float("nan"),
        "detection_ratio_pct": float(100.0 * pred.sum() / field.sum()),
        "n_plots": int(len(field)),
    }


def main():
    seed_everything(0)
    base = pd.read_csv(config.OUT_DIR / "ff3d_detection.csv")[["plot_id", "ff3d_in_plot", "field_n_arv"]]
    dens = pd.read_csv(config.OUT_DIR / "density_count.csv")[["plot_id", "chm_in_plot"]]
    df = base.merge(dens, on="plot_id", how="left").rename(
        columns={"ff3d_in_plot": "baseline", "chm_in_plot": "chm", "field_n_arv": "field"})
    for r in RADII:
        df[f"overlap_r{r}"] = df.plot_id.map(overlap_counts(r))

    methods = {"CHM local maxima": "chm", "FF3D single pass": "baseline",
               "FF3D overlapped r=1.2": f"overlap_r{RADII[0]}",
               "FF3D overlapped r=1.5": f"overlap_r{RADII[1]}"}

    rows = []
    print(f"{len(df)} plots, {int(df.field.sum())} trees tallied in the field\n")
    print(f"{'method':24s} {'MAE':>6} {'95% CI':>16} {'RMSE':>7} {'rel MAE':>8} {'bias':>8} {'r':>6}")
    for label, col in methods.items():
        m = counting_metrics(df.field, df[col])
        rows.append({"method": label, "column": col, **m})
        print(f"{label:24s} {m['MAE']:6.1f} {m['MAE_lo']:7.1f} a {m['MAE_hi']:5.1f} "
              f"{m['RMSE']:7.1f} {m['rMAE_pct']:7.1f}% {m['bias_pct']:+7.1f}% {m['r']:6.2f}")

    res = pd.DataFrame(rows)

    # ---- scatter: predicted vs field count, the figure a counting paper leads with ----
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=170)
    lim = [0, max(df.field.max(), df[[c for c in methods.values()]].max().max()) * 1.08]
    ax.plot(lim, lim, ls="--", lw=1.2, color="#888888", label="perfect count", zorder=1)
    style = {"chm": ("#b0782f", "o"), "baseline": ("#7a8f9e", "s"),
             f"overlap_r{RADII[1]}": ("#1d6b45", "^")}
    for label, col in methods.items():
        if col not in style:
            continue
        c, mk = style[col]
        ax.scatter(df.field, df[col], s=64, color=c, marker=mk, alpha=.9,
                   edgecolor="white", linewidth=.8, label=label, zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("trees tallied in the field in the plot", fontsize=11)
    ax.set_ylabel("automatically detected trees", fontsize=11)
    ax.set_title("Tree count per plot\nEvery method falls below the diagonal, "
                 "because the laser does not see part of the stems", fontsize=11.5, pad=12)
    ax.legend(fontsize=9.5, loc="upper left", frameon=True, framealpha=.95)
    ax.grid(alpha=.25, lw=.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out_png = config.OUT_DIR / "counting_scatter.png"
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.OUT_DIR / "counting_metrics.csv", index=False)
    df.to_csv(config.OUT_DIR / "counting_per_plot.csv", index=False)
    write_manifest(config.OUT_DIR / "counting_metrics_manifest.json",
                   script="exp_counting_metrics",
                   config={"plot_radius_m": PLOT_R, "merge_radii": list(RADII),
                           "centroids": str(CENTROIDS.relative_to(config.REPO))},
                   metrics={"per_method": res.to_dict("records")})
    print(f"\nfigure {out_png}\ntables in {config.OUT_DIR}")


if __name__ == "__main__":
    main()

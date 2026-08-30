#!/usr/bin/env python3
"""Backfill already-computed experiments into W&B (online).

Reads the result CSVs/manifests in OUT_DIR (manual_match/) and logs one W&B run per experiment with a
results table, headline metrics, and the CSV + any figure as artifacts.

Run: PYTHONPATH=. python scripts/wandb_backfill.py
"""
import json
import pandas as pd

from greenvista import config
from greenvista.tracking import log_experiment

OUT = config.OUT_DIR


def _best(df, by, minimize=True):
    if by not in df.columns or not len(df):
        return {}
    row = df.loc[df[by].idxmin() if minimize else df[by].idxmax()]
    name_col = next((c for c in ("model", "tile", "talhao") if c in df.columns), None)
    out = {f"best_{by}": float(row[by])}
    if name_col:
        out["best_model"] = str(row[name_col])
    for c in ("LOPO_R2", "LOPO_rRMSE", "LOTO_R2", "LOTO_rRMSE", "R2", "rRMSE_pct", "cov95", "perm_p"):
        if c in df.columns:
            out[c] = float(row[c])
    return out


# (name, group, csv, extra config, metric extractor, figures)
EXPERIMENTS = [
    ("lidar-only-baseline", "volume", "lidar_only_baseline.csv",
     {"task": "plot Vol_ha", "inputs": "LiDAR only (no age/DBH/stocking)", "N": 13},
     lambda d: _best(d, "LOTO_rRMSE"), []),
    ("model-ladder", "volume", "exp_better_numbers.csv",
     {"task": "plot Vol_ha", "N": 13}, lambda d: _best(d, "LOTO_rRMSE"), ["rumple_verification.png"]),
    ("source-decomposition", "volume", "source_decomposition.csv",
     {"task": "plot Vol_ha", "question": "age vs LiDAR"}, lambda d: _best(d, "LOTO_rRMSE"),
     ["source_decomposition.png"]),
    ("ff3d-detection", "segmentation", "ff3d_detection.csv",
     {"model": "ForestFormer3D", "tiles": 13, "gpu": "RTX 5060 8GB"},
     lambda d: {"overall_detection_rate_pct": round(100 * d.ff3d_in_plot.sum() / d.field_n_arv.sum(), 1),
                "ff3d_stems": int(d.ff3d_in_plot.sum()), "field_stems": int(d.field_n_arv.sum())},
     ["per_tree_dap_audit.png"]),
    ("per-tree-dap", "per-tree", "per_tree_dap.csv",
     {"task": "per-tree DAP", "N": 23, "models": "CNN/PointNet/PointTransformer/allometry"},
     lambda d: _best(d, "R2", minimize=False), ["per_tree_dap_audit.png"]),
]


def main():
    urls = []
    for name, group, csv, cfg, metric_fn, figs in EXPERIMENTS:
        p = OUT / csv
        if not p.exists():
            print(f"  skip {name}: {csv} not found")
            continue
        df = pd.read_csv(p)
        arts = [str(p)] + [str(OUT / f) for f in figs if (OUT / f).exists()]
        url = log_experiment(name, config=cfg, metrics=metric_fn(df), table_csv=str(p),
                             artifacts=arts, group=group, tags=["backfill"])
        print(f"  logged {name}: {url}")
        urls.append(url)

    # Hdom pipeline, read from the JSON manifest
    man = OUT / "hdom_yield_manifest.json"
    if man.exists():
        m = json.loads(man.read_text())["metrics"]
        url = log_experiment("hdom-yield-pipeline", group="volume",
                            config={"model": "Vol_ha ~ ln(Hdom)+ln(age)", "deployable": True},
                            metrics={"LOPO_R2": m["LOPO"]["R2"], "LOPO_rRMSE": m["LOPO"]["rRMSE_pct"],
                                     "LOTO_R2": m["LOTO"]["R2"], "LOTO_rRMSE": m["LOTO"]["rRMSE_pct"],
                                     "cov95": m["cov95"], "perm_p": m["perm_p"], "equation": m["equation"]},
                            artifacts=[str(man)], tags=["backfill", "deliverable"])
        print(f"  logged hdom-yield-pipeline: {url}")
        urls.append(url)

    print(f"\n{len([u for u in urls if u])} runs logged to W&B project '{config.__name__ and 'greenvista-eucalyptus-volume'}'")


if __name__ == "__main__":
    main()

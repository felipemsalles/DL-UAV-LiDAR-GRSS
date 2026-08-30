#!/usr/bin/env python3
"""Per-tree DBH: reproduce the peer's CNN / PointNet / PointTransformer and evaluate them.

Leave-one-tree-out CV on the 23 geolocated destructively scaled trees (stand 001) vs an allometry
baseline. Each net also gets a shuffled-label control (must collapse to R2 ~ 0). Question: does a
point-cloud deep net beat crown allometry for DBH from nadir UAV LiDAR at this N?

Run: PYTHONPATH=. GREENVISTA_LAZ_DIR=<lazdir> python scripts/exp_per_tree_dap.py
"""
import numpy as np, pandas as pd
from greenvista import config
from greenvista.repro import seed_everything, write_manifest
from greenvista.per_tree_dap import data as ptd
from greenvista.per_tree_dap.train import evaluate_all


def main():
    seed_everything(0)
    trees = ptd.load_tree_crowns(radius=2.0)
    print(f"N={len(trees)} geolocated scaled trees (stand 001) | "
          f"DBH {min(t['D_cm'] for t in trees.values()):.1f}–{max(t['D_cm'] for t in trees.values()):.1f} cm | "
          f"crown points median={int(np.median([len(t['points']) for t in trees.values()]))}")

    res = evaluate_all(trees, seeds=(0, 1), n_sample=256, epochs=30, n_aug=4, tta=6)

    rows = sorted(res.items(), key=lambda kv: -kv[1]["R2"])
    print("\n=== Per-tree DBH - leave-one-tree-out (N=23) ===")
    print(f"  {'model':30s} {'R2':>6} {'rRMSE%':>7} {'perm_p':>7}   deep-only: seedR2±sd / shuffledR2")
    for name, mm in rows:
        extra = ""
        if "R2_shuffled" in mm:
            extra = f"   {mm['R2_seed_mean']:+.2f}±{mm['R2_seed_sd']:.2f} / shuf {mm['R2_shuffled']:+.2f}"
        print(f"  {name:30s} {mm['R2']:+6.2f} {mm['rRMSE_pct']:7.1f} {mm['perm_p']:7.3f}{extra}")

    best_allom = max((v for k, v in res.items() if k.startswith("allometry")), key=lambda m: m["R2"])
    # Acceptance rule: a deep net only counts if its real R² clearly exceeds its own shuffled-label
    # control; otherwise the apparent skill is an overfitting artifact.
    valid_deep = {k: v for k, v in res.items()
                  if "R2_shuffled" in v and v["R2"] > v["R2_shuffled"] + 0.10}
    print("\n=== shuffle-control audit (real must exceed shuffled to count) ===")
    for k, v in res.items():
        if "R2_shuffled" in v:
            ok = "VALID" if k in valid_deep else "ARTIFACT (shuffle ≥ real → discard)"
            print(f"  {k:20s} real R²={v['R2']:+.2f}  shuffled R²={v['R2_shuffled']:+.2f}  → {ok}")
    if valid_deep:
        bk, bv = max(valid_deep.items(), key=lambda kv: kv[1]["R2"])
        verdict = (f"{bk} gives a genuine but marginal gain (R²={bv['R2']:+.2f} vs allometry "
                   f"{best_allom['R2']:+.2f})" if bv["R2"] > best_allom["R2"] + 0.02
                   else f"no validated deep net beats allometry ({best_allom['R2']:+.2f})")
    else:
        verdict = f"NO deep net survives the shuffle control — allometry ({best_allom['R2']:+.2f}) is the best model"
    print(f"\nVERDICT: {verdict}")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(model=k, **v) for k, v in res.items()]).to_csv(
        config.OUT_DIR / "per_tree_dap.csv", index=False)
    write_manifest(config.OUT_DIR / "per_tree_dap_manifest.json", script="exp_per_tree_dap.py",
                   config={"N": len(trees), "n_sample": 256, "epochs": 30, "seeds": [0, 1]},
                   metrics=res)
    print(f"wrote {config.OUT_DIR / 'per_tree_dap.csv'} + manifest")


if __name__ == "__main__":
    main()

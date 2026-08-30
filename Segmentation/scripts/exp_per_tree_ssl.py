#!/usr/bin/env python3
"""Does SSL pretraining on FF3D crowns improve per-tree DBH? (our data only)

Pretrain a PointNet on the unlabelled FF3D crown corpus (structure-metric pretext), then fine-tune on
the 23 DBH-labelled trees. Compare PointNet from-scratch vs SSL-finetuned vs SSL-linear-probe, multi-
seed, each with a shuffled-label control. Expected effect at N=23 is small.

Run: PYTHONPATH=. GREENVISTA_LAZ_DIR=<lazdir> python scripts/exp_per_tree_ssl.py
"""
import glob
from pathlib import Path
import numpy as np

from greenvista import config
from greenvista.repro import seed_everything, write_manifest
from greenvista.tracking import log_experiment
from greenvista.eval import metrics, permutation_test
from greenvista.per_tree_dap import data as ptd, ssl
from greenvista.per_tree_dap.train import _loocv_deep


def find_zip():
    cands = [str(Path(config.LAZ_DIR).parent / "ff3d_fullrun_results.zip")]
    cands += sorted(glob.glob(str(config.REPO / "project/models/FF3D_inference/FF3D_oracle/bucket_out_folder/*.zip")))
    return next((c for c in cands if Path(c).exists()), None)


def evaluate(trees, dap, label, seeds=(0, 1, 2), **kw):
    r2s, preds, y = [], None, None
    for s in seeds:
        y, p = _loocv_deep(trees, "PointNet", dap, seed=s, **kw)
        r2s.append(metrics(y, p)["R2"])
        preds = p if preds is None else preds + p
    preds /= len(seeds)
    ys, ps = _loocv_deep(trees, "PointNet", dap, seed=0, shuffle=True, **kw)
    mm = metrics(y, preds)
    print(f"  {label:26s} R²={mm['R2']:+.2f} (seed {np.mean(r2s):+.2f}±{np.std(r2s):.2f}) "
          f"rRMSE={mm['rRMSE_pct']:.1f}%  shuffled R²={metrics(ys, ps)['R2']:+.2f}  "
          f"p={permutation_test(y, preds, n_perm=2000)['p_value']:.3f}")
    return {"model": label, "R2": mm["R2"], "R2_seed_mean": float(np.mean(r2s)),
            "R2_seed_sd": float(np.std(r2s)), "rRMSE_pct": mm["rRMSE_pct"],
            "R2_shuffled": metrics(ys, ps)["R2"]}


def main():
    seed_everything(0)
    z = find_zip()
    if not z:
        print("no FF3D output zip found — run the full pipeline first."); return
    trees = ptd.load_tree_crowns(radius=2.0)
    dap = {a: trees[a]["D_cm"] for a in trees}
    print(f"DBH trees: {len(trees)} | building SSL crown corpus from {Path(z).name} …")
    corpus = ssl.crown_corpus_from_ff3d(z, min_pts=64)
    print(f"SSL corpus: {len(corpus)} unlabelled crowns (median {int(np.median([len(c) for c in corpus]))} points)")

    print("\npretraining PointNet backbone on crown-structure pretext …")
    backbone = ssl.pretrain(corpus, n_sample=256, epochs=30)

    kw = dict(n_sample=256, epochs=30, n_aug=4, tta=4)
    print("\n=== per-tree DBH: does SSL help? (LOOCV, N=23) ===")
    rows = [
        evaluate(trees, dap, "PointNet scratch", **kw),
        evaluate(trees, dap, "PointNet SSL-finetune", init_state=backbone, **kw),
        evaluate(trees, dap, "PointNet SSL-linear-probe", init_state=backbone, freeze_backbone=True, **kw),
    ]
    import pandas as pd
    from greenvista.per_tree_dap.train import _loocv_allometry
    ay, ap = _loocv_allometry(trees, dap, ("zmax", "crown_area"))
    rows.append({"model": "allometry (baseline)", "R2": metrics(ay, ap)["R2"],
                 "rRMSE_pct": metrics(ay, ap)["rRMSE_pct"], "R2_seed_mean": None, "R2_seed_sd": None,
                 "R2_shuffled": None})
    print(f"  {'allometry (baseline)':26s} R²={metrics(ay, ap)['R2']:+.2f} rRMSE={metrics(ay, ap)['rRMSE_pct']:.1f}%")

    res = pd.DataFrame(rows)
    scratch = next(r for r in rows if r["model"] == "PointNet scratch")["R2"]
    best_ssl = max(r["R2"] for r in rows if "SSL" in r["model"])
    verdict = (f"SSL helps (+{best_ssl - scratch:.2f} R² over scratch)" if best_ssl > scratch + 0.03
               else "SSL does not beat scratch at N=23 (as expected) — safe null result")
    print(f"\nVERDICT: scratch R²={scratch:+.2f}, best SSL R²={best_ssl:+.2f} → {verdict}")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.OUT_DIR / "per_tree_ssl.csv", index=False)
    write_manifest(config.OUT_DIR / "per_tree_ssl_manifest.json", script="exp_per_tree_ssl.py",
                   config={"N_trees": len(trees), "corpus": len(corpus), "pretext": list(ssl.PRETEXT_KEYS)},
                   metrics={"scratch_R2": scratch, "best_ssl_R2": best_ssl, "verdict": verdict})
    log_experiment("per-tree-ssl", group="per-tree", tags=["SSL", "our-data"],
                   config={"N_trees": len(trees), "ssl_corpus": len(corpus)},
                   metrics={"scratch_R2": scratch, "best_ssl_R2": best_ssl, "verdict": verdict},
                   table_csv=str(config.OUT_DIR / "per_tree_ssl.csv"))
    print(f"wrote {config.OUT_DIR/'per_tree_ssl.csv'} + logged to W&B")


if __name__ == "__main__":
    main()

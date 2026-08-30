#!/usr/bin/env python3
"""Confidence interval on the recall difference between AdaBN and TENT.

Repeating the run does not answer the relevant question: the run-to-run noise floor of
SegmentAnyTree is 0.2 F1 points (three identical runs), so repetitions give nearly
identical numbers and no interval at all. What varies is which trees fall into the
evaluation, so the resampling unit is the stem.

That is why the target is recall and not F1: recall is indexed by stem, so resampling
stems is direct. Precision depends on the set of detections, which is a different unit,
and mixing the two in the same bootstrap produces an interval with no interpretation.

Caveat to declare: the 211 stems come from two plots, and the nine views of each are the
same group of trees, so the independent units are two and not 211. The bootstrap treats
each stem as a trial, as the detection literature does, and the resulting interval is
optimistic with respect to spatial correlation.

Usage: PYTHONPATH=. python scripts/exp_bootstrap_adaptacao.py A=<dir> B=<dir>
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.argv_bak, sys.argv = sys.argv, [sys.argv[0]]
import importlib.util
sp = importlib.util.spec_from_file_location(
    "tta", Path(__file__).resolve().parent / "exp_tta_comparison.py")
tta = importlib.util.module_from_spec(sp)
sp.loader.exec_module(tta)
sys.argv = sys.argv_bak

LIMIAR, N_BOOT, SEMENTE = 2.0, 10000, 20260821


def casados(ref, pred, limiar):
    """Which stems of `ref` were matched. The `parear` of the other script returns
    only the count, and what matters here is the outcome PER STEM."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    m = np.zeros(len(ref), bool)
    if len(ref) == 0 or len(pred) == 0:
        return m
    d = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    li, ci = linear_sum_assignment(np.where(d <= limiar, d, 1e6))
    m[li[d[li, ci] <= limiar]] = True
    return m


def desfecho_por_fuste(pasta, ctr, REF, modelo="sat"):
    """For each stem in the evaluation square, 1 if it was matched, 0 otherwise.

    The loader depends on the model: SegmentAnyTree delivers LAZ with the field
    PredInstance and ForestFormer3D delivers a panoptic PLY, so pointing the SAT
    loader at a folder of PLY silently returns nothing. The merge radius also
    differs, 1.5 m for SAT and 1.1 m for FF3D (the value with which Table II was
    measured). Passing modelo=ff3d switches both together.
    """
    dados = (tta.centroides_ff3d(Path(pasta), "centroide") if modelo == "ff3d"
             else tta.centroides_sat(Path(pasta)))
    ids, ok = [], []
    for pid in tta.PLOTS:
        if pid not in dados:
            continue
        cents, sizes = dados[pid]
        cx, cy, _ = ctr[pid]
        fundidos = tta.nms_merge(cents, sizes, tta.RAIO_FUSAO)
        ref = tta.no_quadrado(REF, cx, cy)
        pred = tta.no_quadrado(fundidos, cx, cy)
        m = casados(ref, pred, LIMIAR)
        for k, r in enumerate(ref):
            ids.append((pid, round(float(r[0]), 3), round(float(r[1]), 3)))
            ok.append(bool(m[k]))
    return ids, np.array(ok, bool)


def main():
    import geopandas as gpd
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    if len(args) != 2:
        sys.exit("usage: A=<dir> B=<dir>")
    modelo = "ff3d" if any("ff3d" in v or "tent_adam" in v or "tent_sgd" in v
                           for v in args.values()) else "sat"
    if modelo == "ff3d":
        tta.RAIO_FUSAO = 1.1
    print(f"model: {modelo}, merge radius {tta.RAIO_FUSAO} m")
    (na, pa), (nb, pb) = args.items()
    ctr = tta.centros_e_censo()
    REF = np.column_stack(gpd.read_file(tta.MAPA).geometry.apply(lambda g: (g.x, g.y)).tolist())
    REF = REF.T if REF.shape[0] == 2 else REF
    ida, a = desfecho_por_fuste(pa, ctr, REF, modelo)
    idb, b = desfecho_por_fuste(pb, ctr, REF, modelo)
    if ida != idb:
        sys.exit("ERROR: the two sets of stems do not coincide, pairing impossible")
    n = len(a)
    d = a.astype(int) - b.astype(int)
    print(f"stems evaluated: {n}   (from TWO plots, not independent)")
    print(f"  recall {na:<10} {a.mean():.4f}   ({a.sum()} of {n})")
    print(f"  recall {nb:<10} {b.mean():.4f}   ({b.sum()} of {n})")
    print(f"  difference         {a.mean()-b.mean():+.4f}")
    print(f"  disagreement: {na} only {int((d>0).sum())}, {nb} only {int((d<0).sum())}")
    rng = np.random.default_rng(SEMENTE)
    dif = np.array([d[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(dif, [2.5, 97.5])
    print(f"\nstem-paired bootstrap, {N_BOOT} resamplings")
    print(f"  95% CI of the difference: [{lo:+.4f}, {hi:+.4f}]")
    print(f"  fraction of resamplings with {na} >= {nb}: {(dif >= 0).mean():.3f}")
    veredito = ("the difference EXCLUDES zero, it is evidence" if lo > 0 or hi < 0
                else "the interval CONTAINS zero, it is not evidence")
    print(f"  -> {veredito}")


if __name__ == "__main__":
    main()

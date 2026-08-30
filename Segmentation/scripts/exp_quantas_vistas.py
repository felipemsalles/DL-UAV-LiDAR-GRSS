#!/usr/bin/env python3
"""How many views are needed? All 511 combinations of the nine, without a GPU.

The paper uses nine views. The number did not come from a sweep: it is the
geometric consequence of a 3x3 grid of half-tile-step offsets, chosen so that
every point falls well inside at least one tile. This script measures whether
four or five would do the same.

No GPU is needed. The nine per-plot outputs are already on disk, one per view,
carrying `PredInstance`. It is enough to pick subsets, re-fuse them by NMS and
score. That is 2^9 - 1 = 511 subsets per plot, seconds of CPU.

The comparison is paired by construction: every subset sees exactly the same raw
detections, so the only difference between them is the fusion, with no inference
noise in between.

Protocol identical to Table II: SegmentAnyTree with AdaBN, radius 1.5 m, central
26 m square of the two plots that have a stem map, Hungarian matching at 2 m,
aggregated rather than averaged.

Run: PYTHONPATH=. python scripts/exp_quantas_vistas.py
Output: manual_match/quantas_vistas.csv
"""
import importlib.util
import itertools
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from greenvista import config  # noqa: E402

_s = importlib.util.spec_from_file_location("tta", R / "scripts/exp_tta_comparison.py")
tta = importlib.util.module_from_spec(_s); _s.loader.exec_module(tta)

FONTE = R / "work/tent_opt/adabn_r1"
RAIO, LIMIAR, CENTRO = 1.5, 2.0, 4          # g4 is the centred view
SAIDA = config.OUT_DIR / "quantas_vistas.csv"


def por_vista():
    """Centroids and weights of each of the 9 views, per plot."""
    d = {}
    for f in sorted(FONTE.glob("t001_p*_g*_out.laz")):
        pid = f"1_{int(f.name.split('_p')[1][:3])}"
        g = int(f.name.split("_g")[1][0])
        las = laspy.read(str(f))
        inst = np.asarray(las.PredInstance)
        xy = np.column_stack([np.asarray(las.x), np.asarray(las.y)])
        cs, ws = [], []
        for i in np.unique(inst[inst > 0]):
            m = inst == i
            cs.append(xy[m].mean(0)); ws.append(int(m.sum()))
        d[(pid, g)] = (np.asarray(cs).reshape(-1, 2), np.asarray(ws))
    return d


def pontua(sub, vistas, centros, REF):
    tp = n_ref = n_pred = 0
    for pid in tta.PLOTS:
        cx, cy, _ = centros[pid]
        cs = [vistas[(pid, g)][0] for g in sub if len(vistas[(pid, g)][0])]
        ws = [vistas[(pid, g)][1] for g in sub if len(vistas[(pid, g)][0])]
        if not cs:
            continue
        fund = tta.nms_merge(np.vstack(cs), np.concatenate(ws), RAIO)
        ref = tta.no_quadrado(REF, cx, cy)
        pred = tta.no_quadrado(fund, cx, cy)
        if len(ref) and len(pred):
            D = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
            li, ci = linear_sum_assignment(np.where(D <= LIMIAR, D, 1e6))
            tp += int((D[li, ci] <= LIMIAR).sum())
        n_ref += len(ref); n_pred += len(pred)
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_ref if n_ref else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), n_pred


def main():
    centros = tta.centros_e_censo()
    REF = gpd.read_file(tta.MAPA)
    REF = np.column_stack([REF.geometry.x, REF.geometry.y])
    print("reading the nine views of each plot", flush=True)
    vistas = por_vista()

    linhas = []
    for k in range(1, 10):
        for sub in itertools.combinations(range(9), k):
            p, r, f1, npd = pontua(sub, vistas, centros, REF)
            linhas.append(dict(k=k, vistas="".join(map(str, sub)),
                               tem_centro=CENTRO in sub, precisao=p,
                               revocacao=r, f1=f1, n_pred=npd))
    df = pd.DataFrame(linhas)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAIDA, index=False)

    print(f"\n{'k':>2} {'n':>4} {'mean F1':>9} {'F1 min':>7} {'F1 max':>7} "
          f"{'R of best':>12} {'P of best':>12}")
    print("-" * 60)
    for k, g in df.groupby("k"):
        b = g.loc[g.f1.idxmax()]
        print(f"{k:>2} {len(g):>4} {g.f1.mean():9.3f} {g.f1.min():7.3f} "
              f"{g.f1.max():7.3f} {b.revocacao:12.3f} {b.precisao:12.3f}")
    nove = df[df.k == 9].f1.iloc[0]
    print(f"\nall nine together: F1 {nove:.3f}")
    for k in range(1, 9):
        g = df[df.k == k]
        quantos = int((g.f1 >= nove - 0.005).sum())
        print(f"  k={k}: {quantos:>3} of {len(g):>3} subsets come within "
              f"0.5 point of all nine")
    print(f"\nwrote {SAIDA}")


if __name__ == "__main__":
    main()

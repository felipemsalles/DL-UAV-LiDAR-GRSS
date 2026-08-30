#!/usr/bin/env python3
"""Precision, recall and F1 as a function of the matching tolerance. Not part of the paper.

The detector returns no continuous per-candidate score, so there is no ROC curve and no classical
precision-recall curve. The axis available is the tolerance, i.e. how close a detection has to fall
to the stem to count as a hit; this is an evaluation parameter, not a model one.

The chance curve is plotted alongside: at 2 m an uninformed detector already matches 82% of the
stems, so without that reference the 96.9% is overstated. Null = the same number of points, drawn
uniformly over the same extent.

Run: PYTHONPATH=. python scripts/fig_curva_tolerancia.py
Out: figs_en/entrega_cristiano/talhao001_curva_tolerancia.png
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenvista import config  # noqa: E402

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
DETEC = config.REPO / "data/detections/sat_w2w_arvores.csv"
SAIDA = config.ENTREGA_DIR / "talhao001_curva_tolerancia.png"
TOL = np.arange(0.25, 2.51, 0.125)
SORTEIOS, SEMENTE = 10, 11


def pareia(ref, pred, lim):
    D = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    li, ci = linear_sum_assignment(np.where(D <= lim, D, 1e6))
    return int((D[li, ci] <= lim).sum())


def main():
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    d = pd.read_csv(DETEC)
    d1 = d[d.talhao == 1]
    pred = np.column_stack([d1.base_x.values, d1.base_y.values])
    rng = np.random.default_rng(SEMENTE)
    lo, hi = ref.min(0), ref.max(0)

    P, R, F, A = [], [], [], []
    for lim in TOL:
        vp = pareia(ref, pred, lim)
        r, p = vp / len(ref), vp / len(pred)
        R.append(100 * r); P.append(100 * p); F.append(200 * r * p / (r + p))
        A.append(100 * np.mean([pareia(ref, np.column_stack([
            rng.uniform(lo[0], hi[0], len(pred)),
            rng.uniform(lo[1], hi[1], len(pred))]), lim) / len(ref)
            for _ in range(SORTEIOS)]))

    plt.rcParams.update({"font.size": 10, "legend.fontsize": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(TOL, R, "-o", ms=4, color="#1D6B45", label="Recall, fraction of the mapped trees that were found")
    ax.plot(TOL, P, "-s", ms=4, color="#2D5D8F", label="Precision, fraction of the detections that are mapped trees")
    ax.plot(TOL, F, "-^", ms=4, color="#B0304A", label="F1, the mean of the two")
    ax.plot(TOL, A, "--", lw=1.4, color="#999999", label="What chance alone would already reach at the same tolerance")
    ax.axvline(2.0, color="#CCCCCC", lw=1, zorder=0)
    ax.text(1.96, 26, "tolerance used\nin the report ", fontsize=8.5, color="#777777",
            ha="right", va="bottom")
    # spacing mark: planting is 3.0 x 1.8 m, so just above half the within-row spacing the
    # tolerance reaches the neighbouring tree and the matching becomes ambiguous (knee of the curve).
    ax.axvline(0.9, color="#DDDDDD", lw=1, zorder=0)
    ax.text(0.94, 26, "half the within-row\nspacing (1.8 m)", fontsize=8.5, color="#777777",
            ha="left", va="bottom")
    ax.set_xlabel("Matching tolerance (m)")
    ax.set_ylabel("Percentage")
    ax.set_ylim(0, 100)
    ax.set_xlim(TOL[0], TOL[-1])
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", color="#EEEEEE", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Stand 001, detection quality as a function of the proximity requirement",
                 fontsize=11.5, weight="bold", pad=10)
    ax.legend(frameon=False, loc="lower right")
    config.ENTREGA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA, dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    for t, p, r, f1, a in zip(TOL, P, R, F, A):
        print(f"{t:4.2f} m  P {p:5.1f}  R {r:5.1f}  F1 {f1:5.1f}  chance {a:5.1f}")
    print(SAIDA)


if __name__ == "__main__":
    main()

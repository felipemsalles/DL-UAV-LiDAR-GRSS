#!/usr/bin/env python3
"""Spatial map of the hits and errors in stand 001. Not part of the paper.

It is not a confusion matrix, although it shows the same categories: the table of counts is in
`scripts/fig_matriz_confusao_cristiano.py`. What matters here is where each category falls.

Recall on its own hides the false positives, so in this figure they are the main element.

There are two outlines: the outer one is the stand polygon, the same one the inventory uses; the
inner one is how far the stem map reaches, 79% of the stand. Detections between the two are inside
the stand and outside the reference, so they cannot be counted as errors.

The counters live in the legend and the title carries a single line of result.

Run: PYTHONPATH=. python scripts/fig_matriz_cristiano.py
Out: figs_en/entrega_cristiano/talhao001_matriz_confusao.png
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
from shapely.geometry import MultiPoint, Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenvista import config  # noqa: E402

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
POLI = config.DATA / "2-shapes/Areas_plantio/area_plantio.shp"
JULG = config.OUT_DIR / "deteccoes_julgadas_talhao001.csv"
SAIDA = config.ENTREGA_DIR / "talhao001_mapa_acertos_e_erros.png"
LIMIAR = 2.0
VERDE, VERM, LARANJA, CINZA = "#1D6B45", "#B0304A", "#D4761E", "#B4B4B4"


def main():
    if not JULG.exists():
        sys.exit(f"run scripts/exp_matriz_confusao_talhao001.py first ({JULG} does not exist)")
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    d = pd.read_csv(JULG)
    pred = np.column_stack([d.base_x.values, d.base_y.values])
    talhao = gpd.read_file(POLI).query("Talhao == '001'").geometry.iloc[0]
    casco = MultiPoint([Point(*p) for p in ref]).convex_hull

    D = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    li, ci = linear_sum_assignment(np.where(D <= LIMIAR, D, 1e6))
    ok = D[li, ci] <= LIMIAR
    achou = np.zeros(len(ref), bool); achou[li[ok]] = True
    casou = np.zeros(len(pred), bool); casou[ci[ok]] = True
    sem = ~casou
    fora = np.array([not casco.contains(Point(*p)) for p in pred])
    vp, fn = int(achou.sum()), int((~achou).sum())

    lo = np.array(talhao.bounds[:2]); hi = np.array(talhao.bounds[2:])
    plt.rcParams.update({"font.size": 9, "legend.fontsize": 8.5,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    larg = 9.4
    fig, ax = plt.subplots(figsize=(larg, larg * (hi[1] - lo[1]) / (hi[0] - lo[0]) + 2.0))

    ax.plot(*talhao.exterior.xy, color="#333333", lw=1.4,
            label="Boundary of stand 001 (0.74 ha)")
    ax.plot(*casco.exterior.xy, color="#8899AA", lw=1.1, ls="--",
            label="How far the field map reaches (79% of the stand)")
    ax.scatter(*pred[sem & fora].T, s=24, marker="x", color=CINZA, linewidths=0.9,
               label=f"Detection outside the mapped area, no basis for comparison ({int((sem & fora).sum())})")
    ax.scatter(*pred[sem & ~fora].T, s=30, marker="x", color=LARANJA, linewidths=1.3,
               label=f"Detection inside the mapped area, no tree there ({int((sem & ~fora).sum())})")
    ax.scatter(*ref[achou].T, s=5, color=VERDE, linewidths=0,
               label=f"Mapped tree, found ({vp})")
    ax.scatter(*ref[~achou].T, s=44, marker="o", facecolor="none", edgecolor=VERM,
               linewidths=1.3, label=f"Mapped tree, not found ({fn})")

    ax.set_aspect("equal")
    ax.set_xlim(lo[0] - 6, hi[0] + 6)
    ax.set_ylim(lo[1] - 6, hi[1] + 6)
    ax.set_xticks(np.arange(lo[0], hi[0] + 1, 20))
    ax.set_yticks(np.arange(lo[1], hi[1] + 1, 20))
    ax.set_xticklabels([f"{v - lo[0]:.0f}" for v in ax.get_xticks()])
    ax.set_yticklabels([f"{v - lo[1]:.0f}" for v in ax.get_yticks()])
    ax.set_xlabel("Easting distance (m)")
    ax.set_ylabel("Northing distance (m)")
    ax.set_title(f"Stand 001, {vp} of the {len(ref)} trees in the field map were found",
                 fontsize=12, pad=24, weight="bold")
    ax.text(0.5, 1.012, f"One-to-one comparison, tolerance of {LIMIAR:.0f} m",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#555555")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              handletextpad=0.4, columnspacing=1.6, scatterpoints=1)

    config.ENTREGA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA, dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    print(f"TP {vp}  FN {fn}  FP {int(sem.sum())} "
          f"({int((sem & ~fora).sum())} inside the mapped area, {int((sem & fora).sum())} outside)")
    print(SAIDA)


if __name__ == "__main__":
    main()

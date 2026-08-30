#!/usr/bin/env python3
"""Standalone figure for the supplier of the stem map. Not part of the paper.

It supports a single sentence: "the algorithm found 864 of the 892 trees of stand 001".

It differs from Figure 9 of the paper, which crops the evaluation area so that precision is measured
with the same ruler on both sides and which leaves 734 stems in that crop. Here the question is how
many of the 892 mapped trees were found, with no crop.

The unmatched detections are shown, together with the boundary of the survey: showing only the
coloured stems would hide the 212 detections that are not in the map.

The outline is the convex hull of the stems, and it overstates how much of the 212 lies outside the
survey, because a boundary defined by the reference itself hides detections by construction. Measured
against the TLS cloud, 117 of the 212 fall in poorly scanned strips and 76 are detector error. For
the confusion matrix use `scripts/fig_matriz_cristiano.py`, which crops by the actual laser coverage.

Title and labels are in the report language, because the figure circulates without a caption beside it.

Run: PYTHONPATH=. python scripts/fig_para_cristiano.py
Out: manual_match/talhao001_vs_mapa_fustes.png
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
DETEC = config.REPO / "data/detections/sat_w2w_arvores.csv"
SAIDA = config.OUT_DIR / "talhao001_vs_mapa_fustes.png"
LIMIAR = 2.0
VERDE, VERM, CINZA = "#1D6B45", "#B0304A", "#9A9A9A"


def main():
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    d = pd.read_csv(DETEC)
    pred = np.column_stack([d[d.talhao == 1].base_x.values, d[d.talhao == 1].base_y.values])

    D = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    li, ci = linear_sum_assignment(np.where(D <= LIMIAR, D, 1e6))
    ok = D[li, ci] <= LIMIAR
    achou = np.zeros(len(ref), bool); achou[li[ok]] = True
    casou = np.zeros(len(pred), bool); casou[ci[ok]] = True
    tp = int(achou.sum())

    casco = MultiPoint([Point(*p) for p in ref]).convex_hull
    lo, hi = ref.min(0), ref.max(0)

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "legend.fontsize": 8.5, "axes.grid": False,
                         "axes.spines.top": False, "axes.spines.right": False})
    larg = 9.0
    fig, ax = plt.subplots(figsize=(larg, larg * (hi[1] - lo[1]) / (hi[0] - lo[0]) + 1.5))

    ax.plot(*casco.exterior.xy, color="#777777", lw=1.0, ls="--",
            label="Boundary of the terrestrial survey")
    ax.scatter(*pred[~casou].T, s=13, marker="x", color=CINZA, linewidths=0.6,
               label=f"Detection with no tree in the map ({int((~casou).sum())})")
    ax.scatter(*ref[achou].T, s=5, color=VERDE, linewidths=0,
               label=f"Mapped tree, found ({tp})")
    ax.scatter(*ref[~achou].T, s=42, marker="o", facecolor="none", edgecolor=VERM,
               linewidths=1.3, label=f"Mapped tree, not found ({int((~achou).sum())})")

    ax.set_aspect("equal")
    ax.set_xlim(lo[0] - 8, hi[0] + 8)
    ax.set_xticks(np.arange(lo[0], hi[0] + 1, 20))
    ax.set_yticks(np.arange(lo[1], hi[1] + 1, 20))
    ax.set_xticklabels([f"{v - lo[0]:.0f}" for v in ax.get_xticks()])
    ax.set_yticklabels([f"{v - lo[1]:.0f}" for v in ax.get_yticks()])
    ax.set_xlabel("Easting distance (m)")
    ax.set_ylabel("Northing distance (m)")
    ax.set_title(f"Stand 001, {tp} of the {len(ref)} trees found "
                 f"({100 * tp / len(ref):.0f}%)", fontsize=11.5, pad=26, weight="bold")
    # the second subtitle line explains a visual artefact: a large red circle next to a green
    # point appears to contain it, suggesting that the tree was marked as missed while having been
    # found. They are two distinct, nearby stems of the map. In 27 of the 28 cases the detection
    # closest to the missed stem already belonged to the neighbouring stem and there was no free
    # detection nearby: the algorithm found one tree where the map has two (merged neighbours, not
    # a detection failure).
    disputa = sum(1 for i in np.flatnonzero(~achou)
                  if D[i].min() <= LIMIAR and casou[int(D[i].argmin())])
    ax.text(0.5, 1.085,
            "Automatic detection in the drone cloud, compared one to one with the terrestrial "
            f"laser survey, tolerance of {LIMIAR:.0f} m",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color="#444444")
    ax.text(0.5, 1.030,
            f"In {disputa} of the {int((~achou).sum())} not-found cases the algorithm detected "
            "a single tree where the map has two very close together",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color="#444444")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              handletextpad=0.4, columnspacing=2.0, scatterpoints=1)

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA, dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)

    fora = sum(not casco.contains(Point(*p)) for p in pred[~casou])
    print(f"{tp} of {len(ref)} trees found ({100 * tp / len(ref):.1f}%)")
    print(f"{int((~casou).sum())} unmatched detections, {fora} of them outside the survey "
          f"({100 * fora / max(int((~casou).sum()), 1):.0f}%)")
    print(SAIDA)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Confusion matrix as a table of counts. Not part of the paper.

The matrix is incomplete by nature and the figure states that explicitly. The cell "no tree exists
and the algorithm did not detect one" would be the true negative, and it is not countable in
detection: there is no enumerable set of tree-free places. The crossed-out cell prevents anyone from
computing an accuracy by dividing by a total that does not exist.

The false-positive cell is split in two, because a good share of the unmatched detections falls
where the field survey never reached, and there is no reference there to contradict them; the raw
number on its own invites a wrong reading.

The word accuracy does not appear in the figure, because it does not apply to detection.

Run: PYTHONPATH=. python scripts/fig_matriz_confusao_cristiano.py
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
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment
from shapely.geometry import MultiPoint, Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenvista import config  # noqa: E402

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
JULG = config.OUT_DIR / "deteccoes_julgadas_talhao001.csv"
SAIDA = config.ENTREGA_DIR / "talhao001_matriz_confusao.png"
LIMIAR = 2.0
VERDE, VERM, LARANJA, CINZA = "#1D6B45", "#B0304A", "#D4761E", "#AAAAAA"


def celula(ax, x, y, w, h, cor, titulo, valor, nota, riscada=False):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=cor, alpha=0.10 if not riscada else 0.05,
                           edgecolor=cor, lw=1.6, zorder=1))
    ax.text(x + w / 2, y + h - 0.16, titulo, ha="center", va="top", fontsize=10.5,
            color=cor, weight="bold", zorder=3)
    ax.text(x + w / 2, y + h / 2 - 0.06, valor, ha="center", va="center",
            fontsize=30 if not riscada else 13, color=cor,
            weight="bold" if not riscada else "normal", zorder=3)
    if nota:
        ax.text(x + w / 2, y + 0.15, nota, ha="center", va="bottom", fontsize=8.6,
                color="#555555", zorder=3)


def main():
    if not JULG.exists():
        sys.exit(f"run scripts/exp_matriz_confusao_talhao001.py first ({JULG} does not exist)")
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    d = pd.read_csv(JULG)
    pred = np.column_stack([d.base_x.values, d.base_y.values])
    casco = MultiPoint([Point(*p) for p in ref]).convex_hull

    D = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    li, ci = linear_sum_assignment(np.where(D <= LIMIAR, D, 1e6))
    ok = D[li, ci] <= LIMIAR
    casou = np.zeros(len(pred), bool); casou[ci[ok]] = True
    vp, fn, fp = int(ok.sum()), len(ref) - int(ok.sum()), int((~casou).sum())
    fora = int(sum(not casco.contains(Point(*p)) for p in pred[~casou]))

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
    x1, x2, w, h = 2.15, 6.1, 3.7, 1.95
    y1, y2 = 2.75, 0.65

    ax.text(x1 + w / 2, y1 + h + 0.42, "The algorithm detected", ha="center", fontsize=11,
            weight="bold", color="#333333")
    ax.text(x2 + w / 2, y1 + h + 0.42, "The algorithm did not detect", ha="center", fontsize=11,
            weight="bold", color="#333333")
    ax.text(x1 - 0.35, y1 + h / 2, "A tree exists\nin the field map", ha="right", va="center",
            fontsize=11, weight="bold", color="#333333")
    ax.text(x1 - 0.35, y2 + h / 2, "No tree exists\nin the field map", ha="right", va="center",
            fontsize=11, weight="bold", color="#333333")

    celula(ax, x1, y1, w, h, VERDE, "Found correctly", f"{vp}", "of 892 trees in the map")
    celula(ax, x2, y1, w, h, VERM, "Missed", f"{fn}", "trees that exist and were not found")
    celula(ax, x1, y2, w, h, LARANJA, "Flagged with nothing in the map", f"{fp}",
           f"{fora} outside the area the field survey covered\n{fp - fora} inside it")
    celula(ax, x2, y2, w, h, CINZA, "Does not exist",
           "empty places\ncannot be counted", "", riscada=True)

    ax.text(5.0, 5.85, f"Stand 001, comparison with the field map, tolerance of {LIMIAR:.0f} m",
            ha="center", fontsize=12.5, weight="bold")
    ax.text(5.0, 0.18, "Without the fourth cell there is no total, so accuracy cannot be "
            "computed. What describes the result is what the three cells hold",
            ha="center", fontsize=9, color="#555555")

    config.ENTREGA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA, dpi=200, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    print(f"TP {vp}  FN {fn}  FP {fp} ({fora} outside the mapped area)")
    print(SAIDA)


if __name__ == "__main__":
    main()

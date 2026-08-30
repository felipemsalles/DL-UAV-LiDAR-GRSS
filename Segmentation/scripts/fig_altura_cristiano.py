#!/usr/bin/env python3
"""Height of the correct and incorrect detections, to assess whether height works as a cut-off. Not part of the paper.

Two panels: the overlaid histogram answers the diagnosis (are the errors low vegetation?) and the
table on the right answers the decision (how much is gained by cutting?), with six operating points
and the corresponding values.

The result is overlap, not separation: only a minority of the errors are low, and the rest are the
same height as the correct detections, because they are the same tree counted twice.

The two distributions are normalised by their own totals: there are 803 hits against 69 errors, and
in raw counts the orange bar would be imperceptible.

Run: PYTHONPATH=. python scripts/fig_altura_cristiano.py
Out: figs_en/entrega_cristiano/talhao001_altura_das_deteccoes.png
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
JULG = config.OUT_DIR / "deteccoes_julgadas_talhao001.csv"
SAIDA = config.ENTREGA_DIR / "talhao001_altura_das_deteccoes.png"
VERDE, LARANJA = "#1D6B45", "#D4761E"
CORTE = 5.0


def main():
    if not JULG.exists():
        sys.exit(f"run scripts/exp_matriz_confusao_talhao001.py first ({JULG} does not exist)")
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    d = pd.read_csv(JULG)
    pred = np.column_stack([d.base_x.values, d.base_y.values])
    D = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    li, ci = linear_sum_assignment(np.where(D <= 2.0, D, 1e6))
    ok = D[li, ci] <= 2.0
    casou = np.zeros(len(pred), bool); casou[ci[ok]] = True
    casco = MultiPoint([Point(*p) for p in ref]).convex_hull
    dentro = np.array([casco.contains(Point(*p)) for p in pred])

    h = d.z_max.values
    certas = h[casou & dentro]
    erradas = h[~casou & dentro]
    baixas = 100 * (erradas < CORTE).mean()

    # table of the right-hand panel, recomputed here so that it cannot diverge from the histogram
    linhas = []
    for corte in (0, 5, 10, 15, 20, 25):
        k = dentro & (h >= corte)
        P = pred[k]
        Dk = np.hypot(ref[:, None, 0] - P[None, :, 0], ref[:, None, 1] - P[None, :, 1])
        l2, c2 = linear_sum_assignment(np.where(Dk <= 2.0, Dk, 1e6))
        vp = int((Dk[l2, c2] <= 2.0).sum())
        r, pr = vp / len(ref), vp / len(P)
        linhas.append((f"{corte} m" if corte else "no cut-off", len(P),
                       f"{100 * r:.0f}%", f"{100 * pr:.0f}%", f"{200 * r * pr / (r + pr):.0f}%"))

    plt.rcParams.update({"font.size": 10, "legend.fontsize": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, at) = plt.subplots(1, 2, figsize=(13.2, 5.2),
                                 gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.18})
    bins = np.arange(0, 40.1, 2)
    ax.hist(certas, bins=bins, weights=np.full(len(certas), 100 / len(certas)),
            color=VERDE, alpha=0.75, label=f"Found a mapped tree ({len(certas)})")
    ax.hist(erradas, bins=bins, weights=np.full(len(erradas), 100 / len(erradas)),
            color=LARANJA, alpha=0.75, label=f"No tree in the map ({len(erradas)})")
    ax.axvline(CORTE, color="#666666", lw=1.2, ls="--")
    ax.text(CORTE + 0.6, ax.get_ylim()[1] * 0.93, f"cut-off at {CORTE:.0f} m",
            fontsize=9, color="#555555")
    ax.set_xlabel("Height of the detected tree (m)")
    ax.set_ylabel("Percentage of each group")
    ax.set_title("Where the incorrect detections are", fontsize=11.5, weight="bold", pad=14)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    at.axis("off")
    at.set_xlim(0, 1); at.set_ylim(0, 1)
    at.set_title("What each cut-off costs and returns", fontsize=11.5, weight="bold", pad=14)
    # numeric columns right-aligned; left-aligned, the "Detections" header would touch "Found".
    cols = ["Cut-off", "Detections", "Found", "Correct", "F1"]
    xs = [0.02, 0.44, 0.63, 0.82, 0.98]
    ali = ["left", "right", "right", "right", "right"]
    y = 0.88
    for x, c, a in zip(xs, cols, ali):
        at.text(x, y, c, fontsize=10, weight="bold", color="#333333", ha=a)
    at.plot([0, 1], [y - 0.035, y - 0.035], color="#BBBBBB", lw=1.0)
    for i, ln in enumerate(linhas):
        yy = y - 0.10 - i * 0.115
        destaque = ln[0] == f"{CORTE:.0f} m"
        for x, v, a in zip(xs, (ln[0], f"{ln[1]}", ln[2], ln[3], ln[4]), ali):
            at.text(x, yy, v, fontsize=10.5, ha=a,
                    weight="bold" if destaque else "normal",
                    color=LARANJA if destaque else "#333333")
    at.text(0.02, 0.03, "Found = of the 892 mapped trees.  Correct = of the detections that remain.",
            fontsize=8.8, color="#777777")

    fig.suptitle("The incorrect detections are almost the same height as the correct ones",
                 fontsize=13.5, weight="bold", y=1.06)
    fig.text(0.5, 1.00, f"Only {baixas:.0f}% of the errors fall below {CORTE:.0f} m. The rest are "
             "about 30 m tall, because they are the same tree counted twice",
             ha="center", fontsize=10, color="#444444")

    config.ENTREGA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA, dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    print(f"correct {len(certas)} median {np.median(certas):.1f} m | "
          f"incorrect {len(erradas)} median {np.median(erradas):.1f} m, {baixas:.0f}% below {CORTE:.0f} m")
    print(SAIDA)


if __name__ == "__main__":
    main()

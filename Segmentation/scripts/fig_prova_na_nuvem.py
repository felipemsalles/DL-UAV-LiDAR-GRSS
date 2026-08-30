#!/usr/bin/env python3
"""Drone detections over the terrestrial laser cloud. Not part of the paper.

Summary metrics do not settle the objection that, in a plantation with 1.88 m spacing and with
1076 detections for 892 stems, matching at 2 m succeeds by chance: the chance floor at that
tolerance is 82%. The figure shows the marks falling on the stems recorded by the terrestrial
laser itself.

The substrate is the breast-height slice, between 0.8 m and 2.5 m above the ground. Outside that
band understorey comes in below and branches above, and any mark starts to look as if it falls on
something.

The stem is highlighted by a density threshold calibrated on the stem map, and not on the
detections: a 10 cm cell with 400 points or more produces 176 clusters in the window, against 155
mapped stems. Without that cut the slice mixes stems with low branches.

The vertical section accompanies the top-down view, which on its own leaves the doubt of whether
two projections are being compared.

Each detection is linked to its stem by a line: the typical distance between the position estimated
from the crown and the real stem is 0.8 m, so at the zoom level the circle lands beside the cross,
and without the matching link the figure suggests error where there is a hit.

The circles with no line are named in the legend. There are four in the window, all with the nearest
stem between 1.0 m and 1.7 m already taken by another detection, and with heights from 19 to 33 m:
they are extra instances on the same crown, not trees invented in a gap.

The two views use windows of different size. A stem is about 15 cm across, so in a 36 m window it
takes one or two pixels and the top-down view needs a zoom; the side section keeps the window wide,
because height stretches the whole tree and that is where the repetition shows.

The strip of numbers at the top is for the whole stand, to make clear that the 16 by 11 m crop
illustrates a result measured over the 892 trees, and is not the result itself. The window was
chosen for terrestrial-laser coverage, and not for accuracy.

Run: PYTHONPATH=. python scripts/fig_prova_na_nuvem.py
Input: config.TLS_LAS, the terrestrial cloud that lives outside git
Out: figs_en/entrega_cristiano/talhao001_prova_na_nuvem.png
"""
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.optimize import linear_sum_assignment
from shapely.geometry import MultiPoint, Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenvista import config  # noqa: E402

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
DETEC = config.REPO / "data/detections/sat_w2w_arvores.csv"
TLS = config.TLS_LAS
SAIDA = config.ENTREGA_DIR / "talhao001_prova_na_nuvem.png"

CX, CY = 749048.9, 7480012.5      # centre chosen for laser coverage
MEIA_X, MEIA_Y = 18.0, 13.0       # window read from the cloud, 36 x 26 m
ZOOM_X, ZOOM_Y = 8.0, 5.5         # crop of the top-down view, where the stem is visible
FAIXA = 3.0                       # thickness of the vertical-section slice, in metres
H_PEITO = (0.8, 2.5)
GS = 0.10
TRONCO = 400          # points in a 10 cm cell, see the note above
VERDE, LARANJA = "#1D6B45", "#D4761E"


def le_janela():
    """Points of the terrestrial cloud inside the window, and the local terrain model."""
    xs, ys, zs = [], [], []
    with laspy.open(TLS) as fh:
        for p in fh.chunk_iterator(4_000_000):
            x, y, z = np.asarray(p.x), np.asarray(p.y), np.asarray(p.z, dtype=np.float32)
            m = (abs(x - CX) < MEIA_X) & (abs(y - CY) < MEIA_Y)
            if m.any():
                xs.append(x[m]); ys.append(y[m]); zs.append(z[m])
    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    g = 1.0
    nx, ny = int(2 * MEIA_X / g) + 2, int(2 * MEIA_Y / g) + 2
    x0, y0 = CX - MEIA_X, CY - MEIA_Y
    zmin = np.full((nx, ny), 1e9, np.float32)
    np.minimum.at(zmin, (((x - x0) / g).astype(int), ((y - y0) / g).astype(int)), z)
    zmin[zmin > 1e8] = np.nan
    vazio = ~np.isfinite(zmin)
    if vazio.any():
        _, (a, b) = distance_transform_edt(vazio, return_indices=True)
        zmin[vazio] = zmin[a[vazio], b[vazio]]
    zsolo = uniform_filter(zmin, 5)
    h = z - zsolo[np.clip(((x - x0) / g).astype(int), 0, nx - 1),
                  np.clip(((y - y0) / g).astype(int), 0, ny - 1)]
    return x, y, h


def main():
    if not TLS.exists():
        sys.exit(f"TLS cloud not found at {TLS} (set GREENVISTA_TLS_LAS)")
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    d = pd.read_csv(DETEC)
    d1 = d[d.talhao == 1]
    pred = np.column_stack([d1.base_x.values, d1.base_y.values])
    jr = (abs(ref[:, 0] - CX) < MEIA_X) & (abs(ref[:, 1] - CY) < MEIA_Y)
    jp = (abs(pred[:, 0] - CX) < MEIA_X) & (abs(pred[:, 1] - CY) < MEIA_Y)

    print(f"reading {TLS.name} in the {2 * MEIA_X:.0f} x {2 * MEIA_Y:.0f} m window")
    x, y, h = le_janela()
    print(f"{len(x):,} points, {int(jr.sum())} mapped stems, {int(jp.sum())} detections")

    peito = (h > H_PEITO[0]) & (h < H_PEITO[1])
    nx, ny = int(2 * MEIA_X / GS), int(2 * MEIA_Y / GS)
    x0, y0 = CX - MEIA_X, CY - MEIA_Y
    grade = np.zeros((nx, ny), np.int32)
    np.add.at(grade, (np.clip(((x[peito] - x0) / GS).astype(int), 0, nx - 1),
                      np.clip(((y[peito] - y0) / GS).astype(int), 0, ny - 1)), 1)

    tronco = grade >= TRONCO
    print(f"stem cells {int(tronco.sum())}, occupied {int((grade > 0).sum())}")

    # numbers for the whole stand, for the top strip
    Dt = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    lt, ct = linear_sum_assignment(np.where(Dt <= 2.0, Dt, 1e6))
    okt = Dt[lt, ct] <= 2.0
    vp_t = int(okt.sum())
    dist_t = Dt[lt, ct][okt]
    casou_t = np.zeros(len(pred), bool); casou_t[ct[okt]] = True
    casco = MultiPoint([Point(*p) for p in ref]).convex_hull
    fora_t = np.array([not casco.contains(Point(*p)) for p in pred])
    fp_fora = int((~casou_t & fora_t).sum())
    fp_dentro = int((~casou_t & ~fora_t).sum())

    plt.rcParams.update({"font.size": 9.5, "legend.fontsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(9.6, 12.6),
                                     gridspec_kw={"height_ratios": [6.6, 2 * ZOOM_Y * 1.55, 11],
                                                  "hspace": 0.66})

    a0.axis("off"); a0.set_xlim(-0.06, 4.06); a0.set_ylim(-0.10, 1.90)
    a0.text(2.0, 1.88, "Over the whole of stand 001, compared with the "
            f"{len(ref)} trees of the field map",
            ha="center", va="top", fontsize=12.5, weight="bold")
    # label in three short lines, with a rule between blocks: in two lines the text gets wide,
    # the four blocks touch and the strip reads as a single paragraph.
    caixas = [(f"{vp_t} of {len(ref)}", "trees of the field\nmap that were\nfound",
               f"{100 * vp_t / len(ref):.0f}%", VERDE),
              (f"{np.median(dist_t):.2f} m".replace(".", ","),
               "typical distance\nbetween the detection\nand the stem",
               f"RMSE {np.sqrt((dist_t ** 2).mean()):.2f} m".replace(".", ","), "#2D5D8F"),
              (f"{fp_dentro}", "extra detections,\ninside the area\nsurveyed in the field",
               f"{100 * fp_dentro / len(pred):.0f}% of the total", LARANJA),
              (f"{fp_fora}", "detections outside\nthat area, with no\nbasis for comparison",
               f"{100 * fp_fora / len(pred):.0f}% of the total", "#9AA0A6")]
    for k, (num, rot, nota, cor) in enumerate(caixas):
        cx = 0.5 + k
        if k:
            a0.plot([k, k], [-0.08, 1.32], color="#E4E4E4", lw=1.0, zorder=0)
        a0.text(cx, 1.16, num, ha="center", va="center", fontsize=17, color=cor, weight="bold")
        a0.text(cx, 0.52, rot, ha="center", va="center", fontsize=8.8, color="#333333",
                linespacing=1.6)
        a0.text(cx, -0.05, nota, ha="center", va="center", fontsize=8.5, color="#888888")

    ext = [0, 2 * MEIA_X, 0, 2 * MEIA_Y]
    a1.imshow(np.where(grade > 0, 1, np.nan).T, origin="lower", cmap="Greys", vmin=0, vmax=7,
              extent=ext, interpolation="nearest")
    a1.imshow(np.where(tronco, 1, np.nan).T, origin="lower", cmap="Greys", vmin=0, vmax=1.05,
              extent=ext, interpolation="nearest")
    zx, zy = MEIA_X - ZOOM_X, MEIA_Y - ZOOM_Y
    zr = jr & (abs(ref[:, 0] - CX) < ZOOM_X) & (abs(ref[:, 1] - CY) < ZOOM_Y)
    zp = jp & (abs(pred[:, 0] - CX) < ZOOM_X) & (abs(pred[:, 1] - CY) < ZOOM_Y)
    # the matching is the global one, and not one redone inside the window, otherwise the figure
    # would show a correspondence that is not the one that produced the numbers in the report
    Dg = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    li, ci = linear_sum_assignment(np.where(Dg <= 2.0, Dg, 1e6))
    de_quem = {int(j): int(i) for i, j in zip(li, ci) if Dg[i, j] <= 2.0}
    d_par = []
    for j in np.flatnonzero(zp):
        i = de_quem.get(int(j))
        if i is None:
            continue
        a1.plot([ref[i, 0] - x0, pred[j, 0] - x0], [ref[i, 1] - y0, pred[j, 1] - y0],
                color="#8A6A3A", lw=1.1, zorder=2)
        d_par.append(Dg[i, j])
    orfa = np.array([int(j) not in de_quem for j in np.flatnonzero(zp)])
    idp = np.flatnonzero(zp)
    # one detection in the crop matched a stem just outside it, so there is one line more than
    # there are visible crosses (24 to 23); the label records that.
    fora_rec = len(d_par) - int(zr.sum())
    a1.plot([], [], color="#8A6A3A", lw=1.1,
            label=f"Link to the mapped tree ({len(d_par)})" +
                  (f", {fora_rec} just outside the crop" if fora_rec > 0 else ""))
    a1.scatter(pred[idp[~orfa], 0] - x0, pred[idp[~orfa], 1] - y0, s=200, marker="o",
               facecolor="none", edgecolor=LARANJA, linewidths=1.8, zorder=3,
               label=f"Tree detected by the drone ({int((~orfa).sum())})")
    a1.scatter(pred[idp[orfa], 0] - x0, pred[idp[orfa], 1] - y0, s=200, marker="o",
               facecolor="none", edgecolor="#9AA0A6", linewidths=1.6, linestyle=(0, (2, 1.6)),
               zorder=3, label=f"Extra detection on the same crown ({int(orfa.sum())})")
    a1.scatter(ref[zr, 0] - x0, ref[zr, 1] - y0, s=70, marker="+", color=VERDE,
               linewidths=1.8, zorder=4, label=f"Tree of the field map ({int(zr.sum())})")
    print(f"zoom window: {int(zr.sum())} stems, {int(zp.sum())} detections, "
          f"{len(d_par)} links, {int(orfa.sum())} unmatched, "
          f"median link {np.median(d_par):.2f} m")
    a1.set_aspect("equal")
    a1.set_xlim(zx, zx + 2 * ZOOM_X); a1.set_ylim(zy, zy + 2 * ZOOM_Y)
    a1.set_xticks(np.arange(zx, zx + 2 * ZOOM_X + 0.1, 4))
    a1.set_yticks(np.arange(zy, zy + 2 * ZOOM_Y + 0.1, 4))
    a1.set_xticklabels([f"{v - zx:.0f}" for v in a1.get_xticks()])
    a1.set_yticklabels([f"{v - zy:.0f}" for v in a1.get_yticks()])
    a1.set_xlabel("Easting distance (m)"); a1.set_ylabel("Northing distance (m)")
    a1.set_title(f"What this looks like close up, in a {2 * ZOOM_X:.0f} by "
                 f"{2 * ZOOM_Y:.0f} m crop. "
                 "The black is the stems in the terrestrial laser cloud\n"
                 "The line links each tree to what the drone detected",
                 fontsize=10.5, weight="bold", pad=10)
    a1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              scatterpoints=1, columnspacing=2.0, handletextpad=0.5)

    faixa = abs(y - CY) < FAIXA / 2
    sub = faixa & (h > 0.2) & (h < 12)
    a2.scatter(x[sub] - x0, h[sub], s=0.05, color="#5A6570", linewidths=0, alpha=0.35)
    jf = jp & (abs(pred[:, 1] - CY) < FAIXA / 2)
    for xp in pred[jf, 0] - x0:
        a2.plot([xp, xp], [0, 11.4], color=LARANJA, lw=1.4, alpha=0.85, zorder=3)
    a2.plot([], [], color=LARANJA, lw=1.4, label="Position of the tree detected by the drone")
    a2.set_xlim(0, 2 * MEIA_X); a2.set_ylim(0, 12)
    a2.set_xlabel("Easting distance (m)"); a2.set_ylabel("Height (m)")
    a2.set_title(f"The same scene seen from the side, in a {FAIXA:.0f} m band. "
                 "Each column of points is a stem", fontsize=10.5, weight="bold", pad=16)
    # legend outside the axes: inside, it covers the crowns of the trees on the right, and a
    # point-cloud panel has no empty corner.
    a2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16))

    # the top strip is repositioned by hand, and not by hspace: the gridspec hspace is a single
    # value for both gaps, which need to differ — below there is room for the middle panel legend
    # plus the title of the bottom one, above there is room for nothing.
    fig.canvas.draw()
    p0, p1 = a0.get_position(), a1.get_position()
    a0.set_position([p0.x0, p1.y1 + 0.064, p0.width, p0.height])

    config.ENTREGA_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA, dpi=190, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    print(SAIDA)


if __name__ == "__main__":
    main()

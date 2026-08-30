#!/usr/bin/env python3
"""Figure for the group: why the volume chain does not close today.

Left, the tolerance curve, which answers how much the chain needs: with a perfect DBH the volume
error sits at 5.7%, and from ~1 cm of DBH error on it leaves the range accepted in inventory. It is
a curve, and not a number, because the answer depends on how much the DBH is off.

Right, the noise-to-signal ratio: the model error is the same size as the variation it ought to
explain.

Labels in the report language; the English version, for the paper, comes out of fig_relatorio_en.py.

Palette blue #2a78d6 and orange #eb6834, chosen for separation under colour blindness and contrast
on a light background. The marker of "today's model" is dark grey, so as not to spend a third hue on
an element that already carries a written label.

Run: PYTHONPATH=. python scripts/fig_cadeia_volume.py
Out: manual_match/cadeia_volume_grupo.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_cadeia_volume import loto, metricas, prediz, tabela  # noqa: E402

from greenvista import config  # noqa: E402

AZUL, LARANJA, CINZA = "#2a78d6", "#eb6834", "#4A4A4A"
VERDE_FAIXA = "#0f7a54"
SIGMAS = np.arange(0, 3.01, 0.25)
N_DRAWS = 400
RMSE_MODELO = 2.05          # RMSE of the best DBH model of Table IV, scenario S1


def curva(df, n_col, rng):
    y = df.Vol_ha.to_numpy(float)
    med, lo, hi = [], [], []
    for s in SIGMAS:
        r = []
        for _ in range(N_DRAWS if s > 0 else 1):
            d = df.Dbar.to_numpy() + (rng.normal(0, s, len(df)) if s > 0 else 0)
            r.append(metricas(y, loto(df, prediz(df, n_col, d)))["rRMSE"])
        r = np.array(r)
        med.append(r.mean()); lo.append(np.percentile(r, 10)); hi.append(np.percentile(r, 90))
    return np.array(med), np.array(lo), np.array(hi)


def main():
    df = tabela()
    rng = np.random.default_rng(20260827)
    campo, c_lo, c_hi = curva(df, "n_field", rng)
    sat, s_lo, s_hi = curva(df, "n_sat", rng)
    sd_alvo = df.Dbar.std(ddof=1)

    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.6, 4.9), gridspec_kw={"width_ratios": [1.62, 1]})

    # ---------------- left panel, the tolerance curve
    # no label inside the data area: two curves with uncertainty bands fill the whole plane. The y
    # axis runs up to 72 to open an empty strip at the top, the two points of interest become
    # vertical lines labelled in that strip, the values become a text block in the upper-left
    # corner (which the bands never reach) and the legend moves below the axes.
    ax.axhspan(0, 15, color=VERDE_FAIXA, alpha=0.10, zorder=0)
    for x, rot, cor in ((1.0, "target", VERDE_FAIXA), (RMSE_MODELO, "today", CINZA)):
        # the vertical starts at the top of the green band: running down to y=0 it would cross the
        # band caption, the only text still inside the plot. From 15 to 62 it marks x without
        # colliding.
        ax.axvline(x, color=cor, ls="--", lw=1.3, ymin=15 / 72, ymax=62 / 72, zorder=2)
        ax.text(x, 63.5, rot, ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=cor)
    for m, lo, hi, cor, rot in ((campo, c_lo, c_hi, AZUL, "with the field count (perfect)"),
                                (sat, s_lo, s_hi, LARANJA, "with the SegmentAnyTree count")):
        ax.fill_between(SIGMAS, lo, hi, color=cor, alpha=0.15, linewidth=0)
        ax.plot(SIGMAS, m, color=cor, lw=2.4, label=rot, zorder=3)
        for x in (1.0, RMSE_MODELO):
            ax.plot(x, np.interp(x, SIGMAS, m), "o", color=cor, ms=6.5,
                    mec="white", mew=1.4, zorder=4)
    ax.plot(0, campo[0], "o", color=AZUL, ms=6.5, mec="white", mew=1.4, zorder=4)

    # the block of numbers, in the only corner the bands never reach
    # value right-aligned and label left-aligned, at adjacent x: the gap stays constant on every
    # line, whatever the length of the value, and the pair reads as a single line.
    linhas = [(f"{campo[0]:.1f}%".replace(".", ","), "with a perfect DBH", AZUL),
              (f"{np.interp(1.0, SIGMAS, campo):.0f}%", "with 1 cm of error", AZUL),
              (f"{np.interp(RMSE_MODELO, SIGMAS, campo):.0f} to "
               f"{np.interp(RMSE_MODELO, SIGMAS, sat):.0f}%", "with today's DBH", CINZA)]
    # white background behind the whole block, and not a box per text: the "target" vertical passes
    # at x=1.0, in the middle of the labels, and a box per string would leave it visible in the gap
    # between columns.
    ax.add_patch(Rectangle((0.26, 42.4), 1.42, 18.6, facecolor="white", edgecolor="none",
                           alpha=0.92, zorder=4))
    for k, (val, rot, cor) in enumerate(linhas):
        y = 57.5 - k * 5.8
        ax.text(0.80, y, val, fontsize=11.5, color=cor, fontweight="bold", va="center",
                ha="right", zorder=5)
        ax.text(0.88, y, rot, fontsize=9.8, color="#333333", va="center", ha="left", zorder=5)

    # the band caption goes on the right: in the left corner, inside the band, run the blue curve
    # and the target dashed line; above 2.1 cm the band is empty.
    ax.text(2.94, 3.4, "range accepted in forest inventory, up to 15%",
            fontsize=9.4, color=VERDE_FAIXA, ha="right")
    ax.set_xlabel("error of the DBH model  (cm)")
    ax.set_ylabel("error of the estimated volume  (rRMSE, %)")
    ax.set_xlim(-0.08, 3.0); ax.set_ylim(0, 72)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.155),
              ncol=1, fontsize=9.6, handletextpad=0.6, labelspacing=0.35)
    ax.set_title("How much the chain needs", loc="left", fontsize=12, fontweight="bold", pad=12)

    # ---------------- right panel, noise against signal
    barras = [("what there is to explain", "variation of DBH across the plots", sd_alvo, AZUL),
              ("the error of today's model", "RMSE of PointNet, scenario S1", RMSE_MODELO, LARANJA)]
    for k, (titulo, sub, val, cor) in enumerate(barras):
        y = k * 1.35
        bx.barh(y, val, height=0.52, color=cor, zorder=3)
        bx.text(val + 0.07, y, f"{val:.2f} cm".replace(".", ","), va="center",
                fontsize=12.5, fontweight="bold", color=cor)
        # the y axis is inverted, so a smaller y draws higher: the title, which sits on top,
        # calls for the larger offset, not the smaller one.
        bx.text(0, y - 0.80, titulo, va="top", fontsize=10, fontweight="bold", color="#222222")
        bx.text(0, y - 0.52, sub, va="top", fontsize=9.2, color=CINZA)
    bx.set_ylim(2.30, -0.95); bx.set_xlim(0, 2.85)
    bx.set_yticks([]); bx.set_xlabel("centimetres")
    bx.spines["left"].set_visible(False)
    bx.set_title("Why it does not close", loc="left", fontsize=12, fontweight="bold", pad=12)

    # fig.text neither wraps on its own nor respects the edge: short lines, measured to fit the
    # width of the right panel, with the bottom margin reserved in subplots_adjust.
    fig.text(0.612, 0.045,
             "The error is the same size as the signal.\n"
             "Predicting the mean of all plots is wrong by\n"
             "less than the three models evaluated.",
             fontsize=9.6, color=CINZA, linespacing=1.5)
    fig.subplots_adjust(left=0.072, right=0.985, top=0.87, bottom=0.30, wspace=0.14)
    out = config.OUT_DIR / "cadeia_volume_grupo.png"
    fig.savefig(out, dpi=200)
    print(f"written to {out}")
    print(f"  perfect DBH {campo[0]:.1f}%  |  at today's error {np.interp(RMSE_MODELO, SIGMAS, campo):.1f}%"
          f"  |  with the SAT count {np.interp(RMSE_MODELO, SIGMAS, sat):.1f}%")


if __name__ == "__main__":
    main()

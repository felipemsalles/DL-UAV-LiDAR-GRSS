#!/usr/bin/env python3
"""Model ladder for mean DBH and the corresponding effect on volume.

Left, the model ladder for mean DBH: seven standard ABA methods fall below the three networks of
the paper, ordered monotonically in complexity. Right, the effect on volume: three bars (before,
after, ceiling) and one reference line.

Labels stay outside the data area: name on the y axis, value at the tip of the bar.

The 8-metric OLS (RMSE 16.95) is left out of the figure and appears only in the document. Included,
the x axis would run to 17 and the bars of interest would take 7% of the width, making the
comparison unreadable.

The blue/orange palette was chosen for separation under colour blindness and contrast on a light
background.

Run: PYTHONPATH=. python scripts/fig_dap_e_volume.py
Out: manual_match/dap_e_volume_grupo.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from greenvista import config

AZUL, LARANJA, VERDE, CINZA = "#2a78d6", "#eb6834", "#0f7a54", "#4A4A4A"

# (label, RMSE, from the paper?) — from manual_match/dap_literatura.csv, stand removed
MODELOS = [
    ("log allometry on crown heights",   1.25, False),
    ("ridge, 6 flight metrics",          1.33, False),
    ("PLS, full suite",                  1.38, False),
    ("log allometry on mean height",     1.38, False),
    ("elastic net",                      1.51, False),
    ("random forest",                    1.68, False),
    ("PCA and regression",               1.88, False),
    ("the three networks of the paper",  2.02, True),
]
NULO = 2.26

# (label, volume rRMSE, colour)
# 23.7 % and not 36.4: representing a model with R2 near zero as "truth plus noise" lends it
# information it does not have. The correct representation is "predicts the mean", which puts the
# volume at 23.7 %; the advantage of the allometry drops from 18 to 5.6 points and the resampling
# confidence intervals then overlap.
VOLUME = [("with today's DBH model",      23.7, LARANJA),
          ("with a simple allometry",     18.1, AZUL),
          ("ceiling, with measured DBH",   5.7, VERDE)]
ACEITO = 15.0


def main():
    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 4.9), gridspec_kw={"width_ratios": [1.5, 1]})

    # ---------------- left, the model ladder
    ys = np.arange(len(MODELOS))
    for y, (rot, v, artigo) in zip(ys, MODELOS):
        cor = LARANJA if artigo else AZUL
        ax.barh(y, v, height=0.62, color=cor, zorder=3)
        ax.text(v + 0.03, y, f"{v:.2f}".replace(".", ","), va="center", fontsize=10,
                fontweight="bold" if artigo else "normal", color=cor)
    ax.axvline(NULO, color=CINZA, ls="--", lw=1.3, zorder=2)
    # with the y axis inverted, len(MODELOS)-0.35 is the footer and not the top, and the label
    # lands on an x-axis tick; anchoring in axis fraction (get_xaxis_transform) removes the
    # inversion from the calculation.
    ax.text(NULO + 0.05, 0.97, "do nothing,\npredict the mean", transform=ax.get_xaxis_transform(),
            fontsize=9.2, color=CINZA, va="top", linespacing=1.35)
    ax.set_yticks(ys)
    ax.set_yticklabels([r for r, _, _ in MODELOS], fontsize=10)
    for t, (_, _, artigo) in zip(ax.get_yticklabels(), MODELOS):
        if artigo:
            t.set_fontweight("bold"); t.set_color(LARANJA)
    ax.invert_yaxis()
    ax.set_xlim(0, 2.72); ax.set_xlabel("mean DBH error per plot  (cm)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_title("Seven methods from the literature err less than the three networks",
                 loc="left", fontsize=12, fontweight="bold", pad=12)

    # ---------------- right, what that buys
    ys = np.arange(len(VOLUME)) * 1.0
    for y, (rot, v, cor) in zip(ys, VOLUME):
        bx.barh(y, v, height=0.5, color=cor, zorder=3)
        bx.text(v + 0.8, y, f"{v:.1f}%".replace(".", ","), va="center",
                fontsize=12, fontweight="bold", color=cor)
        bx.text(0, y - 0.42, rot, va="bottom", fontsize=9.8, color="#333333")
    # axvline spans edge to edge and would cross the bar labels, which sit above them; the
    # segment is drawn with plot instead, limited to the band of the bars.
    bx.plot([ACEITO, ACEITO], [-0.30, 2.32], color=CINZA, ls="--", lw=1.3, zorder=4)
    # the 15% caption goes below the last bar: the band above each bar is already taken by its
    # own label.
    bx.text(ACEITO + 0.9, 2.46, "range accepted in inventory, 15%", fontsize=9.2,
            color=CINZA, ha="left", va="center")
    bx.set_yticks([]); bx.set_ylim(2.72, -0.72); bx.set_xlim(0, 45)
    bx.set_xlabel("volume error per plot  (rRMSE, %)")
    bx.spines["left"].set_visible(False)
    bx.set_title("And the volume goes from 24% to 18%", loc="left", fontsize=12,
                 fontweight="bold", pad=12)

    fig.subplots_adjust(left=0.245, right=0.985, top=0.86, bottom=0.135, wspace=0.10)
    out = config.OUT_DIR / "dap_e_volume_grupo.png"
    fig.savefig(out, dpi=200)
    print(f"written to {out}")


if __name__ == "__main__":
    main()

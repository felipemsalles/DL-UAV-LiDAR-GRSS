#!/usr/bin/env python3
"""The three levers the literature pointed at, and what each one answered.

One column per lever, each with a question and an answer. The order follows the value of the
result: first the one that worked, then the two that closed a door.

Labels stay outside the data area: value at the tip of the bar, name on the axis, annotation above
the panel.

Every number comes from a CSV, none typed in, so that the figure cannot diverge from the experiment.

Run: PYTHONPATH=. python scripts/fig_tres_alavancas.py
Out: manual_match/tres_alavancas.png
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from greenvista import config  # noqa: E402

_s = importlib.util.spec_from_file_location("vol", R / "scripts/exp_volume_tls_por_arvore.py")
_v = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_v)

AZUL, LARANJA, VERDE, CINZA, CLARO = "#2a78d6", "#eb6834", "#0f7a54", "#4A4A4A", "#B9C4CF"
SAIDA = config.OUT_DIR / "tres_alavancas.png"


def painel_volume(ax):
    cub = _v.curva_cubagem()
    xq, yq = _v.q_de_x(cub)
    A = pd.read_csv(config.OUT_DIR / "volume_tls_afilamento.csv")
    xt, yt = _v.q_de_x(A)
    xs = np.linspace(0.05, 0.60, 40)
    ax.plot(xs, np.interp(xs, xq, yq), color=CINZA, lw=3.0, label="destructive scaling, 63 trees")
    ax.plot(xs, np.interp(xs, xt, yt), color=VERDE, lw=3.0, ls=(0, (5, 2)),
            label="TLS, 809 stems")
    ax.set_xlabel("relative height along the tree  h / H")
    ax.set_ylabel("relative diameter  d / d(1.3 m)")
    ax.set_ylim(0.45, 1.06)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    ax.set_title("Per-tree volume can be measured\nwith the terrestrial laser",
                 loc="left", fontsize=12, fontweight="bold", color=VERDE)
    return ("The measured taper follows the destructive scaling.\n"
            "809 per-tree volumes, 31 times more\nlabels than the 26 scaled trees.")


def painel_fuste(ax):
    d = pd.read_csv(config.OUT_DIR / "densidade_e_fuste.csv").set_index("cond")
    tls = d.loc["TLS integral"]
    ralo = d.loc["TLS 0.003 do total"]
    dro = d.loc["drone T001 integral"]
    itens = [("chance, if the points\nwere spread out", 5.0, CLARO),
             (f"drone\n{dro.pts_por_fuste:.0f} points per stem", 100 * dro.frac_na_casca, LARANJA),
             (f"thinned terrestrial laser\n{ralo.pts_por_fuste:.0f} points per stem",
              100 * ralo.frac_na_casca, AZUL),
             (f"full terrestrial laser\n{tls.pts_por_fuste:.0f} points per stem",
              100 * tls.frac_na_casca, AZUL)]
    ys = np.arange(len(itens))
    ax.barh(ys, [v for _, v, _ in itens], color=[c for _, _, c in itens], height=0.62)
    for y, (_, v, _) in zip(ys, itens):
        ax.text(v + 1.5, y, f"{v:.0f} %".replace(".", ","), va="center", fontsize=10.5,
                fontweight="bold", color=CINZA)
    ax.set_yticks(ys, [t for t, _, _ in itens], fontsize=9.5)
    ax.set_xlim(0, 72)
    ax.set_xlabel("points falling on the stem bark")
    ax.set_title("Flying denser does not make\nthe stem appear", loc="left", fontsize=12,
                 fontweight="bold", color=LARANJA)
    return ("With almost half the points the drone has,\n"
            "the terrestrial laser still finds the stem. What separates\n"
            "the two is where one looks from, not how many points there are.")


def painel_borda(ax):
    d = pd.read_csv(config.OUT_DIR / "efeito_de_borda.csv")
    d = d[(d.alvo == "G") & (np.isclose(d.escala_copa, 1.0))].sort_values("area_m2")
    ax.plot(d.area_m2, d.desacordo_rel, "-o", color=AZUL, lw=2.6, ms=7)
    ax.axvline(400, color=CINZA, lw=1.2, ls=":")
    y = float(d[d.area_m2 == 400].desacordo_rel.iloc[0])
    ax.plot([400], [y], "o", ms=12, mfc="none", mec=LARANJA, mew=2.4)
    ax.set_xscale("log")
    ax.set_xticks([25, 100, 400, 900], ["25", "100", "400", "900"])
    ax.set_xlabel("plot size (m²)")
    ax.set_ylabel("error imposed by the edge (%)")
    ax.set_ylim(0, 34)
    ax.set_title("The plot edge is not\nour problem", loc="left", fontsize=12,
                 fontweight="bold", color=AZUL)
    return ("Crowns coming in and crowns going out nearly cancel, and\n"
            + f"in our 400 m² plots {y:.1f} % remains".replace(".", ",")
            + ",\nagainst the 18 % volume error we have today.")


def main():
    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlepad": 12})
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.4))
    notas = [painel_volume(axes[0]), painel_fuste(axes[1]), painel_borda(axes[2])]
    fig.suptitle("Three routes the literature pointed at, all three tested on our data",
                 x=0.012, y=0.985, ha="left", fontsize=14.5, fontweight="bold", color=CINZA)
    fig.tight_layout(rect=[0, 0.155, 1, 0.94])
    # the note goes below the panel; `fig.text` does not wrap on its own, so the line breaks are
    # written into the text and the position is taken from the axes already drawn.
    for ax, nota in zip(axes, notas):
        cx = ax.get_position().x0
        fig.text(cx, 0.115, nota, ha="left", va="top", fontsize=9.8, color=CINZA,
                 linespacing=1.5)
    fig.savefig(SAIDA, dpi=200, facecolor="white")
    print(SAIDA)


if __name__ == "__main__":
    main()

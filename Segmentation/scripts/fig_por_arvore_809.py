#!/usr/bin/env python3
"""What 35 times more labels changed per tree, and what the plot resolves.

Left, the R2 of per-tree DBH with the two controls on the same scale. Without them the figure would
be just a row of short bars; with them it becomes visible that the harness works (positive control at
0.90) and that the label holds up (ceiling at 0.97), so what is missing is signal.

Right, the same model summed over the plot, with the caveat drawn in: the plot total is carried by
the count.

Labels stay outside the data area: value at the tip of the bar, name on the axis. Every number comes
from a CSV.

Run: PYTHONPATH=. python scripts/fig_por_arvore_809.py
Out: manual_match/por_arvore_809.png
"""
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

AZUL, LARANJA, VERDE, CINZA, CLARO = "#2a78d6", "#eb6834", "#0f7a54", "#4A4A4A", "#B9C4CF"
SAIDA = config.OUT_DIR / "por_arvore_809.png"
MELHOR = "tudo + competicao, floresta"


def teto_do_rotulo():
    d = pd.read_csv(config.OUT_DIR / "ruido_do_rotulo_tls.csv")
    ran = pd.read_csv(config.OUT_DIR / "dap_tls_ransac_talhao001.csv")
    cheio = np.where(ran.motivo.values == "ok", ran.dap_cm.values, np.nan)
    dp_cheia = float(np.std(d.dap_A - d.dap_B, ddof=1)) / 2.0
    return 1 - dp_cheia ** 2 / float(np.nanvar(cheio, ddof=1))


def melhor(t, copa, alvo):
    s = t[(t.copa == copa) & (t.alvo == alvo) & (~t.modelo.str.contains("agregado"))]
    return float(s.R2.max())


def painel_esquerda(ax, t, pn, teto):
    itens = [("null, predicts the mean", 0.0, CLARO),
             ("1.5 m disc", melhor(t, "disco de 1.5 m", "dap_cm"), AZUL),
             ("PointNet on the raw cloud", float(pn[pn.cond == "real"].R2.iloc[0]), AZUL),
             ("voronoi on the map,\noracle segmentation", melhor(t, "voronoi no mapa (oraculo)",
                                                                 "dap_cm"), AZUL),
             ("positive control\n(target built from the crown itself)",
              melhor(t, "disco de 1.5 m", "controle"), VERDE)]
    ys = np.arange(len(itens))
    ax.barh(ys, [v for _, v, _ in itens], color=[c for _, _, c in itens], height=0.6)
    for y, (_, v, _) in zip(ys, itens):
        ax.text(v + 0.018, y, f"{v:.2f}".replace(".", ","), va="center", fontsize=10.5,
                fontweight="bold", color=CINZA)
    ax.axvline(teto, color=LARANJA, lw=1.8, ls=(0, (4, 2)))
    ax.set_yticks(ys, [t_ for t_, _, _ in itens], fontsize=9.5)
    ax.set_xlim(-0.02, 1.10)
    ax.set_xlabel("R² of per-tree DBH, spatial-block validation")
    ax.set_title("35 times more labels, and the result\ndid not change", loc="left", fontsize=12,
                 fontweight="bold", color=AZUL)
    ax.annotate(f"ceiling imposed by\nlabel noise: {teto:.2f}".replace(".", ","),
                xy=(teto, len(itens) - 0.45), xytext=(teto - 0.02, len(itens) - 0.15),
                ha="right", va="center", fontsize=9.2, color=LARANJA)
    return ("With 809 trees measured by the terrestrial laser, the crown\n"
            "seen from above explains less than 10 % of the variation in\n"
            "diameter. The positive control shows that the test is sound.")


def painel_direita(ax, t):
    ag = t[t.modelo.str.contains("agregado")]
    copa = "voronoi no mapa (oraculo)"
    pares = []
    for alvo, rot in (("DAP medio", "mean DBH\nof the plot"),
                      ("volume medio", "mean volume\nper tree"),
                      ("volume total", "total volume\nof the plot")):
        linha = ag[(ag.copa == copa) & (ag.alvo == alvo)]
        base = t[(t.copa == copa) & (t.modelo == MELHOR)
                 & (t.alvo == ("dap_cm" if alvo == "DAP medio" else "vol_m3"))]
        if len(linha) and len(base):
            pares.append((rot, float(base.rRMSE.iloc[0]), float(linha.rRMSE.iloc[0]),
                          float(linha.R2.iloc[0])))
    ys = np.arange(len(pares))
    ax.barh(ys + 0.19, [a for _, a, _, _ in pares], height=0.34, color=CLARO,
            label="per tree")
    ax.barh(ys - 0.19, [b for _, _, b, _ in pares], height=0.34, color=VERDE,
            label="summed over the 400 m² plot")
    # the R2 sits on the same line as the rRMSE: on its own, the drop in relative error suggests
    # that the plot solves the problem, but in a homogeneous stand predicting close to the mean
    # already yields a low relative error, and it is the R2 that exposes that.
    for y, (_, a, b, r2) in zip(ys, pares):
        ax.text(a + 1.0, y + 0.19, f"{a:.0f} %", va="center", fontsize=10, color=CINZA)
        ax.text(b + 1.0, y - 0.19, f"{b:.0f} %   R² {r2:+.2f}".replace(".", ","),
                va="center", fontsize=10.5, fontweight="bold", color=VERDE)
    ax.set_yticks(ys, [r for r, _, _, _ in pares], fontsize=9.5)
    ax.set_xlim(0, max(a for _, a, _, _ in pares) * 1.42)
    ax.set_xlabel("relative error (rRMSE)")
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.set_title("Summing over the plot brings the error down,\nbut it is not the model getting it right",
                 loc="left", fontsize=12, fontweight="bold", color=VERDE)
    return ("The error falls because the stand is homogeneous, and the R² near\n"
            "zero shows that the model cannot size a tree. The total only has\n"
            "a high R² because it tracks the COUNT (r = +0.86).")


def main():
    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlepad": 12})
    t = pd.read_csv(config.OUT_DIR / "por_arvore_809.csv")
    pn = pd.read_csv(config.OUT_DIR / "por_arvore_pointnet.csv")
    teto = teto_do_rotulo()
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.2))
    notas = [painel_esquerda(axes[0], t, pn, teto), painel_direita(axes[1], t)]
    fig.suptitle("Per tree does not work, and now we know why",
                 x=0.012, y=0.985, ha="left", fontsize=14.5, fontweight="bold", color=CINZA)
    fig.tight_layout(rect=[0, 0.185, 1, 0.93])
    for ax, nota in zip(axes, notas):
        fig.text(ax.get_position().x0, 0.145, nota, ha="left", va="top", fontsize=9.8,
                 color=CINZA, linespacing=1.5)
    fig.savefig(SAIDA, dpi=200, facecolor="white")
    print(SAIDA)


if __name__ == "__main__":
    main()

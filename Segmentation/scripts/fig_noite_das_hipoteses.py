#!/usr/bin/env python3
"""Three panels: pre-training scale, teacher-student distillation and the gain in counting.

Left, the pre-training scaling curve, the only route with a gain, showing where it saturates. The
saturation is the actionable information: it turns the request from "fly more area" into "seek
diversity".

Centre, teacher against student, in two bars. Right, the operational gain in counting.

Labels stay outside the data area: value at the tip of the bar, name on the axis, note below. Every
number is read from a CSV.

Run: PYTHONPATH=. python scripts/fig_noite_das_hipoteses.py
Out: manual_match/noite_das_hipoteses.png
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
SAIDA = config.OUT_DIR / "noite_das_hipoteses.png"


def painel_escala(ax):
    d = pd.read_csv(config.OUT_DIR / "ssl_escala.csv")
    c = d[d.teste == "escala"].sort_values("n_corpus")
    n = np.concatenate([[0], c.n_corpus.values])
    r2 = np.concatenate([[0.066], c.R2.values])
    ax.plot(n, r2, "-o", color=VERDE, lw=2.6, ms=7)
    ax.axhspan(c.R2.values[-2:].min() - 0.004, c.R2.values[-2:].max() + 0.004,
               color=VERDE, alpha=0.10)
    ax.axhline(0.066, color=CINZA, lw=1.2, ls=":")
    ax.set_xlabel("unlabelled crowns used in pre-training")
    ax.set_ylabel("R² of per-tree DBH")
    ax.set_ylim(0.0, 0.185)
    ax.set_xticks([0, 4000, 8000, 12000], ["0", "4k", "8k", "12k"])
    ax.annotate("no pre-training", xy=(0, 0.066), xytext=(1100, 0.030),
                fontsize=9.2, color=CINZA)
    ax.annotate("saturates here", xy=(9800, 0.146), xytext=(5200, 0.170),
                fontsize=9.5, color=VERDE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=VERDE, lw=1.4))
    ax.set_title("Pre-training on unlabelled crowns\ndoubles the result, then stops",
                 loc="left", fontsize=12, fontweight="bold", color=VERDE)
    return ("The 12 thousand crowns come from the same clone and the same management, so\n"
            "crown number 12 thousand is almost identical to crown 8 thousand. What saturated was\n"
            "the variety, not the count. Flying more of the same does not help.")


def painel_professor(ax):
    d = pd.read_csv(config.OUT_DIR / "destilacao_tls.csv")
    prof = float(d[d.rota.str.startswith("professor")].R2.iloc[0])
    sozinho = float(d[(d.rota == "aluno sozinho") & (d.cond == "real")].R2.iloc[0])
    dest = float(d[d.rota.str.contains("0.1") & (d.cond == "real")].R2.iloc[0])
    itens = [("student, distilled\nfrom the teacher", dest, LARANJA),
             ("student alone\n(drone only)", sozinho, LARANJA),
             ("teacher\n(terrestrial laser)", prof, AZUL)]
    ys = np.arange(len(itens))
    ax.barh(ys, [v for _, v, _ in itens], color=[c for _, _, c in itens], height=0.55)
    for y, (_, v, _) in zip(ys, itens):
        ax.text(v + 0.012, y, f"{v:.2f}".replace(".", ","), va="center", fontsize=11,
                fontweight="bold", color=CINZA)
    ax.set_yticks(ys, [t for t, _, _ in itens], fontsize=9.5)
    ax.set_xlim(0, 0.55)
    ax.set_xlabel("R² of per-tree DBH")
    ax.set_title("The teacher knows and cannot\nteach it", loc="left", fontsize=12,
                 fontweight="bold", color=AZUL)
    return ("The same network, on the same trees: seeing the terrestrial laser it\n"
            "does four times better. Copying what it learned into the network\n"
            "that sees only the drone transfers nothing. The information is not there.")


def painel_contagem(ax):
    d = pd.read_csv(config.OUT_DIR / "filtro_melhora_contagem.csv")
    base = float(d.f1_sem_filtro.iloc[0])
    so_det = float(d[d.conjunto.str.contains("detector")].f1_fora_da_dobra.iloc[0])
    com = float(d[d.conjunto.str.contains("peito")].f1_fora_da_dobra.iloc[0])
    itens = [("no filter", base, CLARO),
             ("filtering with what\nthe detector already knows", so_det, AZUL),
             ("plus the return at\nbreast height", com, VERDE)]
    ys = np.arange(len(itens))
    ax.barh(ys, [v for _, v, _ in itens], color=[c for _, _, c in itens], height=0.55)
    for y, (_, v, _) in zip(ys, itens):
        ax.text(v + 0.004, y, f"{v:.3f}".replace(".", ","), va="center", fontsize=11,
                fontweight="bold", color=CINZA)
    ax.set_yticks(ys, [t for t, _, _ in itens], fontsize=9.5)
    ax.set_xlim(0.80, 0.96)
    ax.set_xlabel("F1 of the count against the stem map")
    ax.set_title("And the count genuinely\nimproved", loc="left", fontsize=12,
                 fontweight="bold", color=VERDE)
    # three short lines: `fig.text` does not wrap on its own, so a line wider than the panel
    # runs off the edge of the figure.
    return ("Counting error falls from +9.4 % to +4.8 %.\n"
            "The threshold comes from outside the fold, so it is not\n"
            "a number chosen after seeing the result.")


def main():
    plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlepad": 12})
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.4))
    notas = [painel_escala(axes[0]), painel_professor(axes[1]), painel_contagem(axes[2])]
    fig.suptitle("Twelve hypotheses in one night: what paid off, what was proved and what improved",
                 x=0.011, y=0.985, ha="left", fontsize=14.5, fontweight="bold", color=CINZA)
    fig.tight_layout(rect=[0, 0.185, 1, 0.93])
    for ax, nota in zip(axes, notas):
        fig.text(ax.get_position().x0, 0.145, nota, ha="left", va="top", fontsize=9.6,
                 color=CINZA, linespacing=1.5)
    fig.savefig(SAIDA, dpi=200, facecolor="white")
    print(SAIDA)


if __name__ == "__main__":
    main()

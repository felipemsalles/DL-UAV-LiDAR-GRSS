"""Comparison restricted to eucalyptus.

Shinzato et al. 2017 falls below our value, at 58%, and their stand is practically ours: dominant
height 29.79 m against our ~31.7, DBH 16.00 cm against 16.4, and denser, 1708 against 1379 stems/ha,
with a field-inventory reference over 48 plots. That isolates the variable of interest: same species,
same size, denser stand, field reference, and the large difference sits in the sensor, 5 pts/m2 for
them against 350 to 750 for us.

Density of our stand, taken from the field inventory
(Dados_SaoManuel/5-dados_campo/inv_euc.csv): the plots have area_parc = 400 m2, so the 13 plots add
up to 0.520 ha and 717 / 0.520 = 1379 stems/ha. The 12 m radius used to extract the LiDAR metrics
is a different area and must not be substituted for the field-plot area here. The
measured canopy cover (manual_match/plot_structural.csv) gives 0.735 on average, from 0.543 to 0.858,
i.e. 26.5% of the area is gap: a closed canopy, but without crowns touching everywhere. The dominant
height of the 5 mature stands runs from 29.3 to 33.1 m, mean 31.7 m.

Each bar carries alongside it, in prose, the reason why that number is higher, with no legend and no
colour convention to decode.

Sources of the reasons written in the figure:
* da Cunha Neto et al. 2025, Forests 16(11) 1747, Tab. 2: agroforestry system of 357 trees/ha, open
  canopy.
* Zhou et al. 2025, Annals of Forest Science 82:20 (E. urophylla): 380 of the 413 reference trees
  marked by visual interpretation of the point cloud itself, not measured in the field.
* Vauhkonen et al. 2012, Forestry 85(1), Tab. 1 and discussion p. 36: 807 trees/ha, and the winning
  algorithm (97.3%) "used initial estimates of stem density in the tree detection", with commission
  acknowledged from thick branches and without qualitative verification. Their DET% is a count ratio
  against the field plot, equivalent to our metric (p. 30: "The tree positions were not recorded in
  the field in the Brazilian dataset, so that the evaluation was carried out at the plot-level
  only").
* Yan et al. 2024, Forests 15(1) 209: mean DBH 7.76 cm, mean height 10.17 m, small and separated
  crowns. Not to be confused with the "Yan 2024" of the volume figure, Sensors 24(21) 7071, on
  biomass over 22 plots: same first author, same year, different papers.
"""
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
OUT = os.environ.get("GREENVISTA_FIG_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "manual_match", "detection_bars.png"))

GREEN, OUTRO = "#1d6b45", "#b9b3a4"   # a single colour for the others: nothing to decode

# (value, name, reference, reason why the number is higher than ours, colour, is_ours)
dados = [
    (93.0, "Eucalyptus agroforestry system", "da Cunha Neto et al. 2025, Forests",
     "There are 357 trees per hectare, with the crowns separated from one another. "
     "Each whole tree can be seen from above.", OUTRO, False),
    (92.0, "7-year-old eucalyptus", "Zhou et al. 2025, Annals of Forest Science",
     "Of the 413 trees used as reference, 380 were marked by a person "
     "looking at the point cloud on screen, and not measured in the field.", OUTRO, False),
    (86.0, "Veracel plantation, in Bahia", "Vauhkonen et al. 2012, Forestry",
     "There are 807 trees per hectare, and the algorithm that pulled the mean up "
     "was told in advance how many trees each plot held.", OUTRO, False),
    (72.9, "Young eucalyptus, in China", "Yan et al. 2024, Forests",
     "The trees are 8 cm in diameter and 10 metres tall. The crown is still small and "
     "does not touch its neighbour.", OUTRO, False),
    (72.8, "OURS, mature eucalyptus", "São Manuel, 13 plots, 717 trees",
     "There are 1379 trees per hectare, the canopy of the mature stands passes 30 metres "
     "and covers 74% of the area. Success is counted against the 717 trees measured on the ground.",
     GREEN, True),
    (58.0, "Mature eucalyptus, in São Paulo", "Shinzato et al. 2017, iForest",
     "It is the same species as ours, and the stand is denser, with 1708 trees per "
     "hectare and a dominant height of 30 metres. But the laser is airborne and records 5 points "
     "per square metre, against the 350 to 750 of ours.", OUTRO, False),
]

fig, ax = plt.subplots(figsize=(15.4, 9.0), dpi=200)
y = np.arange(len(dados))[::-1]
X_NOME, X_RAZAO = -4, 118          # name gutter on the left, reason gutter on the right

for yi, (v, nome, ref, razao, c, nosso) in zip(y, dados):
    if yi % 2 == 0:
        ax.axhspan(yi - .5, yi + .5, color="#8d8d8608", zorder=0)
    ax.barh([yi], [v], color=c, height=.30, zorder=3)
    ax.text(v + 1.8, yi, f"{v:.1f}".replace(".", ",").replace(",0", "") + "%", va="center",
            ha="left", fontsize=15, fontweight="bold", color="#1c1c1a", zorder=4)
    ax.text(X_NOME, yi + .13, nome, va="center", ha="right", fontsize=12.2,
            fontweight="bold" if nosso else "normal", color=GREEN if nosso else "#2f2f2b")
    ax.text(X_NOME, yi - .17, ref, va="center", ha="right", fontsize=9.6,
            color=GREEN if nosso else "#9a9a93")
    ax.text(X_RAZAO, yi, textwrap.fill(razao, 50), va="center", ha="left", fontsize=10.6,
            color=GREEN if nosso else "#6f6f69", linespacing=1.6)

ax.axvline(0, color="#d5d5ce", lw=1, zorder=1)
ax.set_yticks([]); ax.set_xlim(-104, 300); ax.set_ylim(-0.7, len(dados) - 0.3)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("percentage of the trees the computer finds", fontsize=12.5)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.set_title("These numbers look comparable, but they are not\n"
             "From one study to the next the forest changes, the equipment changes, and so does "
             "the way success is checked",
             fontsize=14, pad=20, loc="left", linespacing=1.6)
ax.grid(axis="x", alpha=.18, lw=.6, zorder=0)
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_xbound(-104, 300)

# any further explanatory text goes in the <figcaption>, outside the image: a `fig.text` here
# would be baked into the PNG, uneditable from the HTML, and would duplicate the figure caption.

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("ok", OUT)

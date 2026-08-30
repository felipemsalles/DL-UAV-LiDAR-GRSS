"""Volume figure, two panels. Left, where the signal comes from. Right, error against number of plots.

Fourth bar of the left panel: the model is the `dominant-H allom` of
manual_match/lidar_only_baseline.csv, with LOTO_R2 = +0.0367, practically zero but positive. The
negative value (-0.1703) belongs to `dom-H + LiDAR-site`, which is not the model cited in section 3 of
the .md (that one cites "dominant-height allometry, LiDAR only" with rRMSE 25.3%, which matches the
dominant-H allom, 25.27%, and not the 27.85% of the other).

Values, all R² under leave-one-stand-out, checked against the CSVs:
* manual_match/lidar_beyond_age.csv    -> base_R2 0.8132 (age only) and combined_R2 0.8119 (age+LiDAR)
* manual_match/lidar_only_baseline.csv -> PLS full suite 0.3415 and dominant-H allom 0.0367

References of the right panel:
* Leite 2020 has 33 plots, not 100: Remote Sensing 12(9):1513, "A total of 33 rectangular field
  plots of ~280 m2 were measured"; in the discussion, "few sample plots (n = 33)".
* Silva 2020 is (63 plots, 12.5%), the Silva et al. 2020 from Suzano, best OLS combination with 40% of
  the plots. Not to be confused with V.S. da Silva 2020 (7.83%), which is a different paper.
* "Yan 2024" here is Sensors 24(21) 7071 (biomass, 22 plots). The "Yan et al. 2024" of the detection
  figure is Forests 15(1) 209: same first author, same year, different papers.

Layout of the right panel: no legend box, each point carries its name beside it; the threshold line
uses its own grey, distinct from the highlight colour of the studies; the Packalén block sits to the
right of the marker, in two short lines, so as not to cover it.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.environ.get("GREENVISTA_FIG_OUT",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "manual_match", "volume_panels.png"))

GREEN, ORANGE, GREY = "#1d6b45", "#b7791f", "#a9b3ac"
LIMIAR = "#8d8d86"          # the reference line's own colour, distinct from the highlight colour

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.9), dpi=170)

# ---------------------------------------------------------------- left panel
# (label, leave-one-stand-out R², colour)
mod = [
    ("Planting date\nonly",          0.8132, ORANGE),
    ("Planting date\nplus LiDAR",    0.8119, ORANGE),
    ("LiDAR only,\nall metrics",     0.3415, GREEN),
    ("LiDAR only,\ncanopy height",   0.0367, GREEN),
]
y = np.arange(len(mod))[::-1]
for yi, (lab, v, c) in zip(y, mod):
    ax1.barh([yi], [v], color=c, height=.52, zorder=3)
    ax1.text(v + .022, yi, f"{v:.2f}".replace(".", ","), va="center", ha="left",
             fontsize=12.5, fontweight="bold", color="#1c1c1a")
    ax1.text(-.03, yi, lab, va="center", ha="right", fontsize=10.4, color="#2f2f2b",
             linespacing=1.5)
ax1.text(0.0367 + .21, y[-1], "practically zero", va="center", ha="left", fontsize=9,
         color="#8a8a84", style="italic")

ax1.set_yticks([]); ax1.set_xlim(-.46, 1.06); ax1.set_ylim(-.62, len(mod) - .38)
ax1.set_xticks([0, .2, .4, .6, .8, 1.0])
ax1.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}".replace(".", ","))
ax1.set_xlabel("model accuracy on a stand it has never seen", fontsize=10.5)
ax1.set_title("The planting date, which is a spreadsheet record,\npredicts volume better than LiDAR",
              fontsize=11, pad=13, loc="left", linespacing=1.6)
ax1.axvline(0, color="#c9c9c2", lw=1, zorder=1)
ax1.grid(axis="x", alpha=.2, lw=.6, zorder=0)
for s in ("top", "right", "left"):
    ax1.spines[s].set_visible(False)
ax1.tick_params(axis="y", length=0)

# ---------------------------------------------------------------- right panel
pub = [
    (22,  23.1,  "Yan 2024",      (0, 15),  "center"),
    (25,   5.42, "Pertille 2025", (0, 15),  "center"),
    (33,  14.4,  "Leite 2020",    (0, -15), "center"),   # the dashed line runs just above
    (63,  12.5,  "Silva 2020",    (0, 15),  "center"),
    (108,  9.99, "Silva 2016",    (0, 15),  "center"),
    (136,  6.14, "Silva 2017",    (0, 15),  "center"),
    (195,  9.0,  "Cosenza 2021",  (16, 0),  "left"),
]
ax2.axhline(15, color=LIMIAR, ls="--", lw=1.8, zorder=2)
for n, r, lab, off, ha in pub:
    ax2.scatter([n], [r], s=110, color=GREY, zorder=4)
    ax2.annotate(lab, (n, r), textcoords="offset points", xytext=off, ha=ha, va="center",
                 fontsize=9.5, color="#7c7c76")

ax2.scatter([195], [11.86], s=170, color=ORANGE, marker="D", zorder=5)
ax2.text(225, 13.6, "Packalén 2011", ha="left", va="bottom", fontsize=10.5,
         fontweight="bold", color=ORANGE)
ax2.text(225, 13.4, "the closest one\nto ours", ha="left", va="top", fontsize=8.6,
         color=ORANGE, linespacing=1.5)

ax2.errorbar([13], [13.2], yerr=[[13.2 - 9.8], [16.3 - 13.2]], fmt="o", ms=13, color=GREEN,
             capsize=7, lw=2.4, zorder=6)
ax2.annotate("ours, with 13 plots", (13, 13.2), textcoords="offset points", xytext=(14, -27),
             ha="left", fontsize=10.5, fontweight="bold", color=GREEN)

ax2.set_xscale("log")
ax2.set_xlim(9, 900); ax2.set_ylim(0, 27)
ax2.set_xticks([10, 20, 50, 100, 200, 500])
ax2.set_xticklabels(["10", "20", "50", "100", "200", "500"])
ax2.set_yticks(range(0, 30, 5))
ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax2.set_xlabel("plots measured in the field", fontsize=10.5)
ax2.set_ylabel("volume error margin", fontsize=10.5)
ax2.set_title("We are within the error the industry accepts.\n"
              "But with 13 plots the uncertainty bar spans almost the whole list",
              fontsize=11, pad=13, loc="left", linespacing=1.6)
ax2.grid(alpha=.2, lw=.6)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
# above the dashed line only our error bar exists, at x=13, so the right side is free
ax2.text(870, 15.35, "error the industry accepts", color=LIMIAR, fontweight="bold",
         fontsize=10, ha="right", va="bottom")

fig.text(.5, -.03, "Every number on the right comes from the test that hides one plot at a time, "
         "which is the test the literature uses.", fontsize=8.4, color="#8a8a84", ha="center")
fig.subplots_adjust(left=.135, right=.985, top=.80, bottom=.20, wspace=.42)

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("ok", os.path.normpath(OUT))

#!/usr/bin/env python3
"""Figures in English for submission, in the IEEE style.

Only the figures are in English; tables and documentation stay in the report language. The
scope is detection, not volume.

No caption is baked into the image: the caption lives in the LaTeX `\\caption{}`, where it is
set in the journal font, enters the list of figures and can be edited without regenerating
the image. The suggested captions are written to `figure_captions.txt`.

Dimensions. An IEEE double column is 7.16 in and a single column 3.5 in. The figures are
drawn at final size, with an 8 pt font, so that they are not rescaled in LaTeX, which is
what makes letter sizes disagree between figures of the same paper.

Run: PYTHONPATH=. python scripts/fig_relatorio_en.py [--out figs_en]
"""
import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_w2w_contagem import nms_malha, parear  # noqa: E402

from greenvista import config  # noqa: E402

VERDE = "#1D6B45"      # SegmentAnyTree
AZUL = "#2C5F8A"       # FF3D
CINZA = "#8A8A8A"      # classical methods
VERM = "#B0304A"
LARANJA = "#D4761E"   # judgeable false positive (fig10)
LIMIAR = 2.0
PR400 = math.sqrt(400 / math.pi)
ANEL = 3.0
COL1, COL2 = 3.5, 7.16
# Printed width, and not the design width: a figure drawn at 3.5 in and placed with
# width=0.50\columnwidth prints at 1.74 in, and the font shrinks with it (8 pt becomes 4 pt,
# half the practical IEEE minimum). Drawing at final size keeps the 8 pt.
# Check with scripts/checa_fonte_figuras.py.
IMP_FIG4 = 0.50 * 3.487
IMP_FIG9 = 1.00 * 3.487

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "lines.linewidth": 1.4,
    # 750 and not 600 because of the `bbox_inches="tight"` in `salvar`: the crop removes the
    # white margin, the saved image covers fewer inches than figsize asked for and LaTeX
    # stretches it back to the full \columnwidth, reducing the effective dpi on the page. At a
    # nominal 600 the measured values were fig7 596, fig4 568, fig9 548 and fig10 503, below
    # the 600 the IEEE asks for line and combination art; at 750 all of them pass.
    # Check with scripts/checa_dpi_figuras.py after changing the size.
    "figure.dpi": 150, "savefig.dpi": 750,
})

LEGENDAS = []


FORMATOS = ("png",)     # PDF dropped on request; 600 dpi is already enough for print


def salvar(fig, out: Path, nome: str, legenda: str, dpi=None, pad=0.02):
    """The `dpi` parameter exists because of the crop: `bbox_inches="tight"` cuts the white,
    the file covers fewer inches than `figsize` asked for and LaTeX stretches it back to
    `\textwidth`. The more white is cut, the greater the stretch and the lower the effective
    dpi; fig10, after aligning the panels, fell to 588 and broke the IEEE floor of 600. Check
    with `scripts/checa_dpi_figuras.py`, which measures pixels per printed inch."""
    out.mkdir(parents=True, exist_ok=True)
    for ext in FORMATOS:
        fig.savefig(out / f"{nome}.{ext}", bbox_inches="tight", pad_inches=pad,
                    **({"dpi": dpi} if dpi else {}))
    plt.close(fig)
    LEGENDAS.append((nome, legenda))
    print(f"  {nome}.{'/.'.join(FORMATOS)}")


def eixo_em_metros(ax, x0, y0, passo, rotulo_y=True):
    """Axes labelled in relative metres, in place of a scale bar.

    A scale bar is a loose object inside the figure: the 200 m one landed near stand 007,
    whose width is 166 m, and the near coincidence led to reading a dimension that does not
    exist. A graduated axis cannot be mistaken for a map feature.

    The origin sits at the corner of the crop, and not in UTM: besides giving smaller
    numbers, the figure then does not print the absolute coordinate of the stand, which is
    under a confidentiality agreement.
    """
    # ticks at multiples of the relative origin, and not of the UTM coordinate: with
    # MultipleLocator they fall at multiples of the absolute value and, since the origin is
    # not a multiple of 200, the labels come out as 20, 220, 420.
    import numpy as _np
    from matplotlib.ticker import FixedLocator, FuncFormatter

    def marcas(eixo, base, lo, hi):
        k0 = _np.ceil((lo - base) / passo)
        k1 = _np.floor((hi - base) / passo)
        eixo.set_major_locator(FixedLocator(
            [base + k * passo for k in _np.arange(k0, k1 + 1)]))

    marcas(ax.xaxis, x0, *ax.get_xlim())
    marcas(ax.yaxis, y0, *ax.get_ylim())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v - x0:.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v - y0:.0f}"))
    ax.set_xlabel("Distance east (m)")
    if rotulo_y:
        ax.set_ylabel("Distance north (m)")
    ax.tick_params(labelsize=7)


# ------------------------------------------------------------------ figure 1
def fig_matched(out):
    """Precision, recall and F1 of the two models under three conditions.

    TENT enters as a negative control: if minimising entropy on the affine parameters adds
    nothing over recomputing the BatchNorm statistics, what was misaligned were the
    statistics and not the weights.

    Each model is merged at its own radius; a single 1.5 m radius for both takes 3.6 points
    off FF3D.
    """
    import geopandas as gpd
    inst = pd.read_csv(config.REPO / "data/detections/tta_t001_instancias.csv")
    inst = inst[inst.posicao == "centroide"]
    fustes = gpd.read_file(config.REPO /
                           "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp")
    ref_all = np.column_stack([fustes.geometry.x, fustes.geometry.y])
    p = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    p["t"] = pd.to_numeric(p.talhao, errors="coerce")
    p["pa"] = pd.to_numeric(p.parcela, errors="coerce")
    C = {f"{int(r.t)}_{int(r.pa)}": (r.geometry.x, r.geometry.y)
         for _, r in p.dropna(subset=["t", "pa"]).iterrows()}
    meia = 32.0 / 2 - 3.0
    RAIO = {"FF3D": 1.1, "SegmentAnyTree": 1.5}

    linhas = []
    for mod, raio in RAIO.items():
        for cond in inst[inst.modelo == mod].condicao.unique():
            tp = npred = nref = 0
            for pid in ("1_1", "1_2"):
                g = inst[(inst.modelo == mod) & (inst.condicao == cond) & (inst.plot_id == pid)]
                if g.empty:
                    continue
                xy = g[["cx", "cy"]].to_numpy(float)
                keep = nms_malha(xy, g.n_pts.to_numpy(float), raio)
                px, py = C[pid]
                dentro = lambda a: a[(np.abs(a[:, 0] - px) <= meia) & (np.abs(a[:, 1] - py) <= meia)]
                pr, rf = dentro(xy[keep]), dentro(ref_all)
                tp += parear(rf, pr, LIMIAR); npred += len(pr); nref += len(rf)
            if npred == 0:
                continue
            prec, rev = tp / npred, tp / nref
            linhas.append({"modelo": mod, "condicao": cond, "precisao": prec, "revocacao": rev,
                           "F1": 2 * prec * rev / (prec + rev) if prec + rev else 0.0})
    d = pd.DataFrame(linhas)
    mapa = {"baseline_pareado": "None", "baseline": "None", "adabn": "AdaBN", "tent": "TENT"}
    ordem = ["None", "AdaBN", "TENT"]
    dados = {}
    for mod, cor in (("FF3D", AZUL), ("SegmentAnyTree", VERDE)):
        sub = d[(d.modelo == mod) & (d.condicao != "baseline_julho")].copy()
        sub["c"] = sub.condicao.map(mapa)
        dados[mod] = (sub.dropna(subset=["c"]).set_index("c").reindex(ordem), cor)

    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.4), sharey=True)
    for ax, (col, tit) in zip(axes, [("precisao", "Precision"), ("revocacao", "Recall"),
                                     ("F1", "F1")]):
        # bar narrower than the step between them: with w = 0.36 and an offset of ±0.18 the
        # two touch and the labels end up crammed together; with w = 0.30 and ±0.19 there is
        # a gap and each label has room.
        x = np.arange(len(ordem)); w, off = 0.30, 0.19
        for k, (mod, (sub, cor)) in enumerate(dados.items()):
            b = ax.bar(x + (2 * k - 1) * off, sub[col].to_numpy(float), w, label=mod,
                       color=cor, edgecolor="white", linewidth=0.5)
            # Two decimals. The third is below the run-to-run noise, 0.9%.
            ax.bar_label(b, fmt="%.2f", fontsize=6.5, padding=1.5)
        ax.set_xticks(x); ax.set_xticklabels(ordem)
        ax.set_ylim(0, 1.10); ax.set_title(tit); ax.set_axisbelow(True)
        ax.set_xlabel("Test-time adaptation")
    axes[0].set_ylabel("Score")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    tab = d.copy()
    tab["condition"] = tab.condicao.map(mapa)
    tab = tab.dropna(subset=["condition"]).rename(
        columns={"modelo": "network", "precisao": "precision", "revocacao": "recall"})
    tab["merge_radius_m"] = tab.network.map(RAIO)
    tab = tab[["network", "condition", "merge_radius_m", "precision", "recall", "F1"]]
    tab = tab.round(3).sort_values(["network", "condition"])
    tab.to_csv(out / "table2_matched_metric.csv", index=False)
    print("  table2_matched_metric.csv")

    salvar(fig, out, "fig5_matched_metric_stand001",
           "Detection accuracy of the two networks under three test-time adaptation settings, "
           "evaluated against the terrestrial laser stem map of stand 001. Reference is 211 stems "
           "inside a 26 m evaluation square, matched one-to-one by Hungarian assignment at a 2 m "
           "threshold. Tiles are centred on the plot, and each network is merged at its own "
           "optimal radius for that scheme, 1.1 m for FF3D and 1.5 m for SegmentAnyTree. "
           "One run per condition. Exact values are given in Table I.")


# ------------------------------------------------------------------ figure 2
def fig_scatter(out):
    """Detected against census, one plot per point, with the 45-degree line.

    A single unswept local-maxima setting (0.5 m raster, 3 px window, sigma 0.6) reaches
    47.7% of the census count. Sweeping resolution and window and choosing by matched F1,
    the same algorithm reaches 97.8%, so a count-based comparison separates the methods
    poorly.

    The difference shows up in the tree-to-tree matching of stand 001, where the swept
    classical method scores F1 0.792 against 0.921 for the full SegmentAnyTree: the advantage
    is in detection quality, and not in counting (figure 8).

    Both baselines are drawn, the swept one and the unswept one.
    """
    sat = pd.read_csv(config.OUT_DIR / "sat_adabn_13parcelas.csv").set_index("parcela")
    ff = pd.read_csv(config.OUT_DIR / "ff3d_adabn_13parcelas.csv")
    ff = ff[ff.raio == 1.1].set_index("parcela")
    cl = pd.read_csv(config.OUT_DIR / "watershed_sweep.csv")
    cl["parcela"] = cl.talhao.astype(str) + "_" + cl.parcela.astype(str)
    cl = cl.set_index("parcela")
    campo = sat.campo

    # the nine views apply to the classical method too: comparing a network with 9 views +
    # AdaBN against single-pass local maxima is asymmetric. The source is the per-plot CSV of
    # the nine-view experiment.
    f = config.OUT_DIR / "maximo_local_nove_vistas_13parcelas.csv"
    if not f.exists():
        print("  (skipping fig4: run exp_maximo_local_nove_vistas_13parcelas.py first)")
        return
    lm = pd.read_csv(f)
    lm = lm[lm.condicao == "nove_vistas"]
    lm["parcela"] = (lm.parcela.str.slice(1, 4).astype(int).astype(str) + "_"
                     + lm.parcela.str.slice(6, 9).astype(int).astype(str))
    def serie_lm(ajuste):
        s = lm[lm.ajuste == ajuste].set_index("parcela").detectado
        return s.reindex(campo.index)

    # Palette local to this figure, and not the global constants. The two local-maxima series
    # in "#4A4A4A" and CINZA are two greys separated only by lightness and, at 1.74 in with a
    # 13 pt marker, read as a single series; the VERDE x AZUL pair (SAT x FF3D) sits at ΔE 13.1
    # under normal vision, below the floor of 15. The four colours below come from a sweep of
    # the 70 combinations of four over the eight reference hues, evaluated with --pairs all,
    # which is the appropriate test for a scatter plot, where any pair may appear side by side.
    #
    # The green is dark because a vivid green collapses with the orange under protanopia
    # (ΔE 3.2); #0f7a54 keeps 7.2, admissible with a secondary encoding, which here is four
    # distinct markers (D ^ s o). Preferred over the light aqua because it passes the 3:1
    # contrast against white paper, where the aqua sits at 2.74.
    #
    # The assignment follows the overlap in the plot: SAT and LM 0.75 m both lie on the 1:1
    # line and take the most distant pair (cool blue x warm orange); FF3D and LM 1.75 m are
    # already separated by position, well below the line.
    AZUL4, LARANJA4, VERDE4, VIOLETA4 = "#2a78d6", "#eb6834", "#0f7a54", "#4a3aa7"
    series = [
        ("LM 0.75 m", serie_lm("0.75 m window"), LARANJA4, "D"),
        ("LM 1.75 m", serie_lm("1.75 m window"), VIOLETA4, "^"),
        ("FF3D", ff.n_r400.reindex(campo.index), VERDE4, "s"),
        ("SAT", sat.adabn_r400, AZUL4, "o"),
    ]
    faltando = [n for n, v, _, _ in series if v.isna().any()]
    if faltando:
        raise SystemExit(f"fig4: incomplete series in {faltando}")
    fig, ax = plt.subplots(figsize=(IMP_FIG4, IMP_FIG4 * 0.98))
    # the axis does not start at zero: the plots range from 33 to 68 trees, and including the
    # origin would leave half the figure empty with the points squeezed into a corner. The
    # 45-degree line keeps the reference explicit.
    todos = np.concatenate([campo.to_numpy()] + [v.to_numpy() for _, v, _, _ in series])
    lo, hi = todos.min() - 4, todos.max() + 4
    ax.plot([lo, hi], [lo, hi], color="#AAAAAA", lw=0.8, ls="--", zorder=0)
    ax.text(lo + 3, lo + 1.2, "1:1", color="#888888", fontsize=7, rotation=45)
    for nome, v, cor, m in series:
        # no percentage in the legend: the axis is in counts and the percentage is a different
        # quantity, the ratio of the summed total against the 717 of the census. The pooled
        # values go in the figure caption and in the table.
        ax.scatter(campo, v, s=13, marker=m, color=cor, alpha=0.85,
                   edgecolor="white", linewidth=0.5, label=nome)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Field stems per plot")
    ax.set_ylabel("Detected per plot")
    # legend outside the axes: with four series it fits in no corner (upper left it touches
    # the diamond of the 53-tree plot, lower right the triangles), and reducing the font would
    # make the letter size disagree with the other figures.
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              ncol=2, handletextpad=0.2, borderpad=0.1, columnspacing=0.6)
    # the per-plot error goes in the table alongside the aggregate ratio: without it the swept
    # baseline looks tied with SegmentAnyTree, when it gets the total right and each plot
    # wrong.
    pd.DataFrame({"method": [n for n, _, _, _ in series],
                  "detected_total": [int(v.sum()) for _, v, _, _ in series],
                  "field_total": int(campo.sum()),
                  "ratio_pct": [round(100 * v.sum() / campo.sum(), 1) for _, v, _, _ in series],
                  "per_plot_MAE": [round(float((v - campo).abs().mean()), 1)
                                   for _, v, _, _ in series],
                  "per_plot_rMAE_pct": [round(float((v - campo).abs().mean() / campo.mean() * 100), 1)
                                        for _, v, _, _ in series],
                  }).to_csv(out / "table1_detection_counts.csv", index=False)
    print("  table1_detection_counts.csv")
    razao = {n: 100 * v.sum() / campo.sum() for n, v, _, _ in series}
    salvar(fig, out, "fig4_detected_vs_field",
           "Detected trees against the field stem count for the 13 inventory plots. Detections "
           "are counted within an 11.28 m radius, enclosing the same 400 m2 as the field plots. "
           "Both networks use nine plot-centred views with AdaBN, each merged at its own radius "
           "for that scheme, 1.1 and 1.5 m. Pooled over the 717 field stems the ratios are "
           + ", ".join(f"{n} {p:.0f}%" for n, p in razao.items()) + ". "
           "Two settings of the classical baseline are shown, each named by its search window, "
           "both under the same nine-view aggregation as the networks. The 0.75 m window is the "
           "best of a sweep scored by matched F1 on the stem map and it nearly matches the field "
           "count. Count agreement therefore separates these methods poorly, and the tree-to-tree "
           "comparison carries the claim, where at full density the 0.75 m window reaches a "
           "matched F1 of 0.81 against 0.92 for SegmentAnyTree.")


# ------------------------------------------------------------------ figure 3
def fig_raio(out):
    """Matched F1 against the merge radius, for both models and both tiling schemes."""
    import geopandas as gpd
    d = pd.read_csv(config.REPO / "data/detections/tta_t001_instancias.csv")
    d = d[(d.condicao == "adabn") & (d.posicao == "centroide")]
    fustes = gpd.read_file(config.REPO /
                           "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp")
    ref_all = np.column_stack([fustes.geometry.x, fustes.geometry.y])
    p = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    p["t"] = pd.to_numeric(p.talhao, errors="coerce")
    p["pa"] = pd.to_numeric(p.parcela, errors="coerce")
    C = {f"{int(r.t)}_{int(r.pa)}": (r.geometry.x, r.geometry.y)
         for _, r in p.dropna(subset=["t", "pa"]).iterrows()}
    meia = 32.0 / 2 - 3.0
    raios = np.arange(0.7, 2.61, 0.1)

    # legend above, and not in the right margin: with the entries to the right of the axes,
    # `bbox_inches="tight"` extends the figure to fit what lies outside it and the file comes
    # out almost twice as wide, with the plot taking two thirds. It is also what figures 4 and
    # 8 do.
    fig, ax = plt.subplots(figsize=(COL2, 3.0))
    for mod, cor in (("FF3D", AZUL), ("SegmentAnyTree", VERDE)):
        f1s = []
        for raio in raios:
            tp = npred = nref = 0
            for pid in ("1_1", "1_2"):
                g = d[(d.modelo == mod) & (d.plot_id == pid)]
                if g.empty:
                    continue
                xy = g[["cx", "cy"]].to_numpy(float)
                keep = nms_malha(xy, g.n_pts.to_numpy(float), raio)
                px, py = C[pid]
                dentro = lambda a: a[(np.abs(a[:, 0] - px) <= meia) & (np.abs(a[:, 1] - py) <= meia)]
                pr, rf = dentro(xy[keep]), dentro(ref_all)
                tp += parear(rf, pr, LIMIAR); npred += len(pr); nref += len(rf)
            prec, rev = tp / max(npred, 1), tp / max(nref, 1)
            f1s.append(2 * prec * rev / (prec + rev) if prec + rev else 0.0)
        f1s = np.array(f1s); i = int(f1s.argmax())
        # short label, because the three entries share a single line
        ax.plot(raios, f1s, color=cor, label=f"{mod}, plot-centred")
        ax.plot(raios[i], f1s[i], "o", color=cor, ms=5, mec="white", mew=0.8)
    w2w = config.OUT_DIR / "w2w_varredura_raio.csv"
    if w2w.exists():
        v = pd.read_csv(w2w).dropna(subset=["f1"]); i = int(v.f1.idxmax())
        ax.plot(v.raio, v.f1, color=VERDE, ls="--", label="SegmentAnyTree, continuous grid")
        ax.plot(v.raio[i], v.f1[i], "s", color=VERDE, ms=5, mec="white", mew=0.8)
    # the spacing label goes at the top, inside the axes, where the three curves have already
    # come down and there is room.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.06)
    ax.axvline(1.88, color=VERM, ls=":", lw=1.0)
    ax.text(1.93, ax.get_ylim()[1], "within-row spacing\n1.88 m", fontsize=6.5,
            color=VERM, va="top", ha="left", linespacing=1.25)
    ax.set_xlabel("Merge radius (m)")
    ax.set_ylabel("Matched F1 at 2 m")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.015), handletextpad=0.5, columnspacing=1.4)
    salvar(fig, out, "fig3_merge_radius_sweep",
           "Matched F1 as a function of the non-maximum-suppression merge radius on stand 001, "
           "both networks with AdaBN. Markers indicate the optimum of each curve, at 1.1, 1.5 and "
           "1.7 m. The optimum shifts with the network and again with the tiling scheme, since in "
           "the continuous grid a tree lies near the edge of several tiles at once. All three "
           "optima fall below the within-row planting spacing.")


# ------------------------------------------------------------------ figure 4
def fig_voo(out):
    """Count against density, FOUR methods, in the same stand.

    The axis is in percentage of the census, and not in relative decline: narrow-window local
    maxima reaches 325% of the census at 20 pts/m², and on a relative axis that would appear
    as "326% retention", read as robustness. With the 100% line marked, falling short and
    overshooting appear on the same ruler.

    There are four curves in order to separate stand from algorithm. Comparing FF3D in a
    closed stand against da Cunha Neto's local maxima in an open stand moves two variables at
    once; running the same family of classical algorithm on the same tiles closes the control.

    Local maxima appears in two settings. The wide window is flat with density, as in the
    reference, and finds only half the trees here; the narrow window finds far more and
    overestimates once the cloud thins. It is the closed stand that forces the narrow window.

    There is no per-tile normalisation. The angle restriction is marked separately because it
    is not spatially uniform, with 44x of density spread within the same condition.
    """
    fonte = config.OUT_DIR / "flight_study_tres_modelos.csv"
    if not fonte.exists():
        fonte = config.OUT_DIR / "flight_study_dois_modelos.csv"
    if not fonte.exists():
        print("  (skipping fig4: run exp_flight_study_dois_modelos.py first)")
        return
    d = pd.read_csv(fonte)
    ang = d.condition.str.startswith("ang")
    thin = d[~ang]

    curvas = [("lm", "CHM local maxima, 0.75 m window", CINZA, ":", "s"),
              ("lm_larga", "CHM local maxima, 1.75 m window", "#4A4A4A", "-.", "D"),
              ("ff3d", "FF3D", AZUL, "--", "o"),
              ("sat", "SegmentAnyTree", VERDE, "-", "o")]
    curvas = [c for c in curvas if f"{c[0]}_pct" in d.columns and d[f"{c[0]}_pct"].notna().any()]

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.9), sharey=True)
    for i, (eixo, subset, rot) in enumerate(
            [(axes[0], thin, "(a) Random thinning"),
             (axes[1], d[ang], "(b) Scan angle restricted")]):
        for col, rotulo, cor, tr, mk in curvas:
            # in panel (b) the thinning curve goes behind, faded: the claim about the angle
            # is that it detects less than thinning at the same density, and without it the
            # panel shows only three loose points.
            if i == 1:
                gt = (thin.groupby("condition")
                      .agg(x=("density_pts_m2", "median"), y=(f"{col}_pct", "mean"))
                      .sort_values("x"))
                eixo.plot(gt.x, gt.y, "-", color=cor, lw=0.8, alpha=0.28, zorder=1)
            g = (subset.groupby("condition")
                 .agg(x=("density_pts_m2", "median"), y=(f"{col}_pct", "mean"))
                 .sort_values("x"))
            eixo.plot(g.x, g.y, tr, marker=mk, color=cor, ms=3.5, label=rotulo, zorder=3)
        eixo.axhline(100, color="#BBBBBB", lw=0.7, ls="--", zorder=0)
        eixo.set_xscale("log"); eixo.set_yscale("log")
        eixo.set_xlabel("Effective point density (pts m$^{-2}$)")
        eixo.set_title(rot)
        eixo.set_yticks([10, 25, 50, 100, 200, 400])
        eixo.set_yticklabels(["10", "25", "50", "100", "200", "400"])
    axes[1].set_xlim(axes[0].get_xlim())     # same scale, otherwise "below" cannot be read
    axes[0].set_ylabel("Detected trees, as % of\nthe field stem count")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.19))

    cheio = {c[0]: d[d.condition == "full"][f"{c[0]}_pct"].mean() for c in curvas}
    salvar(fig, out, "fig6_point_density_degradation",
           "Detected trees as a percentage of the field stem count, against effective point "
           "density, for four methods on the same 13 plots and the same degraded tiles. The dashed "
           "line at 100% is exact agreement with the field census; a method fails by falling below "
           "it or by rising above it. All four run one pass per tile with no view overlap and no "
           "adaptation, so the levels are not those of the complete system; at full density the "
           f"networks reach {cheio.get('ff3d', float('nan')):.0f}% and "
           f"{cheio.get('sat', float('nan')):.0f}% here, against 88% and 103% in Figs. 4 and 5. "
           "The two local-maxima settings differ only in the search window, both chosen on the "
           "full-density data alone. The wide window is nearly flat with density but recovers only "
           f"{cheio.get('lm_larga', float('nan')):.0f}% of the stems in this closed stand, while "
           "the narrow window recovers far more and then over-counts severely once the cloud is "
           "thinned. (b) The same methods on scan-angle-restricted tiles, with each method's "
           "thinning curve from (a) repeated as a faint line on the same axes. Every method falls "
           "below its own curve at equal density, so restricting the angle costs more than "
           "removing the same number of points at random.")


# ------------------------------------------------------------------ figure 8
def fig_casado_voo(out):
    """Precision and recall against density, in stand 001, with matching.

    It complements figure 4, whose axis is a count, and counting cancels errors: losing one
    tree and inventing another leaves the total intact. Wide-window local maxima looks flat
    there, and without matching one cannot tell whether it finds the same trees or swaps trees
    at each density level.

    Here each detection is matched to a stem of the TLS map, so that omission and commission
    appear separately.

    Only stand 001 has a stem map, so there are two tiles, against the 13 plots of figure 4;
    the two captions record the scope of each.
    """
    fonte = config.OUT_DIR / "flight_study_casado_talhao001.csv"
    if not fonte.exists():
        print("  (skipping fig8: run exp_flight_study_casado.py first)")
        return
    d = pd.read_csv(fonte)
    d = d[~d.condition.str.startswith("ang")]      # thinning only, otherwise density is not comparable
    estilo = {
        # the labels are the same as those used in the tables, to allow cross-referencing
        # between figure and table
        "maximo_local_0.75m": ("Local maxima, 0.75 m window", CINZA, ":", "s"),
        "maximo_local_1.75m": ("Local maxima, 1.75 m window", "#4A4A4A", "-.", "D"),
        "ff3d": ("FF3D", AZUL, "--", "o"),
        "segmentanytree": ("SegmentAnyTree", VERDE, "-", "o"),
    }

    # panels stacked for page economy: side by side the figure needs 7.16 in, which under
    # IEEEtran forces `figure*` and costs 236 mm of column, while stacked it fits in
    # `\columnwidth` for 115 mm at the same font, since each panel then spans the full column
    # width. Shrinking the side-by-side version would leave the axes at around 4 pt.
    fig, axes = plt.subplots(2, 1, figsize=(COL1, 3.5), sharex=True)
    for eixo, campo, rot in [(axes[0], "revocacao", "(a) Recall"),
                             (axes[1], "precisao", "(b) Precision")]:
        for met, (rotulo, cor, tr, mk) in estilo.items():
            g = (d[d.metodo == met].groupby("condition")
                 .agg(x=("density_pts_m2", "median"), y=(campo, "mean")).sort_values("x"))
            if g.empty:
                continue
            eixo.plot(g.x, g.y, tr, marker=mk, color=cor, ms=3.5, label=rotulo)
        eixo.set_xscale("log")
        eixo.set_ylim(0, 1.05)
        eixo.set_ylabel("Matched fraction")
        eixo.set_title(rot, fontsize=8, loc="left")
    axes[1].set_xlabel("Effective point density (pts m$^{-2}$)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.10), handlelength=2.0, columnspacing=1.0)

    salvar(fig, out, "fig7_matched_metric_vs_density",
           "Recall and precision against effective point density in stand 001, where a terrestrial "
           "laser stem map allows tree-to-tree matching. Detections are matched one-to-one by "
           "Hungarian assignment at a 2 m threshold inside a 26 m evaluation square, over the two "
           "tiles with stem-map coverage. This resolves what Fig. 6 cannot, since a count can hold "
           "steady while omissions and false detections cancel. The wide local-maxima window is "
           "genuinely density-invariant, holding recall near 0.55 and precision near 0.99 across a "
           "27-fold reduction in density, but it never exceeds that recall in this closed stand. "
           "The networks start higher and decline, and the two curves cross at about "
           "100 pts m$^{-2}$, below which SegmentAnyTree recovers fewer stems than the classical "
           "method. The narrow window inverts: its recall "
           "rises towards 1.0 as thinning proceeds while its precision collapses to 0.41, which is "
           "a detector saturating the plot rather than finding trees.")


# ------------------------------------------------------------------ figure 5
def fig_mapa(out):
    """Map of the detected trees, an overview plus a magnified detail.

    The edge band is a region, and is therefore drawn as a band: colouring the points less
    than 3 m from the boundary produces, with a 0.9 pt marker, a thin outline indistinguishable
    from the polygon line.
    """
    import geopandas as gpd
    d = pd.read_csv(config.REPO / "data/detections/sat_w2w_arvores.csv")
    a = gpd.read_file(config.DATA / "2-shapes/Areas_plantio/area_plantio.shp")
    a["t"] = pd.to_numeric(a.Talhao, errors="coerce")
    sub = a[a.t.isin(sorted(d.talhao.unique()))].copy()
    x0, y0, x1, y1 = sub.total_bounds

    fig = plt.figure(figsize=(COL2, 4.0))
    # the cell proportions follow the data proportions of each panel: both use equal aspect
    # and, with an arbitrary width_ratios, each box shrinks differently inside its cell and the
    # panel baselines end up at different heights.
    # generous wspace: the 005 label sits at the right edge of panel (a) and touches the axis
    # numbers of panel (b).
    asp_a = (x1 - x0 + 295) / (y1 - y0 + 85)      # limits of panel (a), below
    gs = fig.add_gridspec(1, 2, width_ratios=[asp_a, 1.0], wspace=0.13)

    ax = fig.add_subplot(gs[0])
    # the 3 m band is not drawn: in a stand ~100 m wide printed at 3 inches it measures less
    # than the thickness of the polygon line. The value it represents, 12.9% of the trees, goes
    # in the caption and in the table.
    sub.boundary.plot(ax=ax, color="#333333", lw=0.7)
    ax.scatter(d.base_x, d.base_y, s=0.5, color=VERDE, alpha=0.55, linewidths=0)
    # the axis limits come before the labels, because deciding where each label fits requires
    # knowing where the axis ends
    ax.set_aspect("equal", adjustable="box")
    lim_esq, lim_dir = x0 - 60, x1 + 235
    ax.set_xlim(lim_esq, lim_dir); ax.set_ylim(y0 - 30, y1 + 55)

    # Three positions, in this order: to the right is the default; if another stand is right
    # beside it, the label goes left; if on the left the text would leave the axes, it goes
    # above. 004 falls in the third case, with 006 blocking the right and a short left margin.
    LARG = 175.0        # approximate width of "n = 2,811" at this scale, in metres
    for _, r in sub.iterrows():
        n = int((d.talhao == int(r.t)).sum())
        b = r.geometry.bounds
        meio = (b[1] + b[3]) / 2
        viz = any(o.t != r.t and 0 < o.geometry.bounds[0] - b[2] < 170
                  and o.geometry.bounds[1] < meio < o.geometry.bounds[3]
                  for _, o in sub.iterrows())
        if not viz:
            pos, ha, va = (b[2] + 20, meio), "left", "center"
        elif b[0] - 20 - LARG > lim_esq:
            pos, ha, va = (b[0] - 20, meio), "right", "center"
        else:
            pos, ha, va = ((b[0] + b[2]) / 2, b[3] + 18), "center", "bottom"
        ax.annotate(f"{int(r.t):03d}\nn = {n:,}", pos, ha=ha, va=va,
                    fontsize=6.8, color="#222222", linespacing=1.25)
    ax.set_title("(a) Six stands, 13,946 detected trees", loc="left")
    eixo_em_metros(ax, x0, y0, 200)

    ax2 = fig.add_subplot(gs[1])
    g5 = d[d.talhao == 5]
    L = 15.0
    melhor, alvo = -1, (g5.base_x.median(), g5.base_y.median())
    for _, c in g5.sample(min(400, len(g5)), random_state=0).iterrows():
        m = (g5.base_x.between(c.base_x - L, c.base_x + L) &
             g5.base_y.between(c.base_y - L, c.base_y + L))
        if int(m.sum()) > melhor:
            melhor, alvo = int(m.sum()), (c.base_x, c.base_y)
    cx, cy = alvo
    z = g5[g5.base_x.between(cx - L, cx + L) & g5.base_y.between(cy - L, cy + L)]
    ax2.scatter(z.base_x, z.base_y, s=15, color=VERDE, alpha=0.9,
                edgecolor="white", linewidth=0.4)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlim(cx - L, cx + L); ax2.set_ylim(cy - L, cy + L)
    for sp in ax2.spines.values():
        sp.set_color("#888888"); sp.set_visible(True)
    ax2.set_title(f"(b) 30 m detail, stand 005, n = {len(z)}", loc="left")
    # with a y label: omitting it is only valid when the axis is genuinely shared, and here
    # panel (b) has its own system, 0 to 30 m with the origin at a different corner.
    eixo_em_metros(ax2, cx - L, cy - L, 10)

    salvar(fig, out, "fig8_wall_to_wall_map",
           "(a) Stem-base positions of the 13,946 trees detected across the six stands with "
           "delivered point clouds, 9.87 ha in total, where n is the tree count of each stand. "
           "Point clouds are clipped to the stand polygon, so crowns within about 3 m of the "
           "boundary are truncated; those trees are 12.9% of the total. (b) A 30 m "
           "detail inside stand 005, shown because at the scale of (a) each stand renders as a "
           "solid mass and no individual detection is legible. The detections show no "
           "directional structure: the azimuth distribution of near neighbours peaks at "
           "1.44 times uniform, within the 1.31 to 1.53 range of randomly placed points, "
           "so the planting rows are not recovered.")


# ------------------------------------------------------------------ figure 6
def fig_qualitativa(out, tile=None):
    """Point cloud coloured by instance, in plan and in profile.

    It is the only place in the paper where the model output is seen, rather than a number
    about it.

    Two panels: the plan view shows the arrangement of the crowns, which is what the airborne
    view resolves, and the profile shows the vertical structure, where the suppressed tree
    appears with no distinct apex, which supports the occlusion argument.

    The per-instance colour is categorical and cyclic. Two distant trees may receive the same
    colour; neighbours may not, which is why the shuffling uses a fixed seed instead of
    identifier order, which would come out in bands of similar colour.
    """
    import laspy
    if tile is None:
        cands = sorted((config.REPO / "work/sat_showcase").glob("*_out.laz"))
        if not cands:
            cands = sorted((config.REPO / "work/sat_flight_out/full").glob("*_out.laz"))
        if not cands:
            print("  (skipping fig6: no SAT output available)")
            return
        tile = cands[0]
    las = laspy.read(str(tile))
    x, y, z = (np.asarray(las.x), np.asarray(las.y), np.asarray(las.z))
    inst = np.asarray(las.PredInstance)
    x, y = x - x.min(), y - y.min()

    # the percentages are measured here, and not typed into the caption, so that they cannot
    # diverge silently when the tile changes.
    # the ground class comes from PredSemantic, the model output, and not from the LAS
    # `classification` field, which the SegmentAnyTree pipeline returns zeroed; checking via
    # `classification` gives 0% ground.
    sem = np.asarray(las.PredSemantic)
    livre = inst <= 0
    pct_livre = 100 * livre.mean()
    pct_chao = 100 * (livre & (sem == 0)).mean()
    pct_veg = 100 * (livre & (sem != 0)).mean()

    rng = np.random.default_rng(7)
    ids = np.unique(inst[inst > 0])
    cmap = plt.get_cmap("tab20")
    cor = {i: cmap(k % 20) for k, i in enumerate(rng.permutation(ids))}
    cores = np.array([cor.get(i, (0.72, 0.72, 0.72, 1.0)) for i in inst])
    solo = inst <= 0

    # subsampled so the file stays light; the visual reading does not change
    n = len(x)
    sel = rng.choice(n, min(90_000, n), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.9),
                             gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.18})
    a1 = axes[0]
    a1.scatter(x[sel][solo[sel]], y[sel][solo[sel]], s=0.4, color="#CFCFCF", linewidths=0)
    m = ~solo[sel]
    a1.scatter(x[sel][m], y[sel][m], s=0.4, c=cores[sel][m], linewidths=0)
    a1.set_aspect("equal"); a1.grid(False)
    a1.set_xlabel("Distance east (m)"); a1.set_ylabel("Distance north (m)")
    a1.set_title(f"(a) Plan view, {len(ids)} instances")

    # profile of a central slice, otherwise the trees at the back cover those in front
    faixa = (y > y.max() / 2 - 4) & (y < y.max() / 2 + 4)
    idx = np.flatnonzero(faixa)
    idx = rng.choice(idx, min(45_000, len(idx)), replace=False)
    a2 = axes[1]
    a2.scatter(x[idx][inst[idx] <= 0], z[idx][inst[idx] <= 0], s=0.6,
               color="#CFCFCF", linewidths=0)
    mm = inst[idx] > 0
    a2.scatter(x[idx][mm], z[idx][mm], s=0.6, c=cores[idx][mm], linewidths=0)
    a2.set_aspect("equal"); a2.grid(False)
    a2.set_xlabel("Distance east (m)"); a2.set_ylabel("Height above ground (m)")
    a2.set_title("(b) Profile, 8 m slice")
    salvar(fig, out, "fig2_instance_segmentation",
           "Point cloud of one 32 m tile coloured by predicted tree instance, grey for points the "
           "network assigned to ground or to no instance. (a) Plan view, which is what the airborne "
           "sensor resolves best. (b) Vertical profile of a central 8 m slice, where suppressed "
           "trees without a distinct crown apex are visible beneath the dominant canopy. In this "
           f"tile {pct_livre:.1f}% of the points carry no instance, but {pct_chao:.1f} percentage "
           f"points of that are classified as ground by the network itself, leaving {pct_veg:.1f}% "
           "of vegetation unassigned. Colours are categorical and repeat; "
           "they identify neighbouring instances, not tree properties.")


# ------------------------------------------------------------------ figure 7
def fig_fluxo(out):
    """Pipeline diagram, with the parameter that defines each stage inside the block.

    Each box carries the parameter that defines it, so that the chain is reproducible without
    leaving the figure.

    The layout is vertical, and not horizontal, because of page cost: a 7.16 in band forces
    `figure*` under IEEEtran, which in practice cost a whole page of the paper, and not the
    0.26 that the figure area suggested, because the float pushes the text ahead. Stacked, the
    figure fits in `\columnwidth` at the same font, 7.4 pt in the title and 6.0 pt in the
    parameter, since the box then spans the full column width; shrinking the horizontal band
    would leave the internal text at around 4 pt.

    The height is derived, and not chosen by hand: `figsize` comes from the total in axis units
    times the column scale, otherwise there is too much or too little margin and the top or
    bottom box gets clipped.

    `boxstyle="round,pad=..."` draws the fill outside the requested rectangle, so that the box
    starting at the axis limit is drawn beyond it and the crop eats the border. Slack in
    `xlim`/`ylim` and `clip_on=False` are both needed, because slack alone depends on the unit
    of `pad` coinciding with the unit of the data.

    Both grid strides appear in the diagram: 10.67 m is the whole-stand scheme and 16 m is the
    plot-centred scheme used in the other figures.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    # (title, parameter, is this a stage added by this work?)
    passos = [
        ("Normalised\npoint cloud",   "6 stands, 9.87 ha\n418 to 1112 pts m$^{-2}$",    False),
        ("Overlapping\ntiles",        "32 m tiles, 16 or 10.67 m step\n9 views per tree", True),
        ("Instance\nsegmentation",    "pretrained network, no fine-tuning\nAdaBN, no labels", True),
        ("Global merge\nper stand",   "greedy NMS, radius per model\nand per tiling scheme", True),
        ("Tree count\nand positions", "13,946 trees, checked against\n13 field plots", False),
    ]
    n = len(passos)
    W, h, gap = 10.0, 1.22, 0.46           # axis units; W maps to COL1
    FOLGA = 0.12                            # room for the boxstyle pad
    XT, XDIV, XP = 0.42, 3.55, 3.95         # title, divider, parameter
    alt = n * h + (n - 1) * gap

    fig, ax = plt.subplots(figsize=(COL1, COL1 * (alt + 2 * FOLGA) / W))
    ax.set_xlim(-FOLGA, W + FOLGA); ax.set_ylim(-FOLGA, alt + FOLGA); ax.axis("off")
    for k, (tit, par, nosso) in enumerate(passos):
        y = alt - (k + 1) * h - k * gap     # from top to bottom
        ax.add_patch(FancyBboxPatch((0, y), W, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor="#EDF3EF" if nosso else "#F4F4F2",
                                    edgecolor=VERDE if nosso else "#9A9A93",
                                    linewidth=1.0 if nosso else 0.8, clip_on=False))
        ax.plot([XDIV, XDIV], [y + 0.16, y + h - 0.16], color="#C3D2C8" if nosso else "#DCDCD6",
                linewidth=0.8, clip_on=False)
        ax.text(XT, y + h / 2, tit, ha="left", va="center",
                fontsize=7.4, weight="bold", linespacing=1.35)
        ax.text(XP, y + h / 2, par, ha="left", va="center",
                fontsize=6.0, color="#444444", linespacing=1.45)
        if k < n - 1:
            ax.add_patch(FancyArrowPatch((W / 2, y - 0.04), (W / 2, y - gap + 0.04),
                                         arrowstyle="-|>", mutation_scale=8,
                                         color="#666666", linewidth=0.9))
    salvar(fig, out, "fig1_pipeline",
           "Segmentation chain from the raw point cloud to the tree count, with the parameter "
           "that defines each stage. The two networks enter as pretrained inference and the chain "
           "contains no training stage. Shaded blocks mark where this work intervenes, by tiling the "
           "cloud into overlapping views, adapting the normalisation statistics without labels, "
           "and merging the resulting instances; the networks themselves are used as published.")


# ------------------------------------------------------------------ figure 9
def fig_casada_mapa(out):
    """Our detections against the 892 stems of the TLS map, tree by tree.

    It is the only figure in the document that compares the result with independent field
    truth, and not with another number produced here.

    The emphasis is on failure, and not on success: with 850 pairs among 892 stems, drawing
    everything alike produces a green carpet. Missed stems and unmatched detections carry a
    visible marker, and the matched ones sit as background.

    The evaluation area is drawn because, without it, the diagonal band of unmatched
    detections in the upper-left corner reads as false positives, when it is where the TLS did
    not scan.

    The clip is symmetric, being the convex hull of the stems inset by the matching threshold
    itself. A rectangle would include 9% of uncovered area, and clipping only the detections
    would penalise precision without penalising recall; insetting 2 m removes the band where a
    stem could be matched by a detection coming from outside the area.
    """
    import geopandas as gpd
    from scipy.optimize import linear_sum_assignment
    from shapely.geometry import MultiPoint, Point

    mapa = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
    f = gpd.read_file(mapa)
    REF = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    d = pd.read_csv(config.REPO / "data/detections/sat_w2w_arvores.csv")
    PRED = np.column_stack([d[d.talhao == 1].base_x.values, d[d.talhao == 1].base_y.values])

    casco = MultiPoint([Point(*p) for p in REF]).convex_hull
    area = casco.buffer(-LIMIAR)
    ent = lambda pts: np.array([area.contains(Point(*p)) for p in pts])
    mr, mq = ent(REF), ent(PRED)
    ref, pred = REF[mr], PRED[mq]

    D = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    li, ci = linear_sum_assignment(np.where(D <= LIMIAR, D, 1e6))
    ok = D[li, ci] <= LIMIAR
    cr = np.zeros(len(ref), bool); cr[li[ok]] = True
    cq = np.zeros(len(pred), bool); cq[ci[ok]] = True
    tp = int(ok.sum())
    prec, rev = tp / len(pred), tp / len(ref)

    # the edge effect is measured here, and not asserted in the caption: twelve of the 34
    # missed stems are in the interior, so only the per-band rate describes the effect.
    db = np.array([area.exterior.distance(Point(*p)) for p in ref])
    perto = db < 3.0
    borda = 100 * (~cr & perto).sum() / max(perto.sum(), 1)
    miolo = 100 * (~cr & ~perto).sum() / max((~perto).sum(), 1)

    lo, hi = REF.min(0), REF.max(0)
    fig, ax = plt.subplots(figsize=(COL2, COL2 * (hi[1] - lo[1]) / (hi[0] - lo[0]) + 0.6))
    # outside the evaluated area enters as faded context, to show that there is data there
    # and that it was deliberately left out
    ax.scatter(*REF[~mr].T, s=2.5, color="#C9D3CD", linewidths=0)
    ax.scatter(*PRED[~mq].T, s=6, marker="x", color="#DADADA", linewidths=0.5)
    ax.plot(*area.exterior.xy, color="#555555", lw=0.8, ls="--",
            label="Evaluation area")
    ax.scatter(*ref[cr].T, s=3.5, color=VERDE, linewidths=0, alpha=0.8,
               label=f"Matched stem ({tp})")
    ax.scatter(*pred[~cq].T, s=16, marker="x", color=CINZA, linewidths=0.7,
               label=f"Detection with no stem ({int((~cq).sum())})")
    ax.scatter(*ref[~cr].T, s=26, marker="o", facecolor="none", edgecolor=VERM,
               linewidths=0.9, label=f"Missed stem ({int((~cr).sum())})")
    ax.set_aspect("equal"); ax.grid(False)
    eixo_em_metros(ax, lo[0], lo[1], 20)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.20),
              handletextpad=0.2, columnspacing=1.1, scatterpoints=1)

    # table 3 comes from the experiment CSV, and is not recomputed here: the figure draws one
    # of the four evaluation areas and the experiment computes all four. Recomputing would
    # create a second implementation of the same calculation, liable to diverge from it.
    fonte = config.OUT_DIR / "casada_talhao001_completo.csv"
    if fonte.exists():
        v = pd.read_csv(fonte)
        v = v[v.limiar_m == LIMIAR].copy()
        v["evaluation_area"] = v.recuo_m.map(
            lambda x: "none" if str(x) == "sem recorte" else
            "convex hull" if float(x) == 0 else f"convex hull inset {float(x):.0f} m")
        v = v.rename(columns={"ref": "stems", "pred": "detections", "TP": "matched",
                              "precisao": "precision", "revocacao": "recall"})
        v[["evaluation_area", "stems", "detections", "matched",
           "precision", "recall", "F1"]].round(3).to_csv(
            out / "table3_stem_map_comparison.csv", index=False)
        print("  table3_stem_map_comparison.csv")
    else:
        print("  (no table3: run exp_casada_talhao001_completo.py first)")

    salvar(fig, out, "supp_evaluation_area_stand001",
           "Wall-to-wall detections in stand 001 compared tree by tree with the terrestrial laser "
           f"stem map, matched one-to-one by Hungarian assignment at a {LIMIAR:.0f} m threshold. The "
           "stem map came from automatic segmentation of the terrestrial cloud followed by manual "
           "correction, so it is independent of the airborne data evaluated here and includes stems "
           "the airborne sensor cannot reach. The terrestrial survey covered a strip rather than the "
           "whole stand, so the evaluation area is the convex hull of the stem map inset by the "
           "matching threshold, which keeps the comparison symmetric and removes the border band "
           "where a stem could be matched by a detection from outside. Faded marks lie outside that "
           f"area and are excluded. Inside it there are {len(ref)} stems and {len(pred)} detections, "
           f"giving a recall of {rev:.3f} and a precision of {prec:.3f}. Misses are more "
           f"frequent near the boundary, {borda:.1f}% of the stems within 3 m of it against "
           f"{miolo:.1f}% further in, which is where the airborne cloud ends and the crown of "
           "the outermost row is truncated.")


def fig_talhao_completo(out: Path):
    """Figure 9. The whole of stand 001, with no clip, with both outlines.

    The evaluation-area figure is the supplementary one: the two draw the same scene, and
    this one shows the whole stand, the boundary of the reference and the separation of the
    false positives. The dashed outline is the same hull that bounds the evaluation area
    there, up to the 2 m inset.

    There are two outlines. The outer one is the stand polygon, the same one the inventory
    uses; the inner one is how far the field map reaches, 79% of the stand. A detection
    between the two is inside the stand and outside the reference, so it cannot be counted as
    an error.

    False positives are split into inside and outside the map, with counts in the legend: the
    raw number of 212 invites a wrong conclusion, because most of them fall where no reference
    exists to contradict them.
    """
    import geopandas as gpd
    from scipy.optimize import linear_sum_assignment
    from shapely.geometry import MultiPoint, Point

    mapa = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
    f = gpd.read_file(mapa)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    d = pd.read_csv(config.REPO / "data/detections/sat_w2w_arvores.csv")
    pred = np.column_stack([d[d.talhao == 1].base_x.values, d[d.talhao == 1].base_y.values])
    poli = gpd.read_file(config.DATA / "2-shapes/Areas_plantio/area_plantio.shp")
    talhao = poli.query("Talhao == '001'").geometry.iloc[0]
    casco = MultiPoint([Point(*p) for p in ref]).convex_hull

    D = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    li, ci = linear_sum_assignment(np.where(D <= LIMIAR, D, 1e6))
    ok = D[li, ci] <= LIMIAR
    achou = np.zeros(len(ref), bool); achou[li[ok]] = True
    casou = np.zeros(len(pred), bool); casou[ci[ok]] = True
    sem = ~casou
    fora = np.array([not casco.contains(Point(*p)) for p in pred])
    tp, fn = int(achou.sum()), int((~achou).sum())
    n_fora, n_dentro = int((sem & fora).sum()), int((sem & ~fora).sum())
    cobre = 100 * casco.area / talhao.area

    # the edge effect inverts once the clip is removed, which is why it is recomputed here
    # instead of copied: in the inset evaluation area the loss was 9.2% at the edge against
    # 3.3% in the interior, and with no clip it drops to 1.6% at the edge against 3.5% in the
    # interior. The effect was an artefact of the clip, which discarded the outside detections
    # able to match the edge stems.
    db = np.array([casco.exterior.distance(Point(*p)) for p in ref])
    perto = db < 3.0
    borda = 100 * (~achou & perto).sum() / max(perto.sum(), 1)
    miolo = 100 * (~achou & ~perto).sum() / max((~perto).sum(), 1)

    lo = np.array(talhao.bounds[:2]); hi = np.array(talhao.bounds[2:])
    # the height is dominated by the legend, and not by the data: at an aspect of 0.32 the map
    # itself takes 2.3 in of the 3.3 in printed. A one-line legend with short labels and 2 m of
    # slack instead of 5 give back about 0.45 in of page without touching the data.
    # The default is a single column, with the legend stacked below and to the left instead of
    # the horizontal strip on top: it saves ~45% of page space, because the text then runs
    # beside it, and the stems remain separable, at 0.8 mm apart on paper.
    # GV_FIG9_DUAS_COLUNAS=1 restores the wide version.
    import os as _os
    UMA = _os.environ.get("GV_FIG9_DUAS_COLUNAS") != "1"
    LARG = IMP_FIG9 if UMA else COL2
    fig, ax = plt.subplots(figsize=(LARG, LARG * (hi[1] - lo[1]) / (hi[0] - lo[0])
                                    + (1.30 if UMA else 0.52)))
    ax.plot(*talhao.exterior.xy, color="#333333", lw=1.0,
            label=f"Stand ({talhao.area / 1e4:.2f} ha)")
    ax.plot(*casco.exterior.xy, color="#8899AA", lw=0.8, ls="--",
            label=f"Mapped extent ({cobre:.0f}%)")
    ax.scatter(*pred[sem & fora].T, s=13, marker="x", color="#C4C4C4", linewidths=0.5,
               label=f"Unmapped, no stem ({n_fora})")
    ax.scatter(*pred[sem & ~fora].T, s=16, marker="x", color=LARANJA, linewidths=0.75,
               label=f"Mapped, no stem ({n_dentro})")
    ax.scatter(*ref[achou].T, s=3.5, color=VERDE, linewidths=0, alpha=0.85,
               label=f"Matched stem ({tp})")
    ax.scatter(*ref[~achou].T, s=26, marker="o", facecolor="none", edgecolor=VERM,
               linewidths=0.9, label=f"Missed stem ({fn})")
    ax.set_aspect("equal"); ax.grid(False)
    ax.set_xlim(lo[0] - 2, hi[0] + 2); ax.set_ylim(lo[1] - 2, hi[1] + 2)
    eixo_em_metros(ax, lo[0], lo[1], 20)
    if UMA:
        # the offset is a fraction of the axes height, and these axes are flat: at an aspect
        # of 0.32 the box is ~0.65 in tall and -0.30 amounts to 0.2 in, less than the x-axis
        # label, which makes the legend land on top of "Distance east (m)".
        ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(-0.02, -0.50),
                  handletextpad=0.3, labelspacing=0.25, columnspacing=0.8,
                  scatterpoints=1)
    else:
        ax.legend(frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.17),
                  handletextpad=0.2, columnspacing=0.9, scatterpoints=1)

    # larger pad in the single-column variant: the y-axis label is rotated and rises above the
    # plot area, which in this variant is short, and the `tight` crop cuts the "(m)" at the
    # top. Measured with scripts/checa_bordas_figuras.py, 37 px of ink on the first row.
    salvar(fig, out, "fig9_full_stand_stand001",
           "The whole of stand 001 with no evaluation clip, which is the view that answers how many "
           f"of the {len(ref)} stems in the field map were found. Matching is one-to-one by Hungarian "
           f"assignment at a {LIMIAR:.0f} m threshold, giving {tp} matched stems and {fn} missed. Two "
           "boundaries are drawn because precision needs one and the choice changes it. The solid "
           "line is the stand polygon used by the inventory and the dashed line is how far the "
           f"terrestrial survey reached, {cobre:.0f}% of the stand. Of the {int(sem.sum())} "
           f"detections with no matching stem, {n_fora} fall between the two boundaries, inside the "
           "stand but outside the reference, where no stem exists to contradict them; the remaining "
           f"{n_dentro} fall inside the mapped extent and are detector error. Recall is therefore "
           f"{tp / len(ref):.3f} regardless of the clip, while precision is only meaningful once a "
           f"boundary is fixed. With no clip, misses do not concentrate at the edge of the "
           f"mapped extent, {borda:.1f}% of the stems within 3 m of it against {miolo:.1f}% "
           "further in; the higher edge loss reported inside a clipped evaluation area is an "
           "effect of the clip, which discards the outside detections that would match those stems.",
           pad=(0.20 if UMA else 0.02))


def fig_prova_tls(out: Path):
    """Figure 10. The detections over the terrestrial laser cloud, in plan and in section.

    It is the only figure in which the reference itself appears: the others compare number
    with number, and here the detection is shown over the stem recorded by the other sensor.

    The matching link is necessary: the position estimated from the crown lands 0.6 m from the
    stem, and without the link drawn the circle looks wrong.

    The window was chosen by laser coverage, and the caption records that. The criterion is the
    density of the terrestrial cloud measured in the breast-height slice, and not detection
    performance.

    The label is "unmatched", and not "crown split in two": the mechanism is consistent with
    what is measured but is not demonstrated, and the 1.88 m spacing prevents demonstrating it
    by proximity.

    The stem is highlighted by a threshold calibrated on the stem map: a 10 cm cell with 400
    points or more in the breast-height slice produces ~176 clusters against 155 mapped stems
    in the window.

    Input: config.TLS_LAS, the terrestrial cloud that lives outside git.
    """

    import geopandas as gpd
    import laspy
    from scipy.ndimage import distance_transform_edt, uniform_filter
    from scipy.optimize import linear_sum_assignment

    tls = config.TLS_LAS
    if not tls.exists():
        print(f"  (no fig10: TLS cloud not found at {tls})")
        return

    CX, CY = 749048.9, 7480012.5      # centre chosen for laser coverage
    MX, MY = 18.0, 13.0               # window read from the cloud
    # a wide and short crop for page reasons: at 16 by 11 m the figure comes out almost square,
    # does not fit in what remains of the page and floats ahead, leaving half a page blank. At
    # 24 by 11 m it takes the shape a two-column figure calls for.
    ZX, ZY = 12.0, 5.5                # crop of the plan view, where the stem is visible
    FAIXA, GS, TRONCO = 3.0, 0.10, 400
    H0, H1 = 0.8, 2.5

    mapa = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
    f = gpd.read_file(mapa)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    d = pd.read_csv(config.REPO / "data/detections/sat_w2w_arvores.csv")
    pred = np.column_stack([d[d.talhao == 1].base_x.values, d[d.talhao == 1].base_y.values])

    xs, ys, zs = [], [], []
    with laspy.open(tls) as fh:
        for pt in fh.chunk_iterator(4_000_000):
            x, y, z = np.asarray(pt.x), np.asarray(pt.y), np.asarray(pt.z, dtype=np.float32)
            m = (abs(x - CX) < MX) & (abs(y - CY) < MY)
            if m.any():
                xs.append(x[m]); ys.append(y[m]); zs.append(z[m])
    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)

    g = 1.0
    nx, ny = int(2 * MX / g) + 2, int(2 * MY / g) + 2
    x0, y0 = CX - MX, CY - MY
    zmin = np.full((nx, ny), 1e9, np.float32)
    np.minimum.at(zmin, (((x - x0) / g).astype(int), ((y - y0) / g).astype(int)), z)
    zmin[zmin > 1e8] = np.nan
    vazio = ~np.isfinite(zmin)
    if vazio.any():                    # without filling, the slice vanishes at the edges
        _, (a, b) = distance_transform_edt(vazio, return_indices=True)
        zmin[vazio] = zmin[a[vazio], b[vazio]]
    zsolo = uniform_filter(zmin, 5)
    h = z - zsolo[np.clip(((x - x0) / g).astype(int), 0, nx - 1),
                  np.clip(((y - y0) / g).astype(int), 0, ny - 1)]

    peito = (h > H0) & (h < H1)
    NX, NY = int(2 * MX / GS), int(2 * MY / GS)
    grade = np.zeros((NX, NY), np.int32)
    np.add.at(grade, (np.clip(((x[peito] - x0) / GS).astype(int), 0, NX - 1),
                      np.clip(((y[peito] - y0) / GS).astype(int), 0, NY - 1)), 1)

    D = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    li, ci = linear_sum_assignment(np.where(D <= LIMIAR, D, 1e6))
    de_quem = {int(j): int(i) for i, j in zip(li, ci) if D[i, j] <= LIMIAR}
    # the legend accounting has to balance. The matching runs over the whole window, on
    # purpose, otherwise a stem at the edge of the crop would lose its detection by an accident
    # of framing; counting "matched" among the crop detections and "stems" among the crop stems
    # are two independent sets, and produces more matched than stems, impossible in a
    # one-to-one matching.
    #
    # Hence matched means the pair with both ends inside the crop: that way matched <= stems
    # always, and detections = matched + no-stem-in-view.
    zr = (abs(ref[:, 0] - CX) < ZX) & (abs(ref[:, 1] - CY) < ZY)
    zp = (abs(pred[:, 0] - CX) < ZX) & (abs(pred[:, 1] - CY) < ZY)
    idp = np.flatnonzero(zp)
    # orfa = detection in the crop with no visible partner: either it did not match, or it
    # matched a stem outside the frame. The two cases are indistinguishable in the figure, and
    # "no stem in view" describes both.
    orfa = np.array([(int(j) not in de_quem) or (not zr[de_quem[int(j)]])
                     for j in idp], dtype=bool)

    # height measured on the page: at 0.70 and an aspect of 8.0 the figure came out 5.92 in
    # tall in print, 61% of the usable page height of IEEEtran (9.63 in), and more than 75%
    # with the caption. The plan panel is locked by the equal aspect of a 24 by 11 m window, so
    # what can be shortened is the profile, from an aspect of 8.0 to 3.0.
    # The hspace rises with it: shortening the profile brings the two panels closer in absolute
    # terms, and since hspace is a fraction of the axes height, keeping the fraction shrinks
    # the gap. 0.52 gives it back.
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL2, COL2 * 0.425),
                                 gridspec_kw={"height_ratios": [2 * ZY * 1.0, 3.1],
                                              "hspace": 0.55})
    ext = [0, 2 * MX, 0, 2 * MY]
    a1.imshow(np.where(grade > 0, 1, np.nan).T, origin="lower", cmap="Greys",
              vmin=0, vmax=7, extent=ext, interpolation="nearest")
    a1.imshow(np.where(grade >= TRONCO, 1, np.nan).T, origin="lower", cmap="Greys",
              vmin=0, vmax=1.05, extent=ext, interpolation="nearest")
    dpar = []
    for j in idp[~orfa]:
        i = de_quem[int(j)]
        a1.plot([ref[i, 0] - x0, pred[j, 0] - x0], [ref[i, 1] - y0, pred[j, 1] - y0],
                color="#8A6A3A", lw=0.7, zorder=2)
        dpar.append(D[i, j])
    n_par, n_orfa, n_ref = int((~orfa).sum()), int(orfa.sum()), int(zr.sum())
    assert n_par <= n_ref, f"matched {n_par} > stems {n_ref}, the accounting broke"
    assert n_par + n_orfa == len(idp), "detections do not balance"
    a1.plot([], [], color="#8A6A3A", lw=0.7, label="Match link")
    a1.scatter(pred[idp[~orfa], 0] - x0, pred[idp[~orfa], 1] - y0, s=52, marker="o",
               facecolor="none", edgecolor=LARANJA, linewidths=0.9, zorder=3,
               label=f"Detection, matched ({n_par})")
    a1.scatter(pred[idp[orfa], 0] - x0, pred[idp[orfa], 1] - y0, s=52, marker="o",
               facecolor="none", edgecolor="#9AA0A6", linewidths=0.8,
               linestyle=(0, (2, 1.6)), zorder=3,
               label=f"Detection, no stem in view ({n_orfa})")
    a1.scatter(ref[zr, 0] - x0, ref[zr, 1] - y0, s=22, marker="+", color=VERDE,
               linewidths=0.9, zorder=4, label=f"Field stem ({n_ref})")
    print(f"  fig10: {n_ref} stems, {len(idp)} detections = {n_par} matched "
          f"+ {n_orfa} with no stem in view; median link {np.median(dpar):.2f} m")
    zx, zy = MX - ZX, MY - ZY
    a1.set_aspect("equal"); a1.grid(False)
    a1.set_xlim(zx, zx + 2 * ZX); a1.set_ylim(zy, zy + 2 * ZY)
    a1.set_xticks(np.arange(zx, zx + 2 * ZX + 0.1, 6))
    a1.set_yticks(np.arange(zy, zy + 2 * ZY + 0.1, 4))
    a1.set_xticklabels([f"{v - zx:.0f}" for v in a1.get_xticks()])
    a1.set_yticklabels([f"{v - zy:.0f}" for v in a1.get_yticks()])
    a1.set_xlabel("Distance east (m)"); a1.set_ylabel("Distance north (m)")
    # three stacked heights, and moving one pushes the others: bottom to top, axes top,
    # legend and the (a) label. At 1.01 and 1.13 the (a) touches the legend and the legend
    # touches the plot; 1.07 and 1.26 open both gaps, at a cost of 0.1 in of page.
    a1.text(0.005, 1.26, "(a)", transform=a1.transAxes, weight="bold", va="bottom")
    # legend above panel (a), and not below: below it falls between the two panels and starts
    # to look like the legend of (b), which uses different symbols.
    a1.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.07),
              handletextpad=0.3, columnspacing=1.1, scatterpoints=1)

    # both panels need the same window and the same origin: with (a) cropping from zx to
    # zx+2*ZX and labelling 0..24 while (b) shows 0..2*MX and labels 0..36, the same axis name
    # carries a 6 m offset between them. (b) uses the crop of (a), and the cut applies to the
    # cloud and to the detections together (see the jf note below).
    faixa = ((abs(y - CY) < FAIXA / 2) & (h > 0.2) & (h < 12)
             & (x - x0 >= zx) & (x - x0 <= zx + 2 * ZX))
    a2.scatter(x[faixa] - x0, h[faixa], s=0.04, color="#5A6570", linewidths=0, alpha=0.35)
    # the section uses the whole window, and not the plan crop: reusing `zp` here would limit
    # the lines to the central 16 m and the panel would come out with half the stems unmarked,
    # which reads as a detection failure.
    jf = ((pred[:, 0] - x0 >= zx) & (pred[:, 0] - x0 <= zx + 2 * ZX)
          & (abs(pred[:, 1] - CY) < FAIXA / 2))
    for xp in pred[jf, 0] - x0:
        a2.plot([xp, xp], [0, 11.4], color=LARANJA, lw=0.8, alpha=0.9, zorder=3)
    # no legend in this panel, because of a collision: at bbox_to_anchor=(0.5, -0.24) it sat
    # below the axes, and on shortening the panel from an aspect of 8.0 to 3.4 the -0.24 began
    # to land on "Distance east (m)". The figure caption already says that each vertical line
    # is a detection, and panel (a) already defines the orange colour.
    a2.set_xlim(zx, zx + 2 * ZX); a2.set_ylim(0, 12); a2.grid(False)
    a2.set_xticks(np.arange(zx, zx + 2 * ZX + 0.1, 6))
    a2.set_xticklabels([f"{v - zx:.0f}" for v in a2.get_xticks()])
    # three ticks, and not five: panel (b) is short by construction, and with 0/3/6/9/12 the
    # labels touch. 0/6/12 doubles the gap with no loss, because the scale is linear.
    a2.set_yticks([0, 6, 12])
    a2.set_xlabel("Distance east (m)"); a2.set_ylabel("Height (m)")
    a2.text(0.005, 1.02, "(b)", transform=a2.transAxes, weight="bold", va="bottom")

    # (b) is aligned to (a) after drawing, and the order matters: (a) has equal scale on both
    # axes, so matplotlib narrows its box at draw time to respect the 24x11 ratio. The final
    # position only exists after the draw, and `get_position()` without `original=False`
    # returns the requested box, not the one used. Without this block the two panels share the
    # window but appear at different widths.
    fig.canvas.draw()
    cx = a1.get_position(original=False)
    cy = a2.get_position(original=False)
    a2.set_position([cx.x0, cy.y0, cx.width, cy.height])

    salvar(fig, out, "fig10_detections_on_tls",
           "Wall-to-wall detections drawn over the terrestrial laser cloud that the stem map "
           "came from, which is the only figure where the reference itself is shown rather than "
           f"summarised. (a) Plan view of a {2 * ZX:.0f} by {2 * ZY:.0f} m window. Grey is the "
           f"terrestrial cloud between {H0} and {H1} m above ground, where returns are almost "
           "entirely stem, and black marks the cells whose density exceeds the threshold "
           "calibrated on the mapped stems themselves. Each link joins a detection to the stem "
           f"assigned to it, with a median length of {np.median(dpar):.2f} m, because the "
           "detected position derives from the crown seen from above while the reference "
           f"records the stem. Counts close inside the frame, {n_ref} stems and {len(idp)} "
           f"detections of which {n_par} are paired, and a detection whose partner falls "
           "outside the frame is drawn as unpaired. "
           f"(b) The same scene in side view, a {FAIXA:.0f} m thick slab, each column of points "
           "being one stem and each vertical line one detection. The window was chosen by "
           "terrestrial coverage, measured as point density in the breast-height slice, and not "
           "by detection performance.", dpi=1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=config.REPO / "figs_en")
    ap.add_argument("--so", nargs="*")
    args = ap.parse_args()
    quais = set(args.so or ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "supp"])
    print(f"figures in {args.out}")
    # the key is the figure number in the document, which is the reading order, and not the
    # order in which the functions were written in this file: `--so 4` generates
    # fig4_detected_vs_field.
    for k, fn in (("1", fig_fluxo), ("2", fig_qualitativa), ("3", fig_raio),
                  ("4", fig_scatter), ("5", fig_matched), ("6", fig_voo),
                  ("7", fig_casado_voo), ("8", fig_mapa), ("9", fig_talhao_completo),
                  ("10", fig_prova_tls), ("supp", fig_casada_mapa)):
        if k in quais:
            fn(args.out)
    if LEGENDAS:
        # accumulates, so that running only a subset does not erase the other captions
        alvo = args.out / "figure_captions.txt"
        antes = alvo.read_text() if alvo.exists() else ""
        guardadas = dict(ln.split(" :: ", 1) for ln in antes.splitlines() if " :: " in ln)
        guardadas.update(dict(LEGENDAS))
        # a caption for a figure that no longer exists is removed: the merge preserves what
        # came from partial runs and, without this line, would also preserve renumbered or
        # deleted figures.
        guardadas = {n: c for n, c in guardadas.items() if (args.out / f"{n}.png").exists()}
        alvo.write_text("\n".join(f"{n} :: {c}" for n, c in sorted(guardadas.items())) + "\n")
        print(f"  captions in {alvo.name} ({len(guardadas)} figures)")


if __name__ == "__main__":
    main()

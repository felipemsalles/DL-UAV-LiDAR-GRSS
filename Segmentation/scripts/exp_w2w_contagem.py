#!/usr/bin/env python3
"""Tree count over whole stands, from the wall-to-wall run.

Difference with respect to the per-plot scheme: there the NMS merged the 9 views
of one plot and stopped, because the blocks of 9 were isolated from one another.
Here the grid is continuous, so the same tree appears in up to 9 neighbouring
tiles and the merge has to be global within the stand. Merging per tile and
summing would count each tree several times.

The merge radius does not transfer for free, which is why it is chosen here
instead of fixed. The 1.5 m was validated in the per-plot scheme, where the 9
views were all centred on the plot and the counted tree sat in the middle of every
crop. On a continuous grid a tree can be at the edge of several tiles at once,
with its crown cut and its centre displaced, and it then merges worse. Comparing
segmenters with a single radius, without saying whose radius it is, costs FF3D
11 points over the 13 plots.

Selection criterion, in order of strength:
  1. matched metric on stand 001 against the TLS stem map, the only per-tree
     ground truth that exists in the project. Chooses by F1.
  2. count over the 13 plots against the field census, as a cross-check.
Reporting both makes any disagreement visible.

Perimeter ring. The clouds were clipped exactly to the stand polygon: zero points
outside, among 1.8 million sampled across the six. There is not one metre of
context beyond the fence, so the crown of the outermost-row tree is cut at the
boundary, and no grid arrangement fixes that, because data is missing and not
merely cropping. (Measuring coverage in 2 m cells suggests a 1 to 2 m margin, but
that is an artefact: the cell touching the boundary counts in full.) The 3 m ring
is 4094 x 3 = 1.23 ha, or 12.5% of the 9.87 ha.

Run: PYTHONPATH=. python scripts/exp_w2w_contagem.py [--pos base] [--raio 1.5]
"""
import argparse
import math
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point

from greenvista import config

CSV = config.REPO / "data/detections/sat_w2w_instancias.csv"
SAIDA = config.OUT_DIR / "w2w_por_talhao.csv"
VARREDURA_CSV = config.OUT_DIR / "w2w_varredura_raio.csv"
POSICOES = config.REPO / "data/detections/sat_w2w_arvores.csv"
MAPA_FUSTES = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
PR400 = math.sqrt(400 / math.pi)      # 11.28 m, the radius that closes the field 400 m²
ANEL = 3.0                            # width of the perimeter ring analysed
MEIA_AVAL = 32.0 / 2 - 3.0            # central 26 m square, the same ruler as the paired study
LIMIAR_CASADO = 2.0                   # maximum distance for matching a prediction to a stem
PLOTS_TLS = ["1_1", "1_2"]            # where a stem map exists


def nms_malha(xy, peso, raio):
    """Greedy NMS keeping the largest of each cluster, with a grid-based search.

    Same rule as the project's `nms_merge`: it walks in decreasing order of weight
    and accepts the candidate if no already accepted one is closer than `raio`.
    The grid of cells of side `raio` limits the search to the 9 neighbouring
    cells, which trades quadratic cost for linear without changing the result.
    Checked against the project implementation: same indices up to n=20000.
    """
    if len(xy) == 0:
        return np.zeros(0, bool)
    celula = defaultdict(list)
    aceito = np.zeros(len(xy), bool)
    r2 = raio * raio
    for i in np.argsort(-peso):
        p = xy[i]
        cx, cy = int(p[0] // raio), int(p[1] // raio)
        perto = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in celula.get((cx + dx, cy + dy), ()):
                    dd = xy[j] - p
                    if dd[0] * dd[0] + dd[1] * dd[1] <= r2:
                        perto = True
                        break
                if perto:
                    break
            if perto:
                break
        if not perto:
            aceito[i] = True
            celula[(cx, cy)].append(i)
    return aceito


def funde(d, polys, raio, cx, cy, crs):
    """Merges globally per stand and clips to the polygon. Returns a DataFrame."""
    saida = []
    for t, g in d.groupby("talhao"):
        xy = g[[cx, cy]].to_numpy(float)
        keep = nms_malha(xy, g.n_pts.to_numpy(float), raio)
        m = g[keep].copy()
        poly = polys[int(t)]
        pts = gpd.GeoSeries([Point(*p) for p in m[[cx, cy]].to_numpy()], crs=crs)
        dentro = pts.within(poly).to_numpy()
        # `.boundary` and not `.exterior`: stand 006 is a two-part MultiPolygon,
        # and a MultiPolygon has no `.exterior`.
        borda = pts[dentro].distance(poly.boundary).to_numpy()
        m = m[dentro].copy()
        m["talhao"] = int(t)
        m["dist_divisa"] = borda
        m["brutas_no_talhao"] = len(g)
        saida.append(m)
    return pd.concat(saida, ignore_index=True) if saida else pd.DataFrame()


def parear(ref, pred, limiar):
    """Optimal one-to-one assignment, the same rule as exp_matched_metric_talhao001."""
    if len(ref) == 0 or len(pred) == 0:
        return 0
    dist = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    li, ci = linear_sum_assignment(np.where(dist <= limiar, dist, 1e6))
    return int((dist[li, ci] <= limiar).sum())


def metrica_casada(arv, centros, cx, cy):
    """Precision, recall and F1 against the stem map, in the 26 m square.

    It evaluates in the central square and not in the 400 m² circle for the same
    reason as the paired study: the inset removes the edge effect with no special
    rule, and the usable area rises from 400 to 676 m², which nearly doubles the
    sample.
    """
    if not MAPA_FUSTES.exists():
        return None
    fustes = gpd.read_file(MAPA_FUSTES)
    ref_xy = np.column_stack([fustes.geometry.x, fustes.geometry.y])
    tp = npred = nref = 0
    for pid in PLOTS_TLS:
        if pid not in centros:
            continue
        px, py, _ = centros[pid]
        dentro = lambda a: a[(np.abs(a[:, 0] - px) <= MEIA_AVAL) & (np.abs(a[:, 1] - py) <= MEIA_AVAL)]
        r = dentro(ref_xy)
        p = dentro(arv[[cx, cy]].to_numpy(float))
        tp += parear(r, p, LIMIAR_CASADO)
        npred += len(p)
        nref += len(r)
    if npred == 0 or nref == 0:
        return None
    prec, rev = tp / npred, tp / nref
    f1 = 2 * prec * rev / (prec + rev) if prec + rev else 0.0
    return {"tp": tp, "pred": npred, "ref": nref,
            "precisao": prec, "revocacao": rev, "f1": f1}


def centros_e_censo():
    p = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for c in ("talhao", "parcela"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    inv = pd.read_csv(config.DATA / "5-dados_campo/inventario_est.csv")
    p = p.dropna(subset=["talhao", "parcela"]).merge(
        inv[["talhao", "parcela", "n_arv"]], on=["talhao", "parcela"], how="left")
    return {f"{int(r.talhao)}_{int(r.parcela)}":
            (r.geometry.x, r.geometry.y, float(r.n_arv)) for _, r in p.iterrows()}


def contagem_parcelas(arv, centros, cx, cy):
    """Merged detections inside each 11.28 m circle."""
    out = {}
    for pid, (px, py, campo) in centros.items():
        t = int(pid.split("_")[0])
        sub = arv[arv.talhao == t]
        if sub.empty:
            continue
        dd = np.hypot(sub[cx] - px, sub[cy] - py)
        out[pid] = (int((dd <= PR400).sum()), campo)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raio", type=float, default=None,
                    help="fixes the radius; without this, sweeps and chooses by the matched metric")
    ap.add_argument("--varredura", type=float, nargs="*",
                    default=[1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3])
    ap.add_argument("--pos", choices=("base", "cent"), default="base")
    args = ap.parse_args()

    if not CSV.exists():
        raise SystemExit(f"could not find {CSV}. Has the run finished?")
    d = pd.read_csv(CSV)
    # The CSV is written in append mode, one batch at a time. On a runner restart
    # the batches already done are written again; the NMS would suppress the copy,
    # but the raw count and the per-tree redundancy would be inflated, distorting
    # the "views per tree" diagnostic.
    antes = len(d)
    d = d.drop_duplicates(["tile", "inst"])
    if len(d) < antes:
        print(f"{antes - len(d):,} duplicate rows removed (there was a restart)")
    print(f"{len(d):,} raw instances in {d.tile.nunique():,} tiles, "
          f"stands {sorted(int(t) for t in d.talhao.unique())}\n")

    a = gpd.read_file(config.DATA / "2-shapes/Areas_plantio/area_plantio.shp")
    a["t"] = pd.to_numeric(a.Talhao, errors="coerce").astype("Int64")
    polys = {int(r.t): r.geometry for _, r in a.iterrows() if pd.notna(r.t)}
    centros = centros_e_censo()
    cx, cy = f"{args.pos}_x", f"{args.pos}_y"

    # ------------------------------------------------------------- sweep
    raios = [args.raio] if args.raio else args.varredura
    linhas_v = []
    cache = {}
    for raio in raios:
        arv = funde(d, polys, raio, cx, cy, a.crs)
        cache[raio] = arv
        cont = contagem_parcelas(arv, centros, cx, cy)
        tw = sum(v[0] for v in cont.values())
        tc = sum(v[1] for v in cont.values())
        mc = metrica_casada(arv, centros, cx, cy)
        linhas_v.append({
            "raio": raio, "arvores": len(arv),
            "por_ha": round(len(arv) / sum(polys[int(t)].area for t in d.talhao.unique()) * 1e4, 1),
            "nas_parcelas": tw, "campo": int(tc),
            "razao_campo": round(tw / tc, 3) if tc else float("nan"),
            "precisao": round(mc["precisao"], 3) if mc else float("nan"),
            "revocacao": round(mc["revocacao"], 3) if mc else float("nan"),
            "f1": round(mc["f1"], 3) if mc else float("nan"),
        })
    v = pd.DataFrame(linhas_v)
    print("MERGE-RADIUS SWEEP")
    print("(f1 = matched against the TLS stem map, stand 001, 26 m square, 2 m threshold)")
    print(v.to_string(index=False))
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    v.to_csv(VARREDURA_CSV, index=False)

    if args.raio:
        escolhido = args.raio
        porque = "fixed on the command line"
    elif v.f1.notna().any():
        escolhido = float(v.loc[v.f1.idxmax(), "raio"])
        porque = f"highest matched F1 ({v.f1.max():.3f})"
    else:
        escolhido = float(v.loc[(v.razao_campo - 1).abs().idxmin(), "raio"])
        porque = "no stem map, chosen by the ratio against the census"
    print(f"\nRADIUS CHOSEN: {escolhido} m ({porque})\n")

    # ------------------------------------------------------- definitive table
    arv = cache[escolhido]
    linhas = []
    for t, m in arv.groupby("talhao"):
        poly = polys[int(t)]
        ha = poly.area / 1e4
        borda = m.dist_divisa.to_numpy()
        linhas.append({
            "talhao": f"{int(t):03d}", "ha": round(ha, 2),
            # 006 is a two-part MultiPolygon, which gives it twice the boundary
            # per area of 004. Without this column the comparison across stands
            # would read as if all of them had the same exposure.
            "m_divisa_ha": round(poly.length / ha, 0),
            "brutas": int(m.brutas_no_talhao.iloc[0]),
            "arvores": len(m), "por_ha": round(len(m) / ha, 1),
            "no_anel_3m": int((borda <= ANEL).sum()),
            "pct_anel": round(100 * (borda <= ANEL).mean(), 1),
            "vistas_por_arvore": round(m.brutas_no_talhao.iloc[0] / len(m), 2),
        })
    r = pd.DataFrame(linhas)
    tot_ha = r.ha.sum()
    per = sum(polys[int(t)].length for t in arv.talhao.unique())
    r.loc[len(r)] = {
        "talhao": "TOTAL", "ha": round(tot_ha, 2), "m_divisa_ha": round(per / tot_ha, 0),
        "brutas": r.brutas.sum(), "arvores": r.arvores.sum(),
        "por_ha": round(r.arvores.sum() / tot_ha, 1),
        "no_anel_3m": r.no_anel_3m.sum(),
        "pct_anel": round(100 * r.no_anel_3m.sum() / r.arvores.sum(), 1),
        "vistas_por_arvore": round(r.brutas.sum() / r.arvores.sum(), 2)}
    print(r.to_string(index=False))
    r.to_csv(SAIDA, index=False)
    arv.to_csv(POSICOES, index=False)
    print(f"\n{len(arv):,} trees in {POSICOES.name}, summary in {SAIDA.name}")

    # -------------------------------------------- cross-check over the 13 plots
    # The wall-to-wall grid has an arbitrary phase: no plot falls at the centre of
    # a tile, unlike the plot-centred scheme. If the numbers agree, the method does
    # not depend on lucky framing; if they do not, the published 737 is in part an
    # artefact of centring the tile on the plot.
    pub = config.OUT_DIR / "sat_adabn_13parcelas.csv"
    antigo = pd.read_csv(pub).set_index("parcela").adabn_r400 if pub.exists() else None
    cont = contagem_parcelas(arv, centros, cx, cy)
    print(f"\n{'plot':>9} {'field':>6} {'w2w':>5} {'plot-centred':>17} {'diff':>5}")
    print("-" * 48)
    tw = tc = ta = 0
    for pid in sorted(cont, key=lambda s: (int(s.split('_')[0]), int(s.split('_')[1]))):
        n, campo = cont[pid]
        old = int(antigo[pid]) if antigo is not None and pid in antigo.index else 0
        tw += n; tc += int(campo); ta += old
        print(f"{pid:>9} {int(campo):>6} {n:>5} {old:>17} {n - old:>+5}")
    print("-" * 48)
    print(f"{'TOTAL':>9} {tc:>6} {tw:>5} {ta:>17} {tw - ta:>+5}")
    if tc:
        print(f"\naccuracy over the plots: wall-to-wall {100*tw/tc:.1f}%  "
              f"vs plot-centred {100*ta/tc:.1f}%")


if __name__ == "__main__":
    main()

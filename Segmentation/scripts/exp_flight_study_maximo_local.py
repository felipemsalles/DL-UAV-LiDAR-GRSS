#!/usr/bin/env python3
"""Third curve of the flight study: local maxima on the CHM, on the same degraded tiles.

Figure 4 compares the drop in detection with density measured here against the one in da Cunha
Neto (Forests 2025, 16, 1747), who loses almost nothing over a comparable range. In that
comparison two variables change, the stand (closed clonal against open) and the algorithm (deep
network against local maxima on a raster), so the contrast does not identify the cause.

This script runs the same family of algorithm they used, in this stand, over the same 104
degraded tiles that FF3D and SegmentAnyTree have already seen, so that the comparison varies one
thing at a time.

Their technique, verified in the PDF:
  * LAStools 1.8 to merge and clip, then R with lidR and rLiDAR;
  * `lasfilterdecimate` with spatially uniform selection for the 9 densities;
  * `lasground`, `lasnormalize`, `grid_terrain`, `grid_canopy`;
  * "applying a Gaussian-type smoothing filter with sigma 0.5 by the function
    CHMsmoothing";
  * "assess individual tree heights with the FindTreesCHM function", which is a sliding-window
    local maximum over the raster.

They report neither the window size nor the CHM resolution. In a local-maxima detector those two
parameters decide the result on their own, so the study is not reproducible as published; here
both are swept and the baseline is reported at its own best setting.

The setting is chosen once only, at the full-density condition, and stays frozen across the eight
conditions. Choosing per condition would be tuning a parameter on the result.

The selection criterion is the matched F1, and not count agreement. Choosing by the total closest
to the census elects the 0.75 m window, which reaches 96.8% at full density by error cancellation
and explodes to 624% of the census at 20 pts/m2: a total count does not distinguish a detector
from a noise counter. F1 against the stem map does distinguish, and it is the same ruler that
chose the merge radius of the two networks.

The protocol is identical to that of scripts/exp_flight_study_dois_modelos.py, otherwise the three
curves are not comparable: count inside the 11.28 m circle (400 m2), one pass per tile, and each
tile normalised by its own full condition.

Usage: PYTHONPATH=. python scripts/exp_flight_study_maximo_local.py
Output: manual_match/flight_study_tres_modelos.csv
"""
import math
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenvista import config  # noqa: E402
from greenvista.segmentation.local_maxima import build_chm, detect_treetops  # noqa: E402

PR = math.sqrt(400 / math.pi)          # 11.28 m
DEG = config.REPO / "work/ff3d_degraded"
DOIS = config.OUT_DIR / "flight_study_dois_modelos.csv"
SAIDA = config.OUT_DIR / "flight_study_tres_modelos.csv"
SWEEP = config.OUT_DIR / "flight_study_maximo_local_sweep.csv"
CONDS = ["dens_20", "dens_50", "dens_100", "ang10_dens100", "ang_10",
         "dens_200", "ang_20", "full"]

HMIN = 5.0                              # minimum canopy height, as in the rest of the repo
RESOLUCOES = [0.25, 0.5, 1.0]           # not reported by them
JANELAS = [3, 5, 7, 9, 11]              # not reported by them; rLiDAR uses 5 by default
SIGMA = 0.5                             # this one they do report

# selection ruler: TLS stem map in stand 001, same protocol as fig. 1
MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
TILE, MARGEM, LIMIAR = 32.0, 3.0, 2.0   # central 26 m square, matching at 2 m
TILES_REF = ["t001_p001", "t001_p002"]  # the only ones with a stem map


def centros_e_censo():
    p = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for c in ("talhao", "parcela"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    inv = pd.read_csv(config.DATA / "5-dados_campo/inventario_est.csv")
    p = p.dropna(subset=["talhao", "parcela"]).merge(
        inv[["talhao", "parcela", "n_arv"]], on=["talhao", "parcela"], how="left")
    return {f"t{int(r.talhao):03d}_p{int(r.parcela):03d}":
            (r.geometry.x, r.geometry.y, float(r.n_arv)) for _, r in p.iterrows()}


def le_vegetacao(laz):
    """Canopy points of the tile, with ground and noise removed, as in the other baselines."""
    las = laspy.read(str(laz))
    cls = np.asarray(las.classification)
    m = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    return (np.column_stack([np.asarray(las.x)[m], np.asarray(las.y)[m]]),
            np.asarray(las.z)[m])


def preenche(chm):
    """Fills raster holes with the nearest filled neighbour.

    The repo's `build_chm` initialises at zero and only writes where a point fell, so an empty
    cell keeps the value 0. At full density that hardly ever happens; at 20 pts/m2 the 0.25 m
    raster becomes a field of holes, every surviving point is surrounded by zeros and every
    isolated point becomes a local maximum. Without filling, the local maximum scores 624% of the
    census at 20 pts/m2, which is a raster artefact and not the behaviour of the method. lidR,
    used by da Cunha Neto, fills the gaps of `grid_canopy` by neighbourhood.

    It is applied only inside the hull of the points, otherwise the filling would invent canopy at
    the border of the tile where there is no data.
    """
    vazio = chm == 0
    if not vazio.any() or vazio.all():
        return chm
    from scipy.ndimage import distance_transform_edt
    _, (ix, iy) = distance_transform_edt(vazio, return_indices=True)
    out = chm.copy()
    out[vazio] = chm[ix[vazio], iy[vazio]]
    return out


def topos(xy, z, res, janela):
    if len(z) < 100:
        return np.empty((0, 3))
    chm, x0, y0, r = build_chm(xy, z, res=res)
    cheio = preenche(chm)
    # the height filter uses the FILLED raster, otherwise a legitimate peak that fell in an
    # empty cell would be discarded for having zero height
    return detect_treetops(cheio, x0, y0, r, window=janela, hmin=HMIN, smooth=SIGMA)


def conta(xy, z, cx, cy, res, janela):
    t = topos(xy, z, res, janela)
    if len(t) == 0:
        return 0
    return int((np.hypot(t[:, 0] - cx, t[:, 1] - cy) <= PR).sum())


def na_area(xy, cx, cy, meia=TILE / 2 - MARGEM):
    """Central 26 m square, the same extent as the matched metric of figure 1."""
    if len(xy) == 0:
        return xy
    return xy[(np.abs(xy[:, 0] - cx) <= meia) & (np.abs(xy[:, 1] - cy) <= meia)]


def parear(ref, pred, limiar=LIMIAR):
    """Optimal one-to-one assignment, Hungarian, as in the rest of the project."""
    if len(ref) == 0 or len(pred) == 0:
        return 0
    d = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    li, ci = linear_sum_assignment(np.where(d <= limiar, d, 1e6))
    return int((d[li, ci] <= limiar).sum())


def escolhe_ajuste(centros):
    """Sweeps resolution and window at full density and returns the pair with the highest F1.

    The sweep happens only on the two tiles of stand 001, which are the only ones with a stem map,
    and only at the full condition. The degraded conditions are left out of the decision on
    purpose, otherwise the parameter would be tuned on the very result the figure means to measure.
    """
    ref_todo = gpd.read_file(MAPA)
    REF = np.column_stack([ref_todo.geometry.x, ref_todo.geometry.y])
    nuvens = {t: le_vegetacao(DEG / "full" / f"{t}.laz") for t in TILES_REF}

    linhas = []
    for res in RESOLUCOES:
        for jan in JANELAS:
            tp = n_ref = n_pred = 0
            for tile in TILES_REF:
                cx, cy, _ = centros[tile]
                ref = na_area(REF, cx, cy)
                pred = na_area(topos(*nuvens[tile], res, jan)[:, :2], cx, cy)
                tp += parear(ref, pred)
                n_ref += len(ref)
                n_pred += len(pred)
            prec = tp / n_pred if n_pred else 0.0
            rec = tp / n_ref if n_ref else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            linhas.append({"res_m": res, "janela_px": jan, "janela_m": jan * res,
                           "n_ref": n_ref, "n_pred": n_pred, "tp": tp,
                           "precisao": prec, "revocacao": rec, "f1": f1})
            print(f"  res {res:>4} m, window {jan:>2} px ({jan*res:>5.2f} m): "
                  f"pred {n_pred:>4}, P {prec:.3f}  R {rec:.3f}  F1 {f1:.3f}")
    s = pd.DataFrame(linhas).sort_values("f1", ascending=False).reset_index(drop=True)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    s.to_csv(SWEEP, index=False)
    m = s.iloc[0]
    return float(m.res_m), int(m.janela_px), float(m.f1)


def main():
    centros = centros_e_censo()
    print("local-maxima setting sweep, full density, matched F1 against the\n"
          f"TLS stem map in stand 001, matching at {LIMIAR:.0f} m\n")
    res, janela, f1 = escolhe_ajuste(centros)
    print(f"\nsetting chosen and frozen: resolution {res} m, window {janela} px "
          f"({janela*res:.2f} m), F1 {f1:.3f}\n")

    # A second setting, to test the hypothesis that the closed stand forces a small window
    # (narrow crowns pressed together) and that it is that imposition, and not the algorithm, that
    # brings the fragility to raster noise. If so, the wide window stays stable with density and
    # pays for it in recall. 1.75 m is the widest window below the 1.88 m within-row spacing, that
    # is, the widest one that can still separate two neighbours.
    LARGA = (0.25, 7)
    linhas = []
    for cond in CONDS:
        for laz in sorted((DEG / cond).glob("*.laz")):
            tile = laz.stem
            if tile not in centros:
                continue
            cx, cy, _ = centros[tile]
            xy, z = le_vegetacao(laz)
            linhas.append({"condition": cond, "tile": tile,
                           "lm": conta(xy, z, cx, cy, res, janela),
                           "lm_larga": conta(xy, z, cx, cy, *LARGA)})
    lm = pd.DataFrame(linhas)

    if not DOIS.exists():
        sys.exit(f"missing {DOIS}, run exp_flight_study_dois_modelos.py first")
    d = pd.read_csv(DOIS).merge(lm, on=["condition", "tile"], how="left")
    if d.lm.isna().any():
        sys.exit(f"{int(d.lm.isna().sum())} tiles without local maxima, comparison impossible")

    for c in ("lm", "lm_larga"):
        d[f"{c}_pct"] = 100 * d[c] / d.field
        cheio = d[d.condition == "full"].set_index("tile")[f"{c}_pct"]
        d[f"{c}_rel"] = d.apply(lambda r: 100 * r[f"{c}_pct"] / cheio[r.tile]
                                if cheio.get(r.tile, 0) > 0 else np.nan, axis=1)
    d.to_csv(SAIDA, index=False)

    g = d.groupby("condition").agg(
        dens=("density_pts_m2", "median"),
        lm=("lm_pct", "mean"), lm_larga=("lm_larga_pct", "mean"),
        ff3d=("ff3d_pct", "mean"), sat=("sat_pct", "mean"),
        lm_rel=("lm_rel", "mean"), lm_larga_rel=("lm_larga_rel", "mean"),
        ff3d_rel=("ff3d_rel", "mean"),
        sat_rel=("sat_rel", "mean")).sort_values("dens")
    print(f"local maxima at two windows ({janela*res:.2f} m, the highest F1, and "
          f"{LARGA[0]*LARGA[1]:.2f} m), plus the two networks\n"
          "same tiles, same 11.28 m circle, one pass\n"
          "pct = of the field count; rel = of the tile's own full condition\n")
    print(g.round(1).to_string())
    print(f"\n{SAIDA}\n{SWEEP}")


if __name__ == "__main__":
    main()

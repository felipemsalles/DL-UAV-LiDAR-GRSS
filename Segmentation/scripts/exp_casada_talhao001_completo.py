#!/usr/bin/env python3
"""Matched metric over the whole stand 001, against the 892 stems of the TLS map.

A per-plot run reaches only the 211 stems that fit in the central 26 m squares of the
two plot tiles. The whole-stand scan gives detections across the entire area, and the
TLS map has 892 stems, so the comparison here is against the complete reference instead
of against 24% of it.

Provenance of the reference: the shapefile came from TLS and went through manual
adjustment after automatic segmentation, done on the terrestrial cloud and not on the
drone one. The reference is therefore independent of the data being evaluated, and
since the operator sees stems the drone cannot reach, the map includes trees beyond
the reach of aerial detection.

The extent is not the stand, but the area the TLS covered: it scanned a strip inside
the 0.74 ha polygon. Comparing the 1076 detections of the whole stand against 892 stems
would penalise detections made where the TLS never passed.

The extent is symmetric and based on the convex hull, not on a rectangle: the rectangle
includes 9% of uncovered area (visible in figure 9 as a diagonal strip of unmatched
detections) and clipping only the detections penalises precision without penalising
recall.

The setback is the matching threshold itself. A stem less than 2 m from the border of
the evaluated area can be matched by a detection from outside it, which would subject it
to a different rule from the others; setting back exactly 2 m removes the asymmetry.
Same logic as the 3 m setback in the 26 m square of the plots.

The three areas are reported to show that the result does not depend on the choice:
precision stays between 0.89 and 0.91 in all three.

Usage: PYTHONPATH=. python scripts/exp_casada_talhao001_completo.py
Output: manual_match/casada_talhao001_completo.csv
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenvista import config  # noqa: E402

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
DETEC = config.REPO / "data/detections/sat_w2w_arvores.csv"
SAIDA = config.OUT_DIR / "casada_talhao001_completo.csv"
LIMIARES = (1.0, 1.5, 2.0, 2.5)
RECUOS = (None, 0.0, 2.0, 4.0)   # None = no clipping; 2 m = the matching threshold itself
PADRAO = 2.0                  # the (setback, threshold) pair the document reports


def area_avaliada(ref, recuo):
    """Convex hull of the stems, set back. This is where the TLS actually measured.

    `recuo=None` returns None, that is, no clipping: all 892 stems enter and every
    detection of the stand may match. That row measures recall alone against the
    complete reference. The rows with clipping are stricter and are the ones
    reported, because there precision is also measured, and for that the area has to
    be the same on both sides.
    """
    if recuo is None:
        return None
    from shapely.geometry import MultiPoint, Point
    casco = MultiPoint([Point(*p) for p in ref]).convex_hull
    return casco.buffer(-recuo) if recuo else casco


def dentro(area, pts):
    if area is None:
        return np.ones(len(pts), bool)
    from shapely.geometry import Point
    return np.array([area.contains(Point(*p)) for p in pts])


def parear(ref, pred, lim):
    D = np.hypot(ref[:, None, 0] - pred[None, :, 0], ref[:, None, 1] - pred[None, :, 1])
    li, ci = linear_sum_assignment(np.where(D <= lim, D, 1e6))
    ok = D[li, ci] <= lim
    return int(ok.sum()), float(np.sqrt((D[li, ci][ok] ** 2).mean())) if ok.any() else float("nan")


def main():
    f = gpd.read_file(MAPA)
    REF = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    d = pd.read_csv(DETEC)
    PRED = np.column_stack([d[d.talhao == 1].base_x.values, d[d.talhao == 1].base_y.values])

    casco = area_avaliada(REF, 0)
    print(f"TLS map       {len(REF)} stems, strip of {np.ptp(REF[:, 0]):.0f} by "
          f"{np.ptp(REF[:, 1]):.0f} m, convex hull {casco.area:.0f} m²")
    print(f"detections    {len(PRED)} over the whole stand 001\n")

    linhas = []
    for recuo in RECUOS:
        area = area_avaliada(REF, recuo)
        r, q = REF[dentro(area, REF)], PRED[dentro(area, PRED)]
        for lim in LIMIARES:
            tp, rmse = parear(r, q, lim)
            p, rev = tp / len(q), tp / len(r)
            linhas.append({"recuo_m": recuo if recuo is not None else "sem recorte", "limiar_m": lim, "ref": len(r), "pred": len(q),
                           "TP": tp, "FP": len(q) - tp, "FN": len(r) - tp,
                           "precisao": p, "revocacao": rev,
                           "F1": 2 * p * rev / (p + rev) if p + rev else 0.0,
                           "rmse_pos_m": rmse})
            if lim == PADRAO:
                rot = "no clipping" if recuo is None else f"setback {recuo:.0f} m"
                marca = ("  <- the one in the document" if recuo == PADRAO else
                         "  <- the one to quote in conversation" if recuo is None else "")
                print(f"  {rot}   {len(r):>3} stems, {len(q):>4} detections   "
                      f"TP {tp:>3}   precision {p:.3f}   recall {rev:.3f}   "
                      f"F1 {2*p*rev/(p+rev):.3f}{marca}")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

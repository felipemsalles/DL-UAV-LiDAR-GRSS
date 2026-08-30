#!/usr/bin/env python3
"""Precision and recall against density, for the three algorithms, in stand 001.

The flight study compares count against density, and counting cancels error: a tree
that is missed disappears behind an invented one and the total still looks healthy.
That is how the wide-window local maximum appeared stable in the count table, going
from 54% of the census on the full cloud to 74% at 20 pts/m², with no guarantee that
they are the same trees.

Stand 001 has a TLS stem map, so omission can be separated from commission there. It
is the only place where that is possible, and there are only two tiles.

It is also the check missing from da Cunha Neto (Forests 2025, 16, 1747), which reports
only totals by height class against the census, without tree-to-tree matching, so that
the count stability from 2000 down to 25 pts/m² does not separate the two components of
the error.

Same protocol as figure 1: central 26 m square, Hungarian assignment, 2 m threshold.

Usage: PYTHONPATH=. python scripts/exp_flight_study_casado.py
Output: manual_match/flight_study_casado_talhao001.csv
"""
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenvista import config  # noqa: E402
from scripts.exp_flight_study_maximo_local import (  # noqa: E402
    CONDS, DEG, MAPA, le_vegetacao, na_area, parear, topos,
)

FF_OUT = config.REPO / "work/ff3d_degraded_out"
SAT_OUT = config.REPO / "work/sat_flight_out"
SAIDA = config.OUT_DIR / "flight_study_casado_talhao001.csv"
TILES = ["t001_p001", "t001_p002"]
ESTREITA, LARGA = (0.25, 3), (0.25, 7)      # the two frozen settings


def centros():
    p = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for c in ("talhao", "parcela"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    return {f"t{int(r.talhao):03d}_p{int(r.parcela):03d}": (r.geometry.x, r.geometry.y)
            for _, r in p.dropna(subset=["talhao", "parcela"]).iterrows()}


def pos_ff3d(ply, tile_laz):
    """The FF3D PLY comes in local coordinates, shifted by the mean of that tile, and the
    mean changes when the cloud is thinned; the mean of the matching tile must be added back."""
    from greenvista.segmentation.ff3d import load_panoptic_ply
    trees = load_panoptic_ply(ply)
    if not trees:
        return np.empty((0, 2))
    las = laspy.read(str(tile_laz))
    mx, my = float(np.asarray(las.x).mean()), float(np.asarray(las.y).mean())
    return np.array([[t["points"][:, 0].mean() + mx, t["points"][:, 1].mean() + my]
                     for t in trees.values()])


def pos_sat(out_laz):
    las = laspy.read(str(out_laz))
    inst = np.asarray(las.PredInstance)
    xy = np.column_stack([np.asarray(las.x), np.asarray(las.y)])
    cs = [xy[inst == i].mean(0) for i in np.unique(inst[inst > 0])]
    return np.asarray(cs) if cs else np.empty((0, 2))


def main():
    ref_todo = gpd.read_file(MAPA)
    REF = np.column_stack([ref_todo.geometry.x, ref_todo.geometry.y])
    ctr = centros()
    man = pd.read_csv(DEG / "degraded_manifest.csv")

    linhas = []
    for cond in CONDS:
        for tile in TILES:
            laz = DEG / cond / f"{tile}.laz"
            if not laz.exists():
                continue
            cx, cy = ctr[tile]
            ref = na_area(REF, cx, cy)
            xy, z = le_vegetacao(laz)
            m = man[(man.condition == cond) & (man.tile == tile)]
            dens = float(m.iloc[0].density_pts_m2) if len(m) else float("nan")

            ply = FF_OUT / cond / f"{tile}_round2.ply"
            sat_dir = SAT_OUT / cond
            sat_f = next(sat_dir.rglob(f"{tile}_out.laz"), None) if sat_dir.is_dir() else None
            preds = {
                "maximo_local_0.75m": topos(xy, z, *ESTREITA)[:, :2],
                "maximo_local_1.75m": topos(xy, z, *LARGA)[:, :2],
                "ff3d": pos_ff3d(ply, laz) if ply.exists() else None,
                "segmentanytree": pos_sat(sat_f) if sat_f else None,
            }
            for nome, p in preds.items():
                if p is None:
                    continue
                pred = na_area(p, cx, cy)
                tp = parear(ref, pred)
                prec = tp / len(pred) if len(pred) else 0.0
                rec = tp / len(ref) if len(ref) else 0.0
                linhas.append({
                    "condition": cond, "tile": tile, "density_pts_m2": dens,
                    "metodo": nome, "n_ref": len(ref), "n_pred": len(pred), "tp": tp,
                    "precisao": prec, "revocacao": rec,
                    "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0})

    d = pd.DataFrame(linhas)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    d.to_csv(SAIDA, index=False)

    # The aggregate is reported separately: the per-condition table below uses the mean over
    # the two tiles, whereas `table1_matched_metric.csv` for the complete system is aggregated,
    # sum of TP over sum of predictions. Comparing the mean of one against the aggregate of the
    # other gives one to two F1 points of difference, which is the order of magnitude of the
    # effects reported in this project.
    cheia = d[d.condition == "full"]
    print("full condition, one pass per tile, AGGREGATED over the 2 tiles\n"
          "this is the baseline comparable with table1_matched_metric.csv\n")
    for m, g in cheia.groupby("metodo"):
        tp, npred, nref = g.tp.sum(), g.n_pred.sum(), g.n_ref.sum()
        p, r = tp / npred, tp / nref
        print(f"  {m:<22} n_ref={nref}  n_pred={npred}  TP={tp}  "
              f"P {p:.3f}  R {r:.3f}  F1 {2*p*r/(p+r):.3f}")

    print("\ntree-to-tree matching against the stem map, stand 001, 2 tiles\n"
          "central 26 m square, Hungarian assignment, 2 m threshold\n"
          "the per-condition tables below are the MEAN over tiles, not the aggregate\n")
    for nome in d.metodo.unique():
        g = (d[d.metodo == nome].groupby("condition")
             .agg(dens=("density_pts_m2", "median"), n_pred=("n_pred", "sum"),
                  tp=("tp", "sum"), P=("precisao", "mean"),
                  R=("revocacao", "mean"), F1=("f1", "mean"))
             .sort_values("dens", ascending=False))
        print(f"\n{nome}")
        print(g.round(3).to_string())
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

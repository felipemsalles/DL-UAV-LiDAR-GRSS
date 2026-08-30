#!/usr/bin/env python3
"""Wall-to-wall pass over the six stands, covering two open questions in one run.

(A) The population term of the model-assisted estimator.
    `estimation.relative_efficiency` only needs the sampled plots, so the variance gain was already
    measurable. The absolute total needs the model's mean over every cell, which needs the clouds.

    Coverage: the delivered clouds cover ~17 ha of bounding box (14 ha of populated 20 m cells),
    not the 150 ha the inventory reports for the estate. They are dense patches around the
    inventory plots (350 to 750 points per m²), not continuous coverage. The map therefore exists
    only over the flown patches, and the total below is the total of those patches, not an estate
    total. The script prints the coverage on every run.

(B) Validation of the within-stand map.
    Age is constant inside a stand, so any spatial detail inside a stand must come from the
    flight. That is an argument about provenance, not about accuracy. Here the map is looked up at
    each field plot's position and the agreement is split into its between-stand and within-stand
    parts, with a null that permutes plots within each stand.

Both use leave-one-stand-out models: a stand's map is drawn by a model that never saw that stand.

Needs the clouds. Run:
  PYTHONPATH=. GREENVISTA_LAZ_DIR=/tmp/lazall python scripts/exp_map_totals_and_within_stand.py
"""
import warnings

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd

from greenvista import config, estimation
from greenvista.area_based import data as abd, mapping, models as abm, validate as abv
from greenvista.area_based.data import FEATURES
from greenvista.eval import metrics
from greenvista.eval.within_stand import decompose_correlation, within_stand_permutation
from greenvista.repro import seed_everything, write_manifest

warnings.simplefilter("ignore")

CELL = config.ABA.map_cell_m
CELL_AREA_HA = (CELL * CELL) / 1e4


def plot_centres():
    parc = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for c in ("talhao", "parcela"):
        parc[c] = pd.to_numeric(parc[c], errors="coerce").astype("Int64")
    parc["cx"], parc["cy"] = parc.geometry.x, parc.geometry.y
    return parc[["talhao", "parcela", "cx", "cy"]]


def plot_table():
    df = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                        config.DATA / "5-dados_campo/inventario_est.csv")
    return df.merge(plot_centres(), on=["talhao", "parcela"], how="left")


def stand_cells(talhao):
    """Per-cell LiDAR features for one stand, straight off its point cloud."""
    las = laspy.read(str(config.LAZ_DIR / f"SaoManuelTotal_{talhao:03d}.laz"))
    xy = np.column_stack([np.asarray(las.x), np.asarray(las.y)])
    cells, n_nan = mapping.cell_features_from_arrays(xy, np.asarray(las.z), cell=CELL)
    del las, xy
    return cells, n_nan


def cell_at(cells, cx, cy):
    """The cell a plot centre falls in, or the nearest one if the centre lands on a dropped cell."""
    inside = cells[(cells.x0 <= cx) & (cx < cells.x0 + CELL) &
                   (cells.y0 <= cy) & (cy < cells.y0 + CELL)]
    if len(inside):
        return inside.iloc[0], 0.0
    d = np.hypot(cells.x0 + CELL / 2 - cx, cells.y0 + CELL / 2 - cy)
    return cells.iloc[int(d.idxmin())], float(d.min())


def main():
    seed_everything(0)
    df = plot_table()
    y = df.Vol_ha.to_numpy(float)
    G = config.GROUP_COL
    # LiDAR only (a cell has no planting date attached), and restricted to the six metrics that
    # `mapping.stdmetrics_subset` reproduces bit-for-bit against lidR. The headline PLS uses the full
    # lidR suite, but zentropy / zpcum1..9 would have to be reimplemented from scratch to be computed
    # per cell, and a subtly different definition from the training CSV would bias every cell of the
    # map without failing loudly.
    make = lambda: abm.make_pls(cols=FEATURES, max_k=4)

    print(f"{len(df)} plots, {df[G].nunique()} stands, cell {CELL:.0f} m\n")

    pop_rows, at_plot, skipped = [], {}, {}
    for t in sorted(df[G].unique()):
        tr = df[df[G] != t]
        model = make().fit(tr, tr.Vol_ha.to_numpy(float))
        cells, n_nan = stand_cells(int(t))
        vha = model.predict(cells[FEATURES])
        vha = np.clip(vha, 0.0, None)
        cells = cells.assign(vol_ha=vha)
        skipped[int(t)] = n_nan

        pop_rows.append({"talhao": int(t), "n_cells": len(cells),
                         "area_ha": len(cells) * CELL_AREA_HA,
                         "mean_vol_ha": float(vha.mean()),
                         "total_m3": float(vha.sum() * CELL_AREA_HA),
                         "sd_cell_vol_ha": float(vha.std(ddof=1))})
        for _, r in df[df[G] == t].iterrows():
            c, dist = cell_at(cells, r.cx, r.cy)
            at_plot[r.plot_id] = {"map_vol_ha": float(c.vol_ha), "snap_dist_m": dist}
        print(f"  stand {int(t)}  {len(cells):4d} cells  {len(cells)*CELL_AREA_HA:6.1f} ha  "
              f"mean {vha.mean():6.1f} m3/ha  total {vha.sum()*CELL_AREA_HA:8.0f} m3"
              f"  ({n_nan} cells discarded)")

    pop = pd.DataFrame(pop_rows)
    df["map_vol_ha"] = df.plot_id.map(lambda p: at_plot[p]["map_vol_ha"])
    df["snap_dist_m"] = df.plot_id.map(lambda p: at_plot[p]["snap_dist_m"])

    # ---------------- (A) estate total, model-assisted ----------------
    print("\n" + "=" * 78)
    print("A  total over the FLOWN AREA, model-assisted estimate")
    print("=" * 78)
    _, loto = abv.oof_predictions(make, df, group_col=G)
    area_total = float(pop.area_ha.sum())
    pop_mean_weighted = float((pop.mean_vol_ha * pop.area_ha).sum() / area_total)

    field = estimation.field_only_mean(y)
    greg = estimation.model_assisted_mean(y, loto, np.repeat(pop.mean_vol_ha, pop.n_cells))
    eff = estimation.relative_efficiency(y, loto)
    re_ci = estimation.bootstrap_relative_efficiency(y, loto)

    print(f"\n  mapped area {area_total:.1f} ha in {int(pop.n_cells.sum())} cells of {CELL:.0f} m")
    print(f"  WARNING the delivered point clouds cover {area_total:.1f} ha, while the inventory reports "
          f"150 ha of estate.\n  The map exists only over the flown patches, so the total below is NOT "
          f"the estate total.")
    print(f"  model mean over all cells  {pop_mean_weighted:.1f} m3/ha")
    print(f"\n  field only        {field['mean_ha']:6.1f} m3/ha  +/- {field['se']:.1f}"
          f"   total {field['mean_ha']*area_total/1000:7.1f} thousand m3"
          f"   CI {field['lo']*area_total/1000:.1f} to {field['hi']*area_total/1000:.1f}")
    print(f"  field + LiDAR     {greg['mean_ha']:6.1f} m3/ha  +/- {greg['se']:.1f}"
          f"   total {greg['mean_ha']*area_total/1000:7.1f} thousand m3"
          f"   CI {greg['lo']*area_total/1000:.1f} to {greg['hi']*area_total/1000:.1f}")
    print(f"  field correction applied to the model  {greg['field_correction']:+.1f} m3/ha")
    print(f"  relative efficiency {eff['RE']:.2f} (CI {re_ci[0]:.2f} to {re_ci[1]:.2f})"
          f"  standard error {eff['se_reduction_pct']:+.0f}%")

    # ---------------- (B) is there signal INSIDE a stand? ----------------
    print("\n" + "=" * 78)
    print("B  is the map right INSIDE the stand?")
    print("=" * 78)
    print(f"\n  mean plot->cell snap distance {df.snap_dist_m.mean():.1f} m "
          f"(maximum {df.snap_dist_m.max():.1f})")

    rows = []
    for label, pred in (("mapa na posicao da parcela", df.map_vol_ha.to_numpy()),
                        ("modelo na pegada da parcela", loto)):
        dec = decompose_correlation(y, pred, df[G].to_numpy())
        r_w, p_w, sd_w = within_stand_permutation(y, pred, df[G].to_numpy(), seed=0)
        m = metrics(y, pred)
        rows.append({"source": label, **dec, "r_within_p": p_w, "null_sd": sd_w,
                     "rRMSE_pct": m["rRMSE_pct"], "R2": m["R2"]})
        print(f"\n  [{label}]  error {m['rRMSE_pct']:.1f}%")
        print(f"    total correlation      {dec['r_total']:+.2f}")
        print(f"    between stands         {dec['r_between']:+.2f}   (6 stands)")
        print(f"    WITHIN the stand       {dec['r_within']:+.2f}   p={p_w:.3f}"
              f"  (null sd {sd_w:.2f}, {dec['n_within_df']} degrees of freedom)")

    within = pd.DataFrame(rows)

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pop.to_csv(config.OUT_DIR / "map_stand_totals.csv", index=False)
    within.to_csv(config.OUT_DIR / "within_stand_map_validation.csv", index=False)
    df[["plot_id", "talhao", "parcela", "Vol_ha", "map_vol_ha", "snap_dist_m"]].to_csv(
        config.OUT_DIR / "map_at_plots.csv", index=False)
    write_manifest(config.OUT_DIR / "map_totals_manifest.json",
                   script="exp_map_totals_and_within_stand",
                   config={"cell_m": CELL, "model": "PLS on FEATURES (LiDAR only)",
                           "cv": "leave-one-stand-out",
                           "cells_skipped_nan": skipped},
                   metrics={"estate": {"area_ha": area_total, "field": field, "greg": greg,
                                       "RE": eff["RE"], "RE_ci": list(re_ci)},
                            "within_stand": within.to_dict("records")})
    print(f"\nwrote to {config.OUT_DIR}")


if __name__ == "__main__":
    main()

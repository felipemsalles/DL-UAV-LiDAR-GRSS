#!/usr/bin/env python3
"""Flight study with both models, in the same stand and at the same radius.

Figure 4 attributes the drop in detection with density to the stand and not to the
algorithm, without demonstrating it: the comparison puts FF3D in the closed stand
studied here against a classical method in the open stand of da Cunha Neto, where
detection hardly drops at all. Two variables change at the same time, so that is a
contrast and not an isolated cause.

The missing control is a second algorithm on the same degraded tiles: if both collapse,
the algorithmic explanation falls; if SegmentAnyTree holds up, the claim is wrong.

Counting radius: both models are scored at 11.28 m, the radius that closes the 400 m²
measured in the field, which also keeps figure 4 in step with figures 1 and 2. A 12 m
circle covers 452 m² instead and inflates the rates by about 10 points. Rescoring FF3D
costs no GPU time, because the degraded PLYs are still on disk.

Usage: PYTHONPATH=. python scripts/exp_flight_study_dois_modelos.py
"""
import math
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from greenvista import config  # noqa: E402

PR = math.sqrt(400 / math.pi)          # 11.28 m
DEG = config.REPO / "work/ff3d_degraded"
FF_OUT = config.REPO / "work/ff3d_degraded_out"
SAT_OUT = config.REPO / "work/sat_flight_out"
SAIDA = config.OUT_DIR / "flight_study_dois_modelos.csv"
CONDS = ["dens_20", "dens_50", "dens_100", "ang10_dens100", "ang_10",
         "dens_200", "ang_20", "full"]


def centros_e_censo():
    p = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for c in ("talhao", "parcela"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    inv = pd.read_csv(config.DATA / "5-dados_campo/inventario_est.csv")
    p = p.dropna(subset=["talhao", "parcela"]).merge(
        inv[["talhao", "parcela", "n_arv"]], on=["talhao", "parcela"], how="left")
    return {f"t{int(r.talhao):03d}_p{int(r.parcela):03d}":
            (r.geometry.x, r.geometry.y, float(r.n_arv)) for _, r in p.iterrows()}


def conta_ff3d(ply, tile_laz, cx, cy):
    """Centroids of the panoptic PLY, brought back to UTM, inside the circle.

    The converter of the FF3D pipeline shifts each tile to local coordinates by
    subtracting its own mean. To return to UTM one has to add back the mean of that
    tile, which changes by condition, because thinning the cloud changes the mean.
    """
    from greenvista.segmentation.ff3d import load_panoptic_ply
    trees = load_panoptic_ply(ply)          # {id: {"points": (N,3), "semantic": (N,)}}
    if not trees:
        return 0
    las = laspy.read(str(tile_laz))
    mx, my = float(np.asarray(las.x).mean()), float(np.asarray(las.y).mean())
    c = np.array([[t["points"][:, 0].mean() + mx, t["points"][:, 1].mean() + my]
                  for t in trees.values()])
    return int((np.hypot(c[:, 0] - cx, c[:, 1] - cy) <= PR).sum())


def conta_sat(out_laz, cx, cy):
    """One position per predicted instance, already in UTM."""
    las = laspy.read(str(out_laz))
    inst = np.asarray(las.PredInstance)
    xy = np.column_stack([np.asarray(las.x), np.asarray(las.y)])
    cs = [xy[inst == i].mean(0) for i in np.unique(inst[inst > 0])]
    if not cs:
        return 0
    c = np.asarray(cs)
    return int((np.hypot(c[:, 0] - cx, c[:, 1] - cy) <= PR).sum())


def main():
    centros = centros_e_censo()
    man = pd.read_csv(DEG / "degraded_manifest.csv")
    linhas = []
    for cond in CONDS:
        for laz in sorted((DEG / cond).glob("*.laz")):
            tile = laz.stem
            if tile not in centros:
                continue
            cx, cy, campo = centros[tile]
            m = man[(man.condition == cond) & (man.tile == tile)]
            dens = float(m.iloc[0].density_pts_m2) if len(m) else float("nan")

            ff = FF_OUT / cond / f"{tile}_round2.ply"
            sat = SAT_OUT / cond
            sat_f = next(sat.rglob(f"{tile}_out.laz"), None) if sat.is_dir() else None
            linhas.append({
                "condition": cond, "tile": tile, "field": campo,
                "density_pts_m2": dens,
                "ff3d": conta_ff3d(ff, laz, cx, cy) if ff.exists() else np.nan,
                "sat": conta_sat(sat_f, cx, cy) if sat_f else np.nan,
            })
    d = pd.DataFrame(linhas)
    for m in ("ff3d", "sat"):
        d[f"{m}_pct"] = 100 * d[m] / d.field
        cheio = d[d.condition == "full"].set_index("tile")[f"{m}_pct"]
        d[f"{m}_rel"] = d.apply(
            lambda r: 100 * r[f"{m}_pct"] / cheio[r.tile]
            if cheio.get(r.tile, 0) > 0 else np.nan, axis=1)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    d.to_csv(SAIDA, index=False)

    g = d.groupby("condition").agg(
        dens=("density_pts_m2", "median"),
        ff3d=("ff3d_pct", "mean"), sat=("sat_pct", "mean"),
        ff3d_rel=("ff3d_rel", "mean"), sat_rel=("sat_rel", "mean"),
        n_sat=("sat", "count")).sort_values("dens")
    print(f"count inside the {PR:.2f} m circle (400 m²), both models ablated,\n"
          f"one pass per tile, without overlap and without adaptation\n")
    print(g.round(1).to_string())
    print(f"\n{SAIDA}")
    if g.n_sat.min() < 13:
        print("\nsome SegmentAnyTree condition is still incomplete")


if __name__ == "__main__":
    main()

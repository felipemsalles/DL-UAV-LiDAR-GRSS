#!/usr/bin/env python3
"""Build degraded copies of the 13 plot tiles, to answer "how cheap can the flight be?".

Structured degradation, not random decimation
---------------------------------------------
Random point removal does not correspond to any flight an operator would buy: a cheaper flight is
higher and faster, with fewer overlapping passes and a narrower usable swath. The delivered clouds
carry `scan_angle_rank` (±36°, only 24.6% of points inside ±10°) and `gps_time`, so such flights can
be simulated as structured degradations of the geometry rather than as random point loss — a domain
shift with a physical meaning, stacked on the temperate→eucalyptus shift the detector already faces.

Random thinning is still generated, as the control: if detection degrades the same way under random
thinning and under angular restriction at equal final density, only density matters; if it degrades
differently, geometry matters.

Conditions written (one 32 m tile per plot each):
  full            — untouched reference, so the run is self-contained
  dens_<n>        — random thinning to ~n points per m² (the control axis)
  ang_<d>         — keep |scan_angle_rank| <= d degrees (the realistic axis)
  ang<d>_dens<n>  — angular restriction then thinned to a matched density, to separate the two

Run (writes staging dirs only, does not invoke the GPU):
  PYTHONPATH=. GREENVISTA_LAZ_DIR=<lazdir> python scripts/ff3d_make_tiles_degraded.py [size] [outdir]
"""
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd

from greenvista import config
from greenvista.area_based import data as abd
from greenvista.repro import seed_everything

warnings.simplefilter("ignore")

SIZE = float(sys.argv[1]) if len(sys.argv) > 1 else 32.0
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else config.REPO / "work/ff3d_degraded"
SEED = 0

# (name, angle limit in degrees or None, target points per m² or None)
CONDITIONS = [
    ("full",            None, None),
    ("dens_200",        None, 200),
    ("dens_100",        None, 100),
    ("dens_50",         None, 50),
    ("dens_20",         None, 20),
    ("ang_20",          20,   None),
    ("ang_10",          10,   None),
    ("ang10_dens100",   10,   100),
]


def plot_table():
    df = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                        config.DATA / "5-dados_campo/inventario_est.csv")
    inv = pd.read_csv(config.DATA / "5-dados_campo/inventario_est.csv")
    df = df.merge(inv[["talhao", "parcela", "n_arv"]], on=["talhao", "parcela"], how="left")
    parc = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for c in ("talhao", "parcela"):
        parc[c] = pd.to_numeric(parc[c], errors="coerce").astype("Int64")
    parc["cx"], parc["cy"] = parc.geometry.x, parc.geometry.y
    return df.merge(parc[["talhao", "parcela", "cx", "cy"]], on=["talhao", "parcela"], how="left")


def degrade(mask_tile, angle, angle_limit, target_density, area_m2, rng):
    """Return the boolean mask of points to keep, given the tile mask and a condition.

    Angular restriction is applied first because it is the physical constraint; the density target
    is then met by random thinning of what survives. If the angular cut already leaves the cloud
    below the density target, no thinning is applied and the achieved density is reported as is;
    topping it up would fabricate points that flight never collected.
    """
    keep = mask_tile.copy()
    if angle_limit is not None:
        keep &= np.abs(angle) <= angle_limit
    if target_density is not None:
        n_target = int(round(target_density * area_m2))
        idx = np.where(keep)[0]
        if len(idx) > n_target:
            drop = rng.choice(idx, size=len(idx) - n_target, replace=False)
            keep[drop] = False
    return keep


def main():
    seed_everything(SEED)
    rng = np.random.default_rng(SEED)
    df = plot_table()
    area_m2 = SIZE * SIZE
    half = SIZE / 2.0
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for t in sorted(df.talhao.unique()):
        las = laspy.read(str(config.LAZ_DIR / f"SaoManuelTotal_{int(t):03d}.laz"))
        x, y = np.asarray(las.x), np.asarray(las.y)
        ang = np.asarray(las.scan_angle_rank).astype(float)
        for _, r in df[df.talhao == t].iterrows():
            in_tile = (np.abs(x - r.cx) <= half) & (np.abs(y - r.cy) <= half)
            name = f"t{int(t):03d}_p{int(r.parcela):03d}"
            for cond, a_lim, dens in CONDITIONS:
                keep = degrade(in_tile, ang, a_lim, dens, area_m2, rng)
                d = OUT / cond
                d.mkdir(parents=True, exist_ok=True)
                sub = laspy.LasData(las.header)
                sub.points = las.points[keep].copy()
                path = d / f"{name}.laz"
                sub.write(str(path))
                rows.append({"condition": cond, "tile": name, "plot_id": r.plot_id,
                             "talhao": int(t), "parcela": int(r.parcela),
                             "angle_limit_deg": a_lim, "target_density": dens,
                             "n_pts": int(keep.sum()),
                             "density_pts_m2": round(keep.sum() / area_m2, 1),
                             "field_n_arv": float(r.n_arv),
                             "laz_MB": round(path.stat().st_size / 1e6, 2)})
        del las, x, y, ang
        print(f"  stand {int(t)} done")

    man = pd.DataFrame(rows)
    man.to_csv(OUT / "degraded_manifest.csv", index=False)

    print(f"\n{len(man)} tiles in {len(CONDITIONS)} conditions -> {OUT}\n")
    s = man.groupby("condition").agg(densidade=("density_pts_m2", "mean"),
                                     pts=("n_pts", "mean"), MB=("laz_MB", "sum"))
    s["densidade"] = s.densidade.round(0)
    s["pts"] = s.pts.round(0)
    s["MB"] = s.MB.round(0)
    print(s.reindex([c[0] for c in CONDITIONS]).to_string())
    print(f"\nmanifest -> {OUT/'degraded_manifest.csv'}")


if __name__ == "__main__":
    main()

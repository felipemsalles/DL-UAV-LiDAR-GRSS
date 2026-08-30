#!/usr/bin/env python3
"""Wall-to-wall volume maps over stand 001.

Produces two maps for comparison:
  • GPR (the LOPO winner, rRMSE 13%) via the area-based metric mapper.
  • CNN (full-fit) via the sliding-window rasterizer.
Each → GeoPackage + PNG, with a stand total ± CI.

Run (repo root, greenvista env): PYTHONPATH=. python scripts/make_volume_maps.py
"""
import warnings
import numpy as np, laspy
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from greenvista import config
from greenvista.image_cnn import dataset as ds, train as tr, mapping as gmap
from greenvista.area_based import data as abd, models as abm, mapping as abmap

CACHE = config.LAZ_DIR / "plot_samples_r12_s128.npz"
TALHAO = 1


def _png(gdf, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    gdf.plot(column="vol_ha" if "vol_ha" in gdf else "volume_m3", ax=ax, legend=True, cmap="viridis",
             legend_kwds={"label": "Vol_ha (m³/ha)" if "vol_ha" in gdf else "volume (m³/cell)"})
    ax.set_title(title); ax.set_aspect("equal"); ax.set_xlabel("E (m)"); ax.set_ylabel("N (m)")
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def main():
    laz = str(config.LAZ_DIR / f"SaoManuelTotal_{TALHAO:03d}.laz")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- GPR map ---
    print("[GPR] fit on 13 plots + map stand 001 …")
    dfa = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                         config.DATA / "5-dados_campo/inventario_est.csv")
    gpr = abm.make_gpr().fit(dfa, dfa.Vol_ha.to_numpy())
    g_gpr = abmap.make_map(gpr, laz, cell=20.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g_gpr.to_file(str(config.OUT_DIR / f"volume_map_gpr_t{TALHAO:03d}.gpkg"), driver="GPKG")
    _png(g_gpr, f"GPR wall-to-wall volume — stand {TALHAO:03d}\nstand total ≈ {g_gpr.volume_m3.sum():.0f} m³",
         config.OUT_DIR / f"volume_map_gpr_t{TALHAO:03d}.png")
    print(f"  cells={len(g_gpr)}  stand total≈{g_gpr.volume_m3.sum():.0f} m³ "
          f"(CI {g_gpr.vol_lo.sum():.0f}..{g_gpr.vol_hi.sum():.0f})")

    # --- CNN map (comparison) ---
    print("[CNN] full-fit + sliding-window map …")
    samples = np.load(CACHE, allow_pickle=True)["samples"].item()
    ids = ds.build_plot_table().plot_id.tolist()
    net, nrm, resid = tr.fit_full(samples, ids, n_aug=64, epochs=80, seed=0)
    las = laspy.read(laz)
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
    g_cnn = gmap.volume_map(net, nrm, resid, xyz, cell=20.0, min_pts=400)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g_cnn.to_file(str(config.OUT_DIR / f"volume_map_cnn_t{TALHAO:03d}.gpkg"), driver="GPKG")
    _png(g_cnn, f"CNN wall-to-wall volume — stand {TALHAO:03d}\nstand total ≈ {g_cnn.volume_m3.sum():.0f} m³",
         config.OUT_DIR / f"volume_map_cnn_t{TALHAO:03d}.png")
    print(f"  cells={len(g_cnn)}  stand total≈{g_cnn.volume_m3.sum():.0f} m³ "
          f"(CI {g_cnn.vol_lo.sum():.0f}..{g_cnn.vol_hi.sum():.0f})")
    print(f"\nmaps written to {config.OUT_DIR}")


if __name__ == "__main__":
    main()

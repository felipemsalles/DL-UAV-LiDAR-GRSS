#!/usr/bin/env python3
"""Spike S6 — area-based plot/stand volume: model comparison (LOPO-CV) + wall-to-wall map.

Reports leave-one-stand-out next to LOPO, runs the leakage guard, and writes a run manifest +
map provenance (features, coefficients, training envelope) next to the outputs.

Run from the repo root (in the `greenvista` env): PYTHONPATH=. python scripts/spike_s6_area_based.py
"""
from greenvista.area_based import data as abd, validate as abv, mapping as abmap
from greenvista import config
from greenvista.repro import write_manifest, seed_everything

PM = config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv"
INV = config.DATA / "5-dados_campo/inventario_est.csv"
LAZ = config.TILE_LAZ


def main():
    seed_everything(config.ABA.seed)
    df = abd.load_plots(PM, INV)
    kept = abv.leakage_guard(df)                              # leakage guard before CV
    print(f"plots N={len(df)} | features={abd.FEATURES} | leakage-guard kept {len(kept)}/{len(abd.FEATURES)}\n")

    rep = abv.run(df, group_col=config.GROUP_COL, perm=2000)  # grouped CV + permutation p
    print("LOPO-CV model comparison (Vol_ha) — cov95 is conformal (C1/C2):")
    for name, mm in rep.items():
        grp = f" | LOTO R²={mm.get('R2_group', float('nan')):+.2f}" if "R2_group" in mm else ""
        print(f"  {name:14s} R²={mm['R2']:+.2f} (CI {mm['R2_ci'][0]:+.2f}..{mm['R2_ci'][1]:+.2f})"
              f"  rRMSE={mm['rRMSE_pct']:.0f}%  cov95={mm['coverage95']:.2f}"
              f"  p={mm.get('perm_p', float('nan')):.3f}{grp}")

    best = max(rep, key=lambda k: rep[k]["R2"])
    print(f"\nbest by CV R²: {best} -> fitting on all plots and mapping {LAZ}")
    model = abv.DEFAULT_MODELS[best]().fit(df, df.Vol_ha.to_numpy())
    envelope = abmap.feature_envelope(df)                     # reference used to flag extrapolation
    gdf = abmap.make_map(model, LAZ, cell=config.ABA.map_cell_m, envelope=envelope)
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUT_DIR / "s6_volume_map_t001.gpkg"
    gdf.to_file(str(out), driver="GPKG")
    n_extrap = int(gdf.extrapolated.sum()) if "extrapolated" in gdf else 0
    abmap.save_map_provenance(model, abd.FEATURES, config.OUT_DIR / "s6_map_provenance.json",
                              envelope=envelope, extra={"best_model": best})
    write_manifest(config.OUT_DIR / "s6_manifest.json", script="spike_s6_area_based.py",
                   config=config.ABA, seed=config.ABA.seed,
                   metrics={k: {mk: mv for mk, mv in v.items() if mk != "R2_ci"} for k, v in rep.items()})
    print(f"map cells={len(gdf)} ({n_extrap} extrapolated, {gdf.attrs.get('n_skipped_nan',0)} skipped-NaN)  "
          f"stand total≈{gdf.volume_m3.sum():.0f} m³  -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check whether the huber_allom+rumple gain holds across stands (not a single-stand artefact).

- Per-stand leave-one-stand-out residuals for the baseline vs the rumple model.
- Bootstrap CI on the LOTO rRMSE for each.
- A 4-panel figure: two plot CHMs (low- vs high-rumple), observed-vs-predicted (winner), and
  residual-vs-rumple for the baseline (shows the structure signal rumple captures).

Run: PYTHONPATH=. GREENVISTA_LAZ_DIR=<lazdir> python scripts/verify_and_viz.py
"""
import warnings
import numpy as np, pandas as pd, laspy, geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from greenvista import config
from greenvista.repro import seed_everything
from greenvista.area_based import data as abd, models as abm, validate as abv
from greenvista.eval import metrics, bootstrap_ci
from greenvista.image_cnn import footprint
from greenvista.segmentation import build_chm

AGE = abd.AGE


def load_table():
    df = abd.load_plots(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv",
                        config.DATA / "5-dados_campo/inventario_est.csv")
    s = pd.read_csv(config.OUT_DIR / "plot_structural.csv")[["plot_id", "rumple", "cover", "penetration"]]
    return df.merge(s, on="plot_id", how="left")


def main():
    seed_everything(0)
    df = load_table()
    y = df.Vol_ha.to_numpy()

    base = lambda: abm.make_huber_allom(("zmean", "zq90", AGE))
    winner = lambda: abm.make_huber_allom(("zmean", "zq90", AGE, "rumple"))

    loto_base = abv._group_cv(base, df)
    loto_win = abv._group_cv(winner, df)

    print("=== leave-one-stand-out, per stand (Vol_ha) ===")
    print(f"  {'talhao':6} {'obs':>7} {'base':>7} {'+rumple':>8} {'Δ|err|':>7}")
    for t in sorted(df.talhao.unique()):
        m = df.talhao.to_numpy() == t
        e_b = np.abs(loto_base[m] - y[m]).mean(); e_w = np.abs(loto_win[m] - y[m]).mean()
        print(f"  {t:<6} {y[m].mean():7.0f} {loto_base[m].mean():7.0f} {loto_win[m].mean():8.0f} "
              f"{e_b - e_w:+7.0f}")

    mb, mw = metrics(y, loto_base), metrics(y, loto_win)
    cib = bootstrap_ci(y, loto_base); ciw = bootstrap_ci(y, loto_win)
    print(f"\nLOTO base    : R²={mb['R2']:+.2f} rRMSE={mb['rRMSE_pct']:.1f}% "
          f"(rRMSE CI {cib['rRMSE_ci'][0]:.1f}..{cib['rRMSE_ci'][1]:.1f})")
    print(f"LOTO +rumple : R²={mw['R2']:+.2f} rRMSE={mw['rRMSE_pct']:.1f}% "
          f"(rRMSE CI {ciw['rRMSE_ci'][0]:.1f}..{ciw['rRMSE_ci'][1]:.1f})")
    n_better = int((np.abs(loto_win - y) < np.abs(loto_base - y)).sum())
    print(f"per-plot: +rumple beats base on {n_better}/{len(y)} plots")

    # ---- figure ----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parc = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    parc["talhao"] = pd.to_numeric(parc.talhao, errors="coerce").astype("Int64")
    parc["parcela"] = pd.to_numeric(parc.parcela, errors="coerce").astype("Int64")
    parc["cx"], parc["cy"] = parc.geometry.x, parc.geometry.y
    df = df.merge(parc[["talhao", "parcela", "cx", "cy"]], on=["talhao", "parcela"], how="left")

    def chm_for(plot_id):
        r = df[df.plot_id == plot_id].iloc[0]
        las = laspy.read(str(config.LAZ_DIR / f"SaoManuelTotal_{int(r.talhao):03d}.laz"))
        xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
        pts = footprint.extract_footprint(xyz, (r.cx, r.cy), config.PLOT_RADIUS_M)
        c = pts[(pts[:, 2] >= 2) & (pts[:, 2] <= 45)]
        chm, *_ = build_chm(c[:, :2], c[:, 2], res=0.5)
        return chm, r

    lo_id = df.loc[df.rumple.idxmin(), "plot_id"]
    hi_id = df.loc[df.rumple.idxmax(), "plot_id"]
    chm_lo, r_lo = chm_for(lo_id)
    chm_hi, r_hi = chm_for(hi_id)

    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    for a, chm, r in [(ax[0, 0], chm_lo, r_lo), (ax[0, 1], chm_hi, r_hi)]:
        im = a.imshow(chm.T, origin="lower", cmap="viridis", vmin=0, vmax=32)
        a.set_title(f"CHM plot {r.plot_id} (stand {int(r.talhao)}, age {r[AGE]:.1f}y)\n"
                    f"rumple={r.rumple:.1f}, Vol_ha={r.Vol_ha:.0f}")
        a.set_xticks([]); a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=0.046, label="height (m)")

    a = ax[1, 0]
    a.scatter(y, loto_win, c="tab:green", s=60, edgecolor="k", zorder=3, label="+rumple")
    a.scatter(y, loto_base, c="tab:gray", s=40, alpha=0.6, zorder=2, label="base (no rumple)")
    lim = [y.min() * 0.9, y.max() * 1.1]
    a.plot(lim, lim, "k--", lw=1)
    a.set_xlabel("observed Vol_ha (m³/ha)"); a.set_ylabel("LOTO predicted")
    a.set_title(f"Leave-one-stand-out fit\n+rumple R²={mw['R2']:.2f} vs base {mb['R2']:.2f}")
    a.legend(); a.set_xlim(lim); a.set_ylim(lim)

    a = ax[1, 1]
    a.scatter(df.rumple, loto_base - y, c=df[AGE], cmap="autumn", s=60, edgecolor="k")
    a.axhline(0, color="k", lw=1)
    a.set_xlabel("rumple (canopy roughness)"); a.set_ylabel("baseline LOTO residual (pred−obs)")
    a.set_title("Baseline residual vs rumple\n(structure signal rumple adds; colour = age)")
    plt.colorbar(a.collections[0], ax=a, fraction=0.046, label="age (y)")

    fig.suptitle("Why rumple helps: canopy roughness carries cross-stand volume signal (São Manuel, N=13)",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = config.OUT_DIR / "rumple_verification.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote figure {out}")


if __name__ == "__main__":
    main()

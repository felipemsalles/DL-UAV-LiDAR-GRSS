"""CHM watershed as a second classical baseline, head-to-head with CHM local maxima.

Watershed and local maxima are different algorithms in the ITD literature — Guangxi 2024 benchmarks them
side by side as WA and LMA — so reporting a local-maxima detector under the watershed name is wrong.
This script runs watershed proper and reports both, measured with the same ruler on the same data.

Watershed is swept over six settings (five h-maxima tolerances plus no basin merging) so the baseline is
reported at its best setting, not at one that cripples it: lidR's default (tol=1 m) more than halves
detection, and reporting only that would mirror the per-plot parameter tuning criticised in Guangxi
2024.

Protocol (identical to scripts/exp_density_count.py, so the numbers are comparable):
  * clouds clipped to a ±PLOT_RADIUS_M square around each plot centre, ground/noise classes dropped;
  * CHM = max-Z per 0.5 m cell, gaussian smooth 0.6, canopy mask z > 5 m;
  * counted inside the PLOT_RADIUS_M (12 m) circle.

Known bias, quantified 2026-08-14 and not corrected here because that would break comparability with the
stored numbers: detections are counted over a 12 m circle (452 m²) while the field census is the 400 m²
plot (r=11.28 m). The stand-001 TLS stem map puts ~9% more stems inside 12 m than inside 11.28 m, so
every detection rate in this table, ours and the baselines', is optimistic by roughly that much. It hits
all methods equally, so the ranking holds. See docs for the corrected-denominator run.

Run: PYTHONPATH=. GREENVISTA_LAZ_DIR=<lazdir> python scripts/exp_watershed_baseline.py
Out: manual_match/watershed_sweep.csv
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import laspy

from greenvista import config
from greenvista.segmentation import build_chm, detect_treetops, detect_watershed

CHM_RES, CHM_HMIN, CHM_WIN, CHM_SMOOTH = 0.5, 5.0, 3, 0.6
PR = config.PLOT_RADIUS_M
# Two rulers: the 12 m radius is the one the project has used from the start, but it
# covers 452 m²; the field census measured 400 m², i.e. a radius of 11.28 m. Counting
# under both keeps the numbers comparable with the older tables while also giving the
# number with the correct denominator.
import math
PR_400 = math.sqrt(400 / math.pi)
TOLS = [None, 0.25, 0.5, 1.0, 1.5, 2.0]     # None = no basin merging


def _field_counts():
    inv = pd.read_csv(config.DATA / "5-dados_campo/inv_euc.csv")
    return inv[inv.D_cm.notna()].groupby(["talhao", "parcela"]).size()


def _plots():
    parc = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    for c in ("talhao", "parcela"):
        parc[c] = pd.to_numeric(parc[c], errors="coerce").astype("Int64")
    parc["cx"], parc["cy"] = parc.geometry.x, parc.geometry.y
    return parc


def _n_in_circle(pts, cx, cy, r=PR):
    if len(pts) == 0:
        return 0
    return int((np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) <= r).sum())


def run():
    campo, parc = _field_counts(), _plots()
    rows = []
    for t in sorted({int(v) for v in parc.talhao.dropna()}):
        laz = config.LAZ_DIR / f"SaoManuelTotal_{t:03d}.laz"
        if not laz.exists():
            print(f"stand {t:03d}: cloud missing, skipping")
            continue
        las = laspy.read(str(laz))
        X, Y, Z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        cls = np.asarray(las.classification)
        veg = (cls != config.GROUND_NOISE_CLASSES[0]) & (cls != config.GROUND_NOISE_CLASSES[1])
        for _, r in parc[parc.talhao == t].iterrows():
            p = int(r.parcela)
            if (t, p) not in campo.index:
                continue
            m = veg & (np.abs(X - r.cx) <= PR) & (np.abs(Y - r.cy) <= PR)
            chm, x0, y0, res = build_chm(np.column_stack([X[m], Y[m]]), Z[m], res=CHM_RES)
            row = {"talhao": t, "parcela": p, "campo": int(campo[(t, p)])}
            lm = detect_treetops(chm, x0, y0, res, window=CHM_WIN, hmin=CHM_HMIN, smooth=CHM_SMOOTH)
            row["maximo_local"] = _n_in_circle(lm, r.cx, r.cy)
            row["maximo_local_r400"] = _n_in_circle(lm, r.cx, r.cy, PR_400)
            for tol in TOLS:
                ws = detect_watershed(chm, x0, y0, res, hmin=CHM_HMIN, smooth=CHM_SMOOTH, tol=tol)
                nome = "ws_sem_fusao" if tol is None else f"ws_tol{tol}"
                row[nome] = _n_in_circle(ws, r.cx, r.cy)
                row[nome + "_r400"] = _n_in_circle(ws, r.cx, r.cy, PR_400)
            rows.append(row)
            print(f"  stand {t:03d} plot {p:03d}: {row}")

    df = pd.DataFrame(rows).sort_values(["talhao", "parcela"]).reset_index(drop=True)
    out = config.OUT_DIR / "watershed_sweep.csv"
    df.to_csv(out, index=False)

    tot = df.drop(columns=["talhao", "parcela"]).sum()
    campo_tot = tot.pop("campo")
    print(f"\n{len(df)} plots, {campo_tot} trees counted in the field "
          f"({PR:.0f} m circle; see the denominator caveat in the header)\n")
    print(f"{'method':>28} {'trees':>8} {'rate':>7}")
    for k, v in tot.sort_values(ascending=False).items():
        print(f"{k:>28} {int(v):>8} {v / campo_tot * 100:>6.1f}%")
    print(f"\nwritten to {out}")
    return df


if __name__ == "__main__":
    run()

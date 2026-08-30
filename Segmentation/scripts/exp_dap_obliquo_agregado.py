#!/usr/bin/env python3
"""DBH measured on the oblique scan misses the tree, but does it hit the population? Is that useful?

`exp_dap_do_obliquo.py` extracted stem radius from the ring that appears in the oblique band of the
flight. Per tree it correlates weakly with TLS (Spearman between +0.02 and +0.24), but the median
came out at 19.8 cm against 19.0 cm from TLS, 4 % error with no training, no labels and no model.
That is a direct measurement, not a prediction.

Hitting the median is not enough: an estimator can have the right median and still be useless if the
per-tree noise widens the estimated distribution far beyond the real one. What is tested here:
 1. the whole distribution, and not only the median;
 2. the per-plot value, which is the scale at which the inventory decides;
 3. the per-plot basal area, which is what enters the volume chain;
 4. the same estimator applied to random positions, where there is no stem. If it returns 19 cm in
    empty space too, it is measuring the limit of the search grid and not the stem. This control is
    what separates a measurement from an artefact.

Usage: PYTHONPATH=. python scripts/exp_dap_obliquo_agregado.py
Output: manual_match/dap_obliquo_agregado.csv
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from greenvista import config  # noqa: E402

_s = importlib.util.spec_from_file_location("obl", R / "scripts/exp_dap_do_obliquo.py")
_o = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_o)
warnings.filterwarnings("ignore")

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
DRONE = config.REPO / "work/lazall/SaoManuelTotal_001.laz"
SAIDA = config.OUT_DIR / "dap_obliquo_agregado.csv"
CONFIGS = [(0, 36, 1.0, 6.0), (20, 36, 1.0, 6.0), (20, 30, 1.0, 6.0), (0, 36, 2.0, 10.0)]
AREA_PARCELA = 400.0
SEED = 20260828


def estima(X, Y, pos):
    T = cKDTree(np.column_stack([X, Y]))
    out = np.full(len(pos), np.nan)
    for i, (cx, cy) in enumerate(pos):
        idx = np.asarray(T.query_ball_point([cx, cy], _o.R_BUSCA), dtype=int)
        if len(idx) < _o.PTS_MIN:
            continue
        r, _, _ = _o.raio_por_modo(X[idx] - cx, Y[idx] - cy)
        out[i] = 200 * r if np.isfinite(r) else np.nan
    return out


def main():
    rng = np.random.default_rng(SEED)
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])
    ran = pd.read_csv(config.OUT_DIR / "dap_tls_ransac_talhao001.csv")
    dap_tls = np.where(ran.motivo.values == "ok", ran.dap_cm.values, np.nan)

    las = laspy.read(str(DRONE))
    cls = np.asarray(las.classification)
    veg = ~np.isin(cls, config.GROUND_NOISE_CLASSES)
    x, y, z = np.asarray(las.x)[veg], np.asarray(las.y)[veg], np.asarray(las.z)[veg]
    ang = np.abs(np.asarray(las.scan_angle_rank).astype(float))[veg]

    # control positions: drawn inside the same rectangle, far from any stem
    T_ref = cKDTree(ref)
    falsas = []
    while len(falsas) < len(ref):
        p = np.column_stack([rng.uniform(ref[:, 0].min(), ref[:, 0].max(), 400),
                             rng.uniform(ref[:, 1].min(), ref[:, 1].max(), 400)])
        d, _ = T_ref.query(p)
        falsas.extend(p[d > 0.8].tolist())
    falsas = np.array(falsas[:len(ref)])
    print(f"{len(ref)} true stems and {len(falsas)} control positions "
          f"(more than 80 cm from any stem)\n")

    linhas = []
    for a0, a1, h0, h1 in CONFIGS:
        m = (z >= h0) & (z <= h1) & (ang >= a0) & (ang < a1)
        est = estima(x[m], y[m], ref)
        ctl = estima(x[m], y[m], falsas)
        ok = np.isfinite(est) & np.isfinite(dap_tls)
        nome = f"ang {a0}-{a1}, h {h0}-{h1}"
        print(f"=== {nome}   {ok.sum()} stems measured, "
              f"{np.isfinite(ctl).sum()} controls measured")
        print(f"   {'quantile':>9} {'TLS':>8} {'drone':>8} {'control':>9}")
        for q in (10, 25, 50, 75, 90):
            print(f"   {q:8d}% {np.nanpercentile(dap_tls[ok], q):8.1f} "
                  f"{np.nanpercentile(est[ok], q):8.1f} "
                  f"{np.nanpercentile(ctl, q) if np.isfinite(ctl).sum() > 30 else float('nan'):9.1f}")
        print(f"   {'mean':>9} {np.nanmean(dap_tls[ok]):8.1f} {np.nanmean(est[ok]):8.1f} "
              f"{np.nanmean(ctl):9.1f}")
        print(f"   {'sd':>9} {np.nanstd(dap_tls[ok]):8.1f} {np.nanstd(est[ok]):8.1f} "
              f"{np.nanstd(ctl):9.1f}")

        # The control decides: if the empty position returns the same distribution, the estimator is
        # measuring the search grid and not the stem.
        sep = abs(np.nanmedian(est[ok]) - np.nanmedian(ctl))
        print(f"   separation between stem and empty space: {sep:.2f} cm of median")

        # per virtual plot
        Rp = np.sqrt(AREA_PARCELA / np.pi)
        gx = np.arange(ref[:, 0].min() + Rp, ref[:, 0].max() - Rp, Rp)
        gy = np.arange(ref[:, 1].min() + Rp, ref[:, 1].max() - Rp, Rp)
        parc = []
        for cx in gx:
            for cy in gy:
                k = (np.hypot(ref[:, 0] - cx, ref[:, 1] - cy) <= Rp) & ok
                if k.sum() < 15:
                    continue
                parc.append(dict(n=int(k.sum()), tls=float(np.mean(dap_tls[k])),
                                 drone=float(np.mean(est[k])),
                                 g_tls=float(np.sum(np.pi * (dap_tls[k] / 200) ** 2)),
                                 g_drone=float(np.sum(np.pi * (est[k] / 200) ** 2))))
        if len(parc) >= 6:
            P = pd.DataFrame(parc)
            r = float(np.corrcoef(P.tls, P.drone)[0, 1])
            vies = 100 * (P.drone.mean() / P.tls.mean() - 1)
            rr = 100 * float(np.sqrt(np.mean((P.drone - P.tls) ** 2))) / P.tls.mean()
            rg = float(np.corrcoef(P.g_tls, P.g_drone)[0, 1])
            print(f"   per plot ({len(P)} of 400 m2): r(mean DBH) {r:+.3f}, "
                  f"bias {vies:+.1f} %, rRMSE {rr:.1f} %,  r(basal area) {rg:+.3f}")
            linhas.append(dict(cond=nome, n_medidos=int(ok.sum()),
                               mediana_tls=float(np.nanmedian(dap_tls[ok])),
                               mediana_drone=float(np.nanmedian(est[ok])),
                               mediana_controle=float(np.nanmedian(ctl)),
                               separacao_cm=sep, n_parcelas=len(P), r_parcela=r,
                               vies_parcela=vies, rrmse_parcela=rr, r_basal=rg))
        print()

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(linhas).to_csv(SAIDA, index=False)
    print(SAIDA)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract DBH from the terrestrial cloud at the known stem positions, and measure the error.

Extracting DBH from a terrestrial cloud normally requires finding and isolating each tree
first. Here the 892 positions come ready in the supplier's map, so all that remains is to
fit the section, a much narrower problem than the one TreeQSM, SimpleForest and the like
solve.

Six implementation decisions, each matching one source of overestimation:
 1. The fit is centred on the stem. In absolute UTM coordinates (x of order 7e5) the
    system becomes ill-conditioned and the radius comes out in the millions.
 2. Adaptive search radius, updated by the value estimated in the previous iteration. A
    fixed 25 cm radius reaches the neighbour on a 13 cm stem and cuts the trunk on a 40 cm one.
 3. Taubin fit instead of Kasa: the terrestrial laser sees one side of the trunk, and Kasa
    inflates the radius in that case. A geometric refinement follows.
 4. DBH as the median of several thin slices, exploiting the vertical continuity of the
    trunk and discarding contaminated slices.
 5. Terrain on a 0.5 m grid. A minimum over a 2 m cell misses the ground on sloping terrain
    and the breast-height band then samples another part of the trunk.
 6. Quality control: a slice with little angular coverage or a high residual is discarded,
    and a stem with too few good slices comes out as not measured.

Validation is against destructive scaling, and not against another estimate of ours. The 26
scaled trees of stand 001 are georeferenced and all fall inside the cloud, within 8 cm of the
nearest mapped stem.

Usage: PYTHONPATH=. python scripts/exp_dap_tls.py
Output: manual_match/dap_tls_talhao001.csv
"""
import sys
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenvista import config  # noqa: E402

MAPA = config.REPO / "project/models/SegmentAnyTree_blackwell/src_ref/localizacao_arv.shp"
SAIDA = config.OUT_DIR / "dap_tls_talhao001.csv"

G_SOLO = 0.5              # terrain grid, in metres
H_FATIA = (1.0, 1.6)      # section of the trunk used, above the ground
N_FATIAS = 4
R_BUSCA = 0.60            # initial search radius, generous on purpose
COB_MIN = 120.0           # minimum angular coverage of a slice, in degrees
RES_MAX = 0.012           # maximum median residual of a slice, in metres
FATIAS_MIN = 2


def terreno(x, y, z):
    """Terrain by minimum over a 0.5 m cell, holes filled and then smoothed."""
    x0, y0 = x.min() - 1, y.min() - 1
    nx = int((x.max() - x0) / G_SOLO) + 2
    ny = int((y.max() - y0) / G_SOLO) + 2
    zmin = np.full((nx, ny), 1e9, np.float32)
    np.minimum.at(zmin, (((x - x0) / G_SOLO).astype(int), ((y - y0) / G_SOLO).astype(int)), z)
    zmin[zmin > 1e8] = np.nan
    vazio = ~np.isfinite(zmin)
    if vazio.any():
        _, (a, b) = distance_transform_edt(vazio, return_indices=True)
        zmin[vazio] = zmin[a[vazio], b[vazio]]
    return uniform_filter(zmin, 7), x0, y0


def taubin(px, py):
    """Taubin algebraic circle fit. Less biased than Kasa on a partial arc."""
    mx, my = px.mean(), py.mean()
    u, v = px - mx, py - my
    zz = u * u + v * v
    zm = zz.mean()
    m = np.column_stack([zz - zm, u, v])
    _, _, vt = np.linalg.svd(m, full_matrices=False)
    a, b, c = vt[-1]
    if abs(a) < 1e-12:
        return np.nan, np.nan, np.nan
    a2 = 2 * a
    cx, cy = -b / a2, -c / a2
    r2 = cx * cx + cy * cy + zm
    if r2 <= 0:
        return np.nan, np.nan, np.nan
    return np.sqrt(r2), cx + mx, cy + my


def refina(px, py, r, cx, cy):
    """Geometric refinement, minimising the true distance to the circle instead of the algebraic one."""
    def res(p):
        return np.hypot(px - p[0], py - p[1]) - p[2]
    s = least_squares(res, [cx, cy, r], method="lm", max_nfev=60)
    return s.x[2], s.x[0], s.x[1]


def ajusta_fatia(px, py):
    """Fit with an adaptive search radius. Returns radius, residual, coverage and offset."""
    cx = cy = 0.0
    r = None
    for passo in range(4):
        d = np.hypot(px - cx, py - cy)
        sel = d < R_BUSCA if r is None else np.abs(d - r) < max(0.05, 0.25 * r)
        if sel.sum() < 25:
            return (np.nan,) * 4
        r_, cx_, cy_ = taubin(px[sel], py[sel])
        if not np.isfinite(r_) or r_ > 0.35:
            return (np.nan,) * 4
        r, cx, cy = (refina(px[sel], py[sel], r_, cx_, cy_) if passo else (r_, cx_, cy_))
        if not np.isfinite(r) or r <= 0:
            return (np.nan,) * 4
    d = np.hypot(px - cx, py - cy)
    sel = np.abs(d - r) < max(0.05, 0.25 * r)
    ang = np.degrees(np.arctan2(py[sel] - cy, px[sel] - cx))
    cob = len(np.unique((ang // 15).astype(int))) * 15.0
    return r, float(np.median(np.abs(d[sel] - r))), cob, float(np.hypot(cx, cy))


def main():
    if not config.TLS_LAS.exists():
        sys.exit(f"TLS cloud not found at {config.TLS_LAS}")
    f = gpd.read_file(MAPA)
    ref = np.column_stack([f.geometry.x.values, f.geometry.y.values])

    print(f"reading {config.TLS_LAS.name}")
    xs, ys, zs = [], [], []
    with laspy.open(config.TLS_LAS) as fh:
        for p in fh.chunk_iterator(6_000_000):
            xs.append(np.asarray(p.x)); ys.append(np.asarray(p.y))
            zs.append(np.asarray(p.z, dtype=np.float32))
    x, y, z = np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)
    zsolo, x0, y0 = terreno(x, y, z)
    h = z - zsolo[np.clip(((x - x0) / G_SOLO).astype(int), 0, zsolo.shape[0] - 1),
                  np.clip(((y - y0) / G_SOLO).astype(int), 0, zsolo.shape[1] - 1)]

    faixa = (h > H_FATIA[0]) & (h < H_FATIA[1])
    X, Y, H = x[faixa], y[faixa], h[faixa]
    print(f"{len(X):,} points between {H_FATIA[0]} and {H_FATIA[1]} m above the ground")
    T = cKDTree(np.column_stack([X, Y]))
    bordas = np.linspace(*H_FATIA, N_FATIAS + 1)

    linhas = []
    for k, (cx0, cy0) in enumerate(ref):
        idx = np.array(T.query_ball_point([cx0, cy0], R_BUSCA), dtype=int)
        if len(idx) < 60:
            linhas.append({"fuste": k, "dap_cm": np.nan, "n_fatias": 0, "n_pts": len(idx)})
            continue
        px, py, ph = X[idx] - cx0, Y[idx] - cy0, H[idx]
        raios, resid = [], []
        for i in range(N_FATIAS):
            m = (ph >= bordas[i]) & (ph < bordas[i + 1])
            if m.sum() < 30:
                continue
            r, res, cob, desl = ajusta_fatia(px[m], py[m])
            if np.isfinite(r) and res < RES_MAX and cob >= COB_MIN and desl < 0.20:
                raios.append(r); resid.append(res)
        linhas.append({"fuste": k, "x": cx0, "y": cy0,
                       "dap_cm": 200 * np.median(raios) if len(raios) >= FATIAS_MIN else np.nan,
                       "n_fatias": len(raios), "n_pts": len(idx),
                       "residuo_cm": 100 * np.median(resid) if resid else np.nan})
    d = pd.DataFrame(linhas)
    ok = d.dap_cm.notna()
    print(f"\nDBH obtained on {ok.sum()} of the {len(ref)} stems ({100 * ok.mean():.0f}%)")
    print(f"  median {d.dap_cm.median():.1f} cm, mean {d.dap_cm.mean():.1f}, "
          f"sd {d.dap_cm.std():.1f}")
    ab = 1379 * np.pi * (d.dap_cm[ok] / 200) ** 2
    print(f"  basal area implied at 1379 trees/ha: {ab.mean():.1f} m²/ha "
          f"(expected 25 to 35)")

    # validation against the destructively scaled trees
    a = pd.read_csv(config.GEOLOC_CSV)
    a["arv"] = pd.to_numeric(a.Name.astype(str).str.extract(r"^(\d+)")[0], errors="coerce")
    v = pd.read_csv(config.VOLUMES_CSV)
    a = a.join(v[v.talhao == 1].set_index("arv")[["D_cm", "H_m", "V"]], on="arv")
    D = np.hypot(ref[:, None, 0] - a.Easting.values[None, :],
                 ref[:, None, 1] - a.Northing.values[None, :])
    a["dap_tls"] = d.dap_cm.values[D.argmin(0)]
    k = a.dropna(subset=["D_cm", "dap_tls"])
    r = np.corrcoef(k.D_cm, k.dap_tls)[0, 1]
    vies = (k.dap_tls / k.D_cm).mean()
    print(f"\nvalidation on {len(k)} scaled trees")
    print(f"  r = {r:.3f}, r² = {r ** 2:.3f}")
    print(f"  the TLS measures {vies:.2f} times the field ({100 * (vies - 1):+.0f}%)")
    print(f"  raw RMSE {np.sqrt(((k.dap_tls - k.D_cm) ** 2).mean()):.2f} cm")
    print(f"  RMSE after correcting the scale "
          f"{np.sqrt(((k.dap_tls / vies - k.D_cm) ** 2).mean()):.2f} cm")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    d.to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

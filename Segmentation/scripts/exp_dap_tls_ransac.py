#!/usr/bin/env python3
"""DBH by RANSAC cylinder fitting on the terrestrial cloud, at the known stem positions.

The radius estimator is the mode of the radial density per unit area. The alternatives fail for
geometric reasons: least squares chases the envelope of the point shell and inflates by ~30%;
RANSAC scored by inlier count inflates the same, because in a band of fixed width the larger
circle sweeps more area and collects more noise; and scored by arc density (/ 2*pi*r) it starts to
prefer a tiny circle. In the per-area density over concentric rings the trunk surface appears as a
clean peak, while SLAM drift and understorey stay spread out. Validated against the taper, which
is physically correct: 21 cm at 0.4 m down to 11 cm at 6 m of height.

The point shell around the trunk is 3 cm thick as measured, against the 1 to 2 expected, because
handheld SLAM accumulates drift and the same trunk appears as several slightly displaced surfaces.
RANSAC looks for the dominant surface, that is the densest pass, and treats the rest as outliers.

The axis is estimated, not assumed vertical: a leaning tree projected as vertical becomes an
ellipse, and a circle fitted to an ellipse comes out larger than the minor axis. The direction
comes from PCA over the section itself and is only accepted within 25 degrees of the vertical.

The validation is the distribution of the inventory, and not r2 against the destructively scaled
trees. The GPS of the scaled trees is of metric scale and in a 1.88 m plantation any point falls
near some stem, so matching tree with tree returns noise (see `exp_gps_ruler.py`). The inventory
of stand 001 has 120 measured trees, and it is against their distribution that the result is
compared.

Over-bark against over-bark: the inventory DBH is measured with a tape over the bark, which is what
the laser sees, so no correction enters. The `D_cm` of the destructive scaling is under bark and is
not suitable for direct comparison.

Usage: PYTHONPATH=. python scripts/exp_dap_tls_ransac.py
Output: manual_match/dap_tls_ransac_talhao001.csv
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
SAIDA = config.OUT_DIR / "dap_tls_ransac_talhao001.csv"

G_SOLO = 0.5
H_TRECHO = (1.0, 1.6)      # section of trunk used, above the ground
R_BUSCA = 0.55
TOL = 0.010                # half thickness of the inlier band, in metres
BIN = 0.005                # width of the ring in the radial histogram, in metres
# The agreement lock is loose on purpose, and the value was swept: testing from 0.6 to 3.0 cm over
# three dispersion statistics, the yield varies from 14% to 91% and the quality does not change
# (median fixed at 19.0 cm, standard deviation between 4.5 and 4.9, per-quantile error between 9.5
# and 11.5%). The filter does not separate a good measurement from a bad one, it only discards at
# random, so it stays at the value that cuts only the pathological cases. The remaining 8% bias
# belongs to the estimator, and not to a contaminated stem.
DISCORDA_MAX = 0.030       # maximum median absolute deviation between slices, in metres
N_FATIAS = 6
PTS_FATIA = 20             # minimum points for a slice to count
FATIAS_MIN = 2
R_MIN, R_MAX = 0.030, 0.220
COB_MIN = 100.0            # minimum angular coverage of the inliers, in degrees
INLIER_MIN = 80


def terreno(x, y, z):
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


def eixo(P):
    """Direction of the section by PCA. Falls back to vertical if it leans too much, which
    indicates that the crop caught a branch or understorey instead of a trunk."""
    Q = P - P.mean(0)
    d = np.linalg.svd(Q, full_matrices=False)[2][0]
    if d[2] < 0:
        d = -d
    return d if d[2] > np.cos(np.radians(25)) else np.array([0.0, 0.0, 1.0])


def base_ortogonal(d):
    a = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d, a); u /= np.linalg.norm(u)
    return u, np.cross(d, u)


def circunscrito(p1, p2, p3):
    """Circumcircle of each triple, vectorised. An almost collinear triple returns a huge radius
    and is discarded by the radius filter."""
    ax, ay = p1[:, 0], p1[:, 1]
    bx, by = p2[:, 0], p2[:, 1]
    cx, cy = p3[:, 0], p3[:, 1]
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    with np.errstate(divide="ignore", invalid="ignore"):
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
              + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
              + (cx**2 + cy**2) * (bx - ax)) / d
    return ux, uy, np.hypot(ax - ux, ay - uy)


def raio_por_modo(pu, pv, cu=0.0, cv=0.0, minimo=None):
    """Radius from the peak of the radial density, with the centre readjusted at each pass.

    The density is per unit area, that is, the count of the ring divided by its area. Without that
    division the outer ring wins merely by being larger and the radius comes out inflated.
    """
    # The inlier minimum is parameterisable and does not inherit the one of the whole section: a
    # 15 cm slice has four times fewer points, and requiring the same 80 rejects a quarter of the
    # stems because of a threshold designed for another scale.
    minimo = INLIER_MIN if minimo is None else minimo
    aneis = np.arange(0.0, R_MAX + BIN, BIN)
    area = np.pi * (aneis[1:] ** 2 - aneis[:-1] ** 2)
    r = np.nan
    for _ in range(4):
        d = np.hypot(pu - cu, pv - cv)
        hist, _ = np.histogram(d, bins=aneis)
        dens = hist / area
        dens[aneis[:-1] < R_MIN] = 0.0
        if not dens.any():
            return np.nan, np.nan, np.nan, np.zeros(len(pu), bool)
        i = int(dens.argmax())
        r = 0.5 * (aneis[i] + aneis[i + 1])
        m = np.abs(d - r) < TOL * 1.5
        if m.sum() < minimo:
            return np.nan, np.nan, np.nan, m
        # the centre is refined on the peak points; the radius does not come from the fit
        s = least_squares(lambda p: np.hypot(pu[m] - p[0], pv[m] - p[1]) - r,
                          [cu, cv], method="lm", max_nfev=60)
        cu, cv = s.x
    d = np.hypot(pu - cu, pv - cv)
    return r, cu, cv, np.abs(d - r) < TOL * 1.5


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
    k = (h > H_TRECHO[0]) & (h < H_TRECHO[1])
    X, Y, H = x[k], y[k], h[k]
    print(f"{len(X):,} points in the section from {H_TRECHO[0]} to {H_TRECHO[1]} m")
    T = cKDTree(np.column_stack([X, Y]))

    linhas = []
    for j, (cx0, cy0) in enumerate(ref):
        idx = np.array(T.query_ball_point([cx0, cy0], R_BUSCA), dtype=int)
        reg = {"fuste": j, "x": cx0, "y": cy0, "n_pts": len(idx),
               "dap_cm": np.nan, "inliers": 0, "cobertura": np.nan, "residuo_cm": np.nan}
        motivo = "poucos pontos no recorte"
        if len(idx) >= INLIER_MIN:
            P = np.column_stack([X[idx] - cx0, Y[idx] - cy0, H[idx] - np.mean(H_TRECHO)])
            d = eixo(P)
            u, v = base_ortogonal(d)
            pu, pv = P @ u, P @ v
            # Between 1.0 and 1.6 m the real taper is of the order of millimetres, so divergence
            # between the slices indicates contamination: a liana, a fork or a neighbouring stem
            # inside the crop. The lock uses the median of several slices, and not the agreement
            # between two halves: requiring agreement between two rejects ~12% of the stems merely
            # because one half came out sparse, which is scarcity of scanning. With more slices the
            # median absorbs one bad slice and the dispersion still exposes the contamination.
            r, cu, cv, ins = raio_por_modo(pu, pv)
            if not np.isfinite(r):
                motivo = "sem pico radial"
            raios = []
            if np.isfinite(r):
                bordas = np.linspace(P[:, 2].min(), P[:, 2].max(), N_FATIAS + 1)
                for i in range(N_FATIAS):
                    m = (P[:, 2] >= bordas[i]) & (P[:, 2] < bordas[i + 1])
                    if m.sum() < PTS_FATIA:
                        continue
                    ri = raio_por_modo(pu[m], pv[m], cu, cv, minimo=12)[0]
                    if np.isfinite(ri):
                        raios.append(ri)
            if len(raios) < FATIAS_MIN:
                motivo = "fatias válidas de menos"
                concorda = False
            else:
                # Median absolute deviation, and not range or quartiles: with five slices one
                # dirty slice is enough to blow up the range and the quartile, rejecting the stem
                # even though the other four agree. That is the case of the liana that shows up at
                # one height only.
                rr = np.asarray(raios)
                disp = (abs(rr[0] - rr[1]) / 2 if len(rr) == 2
                        else float(np.median(np.abs(rr - np.median(rr)))))
                concorda = disp <= DISCORDA_MAX
                if not concorda:
                    motivo = "fatias discordam"
                else:
                    r = float(np.median(raios))
            if np.isfinite(r) and concorda:
                motivo = ("poucos inliers" if ins.sum() < INLIER_MIN else
                          "raio fora da faixa" if not (R_MIN < r < R_MAX) else motivo)
                if ins.sum() >= INLIER_MIN and R_MIN < r < R_MAX:
                    ang = np.degrees(np.arctan2(pv[ins] - cv, pu[ins] - cu))
                    cob = len(np.unique((ang // 15).astype(int))) * 15.0
                    res = np.median(np.abs(np.hypot(pu[ins] - cu, pv[ins] - cv) - r))
                    if cob >= COB_MIN and np.hypot(cu, cv) < 0.25:
                        reg.update(dap_cm=200 * r, inliers=int(ins.sum()),
                                   cobertura=cob, residuo_cm=100 * res)
                        motivo = "ok"
                    else:
                        motivo = "cobertura baixa" if cob < COB_MIN else "centro deslocado"
        reg["motivo"] = motivo
        linhas.append(reg)
    d = pd.DataFrame(linhas)
    ok = d.dap_cm.notna()
    print("\nwhy each stem was accepted or rejected")
    for k, v in d.motivo.value_counts().items():
        print(f"  {k:32s} {v:4d}  {100 * v / len(d):5.1f}%")
    print(f"\nDBH obtained on {ok.sum()} of the {len(ref)} stems ({100 * ok.mean():.0f}%)")
    print(f"  median {d.dap_cm.median():.1f} cm, mean {d.dap_cm.mean():.1f}, "
          f"sd {d.dap_cm.std():.1f}")
    print(f"  median inliers {d.inliers[ok].median():.0f}, "
          f"median residual {d.residuo_cm.median():.2f} cm")

    inv = pd.read_excel(config.DATA / "5-dados_campo/Inventario_SaoManuel_v030925.xlsx")
    i1 = inv[(inv.talhao == 1) & inv.D_cm.notna()]
    dens = len(i1) / (i1.parcela.nunique() * 0.04)
    print(f"\ninventory of stand 001: {len(i1)} trees, median {i1.D_cm.median():.1f} cm, "
          f"sd {i1.D_cm.std():.1f}, basal area "
          f"{dens * np.pi * (i1.D_cm.mean() / 200) ** 2:.1f} m²/ha")
    print(f"{'quantile':>8} {'field':>8} {'TLS':>8} {'ratio':>7}")
    for q in (10, 25, 50, 75, 90):
        a, b = np.percentile(i1.D_cm, q), np.percentile(d.dap_cm[ok], q)
        print(f"{'p' + str(q):>8} {a:>7.1f} {b:>8.1f} {b / a:>7.2f}")
    g = dens * np.pi * (d.dap_cm[ok].mean() / 200) ** 2
    print(f"\nbasal area implied by the TLS: {g:.1f} m²/ha")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    d.to_csv(SAIDA, index=False)
    print(f"\n{SAIDA}")


if __name__ == "__main__":
    main()

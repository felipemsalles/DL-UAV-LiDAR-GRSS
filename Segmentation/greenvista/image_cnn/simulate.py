"""Synthetic labeled plots for CNN pretraining (Phase 2, transfer learning).

Domain-randomized eucalyptus plots rendered into the same channels as the real rasters (render.py),
labeled with Vol_ha and mean DBH computed from allometries fit on the 63-tree destructive-scaling set
(Schumacher lnV=a+b·lnD+c·lnH, R²≈0.996; H–D lnH=a+b·lnD). Parameter ranges cover and exceed the 13
real plots, so a CNN pretrained here learns canopy-structure→volume with abundant labels and then
fine-tunes on the 13 real plots. The simulation never sees any real plot.
"""
import numpy as np
import pandas as pd
from greenvista import config
from . import render, footprint

PLOT_AREA_HA = config.PLOT_AREA_HA  # 12 m radius → 0.0452 ha


def fit_allometry(volumes_csv=None):
    """Fit Schumacher volume eq + H–D from the destructive-scaling data. Returns coefficients + H residual sd."""
    vol = pd.read_csv(volumes_csv or config.VOLUMES_CSV)
    d = pd.to_numeric(vol["D_cm"], errors="coerce"); h = pd.to_numeric(vol["H_m"], errors="coerce")
    v = pd.to_numeric(vol["V"], errors="coerce")
    ok = d.notna() & h.notna() & v.notna() & (d > 0) & (h > 0) & (v > 0)
    d, h, v = d[ok].to_numpy(), h[ok].to_numpy(), v[ok].to_numpy()
    vc, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(d), np.log(d), np.log(h)]), np.log(v), rcond=None)
    Ah = np.column_stack([np.ones_like(d), np.log(d)])
    hd, *_ = np.linalg.lstsq(Ah, np.log(h), rcond=None)
    return {"vol": vc, "hd": hd, "h_res": float((np.log(h) - Ah @ hd).std())}


def tree_volume(D, H, allo):
    a, b, c = allo["vol"]
    return np.exp(a + b * np.log(D) + c * np.log(H))


def _row_positions(rng, radius, row_sp=3.0, tree_sp=1.8):
    """Plantation lattice (rows) at a random orientation, jittered, clipped to the footprint disc."""
    th = rng.uniform(0, np.pi)
    ct, st = np.cos(th), np.sin(th)
    u = np.arange(-radius - 3, radius + 3, row_sp)
    v = np.arange(-radius - 3, radius + 3, tree_sp)
    uu, vv = np.meshgrid(u, v)
    uu = uu.ravel() + rng.uniform(-0.4, 0.4, uu.size)
    vv = vv.ravel() + rng.uniform(-0.4, 0.4, vv.size)
    x = uu * ct - vv * st
    y = uu * st + vv * ct
    keep = x * x + y * y <= radius * radius
    return np.column_stack([x[keep], y[keep]])


def make_plot(rng, allo, radius=footprint.DEFAULT_RADIUS_M, size=128):
    """One domain-randomized plot → (raster (3,size,size), scalars (3,), labels [Vol_ha, D_cm])."""
    mean_dbh = rng.uniform(9.0, 24.0)
    dbh_sd = rng.uniform(1.5, 3.5)
    pos = _row_positions(rng, radius)
    n_keep = int(np.clip(rng.uniform(30, 85), 1, len(pos)))            # mortality / planting gaps
    pos = pos[rng.choice(len(pos), n_keep, replace=False)]

    pt_density = rng.uniform(40, 110)                                   # crown points per m²
    pts, vols, dbhs = [], [], []
    for (tx, ty) in pos:
        D = float(np.clip(rng.normal(mean_dbh, dbh_sd), 6.0, 30.0))
        a, b = allo["hd"]
        H = float(np.clip(np.exp(a + b * np.log(D) + rng.normal(0, allo["h_res"])), 4.0, 45.0))
        base = H * rng.uniform(0.35, 0.55)
        rc = float(np.clip(0.5 + 0.05 * D, 0.7, 2.2))
        k = int(np.clip(np.pi * rc * rc * pt_density, 60, 500))
        r = rc * np.sqrt(rng.uniform(0, 1, k))
        ang = rng.uniform(0, 2 * np.pi, k)
        z = H - (H - base) * (r / rc) ** 2 + rng.normal(0, 0.15, k)
        x = tx + r * np.cos(ang); y = ty + r * np.sin(ang)
        m = z >= 2.0
        pts.append(np.column_stack([x[m], y[m], z[m]]))
        vols.append(tree_volume(D, H, allo)); dbhs.append(D)

    points = np.concatenate(pts, axis=0) if pts else np.empty((0, 3))
    raster, scal = render.to_raster(points, (0.0, 0.0), radius, size=size)
    vol_ha = float(np.sum(vols) / PLOT_AREA_HA)
    labels = np.array([vol_ha, float(np.mean(dbhs))], np.float32)
    scalars = np.array([scal["zmax"], scal["zmean"], scal["zsd"]], np.float32)
    return raster.astype(np.float32), scalars, labels


def make_dataset(n_plots, seed=0, radius=footprint.DEFAULT_RADIUS_M, size=128):
    """Return {sim_i: {raster, scalars, labels}} — same shape the real loader produces."""
    rng = np.random.default_rng(seed)
    allo = fit_allometry()
    out = {}
    for i in range(n_plots):
        r, s, y = make_plot(rng, allo, radius=radius, size=size)
        out[f"sim_{i}"] = {"raster": r, "scalars": s, "labels": y}
    return out

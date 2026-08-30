"""Operational stand volume from LiDAR dominant height + stand age — the deployable pipeline.

The GreenVista inventory already has the G2 volume model (R²=0.995), a ~717-tree DBH census,
per-stand age (planting records) and a site index; LiDAR's job is wall-to-wall extension, not
replacing the inventory. Dominant height (Hdom) is the mechanistic volume driver, but a LiDAR Hdom
proxy alone does not generalize across stands at N=13 (one young stand breaks the extrapolation),
whereas Hdom + age does (leave-one-stand-out R²≈0.73, ~13% rRMSE). Age is a free, known input for
the company, so this is the deployable model, and it feeds the existing G2 / site-index framework
rather than a black-box fit.

Key functions:
  lidar_hdom      – grid-cell dominant height (mean of the tallest cell-maxima); a stable Hdom proxy,
                    far better than zmax (which is pulled around by single tall returns).
  site_index      – anamorphic Hdom projection to a reference age (the site-index framing).
  StandVolume     – Vol_ha = exp(a + b·ln Hdom + c·ln age): the log-log stand yield model (robust fit,
                    prediction intervals). Fits the greenvista.area_based.validate CV interface.
  volume_map      – wall-to-wall Hdom(per-cell)+age(per-stand) → Vol_ha map.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import HuberRegressor

from greenvista.image_cnn import footprint


def lidar_hdom(xyz, center, radius=12.0, cell=4.0, top_frac=0.5, zmin=2.0, zmax=45.0, min_pts=50):
    """LiDAR dominant height over a circular footprint: grid the canopy into `cell`-m cells, take the
    max height per cell, average the tallest `top_frac` of those cell-maxima. Mimics field Hdom (mean
    height of the dominant trees) and is far more stable across stands than zmax."""
    pts = footprint.extract_footprint(xyz, center, radius)
    z = pts[:, 2]
    canopy = pts[(z >= zmin) & (z <= zmax)]
    if len(canopy) < min_pts:
        return np.nan
    x0, y0 = center[0] - radius, center[1] - radius
    ix = ((canopy[:, 0] - x0) / cell).astype(int)
    iy = ((canopy[:, 1] - y0) / cell).astype(int)
    cellmax = pd.DataFrame({"ix": ix, "iy": iy, "z": canopy[:, 2]}).groupby(["ix", "iy"]).z.max()
    k = max(1, int(len(cellmax) * top_frac))
    return float(cellmax.nlargest(k).mean())


def canopy_cover(z, break_h=2.0):
    """Fraction of returns above `break_h` (m) — a canopy-cover / basal-area proxy."""
    z = np.asarray(z, float)
    return float((z > break_h).mean()) if z.size else np.nan


def site_index(hdom, age, ref_age=7.0, b=1.0):
    """Anamorphic site index: dominant height projected to `ref_age`, SI = Hdom·(ref_age/age)^b.
    `b` is the guide-curve exponent (default 1 = proportional). Fit `b` from the stand's Hdom–age pairs
    for a proper curve; here it gives the site-index *framing* that the company's models expect."""
    return np.asarray(hdom, float) * (ref_age / np.asarray(age, float)) ** b


class StandVolume:
    """Vol_ha = exp(a + Σ b_i ln(x_i)) over (Hdom, age) — the log-log stand yield model.

    Robust (Huber) fit resists the young-stand leverage; normal prediction interval from the log-
    residual std. Implements fit/predict/predict_interval on a DataFrame, so it drops straight into
    `greenvista.area_based.validate` (LOPO / leave-one-stand-out)."""
    calibrated = True

    def __init__(self, cols=("Hdom", "age")):
        self.cols = list(cols)

    def _design(self, X):
        return np.log(np.column_stack([np.asarray(X[c], float) for c in self.cols]))

    def fit(self, X, y):
        L = self._design(X)
        self.hr_ = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000).fit(L, np.log(np.asarray(y, float)))
        self.resid_sd_ = (np.log(np.asarray(y, float)) - self.hr_.predict(L)).std(ddof=1)
        self.coef_ = np.concatenate([[self.hr_.intercept_], self.hr_.coef_])
        return self

    def predict(self, X):
        return np.exp(self.hr_.predict(self._design(X)))

    def predict_interval(self, X, alpha=0.05):
        z = norm.ppf(1 - alpha / 2)
        p = self.hr_.predict(self._design(X))
        return np.exp(p - z * self.resid_sd_), np.exp(p + z * self.resid_sd_)

    def equation(self):
        a, *b = self.coef_
        terms = " + ".join(f"{bi:+.3f}·ln({c})" for bi, c in zip(b, self.cols))
        return f"Vol_ha = exp({a:.3f} {terms})"


def volume_map(model, xyz, ages, cell=20.0, hdom_cell=4.0, radius=12.0, min_pts=300,
               zmin=2.0, zmax=45.0):
    """Wall-to-wall Vol_ha map from a fitted StandVolume model. `ages` is a callable (cx,cy)->age (or a
    constant) supplying the stand age per cell (from planting records). Hdom is computed per grid cell.
    Returns a DataFrame of cell centres with Hdom, age, Vol_ha, and 95% PI."""
    import geopandas as gpd
    from shapely.geometry import box
    from greenvista import config
    x, y = xyz[:, 0], xyz[:, 1]
    x0, y0, x1, y1 = x.min(), y.min(), x.max(), y.max()
    age_fn = ages if callable(ages) else (lambda cx, cy: ages)
    rows, geoms = [], []
    for cx in np.arange(x0 + cell / 2, x1, cell):
        for cy in np.arange(y0 + cell / 2, y1, cell):
            m = (np.abs(x - cx) <= cell / 2) & (np.abs(y - cy) <= cell / 2)
            if m.sum() < min_pts:
                continue
            hd = lidar_hdom(xyz[m], (cx, cy), radius=cell / 2, cell=hdom_cell, zmin=zmin, zmax=zmax)
            if not np.isfinite(hd):
                continue
            age = age_fn(cx, cy)
            X = pd.DataFrame([{"Hdom": hd, "age": age}])
            vha = float(model.predict(X)[0])
            lo, hi = model.predict_interval(X)
            rows.append({"Hdom": hd, "age": age, "vol_ha": vha,
                         "vol_lo": float(lo[0]), "vol_hi": float(hi[0])})
            geoms.append(box(cx - cell / 2, cy - cell / 2, cx + cell / 2, cy + cell / 2))
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=config.CRS)

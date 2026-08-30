"""Wall-to-wall volume map. Per-cell metrics must match lidR .stdmetrics_z on points 2<=z<=45.

Two guards on the output:
  * each cell is flagged `extrapolated=True` when any of its features falls outside the training
    envelope (min/max seen by the model); the prediction interval is only valid inside that
    envelope, so downstream maps can grey out or widen extrapolated cells;
  * a cell whose features come back non-finite is counted and skipped instead of being passed to
    `model.predict`, where a NaN volume would propagate silently into the stand total.
"""
import json
from pathlib import Path

import numpy as np, pandas as pd, geopandas as gpd, laspy
from shapely.geometry import box

from greenvista import config
from .data import FEATURES

CRS = config.CRS


def stdmetrics_subset(z):
    """The FEATURES, defined identically to lidR (z already filtered to [2,45])."""
    z = np.asarray(z, float)
    if z.size < 2:  # degenerate footprint/cell: undefined metrics
        return {k: float("nan") for k in FEATURES}
    return {"zmax": z.max(), "zmean": z.mean(), "zsd": z.std(ddof=1),
            "zq50": np.quantile(z, 0.50), "zq90": np.quantile(z, 0.90),
            "pzabovezmean": 100.0 * np.mean(z > z.mean())}


def plot_footprint_metrics(xy, z, center, radius=config.PLOT_RADIUS_M, zmin=config.Z_MIN_M, zmax=config.Z_MAX_M):
    """Recompute FEATURES on a 12 m-radius plot footprint — used by the consistency check."""
    d2 = (xy[:, 0] - center[0]) ** 2 + (xy[:, 1] - center[1]) ** 2
    sel = (d2 <= radius ** 2) & (z >= zmin) & (z <= zmax)
    return stdmetrics_subset(z[sel])


def feature_envelope(df, features=FEATURES):
    """Training envelope {feature: (min, max)} — a cell outside it is an extrapolation."""
    return {f: (float(df[f].min()), float(df[f].max())) for f in features}


def _is_extrapolated(feats, envelope):
    if not envelope:
        return False
    for f, (lo, hi) in envelope.items():
        v = feats.get(f, np.nan)
        if not np.isfinite(v) or v < lo or v > hi:
            return True
    return False


def cell_features_from_arrays(xy, z, cell=config.ABA.map_cell_m, zmin=config.Z_MIN_M,
                              zmax=config.Z_MAX_M, min_pts=config.ABA.map_min_pts):
    """Bin points to a `cell`-m grid and return the per-cell FEATURES table.

    Shared by `make_map_from_arrays`, by the model-assisted estimator (which needs the population
    mean of the prediction over every cell of a stand) and by the within-stand map validation (which
    needs the cell a field plot falls in), so the three cannot disagree about which points belong to
    which cell.

    Returns (features_df, n_skipped_nan). `features_df` carries the FEATURES columns plus `x0`/`y0`
    (cell lower-left corner) and `n_pts`. Cells below `min_pts` are dropped; cells whose metrics are
    non-finite are dropped and counted.
    """
    m = (z >= zmin) & (z <= zmax)
    x, y, z = xy[m, 0], xy[m, 1], z[m]
    x_min, y_min = x.min(), y.min()
    ix = np.floor((x - x_min) / cell).astype(int)
    iy = np.floor((y - y_min) / cell).astype(int)
    rows, n_skipped = [], 0
    for (cx, cy), idx in pd.DataFrame({"ix": ix, "iy": iy}).groupby(["ix", "iy"]).groups.items():
        zc = z[list(idx)]
        if len(zc) < min_pts:
            continue
        feats = stdmetrics_subset(zc)
        if not np.all(np.isfinite(np.array([feats[f] for f in FEATURES], float))):
            n_skipped += 1
            continue
        rows.append({**feats, "x0": x_min + cx * cell, "y0": y_min + cy * cell, "n_pts": len(zc)})
    return pd.DataFrame(rows), n_skipped


def make_map_from_arrays(model, xy, z, cell=config.ABA.map_cell_m, zmin=config.Z_MIN_M,
                         zmax=config.Z_MAX_M, min_pts=config.ABA.map_min_pts, envelope=None):
    """Bin points to a `cell`-m grid, compute FEATURES per cell, predict volume + 95% PI.

    Returns a GeoDataFrame of cell polygons (EPSG:31982) with volume_m3 / vol_lo / vol_hi (m3/cell),
    plus `extrapolated` (bool) and `vol_ha`. Cells with non-finite features are skipped and counted
    in the GeoDataFrame's `.attrs['n_skipped_nan']`.
    """
    cells, n_skipped = cell_features_from_arrays(xy, z, cell, zmin, zmax, min_pts)
    cell_area_ha = (cell * cell) / 1e4
    rows, geoms = [], []
    for _, c in cells.iterrows():
        feats = c.to_dict()
        Xc = pd.DataFrame([feats])[FEATURES]
        vha = float(model.predict(Xc)[0]); lo, hi = model.predict_interval(Xc, 0.05)
        rows.append({"volume_m3": max(0.0, vha) * cell_area_ha,
                     "vol_lo": max(0.0, float(lo[0])) * cell_area_ha,
                     "vol_hi": max(0.0, float(hi[0])) * cell_area_ha, "vol_ha": vha,
                     "extrapolated": _is_extrapolated(feats, envelope)})
        geoms.append(box(c.x0, c.y0, c.x0 + cell, c.y0 + cell))
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=CRS)
    gdf.attrs["n_skipped_nan"] = n_skipped
    return gdf


def make_map(model, laz_path, cell=config.ABA.map_cell_m, zmin=config.Z_MIN_M, zmax=config.Z_MAX_M,
             envelope=None):
    las = laspy.read(laz_path)
    xy = np.column_stack([np.asarray(las.x), np.asarray(las.y)])
    return make_map_from_arrays(model, xy, np.asarray(las.z), cell, zmin, zmax, envelope=envelope)


def _extract_coefficients(model):
    """Readable coefficients for provenance. Handles the linear/allometric models; returns a type
    tag plus whatever coefficients are inspectable, so the map is reproducible from JSON."""
    info = {"model_type": type(model).__name__}
    if hasattr(model, "coef_") and hasattr(model, "cols"):        # LogAllometric
        info["form"] = "ln(V) = a + sum b_i ln(x_i)"
        info["cols"] = list(model.cols)
        info["coef"] = np.asarray(model.coef_).tolist()
        if hasattr(model, "resid_sd_"):
            info["log_resid_sd"] = float(model.resid_sd_)
    elif hasattr(model, "pipe"):                                   # _SkWrapper / _GPR pipelines
        info["cols"] = list(getattr(model, "cols", []))
        est = model.pipe[-1] if hasattr(model.pipe, "__getitem__") else model.pipe
        if hasattr(est, "coef_"):
            info["coef"] = np.asarray(est.coef_).ravel().tolist()
            info["intercept"] = float(np.ravel(getattr(est, "intercept_", [np.nan]))[0])
    return info


def save_map_provenance(model, features, path, *, envelope=None, extra=None):
    """Persist the map's model provenance as JSON next to the .gpkg so the wall-to-wall map is
    reproducible from a file: feature list, fitted coefficients (where inspectable), and the training
    envelope used for the extrapolation flag."""
    prov = {"features": list(features),
            "model": _extract_coefficients(model),
            "envelope": {k: list(v) for k, v in (envelope or {}).items()}}
    if extra:
        prov.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prov, indent=2, default=str))
    return prov

"""Stem density (trees/ha) from LiDAR detection, calibrated against a field census.

FF3D detects ~61 % of stems (better than CHM) and, unlike per-tree volume, the count has no
occlusion ceiling a correction cannot fix. A raw detector still under-counts (nadir occlusion +
missed suppressed stems), so trees/ha needs a calibration factor learned from field plots. Three
forms of that factor are fitted here:

  * a raw (uncalibrated) count → trees/ha baseline;
  * a global multiplicative correction (one factor, ratio-of-sums or mean-ratio);
  * a proxy correction whose factor depends on an observable LiDAR quantity (canopy cover / return
    density / the detector's own unassigned-point fraction), so it can extrapolate the correction to
    a stand it was never fit on.

The calibration is a `fit`/`predict` estimator so the existing CV harness can hold whole stands out.
The correction for a held-out stand must be learned only from the other stands: a per-stand factor
fit and scored on that same stand is leakage and is trivially perfect. Use
`cv_predict(..., groups=talhao)` for LOTO; the tests enforce this.

Caveat: because the correction is learned from field plots, density estimation still needs some
calibration plots. It needs far fewer than volume (counting is direct, not a saturating allometric
signal), but it is not a zero-field, LiDAR-only estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------------------
# unit helpers
# --------------------------------------------------------------------------------------------------
def to_density(count, area_ha: float) -> np.ndarray:
    """Convert a stem count over `area_ha` hectares to trees/ha."""
    return np.asarray(count, float) / float(area_ha)


def detection_rate(detected_count, true_count) -> float:
    """Aggregate detection rate = sum(detected) / sum(true) (over plots)."""
    d = np.asarray(detected_count, float).sum()
    t = np.asarray(true_count, float).sum()
    return float(d / t) if t else float("nan")


# --------------------------------------------------------------------------------------------------
# the calibration estimator
# --------------------------------------------------------------------------------------------------
class DensityCalibrator:
    """Calibrated stem-density estimator: trees/ha = raw_density x correction.

    Parameters
    ----------
    raw_col : str
        Column of the modeling frame holding the *raw* detector density (trees/ha), e.g.
        ``ff3d_stems_ha_raw`` or ``chm_stems_ha_raw``.
    method : {"raw", "global", "proxy"}
        ``raw``    — identity (uncalibrated detector); the uncorrected floor.
        ``global`` — one multiplicative factor learned from the training plots.
        ``proxy``  — factor is a linear function of a standardized observable proxy, so it can vary
                     across stands and extrapolate to a held-out one.
    proxy_col : str | None
        Required for ``method="proxy"`` — the observable LiDAR quantity (e.g. ``cover``,
        ``pct_unassigned``, return density). Must not be derived from field labels.
    agg : {"ratio", "mean"}
        For ``method="global"``: ``ratio`` = sum(y)/sum(raw) (minimum-variance for counts, the
        default); ``mean`` = mean(y/raw) (equal weight per plot).
    clip : (float, float)
        Bounds on the effective correction factor at predict time — stops a held-out stand whose
        proxy falls outside the training range from producing an absurd extrapolated factor.

    `fit(df, y)` / `predict(df)` take a DataFrame so the estimator drops straight into the project's
    leave-one-*stand*-out CV interface (`greenvista.area_based.validate` style).
    """

    def __init__(self, raw_col: str, method: str = "global", proxy_col: str | None = None,
                 agg: str = "ratio", clip: tuple[float, float] = (0.25, 6.0)):
        if method not in ("raw", "global", "proxy"):
            raise ValueError(f"unknown method {method!r}")
        if method == "proxy" and proxy_col is None:
            raise ValueError("method='proxy' requires proxy_col")
        self.raw_col = raw_col
        self.method = method
        self.proxy_col = proxy_col
        self.agg = agg
        self.clip = clip
        self.calibrated = method != "raw"  # so the CV harness treats it like other calibrated models

    # -- fit --------------------------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y) -> "DensityCalibrator":
        raw = np.asarray(df[self.raw_col], float)
        y = np.asarray(y, float)
        if np.any(raw <= 0):
            # a zero-detection plot can't be scaled; nudge to avoid div-by-zero (rare, flagged).
            raw = np.where(raw > 0, raw, 1e-6)
        if self.method == "raw":
            self.factor_ = 1.0
        elif self.method == "global":
            if self.agg == "ratio":
                self.factor_ = float(y.sum() / raw.sum())
            elif self.agg == "mean":
                self.factor_ = float(np.mean(y / raw))
            else:
                raise ValueError(f"unknown agg {self.agg!r}")
        else:  # proxy: y ~ a*raw + b*(raw*zstd)  → factor(z) = a + b*zstd, from the train rows only
            z = np.asarray(df[self.proxy_col], float)
            self.z_mean_ = float(z.mean())
            self.z_std_ = float(z.std()) or 1.0
            zs = (z - self.z_mean_) / self.z_std_
            F = np.column_stack([raw, raw * zs])          # features that make the target = density
            coef, *_ = np.linalg.lstsq(F, y, rcond=None)  # minimizes density error directly
            self.coef_ = coef                             # coef_[0]=a (base factor), coef_[1]=b (slope)
        return self

    # -- predict ----------------------------------------------------------------------------------
    def factor(self, df: pd.DataFrame) -> np.ndarray:
        """The per-plot multiplicative correction the model would apply (for inspection/plots)."""
        if self.method == "raw":
            return np.ones(len(df))
        if self.method == "global":
            return np.full(len(df), self.factor_)
        z = np.asarray(df[self.proxy_col], float)
        zs = (z - self.z_mean_) / self.z_std_
        fac = self.coef_[0] + self.coef_[1] * zs
        return np.clip(fac, self.clip[0], self.clip[1])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(df[self.raw_col], float)
        return np.clip(raw * self.factor(df), 0.0, None)


# --------------------------------------------------------------------------------------------------
# cross-validation runners — need only fit/predict (no predict_interval), so the shuffle control and
# both CV schemes reuse the same code. groups=None → leave-one-plot-out; groups=array → LOTO.
# --------------------------------------------------------------------------------------------------
def cv_predict(make_model, df: pd.DataFrame, y, groups=None) -> np.ndarray:
    """Out-of-fold predictions. `make_model` is a zero-arg factory (fresh estimator per fold).

    groups=None  → leave-one-plot-out (one row held out per fold).
    groups=array → leave-one-group-out (one whole stand held out; the cross-stand test). The held-out
                   stand's rows are predicted by a model fit only on the other stands, so the
                   correction never sees the stand it is scored on.
    """
    df = df.reset_index(drop=True)
    y = np.asarray(y, float)
    pred = np.empty(len(df), float)
    if groups is None:
        for i in range(len(df)):
            tr = df.index != i
            m = make_model().fit(df[tr], y[tr])
            pred[i] = m.predict(df.iloc[[i]])[0]
    else:
        groups = np.asarray(groups)
        for g in np.unique(groups):
            te = groups == g
            m = make_model().fit(df[~te], y[~te])
            pred[np.where(te)[0]] = m.predict(df[te])
    return pred

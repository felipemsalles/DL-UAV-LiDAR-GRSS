"""Distribution-free conformal prediction intervals for small-N regression.

Everything here consumes out-of-fold (LOPO / LOOCV) predictions, never in-sample residuals: RF/kNN
nearly interpolate the training set, so their in-sample residual std collapses to ~0 and the
resulting interval is meaningless. This module is the single source of the reported `cov95`.

Finite-sample validity: a conformal band over `m` calibration residuals must use the
`ceil((1-alpha)(m+1))/m` empirical quantile (the "+1"), not the plain `1-alpha` quantile — at N=13
that correction is the difference between ~0.88 and ~0.95 coverage.
"""
import numpy as np


def finite_sample_level(m: int, alpha: float = 0.05) -> float:
    """Conformal quantile *level* for `m` calibration residuals: ceil((1-alpha)(m+1))/m, clipped."""
    if m <= 0:
        return 1.0
    k = np.ceil((1.0 - alpha) * (m + 1))
    return float(min(max(k / m, 0.0), 1.0))


def _quantile(res: np.ndarray, alpha: float) -> float:
    """Finite-sample-valid upper quantile of |residuals| (conservative 'higher' interpolation)."""
    res = np.asarray(res, float)
    m = len(res)
    if m == 0:
        return float("inf")
    return float(np.quantile(res, finite_sample_level(m, alpha), method="higher"))


def jackknife_plus_interval(y_true, y_pred, alpha: float = 0.05):
    """Symmetric jackknife+ (1-alpha) PI from leave-one-out residuals.

    Each point's half-width is the finite-sample (1-alpha) quantile of the other points' absolute
    LOO residuals (no self-inclusion, hence no optimism). Returns (lo, hi) arrays.
    """
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    res = np.abs(y - p)
    n = len(res)
    if n < 2:
        w = np.full(n, np.inf)
        return p - w, p + w
    half = np.array([_quantile(np.delete(res, i), alpha) for i in range(n)])
    return p - half, p + half


def mondrian_interval(y_true, y_pred, groups, alpha: float = 0.05):
    """Locally-valid (Mondrian) conformal: the residual quantile is taken within each group bin, so
    the width adapts to heteroscedastic error (e.g. wider where volume is high). `groups` is a label
    per point (int/str). Singleton bins fall back to the global finite-sample quantile.
    """
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    g = np.asarray(groups)
    res = np.abs(y - p)
    n = len(res)
    glob = _quantile(res, alpha)
    idx = np.arange(n)
    half = np.empty(n)
    for i in range(n):
        others = res[(g == g[i]) & (idx != i)]
        half[i] = _quantile(others, alpha) if len(others) >= 2 else glob
    return p - half, p + half


def locally_weighted_interval(y_true, y_pred, scale, alpha: float = 0.05):
    """Normalized (locally-weighted) conformal: half-width scales with a per-point difficulty
    estimate `scale` (>0) — e.g. predicted volume or a local residual estimate — giving
    heteroscedastic width with a single global finite-sample coverage guarantee.
    """
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    s = np.asarray(scale, float)
    s = np.where(s > 1e-9, s, 1e-9)
    norm_res = np.abs(y - p) / s
    q = _quantile(norm_res, alpha)
    half = q * s
    return p - half, p + half


def coverage(y_true, lo, hi) -> float:
    """Empirical coverage: fraction of truths inside [lo, hi]."""
    y = np.asarray(y_true, float)
    return float(((y >= np.asarray(lo, float)) & (y <= np.asarray(hi, float))).mean())


def interval_score(y_true, lo, hi, alpha: float = 0.05) -> float:
    """Mean Winkler / interval score — rewards narrow intervals, penalizes misses. Lower is better;
    a single number to compare interval methods (calibration *and* sharpness together)."""
    y = np.asarray(y_true, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    width = hi - lo
    below = (2.0 / alpha) * (lo - y) * (y < lo)
    above = (2.0 / alpha) * (y - hi) * (y > hi)
    return float(np.mean(width + below + above))

"""Validation utilities for small-N: LOOCV predictions, bootstrap CIs, leakage guard.

At N≈23 a bare point estimate is meaningless — always report bootstrap CIs. Any feature or
hyperparameter selection must happen inside the LOOCV fold, never on the reported CV;
`drop_leaky_columns` guards against target leakage.
"""
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def assert_finite(a, name: str = "array"):
    """Fail loudly on NaN/Inf instead of letting it propagate silently into a metric or a map cell.
    Returns the array unchanged so it can wrap an expression inline."""
    a = np.asarray(a, float)
    if not np.all(np.isfinite(a)):
        bad = int((~np.isfinite(a)).sum())
        raise ValueError(f"{name} has {bad} non-finite value(s) — refusing to continue silently")
    return a


def metrics(y, yhat) -> dict:
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    bias = float(np.mean(yhat - y))  # mean signed error (forestry convention: pred − obs)
    return {"R2": float(r2_score(y, yhat)), "RMSE": rmse,
            "rRMSE_pct": float(rmse / np.mean(y) * 100),
            "bias": bias, "bias_pct": float(bias / np.mean(y) * 100), "n": int(len(y))}

def loocv_predict(make_model, X, y) -> np.ndarray:
    """Leave-one-out predictions. `make_model` is a zero-arg factory returning a fresh fit/predict model.
    Any per-fold feature selection should live inside `make_model`/the loop, not outside."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n = len(y)
    yhat = np.empty(n, float)
    for i in range(n):
        tr = np.arange(n) != i
        model = make_model()
        model.fit(X[tr], y[tr])
        yhat[i] = np.asarray(model.predict(X[i:i + 1]), float)[0]
    return yhat

def bootstrap_ci(y, yhat, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap CIs for RMSE / rRMSE (primary) and R² (secondary) over resampled pairs.

    At N≈13 the R² bootstrap is unstable: a near-constant resample makes r2_score blow up to a huge
    negative value or NaN. RMSE/rRMSE stay bounded and are the primary headline. Non-finite resample
    statistics are dropped so a single degenerate draw cannot poison a percentile, and the number
    dropped is reported. Report RMSE/rRMSE as primary and the R² CI as a caveated secondary.
    """
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    r2s, rmses, rrmses = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.ptp(y[idx]) == 0:          # constant target → R² undefined; skip (would emit -inf/NaN)
            continue
        m = metrics(y[idx], yhat[idx])
        if not np.isfinite(m["R2"]):
            continue
        r2s.append(m["R2"]); rmses.append(m["RMSE"]); rrmses.append(m["rRMSE_pct"])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def _ci(vals):
        return (float(np.percentile(vals, lo)), float(np.percentile(vals, hi))) if vals else (float("nan"), float("nan"))

    return {
        "RMSE_ci": _ci(rmses),          # primary — bounded, stable
        "rRMSE_ci": _ci(rrmses),        # primary
        "R2_ci": _ci(r2s),              # secondary — wide/unstable at small N, use with care
        "n_boot_valid": len(r2s), "n_boot_dropped": n_boot - len(r2s),
    }

def drop_leaky_columns(X, y, thresh: float = 0.999, warn: bool = True):
    """Drop feature columns whose |Pearson r| with y exceeds thresh (target leakage guard).
    Returns (X_clean, kept_col_indices).

    A column with a NaN correlation (e.g. it contains NaN) is dropped and the drop is surfaced via a
    warning, so a NaN feature cannot vanish unnoticed.
    """
    import warnings

    X = np.asarray(X, float); y = np.asarray(y, float)
    kept, dropped_nan, dropped_leak = [], [], []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.std(col) == 0:
            kept.append(j); continue
        r = abs(np.corrcoef(col, y)[0, 1])
        if not np.isfinite(r):
            dropped_nan.append(j); continue
        if r <= thresh:
            kept.append(j)
        else:
            dropped_leak.append(j)
    if warn and dropped_nan:
        warnings.warn(f"drop_leaky_columns: dropped {len(dropped_nan)} column(s) with NaN correlation "
                      f"(indices {dropped_nan}) — check for NaN features", stacklevel=2)
    if warn and dropped_leak:
        warnings.warn(f"drop_leaky_columns: dropped {len(dropped_leak)} leaky column(s) |r|>{thresh} "
                      f"(indices {dropped_leak})", stacklevel=2)
    kept = np.array(kept, int)
    return X[:, kept], kept

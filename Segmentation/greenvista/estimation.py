"""Model-assisted (GREG) estimation of stand and estate totals.

Rationale
---------
Per-plot R² / rRMSE at N=13 is very uncertain: the R² bootstrap interval straddles zero and the
rRMSE interval (9.8 to 16.3) spans most of the published literature. That limits *prediction*
accuracy, which is not the quantity a forest inventory needs — it needs the total volume of a stand
with a confidence interval, and for that the model-assisted / difference (GREG) estimator turns even
a weak model into a defensible gain (Särndal et al. 1992; Ståhl et al. 2016 for the LiDAR case).

The field plots stay the source of truth (so the estimate remains design-unbiased even if the model
is bad) and the wall-to-wall LiDAR only reduces the variance of the mean:

    mean_greg = mean(yhat over every cell of the stand)  +  mean(y_i - yhat_i over the n plots)
                └─ the model, applied everywhere ─┘         └─ the field, correcting its bias ─┘

The first term needs no field data; the second is a bias correction that vanishes in expectation.
The estimator's variance depends on the spread of the residuals, not of the volumes, so

    relative efficiency  RE = var(y) / var(y - yhat)

is the headline. RE = 2 means the flight halves the variance, i.e. it is worth as many extra field
plots as already exist. A useless model gives RE ≈ 1, never worse in expectation.

Requirements
------------
* `yhat` for the sampled plots must be out-of-fold (leave-one-plot-out or leave-one-stand-out).
  In-sample predictions shrink the residuals artificially and inflate RE. `relative_efficiency`
  cannot check this, so the caller is responsible.
* The plots are treated as a simple random sample of the stand. São Manuel's plots are a systematic
  inventory grid, which is standard practice to analyse this way and generally makes the variance
  estimate slightly conservative.
* The population mean of `yhat` needs wall-to-wall predictions over the stand. Without them only
  RE (a variance ratio) is available.
"""
from __future__ import annotations

import numpy as np


def _var(a):
    """Sample variance with the n-1 denominator, 0.0 for degenerate input."""
    a = np.asarray(a, float)
    return float(np.var(a, ddof=1)) if len(a) > 1 else 0.0


def relative_efficiency(y, yhat):
    """How much variance the model removes, expressed in units a forester can act on.

    `yhat` must be out-of-fold (see module docstring). Returns a dict with
      RE                 — var(y) / var(residual); 1.0 means the model bought nothing
      equivalent_plots   — extra field plots that would buy the same precision, n*(RE-1)
      se_field_only      — standard error of the plain field mean
      se_model_assisted  — standard error of the GREG mean
      se_reduction_pct   — how much the standard error shrinks
    """
    y = np.asarray(y, float)
    resid = y - np.asarray(yhat, float)
    n = len(y)
    v_y, v_e = _var(y), _var(resid)
    re = float(v_y / v_e) if v_e > 0 else float("inf")
    se_f = float(np.sqrt(v_y / n)) if n else float("nan")
    se_m = float(np.sqrt(v_e / n)) if n else float("nan")
    return {
        "RE": re,
        "equivalent_plots": float(n * (re - 1.0)),
        "se_field_only": se_f,
        "se_model_assisted": se_m,
        "se_reduction_pct": float(100.0 * (1.0 - se_m / se_f)) if se_f > 0 else 0.0,
        "n": n,
        "mean_y": float(y.mean()),
        "residual_bias": float(resid.mean()),
    }


def model_assisted_mean(y, yhat, yhat_population, alpha=0.05):
    """GREG mean per hectare with its design-based standard error and normal interval.

    Parameters
    ----------
    y, yhat          : field volumes and their out-of-fold predictions on the n sampled plots.
    yhat_population  : predictions over every cell of the stand (array), or their mean (scalar).
    """
    y = np.asarray(y, float)
    resid = y - np.asarray(yhat, float)
    n = len(y)
    pop_mean = float(np.mean(yhat_population))
    mean = pop_mean + float(resid.mean())
    se = float(np.sqrt(_var(resid) / n)) if n else float("nan")
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else float(
        __import__("scipy.stats", fromlist=["norm"]).norm.ppf(1 - alpha / 2))
    return {
        "mean_ha": mean,
        "se": se,
        "lo": mean - z * se,
        "hi": mean + z * se,
        "pop_model_mean": pop_mean,
        "field_correction": float(resid.mean()),
        "n": n,
    }


def field_only_mean(y, alpha=0.05):
    """The baseline every inventory already has: the plain plot mean and its standard error."""
    y = np.asarray(y, float)
    n = len(y)
    mean = float(y.mean())
    se = float(np.sqrt(_var(y) / n)) if n else float("nan")
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else float(
        __import__("scipy.stats", fromlist=["norm"]).norm.ppf(1 - alpha / 2))
    return {"mean_ha": mean, "se": se, "lo": mean - z * se, "hi": mean + z * se, "n": n}


def bootstrap_relative_efficiency(y, yhat, n_boot=4000, alpha=0.05, seed=0):
    """Percentile interval for RE — a variance ratio is very unstable at n=13."""
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v_y, v_e = _var(y[idx]), _var(y[idx] - yhat[idx])
        if v_e > 0 and np.isfinite(v_y / v_e):
            vals.append(v_y / v_e)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, lo)), float(np.percentile(vals, hi)))

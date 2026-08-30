"""Significance / stability tests for small-N model comparison.

At N=13 a bootstrap R² CI is wide and unstable; a permutation test answers "is this skill better
than chance?" more convincingly. Two flavours:

  * `permutation_test` — cheap post-hoc test on final (y, pred): permute y, recompute the statistic.
  * `cv_permutation_test` — re-runs the whole CV under permuted labels, so the null accounts for the
    model's capacity to fit noise.

`multiseed_summary` aggregates a metric over seeds → mean ± sd; a single-seed R² at N=13 is high
variance, so the spread is reported.
"""
import numpy as np

from .validation import metrics


def permutation_test(y, pred, n_perm: int = 5000, seed: int = 0, stat: str = "R2") -> dict:
    """Label-permutation test on fixed predictions. Returns {stat, p_value, n_perm}.

    p = (1 + #{perm_stat >= observed}) / (n_perm + 1)  — the standard add-one estimator (never 0).
    """
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    rng = np.random.default_rng(seed)
    obs = metrics(y, pred)[stat]
    ge = 0
    for _ in range(n_perm):
        if metrics(rng.permutation(y), pred)[stat] >= obs:
            ge += 1
    return {"stat": stat, "observed": float(obs), "p_value": (1 + ge) / (n_perm + 1), "n_perm": n_perm}


def cv_permutation_test(run_cv, y, n_perm: int = 200, seed: int = 0, stat: str = "R2") -> dict:
    """Permutation test with the CV inside the null: `run_cv(y_perm) -> pred` is called under each
    label permutation, so the null reflects the model's ability to fit shuffled labels.

    `run_cv` takes a permuted target vector and returns out-of-fold predictions aligned to it.
    """
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    obs = metrics(y, np.asarray(run_cv(y), float))[stat]
    ge = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        if metrics(yp, np.asarray(run_cv(yp), float))[stat] >= obs:
            ge += 1
    return {"stat": stat, "observed": float(obs), "p_value": (1 + ge) / (n_perm + 1), "n_perm": n_perm}


def multiseed_summary(values, key: str = "R2") -> dict:
    """Aggregate a list of metric dicts (or floats) over seeds → mean/sd/min/max for `key`."""
    arr = np.array([v[key] if isinstance(v, dict) else float(v) for v in values], float)
    return {f"{key}_mean": float(arr.mean()), f"{key}_sd": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
            f"{key}_min": float(arr.min()), f"{key}_max": float(arr.max()), "n_seeds": int(len(arr))}

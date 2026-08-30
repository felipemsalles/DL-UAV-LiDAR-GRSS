"""Does the LiDAR add anything beyond a cheap tabular covariate (stand age)?

Motivation
----------
Two simpler framings do not answer this cleanly:

1. LiDAR alone (PLS, LOTO rRMSE ~21%) shows that LiDAR carries signal, but not whether that signal
   is anything other than "this stand is young / this stand is mature", which the planting record
   already states for free.
2. LiDAR and age in one model: the canopy metrics are near-copies of age (r = 0.75..0.91 on the
   13 plots), so the joint fit is badly collinear. Coefficients split arbitrarily across duplicated
   predictors, the split is chosen by training-set noise, and it does not transfer to a held-out
   stand. Measured: adding LiDAR features to age degrades LOTO monotonically
   (0.81 -> 0.81 -> 0.73 -> 0.49 as features are added). That degradation is a variance artifact,
   not evidence about information content.

The orthogonalised question is the clean one: fit the age model, then ask whether the LiDAR can
predict the error the age model made. The residual target is orthogonal to age by construction, so
collinearity cannot confound the answer, and a null result here is a real null.

Everything is leave-one-stand-out. The base model is refit inside every fold; the residual model
never sees the held-out stand. A label-shuffle control and a permutation p-value guard against
reading noise as signal at N=13 (same discipline as `eval.significance`).
"""
from __future__ import annotations

import numpy as np

from ..eval import metrics
from ..eval.validation import bootstrap_ci


def _fit_predict(make_model, tr, te, y_tr):
    return make_model().fit(tr, y_tr).predict(te)


def base_and_residual_cv(make_base, make_resid, df, target="Vol_ha", group_col="talhao"):
    """Two-stage leave-one-group-out.

    Stage 1 fits `make_base` (e.g. volume from stand age) on the training stands.
    Stage 2 fits `make_resid` (the LiDAR model) on the *training* residuals of stage 1, then adds
    its prediction on top of the base prediction for the held-out stand.

    Returns (base_pred, combined_pred, resid_pred, resid_true) — all out-of-fold, length len(df).

    `resid_true` is the target the LiDAR would have to hit: the held-out error the age model
    actually made. `resid_pred` is what the LiDAR predicted for it. Correlating those two is the
    orthogonalised test; `combined_pred` shows whether acting on it would help end to end.
    """
    y = df[target].to_numpy(float)
    n = len(df)
    base_pred = np.empty(n)
    resid_pred = np.empty(n)
    groups = df[group_col].to_numpy()

    for g in np.unique(groups):
        te_mask = groups == g
        tr, te = df[~te_mask], df[te_mask]
        y_tr = y[~te_mask]
        idx = np.where(te_mask)[0]

        base_pred[idx] = _fit_predict(make_base, tr, te, y_tr)
        # residuals of the base model on the training stands (in-sample for stage 1, but the whole
        # two-stage object is still evaluated only on the untouched held-out stand)
        tr_resid = y_tr - make_base().fit(tr, y_tr).predict(tr)
        resid_pred[idx] = _fit_predict(make_resid, tr, te, tr_resid)

    return base_pred, base_pred + resid_pred, resid_pred, y - base_pred


def _corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def shuffled_null(make_base, make_resid, df, target="Vol_ha", group_col="talhao",
                  n_rep=200, seed=0):
    """Null distribution of `resid_r`, obtained by refitting the whole two-stage pipeline on
    permuted volumes.

    Permuting the output vector instead gives a null centred on zero, which is the wrong null here:
    leave-one-group-out with few groups induces a structural negative correlation between a held-out
    value and any prediction built from the other groups, because when a stand is above average the
    mean of the remaining stands is necessarily pulled below it. On this data the refit null sits
    near r = -0.8, not 0, so a naive permutation p-value makes a meaningless r = +0.25 look almost
    significant.

    Returns the array of null correlations; an observed r must be compared against it.
    """
    rng = np.random.default_rng(seed)
    y = df[target].to_numpy(float)
    out = np.empty(n_rep)
    work = df.copy()
    for i in range(n_rep):
        work[target] = rng.permutation(y)
        _, _, pred, true = base_and_residual_cv(make_base, make_resid, work, target, group_col)
        out[i] = _corr(pred, true)
    return out


def lidar_beyond_base(make_base, make_resid, df, target="Vol_ha", group_col="talhao",
                      n_rep=200, seed=0):
    """Full orthogonalised report: does `make_resid` explain the held-out error of `make_base`?

    Returns a dict with
      base / combined   — metrics of the age model and of age+LiDAR-on-residual
      resid_r           — correlation between predicted and actual held-out residual (the direct test)
      resid_R2          — R² of the residual prediction (negative = worse than predicting no error)
      null_r_mean/sd    — the refit shuffled-label null for `resid_r` (see `shuffled_null`)
      p_vs_null         — fraction of shuffled refits reaching `resid_r` or better (one-sided)
      z_vs_null         — how many null standard deviations above the null mean `resid_r` sits
    """
    y = df[target].to_numpy(float)
    base_pred, comb_pred, resid_pred, resid_true = base_and_residual_cv(
        make_base, make_resid, df, target, group_col)
    r = _corr(resid_pred, resid_true)

    null = shuffled_null(make_base, make_resid, df, target, group_col, n_rep=n_rep, seed=seed)
    sd = float(null.std(ddof=1)) or float("nan")

    return {
        "base": metrics(y, base_pred),
        "combined": metrics(y, comb_pred),
        "base_rRMSE_ci": bootstrap_ci(y, base_pred, seed=seed)["rRMSE_ci"],
        "combined_rRMSE_ci": bootstrap_ci(y, comb_pred, seed=seed)["rRMSE_ci"],
        "resid_r": r,
        "resid_R2": float(metrics(resid_true, resid_pred)["R2"]),
        "null_r_mean": float(null.mean()),
        "null_r_sd": sd,
        "p_vs_null": float((null >= r).mean()),
        "z_vs_null": float((r - null.mean()) / sd) if np.isfinite(sd) else float("nan"),
        "n": len(df),
        "n_groups": int(df[group_col].nunique()),
    }

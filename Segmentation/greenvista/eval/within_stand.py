"""Does the wall-to-wall map carry information inside a stand, or only between stands?

The claim under test is that the LiDAR's contribution is the within-stand map, on the argument that
stand age is constant across a stand, so any spatial detail inside it can only come from the
flight. That argument is about provenance, not accuracy. The source decomposition found that canopy
metrics correlate ~0 with volume once age is held fixed, which is precisely the regime the
within-stand map lives in, so the map could be drawing confident-looking spatial patterns out of
noise.

The test is direct: take the field plots, subtract each stand's own mean from both the observed
volume and the prediction, and ask whether the leftovers still line up. Removing the stand mean
removes everything age could have explained; what survives is within-stand signal or nothing.

The null must permute within each stand. Permuting globally also destroys the between-stand
structure, which makes the null far too easy to beat and turns "the model knows stand 4 is young"
into apparent within-stand skill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def center_by_group(values, groups):
    """Deviations from each group's own mean — the within-stand part of a signal."""
    s = pd.Series(np.asarray(values, float))
    g = pd.Series(np.asarray(groups))
    return (s - s.groupby(g.values).transform("mean")).to_numpy()


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def decompose_correlation(y, yhat, groups):
    """Split the predicted/observed agreement into its between-stand and within-stand parts.

    Returns dict with
      r_total    — the raw correlation, which is what a headline R² is built from
      r_between  — correlation of the stand means (how well the model ranks whole stands)
      r_within   — correlation after removing stand means (what the map claims to deliver)
      n_within_df — residual degrees of freedom available for the within test (n - n_groups)
    """
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    g = pd.Series(np.asarray(groups))
    means = pd.DataFrame({"y": y, "yhat": yhat, "g": g.values}).groupby("g").mean()
    return {
        "r_total": _corr(y, yhat),
        "r_between": _corr(means.y.to_numpy(), means.yhat.to_numpy()),
        "r_within": _corr(center_by_group(y, g.values), center_by_group(yhat, g.values)),
        "n": len(y),
        "n_groups": int(g.nunique()),
        "n_within_df": int(len(y) - g.nunique()),
    }


def within_stand_permutation(y, yhat, groups, n_perm=5000, seed=0):
    """p-value for `r_within`, permuting the observed volumes within each stand.

    Stand means are preserved by construction, so this asks exactly one question: given that the
    model knows which stand is which, does it also know which plot inside a stand has more wood?
    Returns (r_within, p_two_sided, null_sd).
    """
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    g = np.asarray(groups)
    obs = _corr(center_by_group(y, g), center_by_group(yhat, g))
    if not np.isfinite(obs):
        return obs, np.nan, np.nan

    rng = np.random.default_rng(seed)
    idx_by_group = [np.where(g == u)[0] for u in np.unique(g)]
    null = np.empty(n_perm)
    yp = y.copy()
    for i in range(n_perm):
        for idx in idx_by_group:
            yp[idx] = rng.permutation(y[idx])
        null[i] = _corr(center_by_group(yp, g), center_by_group(yhat, g))
    null = null[np.isfinite(null)]
    p = float((np.abs(null) >= abs(obs)).mean()) if len(null) else np.nan
    return obs, p, float(null.std(ddof=1)) if len(null) > 1 else np.nan

"""Site-index / yield-curve framing for the mechanistic Hdom(+age) -> Vol_ha model.

Thin helper layer on top of `greenvista.stand_volume`, which stays authoritative for `StandVolume`,
`lidar_hdom`, `site_index` and `volume_map`. It shows that the constrained log-log form

    Vol_ha = exp(a + b*ln(Hdom) + c*ln(age))

— the same family as a Clutter/Schumacher yield curve keyed to site index, which the company's G2
inventory already uses at the tree level (Schumacher-Hall, R^2=0.995) — extrapolates sanely to the one
young stand (stand 4, ~4.8 yr) that breaks every unconstrained empirical fit (PLS, GPR) under
leave-one-stand-out.

Functions:
  mechanistic_model       - `make_*`-style factory for StandVolume (matches area_based.models convention
                            so it drops straight into greenvista.area_based.validate's CV harness).
  yield_curve             - Vol_ha(Hdom) at a fixed age for a fitted model — the curve itself. Used both
                            to check monotonicity/extrapolation (tests/test_site_index_yield.py) and
                            to plot how each candidate model behaves outside its training Hdom range.
  metrics_for_group /
  metrics_excluding_group - split an out-of-fold prediction vector by group, so "does it fail only on
                            the young stand?" has a direct answer: metrics on the other 5 stands
                            vs on stand 4 alone (rather than one pooled LOTO number hiding the split).
  holdout_table           - tidy per-plot observed/predicted/bias% table for one held-out group (T4).
  stand_site_index        - per-stand mean site index (Hdom projected to a reference age) — the
                            productivity stratifier the company's G2/site-index inventory expects.
"""
import numpy as np
import pandas as pd

from greenvista.stand_volume import StandVolume, site_index
from greenvista.eval import metrics as _metrics


def mechanistic_model(cols=("Hdom", "age")):
    """`make_*`-style factory for the mechanistic yield-curve model: Vol_ha = exp(a + b*ln(Hdom) +
    c*ln(age)), fit by `StandVolume` (robust Huber log-log regression). A zero-arg callable, matching
    `greenvista.area_based.models.make_*` so it plugs directly into `area_based.validate` (`_lopo`,
    `_group_cv`, `oof_predictions`)."""
    return StandVolume(cols=cols)


def yield_curve(model, hdom_grid, age):
    """Vol_ha predicted by a fitted `model` across a grid of Hdom values at fixed `age`.

    Works for any model whose `.predict` accepts a DataFrame with 'Hdom' and 'age' columns
    (StandVolume, or a GPR/log-GPR fit on the same two columns) — the interface the CV harness already
    requires. This is the yield curve itself: hold age fixed, vary dominant height, read off volume.
    """
    hdom_grid = np.asarray(hdom_grid, float)
    X = pd.DataFrame({"Hdom": hdom_grid, "age": np.full(hdom_grid.shape, float(age))})
    return np.asarray(model.predict(X), float)


def metrics_for_group(df, y, pred, group_col, value):
    """Metrics computed only on rows where `group_col == value` (e.g. the T4 hold-out fold alone).
    N is tiny (2 plots for stand 4) so R^2 is not meaningful here — read bias_pct / rRMSE_pct."""
    m = (df[group_col] == value).to_numpy()
    return _metrics(np.asarray(y)[m], np.asarray(pred)[m])


def metrics_excluding_group(df, y, pred, group_col, value):
    """Metrics on every row except `group_col == value` — separates "does this model work everywhere
    but the excluded stand?" from the single pooled leave-one-stand-out number, which can hide a
    collapse concentrated entirely in one stand."""
    m = (df[group_col] != value).to_numpy()
    return _metrics(np.asarray(y)[m], np.asarray(pred)[m])


def holdout_table(df, y, pred, group_col, value, id_cols=("talhao", "parcela", "age", "Hdom")):
    """Tidy per-plot table (observed, predicted, bias%, abs%err) for the rows where `group_col ==
    value` — the T4-fold detail behind the headline metrics, for the results doc / figure."""
    m = (df[group_col] == value).to_numpy()
    out = df.loc[m, [c for c in id_cols if c in df.columns]].copy().reset_index(drop=True)
    yv = np.asarray(y, float)[m]
    pv = np.asarray(pred, float)[m]
    out["observed_Vol_ha"] = yv
    out["predicted_Vol_ha"] = pv
    out["bias_pct"] = (pv - yv) / yv * 100.0
    out["abs_pct_err"] = np.abs(out["bias_pct"])
    return out


def stand_site_index(df, hdom_col="Hdom", age_col="age", ref_age=7.0, group_col="talhao"):
    """Per-stand mean site index (Hdom projected to `ref_age` via `stand_volume.site_index`) — the
    productivity stratifier the company's G2/site-index inventory expects (illustrative guide-curve
    exponent b=1; see `stand_volume.site_index` docstring for the caveat). Returns a DataFrame indexed
    by stand with mean Hdom, mean age, and SI@ref_age.
    """
    si = site_index(df[hdom_col], df[age_col], ref_age=ref_age)
    tmp = df[[group_col, hdom_col, age_col]].copy()
    tmp["SI"] = si
    return (tmp.groupby(group_col).mean(numeric_only=True)
               .rename(columns={hdom_col: "Hdom_mean", age_col: "age_mean"}))

"""Leave-one-plot-out CV with bootstrap CIs and interval calibration (N=13).

Reported `cov95` always comes from out-of-fold jackknife+ conformal residuals — the identical method
for every model. A parametric interval from the in-sample residual std is meaningless for
near-interpolating RF/kNN, so it is reported separately as `coverage95_parametric`.

Leave-one-stand-out group CV is reported next to LOPO. LOPO trains on same-stand siblings (same
age/clone/spatial block) and therefore overstates cross-stand generalization; the group split (one
whole stand out) is the transfer number.
"""
import numpy as np

from greenvista.eval import (metrics, bootstrap_ci, drop_leaky_columns,
                             jackknife_plus_interval, coverage, interval_score, permutation_test)
from greenvista import config
from .data import FEATURES
from .models import (make_log_allometric, make_elasticnet, make_gpr, make_baseline_lm,
                     make_rf, make_knn, make_pca_ols, make_quantile_gbm)
from greenvista.eval.hierarchical import make_hierarchical
from greenvista.mechanistic_volume import make_mechanistic

DEFAULT_MODELS = {"log_allometric": make_log_allometric, "elasticnet": make_elasticnet,
                  "gpr": make_gpr, "rf": make_rf, "knn": make_knn, "pca_ols": make_pca_ols,
                  "quantile_gbm": make_quantile_gbm, "baseline_lm": make_baseline_lm}

# --- Hdom-augmented models (opt-in; not in DEFAULT_MODELS) --------------------------------------
# These consume columns the standard load_plots() table does not carry, so adding them to
# DEFAULT_MODELS would break every run(df) on a plain metric table. Opt in with
# `run(df_aug, models=HDOM_MODELS)` on a table augmented with LiDAR-derived columns:
#   "hier_bayes"  needs  "Hdom", "age" + the group column (the stand id)  — build via
#                 scripts/hdom_yield_pipeline.build_table
#   "mechanistic" needs  "Hdom", "Hmean", "ff3d_n_ha", "field_n_ha" — build via
#                 scripts/exp_mechanistic_decomposition
# hier_bayes fits a full MCMC per fold (LOPO ≈ minutes). Its run()-reported cov95 is the harness
# jackknife+ conformal; the richer posterior-predictive coverage comes from the estimator's own
# predict_interval (or the experiment script / manifest).
HDOM_MODELS = {
    "hier_bayes": lambda: make_hierarchical(("Hdom", "age")),
    "mechanistic": make_mechanistic,
}


def leakage_guard(df, target="Vol_ha", cols=FEATURES, thresh=0.999):
    """Run the target-leakage tripwire on the modeling table before CV. Returns the kept feature
    names; warns (via drop_leaky_columns) if any feature is ~perfectly correlated with the target or
    is NaN. Not destructive — it surfaces problems and leaves the caller to act."""
    X = df[list(cols)].to_numpy(float)
    y = df[target].to_numpy(float)
    _, kept = drop_leaky_columns(X, y, thresh=thresh)
    return [list(cols)[j] for j in kept]


def _lopo(make_model, df, target="Vol_ha"):
    """Leave-one-plot-out predictions + 95% *parametric* PI bounds for each held-out plot."""
    n = len(df)
    pred = np.empty(n); lo = np.empty(n); hi = np.empty(n)
    for i in range(n):
        tr = df.drop(df.index[i]); te = df.iloc[[i]]
        m = make_model().fit(tr, tr[target].to_numpy())
        pred[i] = m.predict(te)[0]
        l, h = m.predict_interval(te, alpha=0.05)
        lo[i], hi[i] = l[0], h[0]
    return pred, lo, hi


def _group_cv(make_model, df, target="Vol_ha", group_col=config.GROUP_COL):
    """Leave-one-group-out predictions. One whole `group_col` value (one stand) held out per fold."""
    pred = np.empty(len(df))
    groups = df[group_col].to_numpy()
    for g in np.unique(groups):
        te_mask = groups == g
        tr = df[~te_mask]; te = df[te_mask]
        m = make_model().fit(tr, tr[target].to_numpy())
        pred[np.where(te_mask)[0]] = m.predict(te)
    return pred


def group_cv_intervals(make_model, df, target="Vol_ha", group_col=config.GROUP_COL, alpha=0.05):
    """Leave-one-group-out predictions with held-out prediction intervals (companion to `_group_cv`,
    which returns point predictions only). For models whose `predict_interval` widens for an unseen
    stand — the hierarchical partial-pooling model draws a fresh u~N(0,σ_stand) for a held-out
    stand — this is the cross-stand uncertainty band and a natural map-extrapolation flag.
    Returns (pred, lo, hi)."""
    n = len(df)
    pred = np.empty(n); lo = np.empty(n); hi = np.empty(n)
    groups = df[group_col].to_numpy()
    for g in np.unique(groups):
        te_mask = groups == g
        tr = df[~te_mask]; te = df[te_mask]
        m = make_model().fit(tr, tr[target].to_numpy())
        idx = np.where(te_mask)[0]
        pred[idx] = m.predict(te)
        l, h = m.predict_interval(te, alpha=alpha)
        lo[idx] = l; hi[idx] = h
    return pred, lo, hi


def oof_predictions(make_model, df, target="Vol_ha", group_col=config.GROUP_COL):
    """Out-of-fold predictions under both schemes → (lopo_pred, loto_pred). Ensembling averages these
    out-of-fold vectors, so every component prediction of the ensemble stays held-out."""
    lopo = _lopo(make_model, df, target)[0]
    loto = (_group_cv(make_model, df, target, group_col)
            if group_col is not None and group_col in df.columns and df[group_col].nunique() > 1 else None)
    return lopo, loto


def run(df, models=None, target="Vol_ha", group_col=config.GROUP_COL, perm=0, seed=0):
    """Return {model_name: metrics} under identical LOPO-CV, with conformal cov95 (single source),
    leave-one-stand-out metrics, bootstrap CIs, and (optional) a permutation p-value.
    """
    models = models or DEFAULT_MODELS
    y = df[target].to_numpy()
    out = {}
    for name, make in models.items():
        pred, lo, hi = _lopo(make, df, target)
        mm = metrics(y, pred)
        ci = bootstrap_ci(y, pred, seed=seed)
        mm["R2_ci"] = ci["R2_ci"]; mm["rRMSE_ci"] = ci["rRMSE_ci"]

        # reported coverage = out-of-fold conformal, identical method for every model
        c_lo, c_hi = jackknife_plus_interval(y, pred, alpha=0.05)
        mm["coverage95"] = coverage(y, c_lo, c_hi)
        mm["interval_score"] = interval_score(y, c_lo, c_hi, alpha=0.05)
        # parametric interval kept for reference and for the map cell band, under its own key
        calibrated = getattr(make(), "calibrated", True)
        mm["coverage95_parametric"] = float(((y >= lo) & (y <= hi)).mean()) if calibrated else float("nan")

        # cross-stand generalization
        if group_col is not None and group_col in df.columns and df[group_col].nunique() > 1:
            gp = _group_cv(make, df, target, group_col)
            gm = metrics(y, gp)
            mm["R2_group"] = gm["R2"]; mm["rRMSE_group_pct"] = gm["rRMSE_pct"]
            mm["n_groups"] = int(df[group_col].nunique())

        # permutation test: is LOPO skill better than chance?
        if perm:
            mm["perm_p"] = permutation_test(y, pred, n_perm=perm, seed=seed)["p_value"]

        out[name] = mm
    return out

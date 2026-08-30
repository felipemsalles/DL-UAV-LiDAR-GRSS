"""Registration smoke test for the Hdom-augmented model zoo.

Verifies the two estimators are registered in `greenvista.area_based.validate.HDOM_MODELS`,
conform to the fit/predict/predict_interval contract, plug into `run()` and the
`group_cv_intervals` hook, and are kept out of `DEFAULT_MODELS`, so a plain `run(df)` on a metric
table is unaffected.

Only the cheap `mechanistic` model is actually fit; `hier_bayes` (a full MCMC per fold) is
construct-checked only, since its sampling interface is exercised in
`tests/test_hierarchical_bayes.py`.
"""
import numpy as np
import pandas as pd

from greenvista import config
from greenvista.area_based import validate as V


def _augmented_table(n_groups=3, per=3):
    """Tiny synthetic Hdom-augmented plot table with every column the two models read."""
    rng = np.random.default_rng(0)
    rows = []
    for g in range(1, n_groups + 1):
        hdom0 = 20.0 + 4.0 * g
        for _ in range(per):
            hdom = hdom0 + rng.normal(0, 0.5)
            field_n = 1400.0 - 80.0 * g + rng.normal(0, 30)
            ff3d_n = 0.6 * field_n + rng.normal(0, 20)
            rows.append({config.GROUP_COL: g, "Hdom": hdom, "Hmean": hdom - 3.0, "age": 10.0,
                         "ff3d_n_ha": ff3d_n, "field_n_ha": field_n,
                         "Vol_ha": 12.0 * hdom + 0.05 * field_n + rng.normal(0, 5)})
    return pd.DataFrame(rows)


def test_hdom_models_registered_and_kept_out_of_default():
    assert set(V.HDOM_MODELS) == {"hier_bayes", "mechanistic"}
    # opt-in only: they must not leak into the default zoo (would break run(df) on a metric table)
    assert "hier_bayes" not in V.DEFAULT_MODELS and "mechanistic" not in V.DEFAULT_MODELS
    for _name, make in V.HDOM_MODELS.items():
        m = make()  # cheap: construction only (hier_bayes imports PyMC lazily inside fit)
        assert all(hasattr(m, a) for a in ("fit", "predict", "predict_interval"))
        assert getattr(m, "calibrated", False) is True


def test_mechanistic_runs_through_validate_run():
    df = _augmented_table()
    out = V.run(df, models={"mechanistic": V.HDOM_MODELS["mechanistic"]}, perm=0)
    m = out["mechanistic"]
    for k in ("R2", "rRMSE_pct", "coverage95", "R2_group", "rRMSE_group_pct"):
        assert k in m
    assert np.isfinite(m["R2"]) and np.isfinite(m["rRMSE_pct"])


def test_group_cv_intervals_hook_brackets_predictions():
    df = _augmented_table()
    pred, lo, hi = V.group_cv_intervals(V.HDOM_MODELS["mechanistic"], df)
    assert pred.shape == lo.shape == hi.shape == (len(df),)
    assert np.all(hi >= lo)
    assert np.all(np.isfinite(pred)) and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))

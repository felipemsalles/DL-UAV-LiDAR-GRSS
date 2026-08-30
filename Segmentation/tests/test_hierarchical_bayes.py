"""Smoke tests for the Bayesian hierarchical stand-volume model.

Few draws, so the file samples in a couple of seconds. On synthetic data generated with a real
stand effect the model must (a) recover sigma_stand > 0, (b) produce finite point predictions and
finite predictive intervals, and (c) plug into the project CV harness (_lopo/_group_cv), including
the unseen-stand (LOTO) regime.
"""
import numpy as np
import pandas as pd
import pytest

from greenvista.eval.hierarchical import HierarchicalStandVolume
from greenvista.area_based import validate as abv


def _synth(n_stands=5, per=4, sigma_stand=0.30, sigma_resid=0.06, seed=0):
    """log Vol = a + b*ln(Hdom) + c*ln(age) + u[stand] + eps, with a genuine between-stand effect."""
    rng = np.random.default_rng(seed)
    u = rng.normal(0, sigma_stand, n_stands)                 # true stand effects
    rows = []
    for s in range(n_stands):
        age = 5.0 + 5.0 * rng.random()                       # stand-constant age
        for _ in range(per):
            hdom = 20.0 + 15.0 * rng.random()
            logv = 4.0 + 0.8 * np.log(hdom) + 0.5 * np.log(age) + u[s] + rng.normal(0, sigma_resid)
            rows.append({"talhao": s, "Hdom": hdom, "age": age, "Vol_ha": float(np.exp(logv))})
    return pd.DataFrame(rows)


def _fit(df, **kw):
    return HierarchicalStandVolume(("Hdom", "age"), draws=60, tune=60, chains=2,
                                   target_accept=0.9, seed=0, **kw).fit(df, df.Vol_ha.to_numpy())


def test_recovers_stand_variance_and_predicts_finite():
    df = _synth()
    m = _fit(df)
    vc = m.variance_components()
    # (a) a real stand effect → posterior sigma_stand mass is strictly positive
    assert vc["sigma_stand"]["mean"] > 0.0
    assert vc["sigma_stand"]["q97.5"] > vc["sigma_stand"]["q2.5"] >= 0.0
    # sign of the log-log elasticities is recovered (Hdom & age both raise volume)
    assert vc["coef_ln_Hdom"]["mean"] > 0.0
    assert vc["coef_ln_age"]["mean"] > 0.0
    # (b) finite point predictions
    pred = m.predict(df)
    assert pred.shape == (len(df),)
    assert np.all(np.isfinite(pred)) and np.all(pred > 0)
    # ICC is a valid fraction
    assert 0.0 <= vc["ICC"]["mean"] <= 1.0


def test_predict_interval_finite_and_brackets_point():
    df = _synth()
    m = _fit(df)
    pred = m.predict(df)
    lo, hi = m.predict_interval(df, alpha=0.05)
    assert np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))
    assert np.all(hi > lo)
    assert np.all((pred >= lo) & (pred <= hi))       # point sits inside its own 95% PPD interval


def test_unseen_stand_interval_is_wider_than_seen():
    """LOTO: an unseen stand folds sigma_stand into the interval → wider than an in-sample row
    of the same design."""
    df = _synth(seed=1)
    tr = df[df.talhao != 0]
    m = HierarchicalStandVolume(("Hdom", "age"), draws=80, tune=80, chains=2,
                                target_accept=0.9, seed=0).fit(tr, tr.Vol_ha.to_numpy())
    te = df[df.talhao == 0]
    lo_u, hi_u = m.predict_interval(te)              # unseen stand
    seen = tr.iloc[[0]]
    lo_s, hi_s = m.predict_interval(seen)            # a seen stand
    assert np.all(np.isfinite(lo_u)) and np.all(np.isfinite(hi_u))
    assert (hi_u - lo_u).mean() > (hi_s - lo_s).mean()


def test_plugs_into_lopo_and_group_cv():
    """Interface conformance with the project harness (fit/predict/predict_interval on a DataFrame)."""
    df = _synth(n_stands=4, per=3, seed=2)
    make = lambda: HierarchicalStandVolume(("Hdom", "age"), draws=50, tune=50, chains=2,
                                           target_accept=0.9, seed=0)
    lopo, lo, hi = abv._lopo(make, df)
    assert lopo.shape == (len(df),) and np.all(np.isfinite(lopo))
    assert np.all(hi >= lo)
    loto = abv._group_cv(make, df)                   # whole-stand held out per fold
    assert loto.shape == (len(df),) and np.all(np.isfinite(loto))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:warnings"]))

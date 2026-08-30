"""Tests for the LOTO stem-density calibrator, on synthetic data.

Three properties:
  1. recovery — with one detection ratio shared across stands, the global correction learned
     under LOTO recovers field density almost exactly.
  2. no peek — with a different ratio per stand, the correction learned only on the other stands
     carries real error, while an oracle fitted on the held-out stand itself is near perfect.
     The calibration must not fit a per-stand factor and then be scored on that same stand.
  3. proxy — when the correction is a function of an observable proxy, the proxy model
     extrapolates it to the held-out stand and beats the global correction.
"""
import numpy as np
import pandas as pd
import pytest

from greenvista.density import DensityCalibrator, cv_predict, to_density, detection_rate
from greenvista.eval import metrics


def _panel(ratios, n_per=3, seed=0, y_lo=1000.0, y_hi=1700.0, z=None):
    """Build a synthetic stand panel: `ratios` is a per-stand detection ratio.

    raw_stems_ha = field_stems_ha * ratio[stand]; a proxy column `z` (one value per stand) is
    attached when given. Returns (df, y_field)."""
    rng = np.random.default_rng(seed)
    rows = []
    for g, r in enumerate(ratios):
        yg = rng.uniform(y_lo, y_hi, n_per)
        for k in range(n_per):
            row = {"talhao": g, "field_stems_ha": yg[k], "raw": yg[k] * r}
            if z is not None:
                row["z"] = z[g]
            rows.append(row)
    df = pd.DataFrame(rows)
    return df, df.field_stems_ha.to_numpy()


def test_unit_helpers_and_estimator():
    assert to_density(45, 0.045).round(0) == 1000
    assert detection_rate([6, 4], [12, 6]) == pytest.approx(10 / 18)
    # raw method is identity
    df, y = _panel([0.6] * 4)
    raw = DensityCalibrator("raw", method="raw").fit(df, y).predict(df)
    assert np.allclose(raw, df.raw.to_numpy())
    # global ratio factor == sum(y)/sum(raw)
    m = DensityCalibrator("raw", method="global", agg="ratio").fit(df, y)
    assert m.factor_ == pytest.approx(y.sum() / df.raw.sum())


def test_recovery_shared_ratio_loto_global():
    # one shared ratio → the correction learned on other stands transfers perfectly.
    df, y = _panel([0.6, 0.6, 0.6, 0.6, 0.6])
    make = lambda: DensityCalibrator("raw", method="global")
    pred = cv_predict(make, df, y, groups=df.talhao.to_numpy())
    assert metrics(y, pred)["rRMSE_pct"] < 1.0     # ~exact recovery under LOTO


def test_no_peek_leakage_guard():
    # different ratio per stand: a peeking oracle is perfect; LOTO is not.
    ratios = [0.40, 0.52, 0.61, 0.72, 0.83]
    df, y = _panel(ratios, seed=1)
    g = df.talhao.to_numpy()
    make = lambda: DensityCalibrator("raw", method="global")

    honest = cv_predict(make, df, y, groups=g)                     # correction from other stands only
    # oracle that peeks: fits the per-stand factor on the held-out stand itself, then scores it.
    peek = np.empty(len(df))
    for gg in np.unique(g):
        te = g == gg
        m = make().fit(df[te], y[te])          # <-- leakage: fit on the same rows we score
        peek[te] = m.predict(df[te])

    r_honest = metrics(y, honest)["rRMSE_pct"]
    r_peek = metrics(y, peek)["rRMSE_pct"]
    assert r_peek < 1.0                        # peeking is near perfect
    assert r_honest > 8.0                      # LOTO carries the real cross-stand error
    assert r_honest > 5 * r_peek               # and is much worse than the leaked version
    # the held-out prediction must differ from the leaked one (proves no peek)
    assert not np.allclose(honest, peek, rtol=0.02)


def test_proxy_beats_global_when_correction_is_observable():
    # correction factor c = field/raw is an exact linear function of an observable proxy z.
    z = np.array([0.5, 0.7, 0.9, 1.1, 1.3])         # one observable value per stand
    c = 1.0 + 1.2 * z                               # true correction factor per stand (>1)
    ratios = list(1.0 / c)                          # raw = field * (1/c) = field / c
    df, y = _panel(ratios, seed=2, z=z)
    g = df.talhao.to_numpy()

    glob = cv_predict(lambda: DensityCalibrator("raw", method="global"), df, y, groups=g)
    prox = cv_predict(lambda: DensityCalibrator("raw", method="proxy", proxy_col="z"), df, y, groups=g)

    r_glob = metrics(y, glob)["rRMSE_pct"]
    r_prox = metrics(y, prox)["rRMSE_pct"]
    assert r_prox < r_glob                          # the observable proxy generalizes the correction
    assert r_prox < 3.0                             # near-exact when the relationship is real+linear

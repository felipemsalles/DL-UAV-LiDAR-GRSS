"""Tests for the Hdom+age stand-volume pipeline (greenvista/stand_volume.py)."""
import numpy as np
import pandas as pd
import pytest

from greenvista.stand_volume import lidar_hdom, canopy_cover, site_index, StandVolume


def test_lidar_hdom_recovers_dominant_height():
    """A canopy with tallest points ~30 m → dominant-height proxy near 30, above the mean."""
    rng = np.random.default_rng(0)
    n = 4000
    # understory at ~10 m + dominant crowns at ~30 m
    z = np.where(rng.random(n) < 0.5, rng.uniform(4, 12, n), rng.uniform(26, 30, n))
    xyz = np.column_stack([rng.uniform(-10, 10, n), rng.uniform(-10, 10, n), z])
    hd = lidar_hdom(xyz, (0.0, 0.0), radius=11.3, cell=4.0)
    assert 24 < hd <= 30                       # tracks the dominant stratum, not the ~19 m overall mean


def test_canopy_cover_and_site_index():
    z = np.array([1.0, 1.5, 5.0, 20.0, 28.0])
    assert canopy_cover(z) == pytest.approx(0.6)
    # SI projects Hdom to ref age: a younger stand's Hdom is scaled up toward the reference
    assert site_index(20.0, 5.0, ref_age=7.0, b=1.0) > 20.0
    assert site_index(30.0, 10.0, ref_age=7.0, b=1.0) < 30.0


def test_stand_volume_fit_predict_interval_and_generalizes():
    """StandVolume recovers a clean Vol ~ Hdom^b · age^c law and honours the CV interface."""
    rng = np.random.default_rng(0)
    rows = []
    for t in range(1, 7):
        age = 4 + 1.2 * t
        for p in range(3):
            hd = rng.uniform(18, 34)
            vol = np.exp(2.5 + 0.4 * np.log(hd) + 0.9 * np.log(age)) * rng.uniform(0.97, 1.03)
            rows.append({"talhao": t, "Hdom": hd, "age": age, "Vol_ha": vol})
    df = pd.DataFrame(rows)
    y = df.Vol_ha.to_numpy()
    m = StandVolume(("Hdom", "age")).fit(df, y)
    p = m.predict(df)
    assert np.corrcoef(p, y)[0, 1] > 0.97
    lo, hi = m.predict_interval(df)
    assert (lo <= hi).all() and "Vol_ha = exp" in m.equation()
    # generalizes under leave-one-stand-out
    from greenvista.area_based.validate import _group_cv
    from greenvista.eval import metrics
    pred = _group_cv(lambda: StandVolume(("Hdom", "age")), df)
    assert metrics(y, pred)["R2"] > 0.6

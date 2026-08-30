import numpy as np
from greenvista.volume import CrownAllometry, MeanRegressor, CrownGPR

def test_allometry_recovers_power_law():
    rng = np.random.default_rng(0)
    area = rng.uniform(5, 40, 60); H = rng.uniform(12, 32, 60)
    V = 0.05 * area ** 0.8 * H ** 1.2          # known log-linear law
    m = CrownAllometry().fit(np.column_stack([area, H]), V)
    pred = m.predict(np.column_stack([area, H]))
    assert np.corrcoef(pred, V)[0, 1] > 0.99

def test_allometry_rejects_nonpositive():
    import pytest
    with pytest.raises(ValueError):
        CrownAllometry().fit(np.array([[1.0, -2.0]]), np.array([1.0]))

def test_mean_regressor():
    m = MeanRegressor().fit(np.zeros((4, 2)), np.array([1.0, 2, 3, 4]))
    out = m.predict(np.zeros((3, 2)))
    assert np.allclose(out, 2.5) and len(out) == 3

def test_gpr_runs_with_uncertainty():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, (40, 2)); y = X[:, 0] * 0.3 + X[:, 1] * 0.1 + 1
    m = CrownGPR().fit(X, y)
    mu, sd = m.predict(X[:5], return_std=True)
    assert np.all(np.isfinite(mu)) and np.all(sd >= 0)

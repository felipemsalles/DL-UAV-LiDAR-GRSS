"""Leakage guard: (a) a leaked target inflates LOOCV to ~1, and (b) drop_leaky_columns removes
it so the true signal remains."""
import numpy as np
from sklearn.linear_model import LinearRegression
from greenvista.eval import loocv_predict, metrics, drop_leaky_columns

def _make():
    rng = np.random.default_rng(0)
    x = rng.uniform(5, 40, 40)
    y = 0.05 * x ** 1.1 + rng.normal(0, 0.2, 40)
    return x, y

def test_leak_inflates_then_guard_removes_it():
    x, y = _make()
    X_leak = np.column_stack([x, y.copy()])           # second column is the target
    # with the leak, LOOCV looks perfect
    yhat_leak = loocv_predict(LinearRegression, X_leak, y)
    assert metrics(y, yhat_leak)["R2"] > 0.99
    # guard removes the leaky column, leaving the true (lower) performance
    X_clean, kept = drop_leaky_columns(X_leak, y)
    assert kept.tolist() == [0]
    yhat = loocv_predict(LinearRegression, X_clean, y)
    assert metrics(y, yhat)["R2"] < 0.99

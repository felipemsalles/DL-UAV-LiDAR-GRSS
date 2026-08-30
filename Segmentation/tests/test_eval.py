import numpy as np
from greenvista.eval import metrics, loocv_predict, bootstrap_ci, drop_leaky_columns
from greenvista.volume import CrownAllometry

def test_metrics_perfect():
    y = np.array([1.0, 2, 3, 4])
    m = metrics(y, y)
    assert abs(m["R2"] - 1.0) < 1e-9 and m["rRMSE_pct"] < 1e-6

def test_loocv_predict_linear():
    rng = np.random.default_rng(0)
    a = rng.uniform(5, 40, 40); H = rng.uniform(12, 32, 40)
    V = 0.05 * a ** 0.8 * H ** 1.2
    X = np.column_stack([a, H])
    yhat = loocv_predict(CrownAllometry, X, V)
    assert metrics(V, yhat)["R2"] > 0.95

def test_bootstrap_ci_brackets_and_orders():
    rng = np.random.default_rng(0)
    y = rng.uniform(0.1, 1.0, 50); yhat = y + rng.normal(0, 0.05, 50)
    ci = bootstrap_ci(y, yhat, n_boot=500)
    lo, hi = ci["R2_ci"]
    assert lo < hi and -1 <= lo and hi <= 1

def test_drop_leaky_columns_removes_target_copy():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 1, 50)
    X = np.column_stack([rng.uniform(0, 1, 50), y.copy()])   # col 1 == target (leak)
    Xc, kept = drop_leaky_columns(X, y, thresh=0.999)
    assert kept.tolist() == [0] and Xc.shape[1] == 1

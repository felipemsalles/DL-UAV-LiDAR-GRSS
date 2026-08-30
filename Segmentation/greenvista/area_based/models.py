"""Low-parameter regressors for area-based volume, with prediction intervals.

All take a DataFrame X containing FEATURES; ElasticNet/GPR tune internally (inner CV / marginal
likelihood) so feature/penalty selection happens inside each CV fold (no leakage).

Interval note: the parametric `predict_interval` here uses the training residual std, which is
adequate for the well-behaved parametric models but structurally over-confident for
near-interpolating ones (RF, kNN). The reported coverage instead comes from out-of-fold conformal
intervals (`greenvista.eval.conformal`), computed at the CV level in `validate.run`, which is the
single source of `cov95`. `predict_interval` is kept for the map's per-cell band and the unit tests.
"""
from itertools import combinations

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import ElasticNetCV, LinearRegression, HuberRegressor, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, ConstantKernel as C, WhiteKernel,
                                              DotProduct, Matern)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from greenvista import config
from .data import FEATURES, BASELINE_COLS, ALL_METRICS, FEATURES_AGE, AGE

# Positive-valued candidate predictors for the log-allometric family (no zskew/zkurt — sign-changing).
ALLOM_CANDIDATES = ("zmean", "zq90", "zmax", "zq50", "zq70", "zsd", "pzabovezmean", AGE)


class LogAllometric:
    """ln(Vol) = a + sum b_i ln(x_i). Normal PI from training log-residual std."""
    calibrated = True  # produces a real prediction interval

    def __init__(self, cols=("zmean", "zq90")):
        self.cols = list(cols)

    def fit(self, X, y):
        L = np.log(X[self.cols].to_numpy())
        A = np.column_stack([np.ones(len(L)), L])
        self.coef_, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
        self.resid_sd_ = (np.log(y) - A @ self.coef_).std(ddof=1)
        return self

    def _pred_log(self, X):
        L = np.log(X[self.cols].to_numpy())
        A = np.column_stack([np.ones(len(L)), L])
        return A @ self.coef_

    def predict(self, X):
        return np.exp(self._pred_log(X))

    def predict_interval(self, X, alpha=0.05):
        z = norm.ppf(1 - alpha / 2)
        pl = self._pred_log(X)
        return np.exp(pl - z * self.resid_sd_), np.exp(pl + z * self.resid_sd_)


class _SkWrapper:
    """Wrap an sklearn regressor on a chosen feature set; normal PI from training residual std."""
    calibrated = True

    def __init__(self, pipe, cols=FEATURES):
        self.pipe = pipe
        self.cols = list(cols)

    def fit(self, X, y):
        self.pipe.fit(X[self.cols].to_numpy(), y)
        self.resid_sd_ = (y - self.pipe.predict(X[self.cols].to_numpy())).std(ddof=1)
        return self

    def predict(self, X):
        return self.pipe.predict(X[self.cols].to_numpy())

    def predict_interval(self, X, alpha=0.05):
        z = norm.ppf(1 - alpha / 2)
        p = self.predict(X)
        return p - z * self.resid_sd_, p + z * self.resid_sd_


class _GPR:
    """GPR on a chosen feature set; PI from posterior std (the headline uncertainty product)."""
    calibrated = True

    def __init__(self, cols=FEATURES):
        self.cols = list(cols)
        # Isotropic RBF (one length-scale): only 3 hyperparameters, identifiable at N=13.
        kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(1.0)
        self.pipe = Pipeline([
            ("sc", StandardScaler()),
            ("gp", GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                            n_restarts_optimizer=2, random_state=0)),
        ])

    def fit(self, X, y):
        self.pipe.fit(X[self.cols].to_numpy(), y)
        return self

    def predict(self, X):
        return self.pipe.predict(X[self.cols].to_numpy())

    def predict_interval(self, X, alpha=0.05):
        Xs = self.pipe[:-1].transform(X[self.cols].to_numpy())
        mu, sd = self.pipe[-1].predict(Xs, return_std=True)
        z = norm.ppf(1 - alpha / 2)
        return mu - z * sd, mu + z * sd


class _QuantileGBM:
    """Gradient-boosted quantile regression → heteroscedastic intervals.

    Three shallow GBMs (lower/median/upper pinball loss). Unlike a normal PI, the width adapts to the
    input, so it can be wider where volume is high or data is sparse. Kept to stumps at N=13, offered
    as an option rather than the default; quantile crossing is clamped.
    """
    calibrated = True

    def __init__(self, cols=FEATURES, alpha=0.05, max_depth=1, n_estimators=200, learning_rate=0.03):
        self.cols = list(cols)
        self.alpha = alpha
        self._kw = dict(loss="quantile", max_depth=max_depth, n_estimators=n_estimators,
                        learning_rate=learning_rate, random_state=config.ABA.seed)

    def fit(self, X, y):
        Xv = X[self.cols].to_numpy()
        y = np.asarray(y, float)
        self.lo_ = GradientBoostingRegressor(alpha=self.alpha / 2, **self._kw).fit(Xv, y)
        self.mid_ = GradientBoostingRegressor(alpha=0.5, **self._kw).fit(Xv, y)
        self.hi_ = GradientBoostingRegressor(alpha=1 - self.alpha / 2, **self._kw).fit(Xv, y)
        return self

    def predict(self, X):
        return self.mid_.predict(X[self.cols].to_numpy())

    def predict_interval(self, X, alpha=0.05):
        Xv = X[self.cols].to_numpy()
        lo = self.lo_.predict(Xv)
        hi = self.hi_.predict(Xv)
        return np.minimum(lo, hi), np.maximum(lo, hi)   # clamp any quantile crossing


class _PCAReg:
    """PCA on the full metric suite → OLS, with n_components chosen by inner LOOCV (nested, no leakage).

    The literature's "rich metric suite + principled dimension reduction" recipe (Silva 2016
    PCA→MLR) made leakage-safe: selection happens entirely inside fit(), so calling it per outer fold
    keeps the choice out of the reported CV.
    """
    calibrated = True

    def __init__(self, cols=ALL_METRICS, max_k=6):
        self.cols = list(cols)
        self.max_k = max_k

    def fit(self, X, y):
        Xv = X[self.cols].to_numpy()
        y = np.asarray(y, float)
        n = len(y)
        kmax = max(1, min(self.max_k, Xv.shape[1], n - 2))
        best_k, best_err = 1, np.inf
        for k in range(1, kmax + 1):
            err = 0.0
            for i in range(n):                       # inner LOOCV error for this k
                tr = np.arange(n) != i
                sc = StandardScaler().fit(Xv[tr])
                Z = sc.transform(Xv)
                pc = PCA(n_components=k).fit(Z[tr])
                T = pc.transform(Z)
                lr = LinearRegression().fit(T[tr], y[tr])
                err += float((lr.predict(T[i:i + 1])[0] - y[i]) ** 2)
            if err < best_err:
                best_err, best_k = err, k
        self.k_ = best_k
        self.sc_ = StandardScaler().fit(Xv)
        Z = self.sc_.transform(Xv)
        self.pca_ = PCA(n_components=best_k).fit(Z)
        T = self.pca_.transform(Z)
        self.lr_ = LinearRegression().fit(T, y)
        self.resid_sd_ = (y - self.lr_.predict(T)).std(ddof=1)
        return self

    def _T(self, X):
        return self.pca_.transform(self.sc_.transform(X[self.cols].to_numpy()))

    def predict(self, X):
        return self.lr_.predict(self._T(X))

    def predict_interval(self, X, alpha=0.05):
        z = norm.ppf(1 - alpha / 2)
        p = self.predict(X)
        return p - z * self.resid_sd_, p + z * self.resid_sd_


class _BestSubsetAllom:
    """Nested best-subset log-linear allometry: ln V = a + Σ b_i ln(x_i), predictor subset (size
    1..max_k) chosen inside fit() → leakage-safe. Literature-standard ABA selection (Silva 2016;
    Cosenza 2021) on the log scale.

    `select_by="group"` chooses the subset by inner leave-one-stand-out rather than
    leave-one-plot-out. Selecting by LOPO picks age-only subsets: age is nearly constant within a
    stand, so it minimises LOPO error but cannot predict a held-out stand's age (LOTO R²<0).
    Selecting by the same cross-stand criterion that is reported picks a height-driven subset that
    generalizes. `robust=True` fits the final coefficients with a Huber loss to resist the
    young-stand leverage.
    """
    calibrated = True

    def __init__(self, candidates=ALLOM_CANDIDATES, max_k=3, select_by="group", robust=False, group_col=None):
        self.candidates = list(candidates)
        self.max_k = max_k
        self.select_by = select_by
        self.robust = robust
        self.group_col = group_col or config.GROUP_COL

    @staticmethod
    def _design(Xlog, combo):
        return np.column_stack([np.ones(len(Xlog))] + [Xlog[:, j] for j in combo])

    def _inner_folds(self, X, n):
        """Leave-one-group-out folds when a group column is present, else leave-one-out."""
        if self.select_by == "group" and self.group_col in X.columns and X[self.group_col].nunique() > 1:
            g = X[self.group_col].to_numpy()
            return [(g != v, g == v) for v in np.unique(g)]
        return [(np.arange(n) != i, np.arange(n) == i) for i in range(n)]

    def _fit_coef(self, A, ylog):
        if self.robust and A.shape[0] > A.shape[1] + 1:
            hr = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000).fit(A[:, 1:], ylog)
            return np.concatenate([[hr.intercept_], hr.coef_])
        coef, *_ = np.linalg.lstsq(A, ylog, rcond=None)
        return coef

    def fit(self, X, y):
        y = np.asarray(y, float)
        Xlog = np.log(X[self.candidates].to_numpy(float))
        ylog = np.log(y)
        n, p = len(y), Xlog.shape[1]
        kmax = max(1, min(self.max_k, p, n - 2))
        folds = self._inner_folds(X, n)
        best = None
        for k in range(1, kmax + 1):
            for combo in combinations(range(p), k):
                err = 0.0
                for tr_mask, te_mask in folds:                # error in RAW units (matches rRMSE)
                    coef = self._fit_coef(self._design(Xlog[tr_mask], combo), ylog[tr_mask])
                    pred = np.exp(self._design(Xlog[te_mask], combo) @ coef)
                    err += float(np.sum((pred - y[te_mask]) ** 2))
                if best is None or err < best[0]:
                    best = (err, combo)
        self.combo_ = best[1]
        self.sel_cols_ = [self.candidates[j] for j in self.combo_]
        A = self._design(Xlog, self.combo_)
        self.coef_ = self._fit_coef(A, ylog)
        self.resid_sd_ = (ylog - A @ self.coef_).std(ddof=1)
        return self

    def _pred_log(self, X):
        Xlog = np.log(X[self.candidates].to_numpy(float))
        return self._design(Xlog, self.combo_) @ self.coef_

    def predict(self, X):
        return np.exp(self._pred_log(X))

    def predict_interval(self, X, alpha=0.05):
        z = norm.ppf(1 - alpha / 2)
        pl = self._pred_log(X)
        return np.exp(pl - z * self.resid_sd_), np.exp(pl + z * self.resid_sd_)


class _HuberAllom(LogAllometric):
    """Robust (Huber) log-linear allometry — downweights high-leverage plots (the young stand-4 pair
    dominates the fit at N=13). Same interface/interval as LogAllometric."""
    def fit(self, X, y):
        L = np.log(X[self.cols].to_numpy())
        self.hr_ = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=2000).fit(L, np.log(y))
        self.resid_sd_ = (np.log(y) - self.hr_.predict(L)).std(ddof=1)
        return self

    def _pred_log(self, X):
        return self.hr_.predict(np.log(X[self.cols].to_numpy()))


class _PLSNested:
    """Partial least squares with n_components chosen by inner LOOCV (nested). The Cosenza (2021) ABA
    workhorse: extracts a few Vol-correlated latent components from the full collinear metric suite."""
    calibrated = True

    def __init__(self, cols=ALL_METRICS, max_k=4):
        self.cols = list(cols)
        self.max_k = max_k

    def fit(self, X, y):
        Xv = X[self.cols].to_numpy(float)
        y = np.asarray(y, float)
        n = len(y)
        kmax = max(1, min(self.max_k, Xv.shape[1], n - 2))
        best_k, best_err = 1, np.inf
        for k in range(1, kmax + 1):
            err = 0.0
            for i in range(n):
                tr = np.arange(n) != i
                sc = StandardScaler().fit(Xv[tr]); Z = sc.transform(Xv)
                pls = PLSRegression(n_components=k).fit(Z[tr], y[tr])
                err += float((np.ravel(pls.predict(Z[i:i + 1]))[0] - y[i]) ** 2)
            if err < best_err:
                best_err, best_k = err, k
        self.k_ = best_k
        self.sc_ = StandardScaler().fit(Xv); Z = self.sc_.transform(Xv)
        self.pls_ = PLSRegression(n_components=best_k).fit(Z, y)
        self.resid_sd_ = (y - self.pls_.predict(Z).ravel()).std(ddof=1)
        return self

    def predict(self, X):
        return self.pls_.predict(self.sc_.transform(X[self.cols].to_numpy(float))).ravel()

    def predict_interval(self, X, alpha=0.05):
        z = norm.ppf(1 - alpha / 2)
        p = self.predict(X)
        return p - z * self.resid_sd_, p + z * self.resid_sd_


class _LogGPR:
    """GPR on the log target with a near-linear kernel (DotProduct + small Matern). The log target
    stabilizes the variance; the near-linear kernel keeps the model from memorizing stand identity
    the way the plain RBF-GPR does (that one wins LOPO but collapses under leave-one-stand-out), so
    the intervals stay calibrated while the fit still transfers across stands. PI from the
    log-posterior std."""
    calibrated = True

    def __init__(self, cols=FEATURES_AGE):
        self.cols = list(cols)
        kernel = C(1.0) * DotProduct(sigma_0=1.0) + C(0.3) * Matern(length_scale=2.0, nu=1.5) + WhiteKernel(0.1)
        self.pipe = Pipeline([("sc", StandardScaler()),
                              ("gp", GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                                              n_restarts_optimizer=3, random_state=0))])

    def fit(self, X, y):
        self.pipe.fit(X[self.cols].to_numpy(), np.log(np.asarray(y, float)))
        return self

    def predict(self, X):
        return np.exp(self.pipe.predict(X[self.cols].to_numpy()))

    def predict_interval(self, X, alpha=0.05):
        Xs = self.pipe[:-1].transform(X[self.cols].to_numpy())
        mu, sd = self.pipe[-1].predict(Xs, return_std=True)
        z = norm.ppf(1 - alpha / 2)
        return np.exp(mu - z * sd), np.exp(mu + z * sd)      # lognormal interval (asymmetric, positive)


def make_log_allometric(cols=("zmean", "zq90")):
    return LogAllometric(cols=cols)


def make_bestsubset_allom(candidates=ALLOM_CANDIDATES, max_k=3, select_by="group", robust=False):
    """Nested best-subset log-allometry. `select_by='group'` selects by leave-one-stand-out
    (cross-stand); `robust=True` uses a Huber final fit."""
    return _BestSubsetAllom(candidates=candidates, max_k=max_k, select_by=select_by, robust=robust)


def make_huber_allom(cols=("zmean", "zq90", AGE)):
    """Robust log-allometry — resists the young-stand leverage points."""
    return _HuberAllom(cols=cols)


def make_pls(cols=ALL_METRICS, max_k=4):
    """PLS with nested component selection (Cosenza 2021)."""
    return _PLSNested(cols=cols, max_k=max_k)


def make_loggpr(cols=FEATURES_AGE):
    """Log-target GPR with a near-linear kernel → calibrated intervals that also generalize."""
    return _LogGPR(cols=cols)


def make_elasticnet(cols=FEATURES):
    return _SkWrapper(Pipeline([
        ("sc", StandardScaler()),
        ("en", ElasticNetCV(l1_ratio=[.2, .5, .8, 1.0], cv=4, max_iter=50000)),
    ]), cols=cols)


def make_ridge(cols=FEATURES, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
    """Standardised ridge on a chosen feature set — the linear model for signed targets.

    The log-allometric family (`make_huber_allom`, `make_log_allometric`) cannot be used when the
    target can go negative, which is the case for the two-stage residual test in
    `greenvista.eval.orthogonal` (an age model's error is signed). Ridge keeps the variance down at
    N=13 while staying valid on signed targets; the penalty is chosen by inner CV, so selection
    stays inside the fold.
    """
    return _SkWrapper(Pipeline([
        ("sc", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.asarray(alphas, float))),
    ]), cols=cols)


def make_gpr(cols=FEATURES):
    return _GPR(cols=cols)


def make_rf(cols=FEATURES):
    """Random Forest — the eucalyptus literature's best performer (Pertille 2025, V.S.da Silva 2020).
    Shallow leaves + many trees to keep the variance down at N=13 (hyperparameters in config.ABA)."""
    return _SkWrapper(RandomForestRegressor(
        n_estimators=config.ABA.rf_n_estimators, min_samples_leaf=config.ABA.rf_min_samples_leaf,
        max_features=config.ABA.rf_max_features, random_state=config.ABA.seed), cols=cols)


def make_knn(cols=FEATURES):
    """k-NN imputation — the operational inventory baseline (Cosenza 2021; consistently the weakest)."""
    return _SkWrapper(Pipeline([
        ("sc", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=config.ABA.knn_neighbors, weights="distance")),
    ]), cols=cols)


def make_pca_ols(cols=ALL_METRICS):
    return _PCAReg(cols=cols)


def make_quantile_gbm(cols=FEATURES):
    """Heteroscedastic-interval model — quantile GBM; width adapts to the input."""
    return _QuantileGBM(cols=cols)


def make_baseline_lm():
    """Existing UNESP lm, on its 8 CSV columns — for fair LOPO comparison only (no interval)."""
    class _LM(_SkWrapper):
        calibrated = False  # placeholder interval — coverage is not meaningful, report as N/A

        def fit(self, X, y):
            self.pipe.fit(X[BASELINE_COLS].to_numpy(), y)
            self.resid_sd_ = 1.0
            return self

        def predict(self, X):
            return self.pipe.predict(X[BASELINE_COLS].to_numpy())

    return _LM(LinearRegression())

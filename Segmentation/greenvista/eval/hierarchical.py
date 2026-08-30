"""Bayesian hierarchical (partial-pooling) stand-volume model — the missing right tool at N=13.

At N=13 plots across 6 stands, the statistically correct instrument is a *hierarchical* log-log yield
model with a stand-level random intercept (partial pooling):

    ln Vol_ha = a + b·ln(Hdom) + c·ln(age) + u[stand] + eps
        u[stand] ~ Normal(0, sigma_stand)      (partial pooling across stands)
        eps       ~ Normal(0, sigma_resid)

It (1) borrows strength across the 6 stands instead of fitting each independently or fully pooling;
(2) yields posterior-predictive intervals; and (3) decomposes the variance into a between-stand piece
(sigma_stand^2) and a plot/residual piece (sigma_resid^2), turning the qualitative "the numbers are
carried by age, not LiDAR" finding into a posterior quantity — add ln(age) to the mean and
sigma_stand collapses.

Prediction is done in NumPy from the posterior draws (not `pm.set_data`), for explicit control over
the two regimes the CV harness needs:

  * Seen stand (leave-one-plot-out): the held-out plot's stand still has siblings in the training
    set, so that stand's posterior random intercept u[stand] is used.
  * Unseen stand (leave-one-stand-out): the whole stand is held out, so there is no observed random
    intercept. The cross-stand prediction is the population mean (u=0) for the point estimate, with
    the between-stand variance folded into the predictive interval by drawing a fresh
    u ~ Normal(0, sigma_stand) per posterior sample.

Note: 6 groups only weakly identify sigma_stand. The prior on it is a weakly-informative HalfNormal
and the group variance should be reported as weakly identified; check divergences / R-hat / ESS via
`diagnostics()`.

The estimator exposes fit(df, y) / predict(df) / predict_interval(df, alpha) on a DataFrame, so it drops
straight into `greenvista.area_based.validate._lopo` and `._group_cv` alongside `StandVolume`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _quantile_summary(x: np.ndarray) -> dict:
    """Posterior summary of a 1-D draw vector: mean/sd + equal-tailed 95% credible interval."""
    x = np.asarray(x, float)
    return {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "q2.5": float(np.percentile(x, 2.5)),
        "q97.5": float(np.percentile(x, 97.5)),
    }


class HierarchicalStandVolume:
    """Partial-pooling log-log yield model in PyMC, with a stand-level random intercept.

    ln Vol_ha ~ a + Σ b_i·ln(x_i) + u[group] + eps, with u ~ N(0, sigma_stand), eps ~ N(0, sigma_resid).

    Fits the `greenvista.area_based.validate` CV interface (fit/predict/predict_interval on a DataFrame).
    Prediction from posterior draws distinguishes seen stands (posterior u[stand]) from unseen stands
    (population mean u=0 for the point, u~N(0,sigma_stand) folded into the interval) — the LOTO
    behaviour.

    Parameters
    ----------
    cols : tuple[str]
        Predictor columns entered as ln(x). Default ("Hdom", "age"). May be empty () for an
        intercept-only + random-intercept model (used for the variance decomposition).
    group_col : str
        Column holding the stand id (the random-effect group). Default "talhao".
    draws, tune, chains, target_accept : MCMC controls.
    sigma_stand_prior, sigma_resid_prior : HalfNormal prior sigmas (weakly-informative on the log scale).
    coef_prior_sd : Normal prior sd for the log-log elasticities b (weakly-informative; O(1-3) expected).
    seed : reproducibility seed for sampling and for the predictive-interval Monte-Carlo draws.
    """

    calibrated = True  # exposes a proper predict_interval (validate.run parametric-coverage path)

    def __init__(self, cols=("Hdom", "age"), group_col="talhao", *, draws: int = 1000, tune: int = 1500,
                 chains: int = 4, target_accept: float = 0.95, sigma_stand_prior: float = 0.5,
                 sigma_resid_prior: float = 0.5, coef_prior_sd: float = 2.0, seed: int = 0,
                 progressbar: bool = False):
        self.cols = list(cols)
        self.group_col = group_col
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.target_accept = target_accept
        self.sigma_stand_prior = sigma_stand_prior
        self.sigma_resid_prior = sigma_resid_prior
        self.coef_prior_sd = coef_prior_sd
        self.seed = seed
        self.progressbar = progressbar

    # ------------------------------------------------------------------ design --
    def _design(self, X) -> np.ndarray:
        """ln of the predictor columns → (n, p) design (p may be 0)."""
        if not self.cols:
            return np.empty((len(X), 0), float)
        return np.log(np.column_stack([np.asarray(X[c], float) for c in self.cols]))

    # --------------------------------------------------------------------- fit --
    def fit(self, X, y):
        import pymc as pm  # imported here so module import never fails if PyMC is absent

        L = self._design(X)
        self.feat_mean_ = L.mean(axis=0) if L.shape[1] else np.zeros(0)
        Lc = L - self.feat_mean_                     # centre predictors → stable, well-identified a
        logy = np.log(np.asarray(y, float))
        groups = np.asarray(X[self.group_col])
        self.stands_ = list(pd.unique(groups))
        idx = np.array([self.stands_.index(g) for g in groups], dtype=int)
        n_stands = len(self.stands_)
        p = Lc.shape[1]

        with pm.Model() as model:
            a = pm.Normal("a", mu=float(logy.mean()), sigma=1.0)
            sigma_stand = pm.HalfNormal("sigma_stand", sigma=self.sigma_stand_prior)
            z = pm.Normal("z", 0.0, 1.0, shape=n_stands)          # non-centered → fewer divergences
            u = pm.Deterministic("u", z * sigma_stand)
            sigma = pm.HalfNormal("sigma", sigma=self.sigma_resid_prior)
            mu = a + u[idx]
            if p:
                b = pm.Normal("b", mu=0.0, sigma=self.coef_prior_sd, shape=p)
                mu = mu + pm.math.dot(Lc, b)
            pm.Normal("y", mu=mu, sigma=sigma, observed=logy)
            idata = pm.sample(self.draws, tune=self.tune, chains=self.chains, cores=1,
                              target_accept=self.target_accept, random_seed=self.seed,
                              progressbar=self.progressbar)

        self.model_ = model
        self.idata_ = idata
        post = idata.posterior
        # flatten (chain, draw) → S posterior samples
        self.a_ = post["a"].stack(s=("chain", "draw")).values                       # (S,)
        self.u_ = post["u"].stack(s=("chain", "draw")).values.T                     # (S, n_stands)
        self.sigma_stand_ = post["sigma_stand"].stack(s=("chain", "draw")).values   # (S,)
        self.sigma_resid_ = post["sigma"].stack(s=("chain", "draw")).values         # (S,)
        self.b_ = (post["b"].stack(s=("chain", "draw")).values.T if p
                   else np.zeros((self.a_.shape[0], 0)))                             # (S, p)
        return self

    # --------------------------------------------------------------- internals --
    def _linear(self, X) -> np.ndarray:
        """Fixed-effect linear predictor per row × posterior sample → (m, S), on the log scale."""
        Lc = self._design(X) - self.feat_mean_ if self.cols else np.empty((len(X), 0))
        lin = self.a_[None, :] + (Lc @ self.b_.T if self.b_.shape[1] else 0.0)       # (m, S)
        return np.broadcast_to(lin, (len(X), self.a_.shape[0])).copy()

    def _group_index(self, X):
        """For each row: index into stands_ if seen, else -1 (unseen stand → population mean)."""
        groups = np.asarray(X[self.group_col])
        return np.array([self.stands_.index(g) if g in self.stands_ else -1 for g in groups], dtype=int)

    # ----------------------------------------------------------------- predict --
    def predict(self, X) -> np.ndarray:
        """Point prediction = exp(posterior-mean log-volume). Seen stands use u[stand]; unseen use u=0."""
        lin = self._linear(X)                        # (m, S)
        gidx = self._group_index(X)
        mu = lin.copy()
        for j, gi in enumerate(gidx):
            if gi >= 0:
                mu[j] += self.u_[:, gi]              # seen stand → its posterior random intercept
            # unseen stand → population mean (u = 0) for the point estimate
        return np.exp(mu.mean(axis=1))

    def predict_interval(self, X, alpha: float = 0.05, n_extra: int | None = None):
        """95%(=1-alpha) posterior-predictive interval on Vol_ha, per held-out row.

        Folds in parameter uncertainty (a, b, u posteriors), residual noise (sigma_resid) and, for
        unseen stands, the between-stand spread (a fresh u ~ N(0, sigma_stand) per posterior sample),
        so the unseen-stand interval comes out wider than the seen-stand one.
        """
        lin = self._linear(X)                        # (m, S)
        gidx = self._group_index(X)
        rng = np.random.default_rng(self.seed)
        lo = np.empty(len(X)); hi = np.empty(len(X))
        for j, gi in enumerate(gidx):
            if gi >= 0:
                u_draw = self.u_[:, gi]                              # seen stand
            else:
                u_draw = rng.normal(0.0, self.sigma_stand_)         # unseen stand: population spread
            eps = rng.normal(0.0, self.sigma_resid_)                # residual (aleatoric) noise
            v = np.exp(lin[j] + u_draw + eps)                       # (S,) predictive draws of Vol_ha
            lo[j] = np.percentile(v, 100 * alpha / 2)
            hi[j] = np.percentile(v, 100 * (1 - alpha / 2))
        return lo, hi

    # ------------------------------------------------------------- diagnostics --
    def diagnostics(self) -> dict:
        """Convergence diagnostics from ArviZ: max R-hat, min bulk-ESS, #divergences."""
        import arviz as az
        var_names = [v for v in ["a", "b", "sigma_stand", "sigma", "u"] if v in self.idata_.posterior]
        s = az.summary(self.idata_, var_names=var_names)
        return {
            "max_rhat": float(s["r_hat"].max()),
            "min_ess_bulk": float(s["ess_bulk"].min()),
            "min_ess_tail": float(s["ess_tail"].min()),
            "n_divergences": int(self.idata_.sample_stats["diverging"].sum()),
            "n_samples": int(self.a_.shape[0]),
        }

    def variance_components(self) -> dict:
        """Posterior variance decomposition + coefficient posteriors.

        ICC = sigma_stand^2 / (sigma_stand^2 + sigma_resid^2) — the fraction of residual (i.e. after
        the fixed effects) variance that sits at the stand level. `coef_ln_<col>` are the log-log
        elasticities."""
        ss, sr = self.sigma_stand_, self.sigma_resid_
        icc = ss ** 2 / (ss ** 2 + sr ** 2)
        out = {
            "sigma_stand": _quantile_summary(ss),
            "sigma_resid": _quantile_summary(sr),
            "sigma_stand_sq": _quantile_summary(ss ** 2),
            "sigma_resid_sq": _quantile_summary(sr ** 2),
            "ICC": _quantile_summary(icc),
            "intercept": _quantile_summary(self.a_),
        }
        for j, c in enumerate(self.cols):
            out[f"coef_ln_{c}"] = _quantile_summary(self.b_[:, j])
        return out

    def equation(self) -> str:
        a = self.a_.mean()
        terms = " + ".join(f"{self.b_[:, j].mean():+.3f}·ln({c})" for j, c in enumerate(self.cols))
        tail = f" {terms}" if self.cols else ""
        return (f"ln Vol_ha = {a:.3f}{tail} + u[{self.group_col}]   "
                f"(sigma_stand≈{self.sigma_stand_.mean():.3f}, sigma_resid≈{self.sigma_resid_.mean():.3f})")


def make_hierarchical(cols=("Hdom", "age"), **kw):
    """Zero-arg-friendly factory for the CV harness: `lambda: make_hierarchical(...)` style."""
    return HierarchicalStandVolume(cols=cols, **kw)

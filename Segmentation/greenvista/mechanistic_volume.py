"""Mechanistic volume decomposition:  V_ha = (N_stems / area) × v̄(D̄, H̄).

Every other model in this project regresses ``Vol_ha`` directly from LiDAR/age.  This module instead
decomposes plot volume into physically-grounded factors, each supplied by LiDAR or by the company's
fixed inventory equations:

    N_stems  ← FF3D stem detection (61%, stand-biased) → calibrated to true density
    H̄, Hdom  ← LiDAR (greenvista.stand_volume.lidar_hdom; Hdom=top 50% cell-maxima, H̄=all cell-maxima)
    D̄        ← invert the Campos hypsometric relation (H from D) to get D̄ from (H̄, Hdom)
    v̄        ← the G2 volume equation (Schumacher-family) applied on (D̄, H̄)

The Campos and G2 coefficients are fixed from the São Manuel inventory report (fitted by the company
on 63 cubed trees / ~717 census trees, not on the 13 plots used here).  The only thing fit to those
13 plots is the FF3D count calibration (plus an optional multiplicative bias term), and it is fit
under the outer CV fold: the correction is learned on held-in stands and applied to the held-out one,
never fit and tested on the same stand.  The estimate therefore extrapolates to an unseen stand
through mechanistic forestry models rather than through a 13-point empirical surface.

------------------------------------------------------------------------------------------------
COEFFICIENT PROVENANCE  (all located inside ``Dados_SãoManuel/``)
------------------------------------------------------------------------------------------------
Campos hypsometric   ln(H) = b0 + b1·(1/D) + b2·ln(Hdom)      (R²_gen = 0.8165, best of 7)
    b0 = 2.6939602,  b1 = -9.9588349,  b2 = 0.3348264
    • coefficients hardcoded in the inventory pipeline:  Codigos/Cubagem.R  lines 110-118
    • model form:  Codigos/Modelos_Hipsométricos.R  ->  ``Campos <- lm(ln_H ~ inv_D + ln_H_dom)``
    • R²/ranking:  Resultados do APP GREEN VISTA- Dados São Manuel/relatorio_hipsometricos_parametros.md
                   (+ ranking_modelos_hipsometricos.csv)

G2 volume            V = c0 + c1·D² + c2·D²·H + c3·H          (R² = 0.9950, best of 4)
    • model form:  Codigos/Cubagem.R  line 72  ->  ``mod_V_G2 <- lm(V ~ D2 + D2_H + H_m)``
    • R²/RMSE:      Resultados do APP GREEN VISTA- Dados São Manuel/relatorio_volumetricos_completo.md
    • refit here from the destructive-scaling data itself (5-dados_campo/volumes.csv, N=63) —
      ``fit_g2_from_cubagem``
      reproduces R²=0.9950 / RMSE=0.0189 and matches
      Resultados…/dados_predicoes_volumetricas.csv::pred_G2 to 1.6e-4.  Refit (rather than transcribe)
      because the report publishes only R²/RMSE, not the four coefficients.

Caveat: the ground-truth ``Vol_ha`` was built by the company with the Schumacher-Hall per-tree model
summed over individual trees (Cubagem.R), whereas this decomposition applies G2 to the mean tree.
Two consequences, both quantified in the experiment: (a) G2≠Schumacher, (b) v̄(D̄) < mean_i v(D_i) by
Jensen's inequality (volume is convex in D), so the mean-tree route structurally under-predicts.
Even with exact field D̄,H̄,N it only reaches R²≈0.74 / −11.5% bias against Vol_ha; that is the ceiling
of the decomposition, which the optional multiplicative bias term corrects for.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# --- fixed inventory coefficients ---------------------------------------------------------------
CAMPOS = (2.6939602, -9.9588349, 0.3348264)          # (b0, b1, b2)  Cubagem.R L110-112
# G2 coefficients refit from volumes.csv (N=63) — see fit_g2_from_cubagem; R²=0.9950, RMSE=0.0189.
G2_DEFAULT = (-4.90367e-03, 1.40300e-04, 3.21600e-05, 3.57810e-04)   # (c0, cD2, cD2H, cH)

# Default DataFrame column names the estimator reads.
COUNT_COL = "ff3d_n_ha"        # FF3D-observed stems/ha (biased, ~61% detection)
TRUTH_COUNT_COL = "field_n_ha" # field stems/ha — the calibration TARGET (train folds only)
HDOM_COL = "Hdom"              # LiDAR dominant height  (lidar_hdom top_frac=0.5)
HMEAN_COL = "Hmean"            # LiDAR mean canopy-top height (lidar_hdom top_frac=1.0)

D_MIN_CM, D_MAX_CM = 4.0, 45.0  # plausible eucalyptus DBH clamp (matches inventory D>=4 filter)


# ================================================================================================
# Campos hypsometric relation and its inverse
# ================================================================================================
def campos_height(D, Hdom, coef=CAMPOS):
    """Forward Campos: predict individual tree height H from DBH D (cm) and plot Hdom (m)."""
    b0, b1, b2 = coef
    D = np.asarray(D, float)
    return np.exp(b0 + b1 / D + b2 * np.log(np.asarray(Hdom, float)))


def campos_invert_diameter(H, Hdom, coef=CAMPOS, clip=(D_MIN_CM, D_MAX_CM)):
    """Invert Campos to recover DBH from a (mean) height H and the plot Hdom.

    ln(H) = b0 + b1/D + b2·ln(Hdom)  ⇒  D = b1 / (ln(H) − b0 − b2·ln(Hdom)).
    Since b1<0 and the denominator is negative for real trees, D>0.  The denominator is nudged away
    from zero and D is clamped to a plausible DBH range (returns are finite by construction).
    """
    b0, b1, b2 = coef
    H = np.asarray(H, float)
    denom = np.log(H) - b0 - b2 * np.log(np.asarray(Hdom, float))
    denom = np.where(np.abs(denom) < 1e-3, -1e-3, denom)   # guard the singularity
    D = b1 / denom
    if clip is not None:
        D = np.clip(D, clip[0], clip[1])
    return D


# ================================================================================================
# G2 volume equation
# ================================================================================================
def g2_volume(D, H, coef=G2_DEFAULT):
    """G2 mean-tree volume (m³):  V = c0 + c1·D² + c2·D²·H + c3·H  (D in cm, H in m)."""
    c0, c1, c2, c3 = coef
    D = np.asarray(D, float)
    H = np.asarray(H, float)
    return c0 + c1 * D ** 2 + c2 * D ** 2 * H + c3 * H


def fit_g2_from_cubagem(volumes_csv, d_min=D_MIN_CM):
    """Refit the G2 model  V ~ D² + D²·H + H  from the destructive-scaling data (volumes.csv) and return
    (coef, r2, rmse, n).  Refit — not transcribed — because the report publishes only R²/RMSE.
    """
    import pandas as pd
    v = pd.read_csv(volumes_csv)
    v = v[v.D_cm >= d_min]
    D2 = (v.D_cm ** 2).to_numpy()
    D2H = (v.D_cm ** 2 * v.H_m).to_numpy()
    H = v.H_m.to_numpy()
    y = v.V.to_numpy()
    A = np.column_stack([np.ones(len(v)), D2, D2H, H])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    r2 = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    rmse = float(np.sqrt(((y - pred) ** 2).mean()))
    return tuple(float(c) for c in coef), r2, rmse, int(len(v))


# ================================================================================================
# FF3D stem-count calibration (fit on held-in stands only)
# ================================================================================================
class CountCalibrator:
    """Map FF3D-observed stem density → true stem density.

    FF3D detects ~61% of stems and is stand-biased (dense stands under-detect), so a single global
    ÷0.61 does not remove the bias.  Modes:
      * ``"ratio"``  – multiplicative:  N̂ = k·N_obs,  k = Σ N_true / Σ N_obs  (robust, 1 param).
      * ``"linear"`` – affine OLS:  N̂ = a + b·N_obs  (2 params; can partially undo the slope bias).
      * ``"none"``   – pass the raw FF3D density through (uncalibrated control).
    Fit uses only the rows handed to it (the outer training fold), so it never sees the held-out stand.
    """

    def __init__(self, mode="ratio"):
        self.mode = mode

    def fit(self, obs, truth):
        obs = np.asarray(obs, float)
        truth = np.asarray(truth, float)
        if self.mode == "ratio":
            self.k_ = float(truth.sum() / obs.sum())
        elif self.mode == "linear":
            A = np.column_stack([np.ones(len(obs)), obs])
            self.a_, self.b_ = np.linalg.lstsq(A, truth, rcond=None)[0]
        elif self.mode == "none":
            pass
        else:
            raise ValueError(f"unknown count-correction mode {self.mode!r}")
        return self

    def predict(self, obs):
        obs = np.asarray(obs, float)
        if self.mode == "ratio":
            return self.k_ * obs
        if self.mode == "linear":
            return self.a_ + self.b_ * obs
        return obs  # "none"


# ================================================================================================
# The mechanistic estimator (fit / predict / predict_interval — area_based.validate compatible)
# ================================================================================================
class MechanisticVolume:
    """V̂_ha = bias · N̂_stems_ha · v̄(D̄, H̄),  with N̂ calibrated and (D̄→v̄) from fixed equations.

    Fits the ``greenvista.area_based.validate`` interface (``fit(df, y)`` / ``predict(df)`` /
    ``predict_interval(df, alpha)``), so it drops straight into the LOPO / leave-one-stand-out /
    conformal / permutation harness used by every other model.

    Free parameters fit to the 13 plots (all learned on the held-in fold only):
      * the count calibration  (``count_correction``:  1 param for ratio, 2 for linear, 0 for none)
      * an optional multiplicative volume-bias term (``fit_bias``:  1 param) that absorbs the
        Jensen / G2-vs-Schumacher offset so the model competes on cross-stand *shape*, not offset.
    Everything else (Campos, G2) is fixed forestry.
    """
    calibrated = True

    def __init__(self, g2_coef=G2_DEFAULT, campos=CAMPOS, count_correction="ratio", fit_bias=True,
                 count_col=COUNT_COL, truth_count_col=TRUTH_COUNT_COL,
                 hdom_col=HDOM_COL, hmean_col=HMEAN_COL):
        self.g2_coef = tuple(g2_coef)
        self.campos = tuple(campos)
        self.count_correction = count_correction
        self.fit_bias = fit_bias
        self.count_col = count_col
        self.truth_count_col = truth_count_col
        self.hdom_col = hdom_col
        self.hmean_col = hmean_col

    # --- the mechanistic chain ------------------------------------------------------------------
    def dbar(self, X):
        """Mean DBH (cm) from inverting Campos on LiDAR (H̄, Hdom)."""
        return campos_invert_diameter(X[self.hmean_col].to_numpy(float),
                                      X[self.hdom_col].to_numpy(float), self.campos)

    def vbar(self, X):
        """Mean-tree volume (m³) = G2(D̄, H̄)."""
        return g2_volume(self.dbar(X), X[self.hmean_col].to_numpy(float), self.g2_coef)

    def count_ha(self, X):
        """Calibrated stems/ha."""
        return self.cal_.predict(X[self.count_col].to_numpy(float))

    def _raw(self, X):
        """N̂_ha · v̄ before the bias term."""
        return self.count_ha(X) * self.vbar(X)

    # --- estimator API --------------------------------------------------------------------------
    def fit(self, X, y):
        y = np.asarray(y, float)
        self.cal_ = CountCalibrator(self.count_correction).fit(
            X[self.count_col].to_numpy(float), X[self.truth_count_col].to_numpy(float))
        raw = self._raw(X)
        self.bias_ = float(y.sum() / raw.sum()) if self.fit_bias else 1.0
        pred = raw * self.bias_
        # log-scale residual sd → positive (lognormal) prediction interval
        self.resid_sd_ = float((np.log(y) - np.log(np.clip(pred, 1e-6, None))).std(ddof=1))
        return self

    def predict(self, X):
        return self._raw(X) * self.bias_

    def predict_interval(self, X, alpha=0.05):
        z = norm.ppf(1 - alpha / 2)
        lp = np.log(np.clip(self.predict(X), 1e-6, None))
        return np.exp(lp - z * self.resid_sd_), np.exp(lp + z * self.resid_sd_)

    def equation(self):
        b0, b1, b2 = self.campos
        c0, c1, c2, c3 = self.g2_coef
        cc = self.count_correction
        return (f"V_ha = {self.bias_:.3f} · N̂_ha · G2(D̄,H̄);  "
                f"D̄ = {b1:.3f}/(ln H̄ − {b0:.3f} − {b2:.3f}·ln Hdom);  "
                f"G2=({c0:.2e},{c1:.2e},{c2:.2e},{c3:.2e});  count={cc}")


def make_mechanistic(g2_coef=G2_DEFAULT, count_correction="ratio", fit_bias=True, **kw):
    """Factory matching the ``make_*`` convention in greenvista.area_based.models."""
    return MechanisticVolume(g2_coef=g2_coef, count_correction=count_correction,
                             fit_bias=fit_bias, **kw)

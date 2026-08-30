"""Error analysis for the small-N volume models.

  * `residual_table` / `residual_correlations` — where does the model err? Correlate |residual| with
    covariates (age, zmax, stand) so systematic error is visible, not just a single R².
  * `leverage` — at N=13 one plot can flip R². This reports each point's leave-one-out ΔR² so the
    high-leverage plots are named.
  * `hard_truth_from_trees` / `compare_to_hard_truth` — the plot `Vol_ha` is model-derived
    (Schumacher), so the only hard ground truth is the cubed trees. This aggregates measured tree
    volumes to plot level and scores predictions against them where plots overlap, which makes the
    semi-circularity explicit.
"""
import numpy as np
import pandas as pd

from .validation import metrics


def residual_table(y, pred, covariates: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-point residuals (signed + absolute) optionally joined to covariates for inspection."""
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    df = pd.DataFrame({"y": y, "pred": pred, "residual": pred - y, "abs_residual": np.abs(pred - y)})
    if covariates is not None:
        df = pd.concat([df.reset_index(drop=True), covariates.reset_index(drop=True)], axis=1)
    return df


def residual_correlations(y, pred, covariates: pd.DataFrame) -> dict:
    """Pearson r between |residual| and each numeric covariate — flags systematic error patterns."""
    res = np.abs(np.asarray(pred, float) - np.asarray(y, float))
    out = {}
    for c in covariates.columns:
        col = pd.to_numeric(covariates[c], errors="coerce").to_numpy(float)
        if np.all(np.isfinite(col)) and np.std(col) > 0 and np.std(res) > 0:
            out[c] = float(np.corrcoef(res, col)[0, 1])
    return out


def leverage(y, pred, ids=None) -> pd.DataFrame:
    """Per-point leave-one-out influence on R²: delta = R²_full − R²_without_i. Large positive delta
    ⇒ dropping the point lowers R² (it was propping the fit up); large negative ⇒ it was hurting.
    Sorted by |delta| so the plots that dominate the headline are named."""
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    n = len(y)
    full = metrics(y, pred)["R2"]
    ids = list(ids) if ids is not None else list(range(n))
    rows = []
    for i in range(n):
        keep = np.arange(n) != i
        r2_wo = metrics(y[keep], pred[keep])["R2"] if np.ptp(y[keep]) > 0 else np.nan
        rows.append({"id": ids[i], "y": y[i], "pred": pred[i], "residual": pred[i] - y[i],
                     "R2_without": r2_wo, "delta_R2": full - r2_wo})
    return pd.DataFrame(rows).sort_values("delta_R2", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


def hard_truth_from_trees(tree_df: pd.DataFrame, plot_col: str, vol_col: str = "V",
                          area_ha: float | None = None) -> pd.DataFrame:
    """Aggregate *measured* cubed-tree volumes to plot level. If `area_ha` is given, returns Vol_ha
    (sum / area); otherwise the raw summed volume. This is the only non-model-derived reference."""
    g = tree_df.groupby(plot_col)[vol_col].agg(sum_V="sum", n_trees="count").reset_index()
    if area_ha:
        g["Vol_ha_hard"] = g["sum_V"] / area_ha
    return g


def compare_to_hard_truth(pred_by_plot: dict, hard_df: pd.DataFrame, plot_col: str,
                          hard_col: str = "Vol_ha_hard") -> dict:
    """Score plot predictions `{plot_id: pred}` against the measured (hard) plot volumes on the
    overlapping plots. Returns metrics + the number of plots compared. The training target was the
    model-derived Vol_ha, so this is an out-of-distribution check, not the fitting objective."""
    rows = [(hard[hard_col], pred_by_plot[str(hard[plot_col])])
            for _, hard in hard_df.iterrows() if str(hard[plot_col]) in pred_by_plot]
    if not rows:
        return {"n_compared": 0}
    yh, pp = np.array([r[0] for r in rows], float), np.array([r[1] for r in rows], float)
    return {**metrics(yh, pp), "n_compared": len(rows),
            "note": "scored vs MEASURED cubed-tree volume (hard truth); training target was model-derived Vol_ha"}

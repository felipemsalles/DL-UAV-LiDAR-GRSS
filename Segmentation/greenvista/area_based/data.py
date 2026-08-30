"""Load and reconcile the 13-plot LiDAR-metric + field-Vol_ha table."""
import pandas as pd

# Replication-safe lidR .stdmetrics_z features (pzabove2 excluded — near-degenerate).
FEATURES = ["zmax", "zmean", "zsd", "zq50", "zq90", "pzabovezmean"]
# Columns the existing UNESP lm baseline uses (consumed straight from the CSV, not recomputed).
BASELINE_COLS = ["zmax", "zmean", "zsd", "zskew", "zkurt", "zentropy", "zq70", "zpcum7"]
# Stand age (years) — the strongest non-LiDAR covariate (Silva 2016: stratification 10%→5% rRMSE).
AGE = "idade_anos"
FEATURES_AGE = FEATURES + [AGE]
# Full lidR .stdmetrics_z suite present in the plot CSV (for nested PCA/AICc selection). pzabove2 excluded.
ALL_METRICS = (["zmax", "zmean", "zsd", "zskew", "zkurt", "zentropy", "pzabovezmean"]
               + [f"zq{q}" for q in range(5, 100, 5)]
               + [f"zpcum{i}" for i in range(1, 10)])

# Eucalyptus basic wood density (Mg/m3), for an indicative volume→AGB conversion.
EUC_BASIC_DENSITY = 0.50


def volume_to_agb(vol_ha, basic_density=EUC_BASIC_DENSITY, expansion=1.0):
    """Indicative stem AGB (Mg/ha) = Vol_ha (m3/ha) × basic density × biomass-expansion factor.

    Allows comparison with the AGB-reporting literature. rRMSE is invariant to this linear scaling,
    so model accuracy is unchanged — only the magnitude/units differ. `expansion`>1 adds
    branches/bark/foliage (total AGB); leave at 1.0 for stem-only.
    """
    return vol_ha * basic_density * expansion


def load_plots(plot_metric_csv, inventario_csv, tol=0.5):
    """Return the modeling table: keys + the full metric suite + age + Vol_ha (one row per plot).

    Cross-checks Vol_ha against the independent inventory summary and fails loudly on mismatch.
    """
    pm = pd.read_csv(plot_metric_csv)
    inv = pd.read_csv(inventario_csv)
    cols = sorted(set(FEATURES + BASELINE_COLS + ALL_METRICS + [AGE]) & set(pm.columns))
    keep = ["talhao", "parcela"] + cols + ["Vol_ha"]
    df = pm[keep].copy()
    # cross-check the target against the inventory summary
    m = df[["talhao", "parcela", "Vol_ha"]].merge(
        inv[["talhao", "parcela", "Vol_ha"]], on=["talhao", "parcela"], suffixes=("_pm", "_inv"))
    assert len(m) == len(df), "plot/inventory join lost rows"
    assert (m.Vol_ha_pm - m.Vol_ha_inv).abs().max() <= tol, "Vol_ha disagrees between sources"
    df["plot_id"] = df.talhao.astype(str) + "_" + df.parcela.astype(str)
    return df.reset_index(drop=True)

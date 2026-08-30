"""Synthetic-ground-truth integration harness.

Builds trees whose volume is a known function of crown size, runs the whole chain
(crown_features -> CrownAllometry -> LOOCV), and asserts it recovers the relationship. Catches
integration, leakage and scale errors that per-module smoke tests miss.
"""
import pandas as pd
from greenvista.features import crown_features
from greenvista.volume import CrownAllometry
from greenvista.eval import loocv_predict, metrics

def test_pipeline_recovers_known_volume_law(synthetic_dataset):
    rows, clouds = synthetic_dataset
    feats = [crown_features(c) for c in clouds]
    df = pd.DataFrame(feats)
    df["V"] = [r["V"] for r in rows]
    # use crown_area + H_max (both derivable from points; V_true ∝ area * height)
    X = df[["crown_area", "H_max"]].values
    y = df["V"].values
    yhat = loocv_predict(CrownAllometry, X, y)
    m = metrics(y, yhat)
    # known deterministic law (modulo cone-sampling noise) -> recovery should be strong
    assert m["R2"] > 0.85, f"integration R²={m['R2']:.3f} too low — chain broken"
    assert m["rRMSE_pct"] < 25

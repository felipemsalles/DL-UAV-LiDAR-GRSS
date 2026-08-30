"""Tests for the W&B tracking wrapper — offline so no cloud run is created during CI."""
import pandas as pd


def test_tracking_available():
    from greenvista import tracking
    assert tracking.available()          # wandb is an env dependency


def test_log_experiment_offline(tmp_path, monkeypatch):
    from greenvista import tracking
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_MODE", "offline")
    csv = tmp_path / "res.csv"
    pd.DataFrame({"model": ["a", "b"], "LOTO_R2": [0.7, 0.3]}).to_csv(csv, index=False)
    # should run to completion without raising, in offline mode (no network)
    url = tracking.log_experiment("unit-test", config={"n": 13}, metrics={"LOTO_R2": 0.7},
                                  table_csv=str(csv), mode="offline", tags=["test"])
    assert url is None or isinstance(url, str)

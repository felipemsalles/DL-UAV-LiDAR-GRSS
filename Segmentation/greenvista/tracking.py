"""Weights & Biases experiment tracking (online by default).

Thin wrapper so every experiment persists to W&B with: config, summary metrics, a results table, and
the result file(s) as artifacts. Complements (does not replace) the local JSON manifests — W&B is the
shareable, off-machine record. Set WANDB_MODE=offline to log locally without pushing.
"""
import os
from pathlib import Path

PROJECT = os.environ.get("WANDB_PROJECT", "greenvista-eucalyptus-volume")


def available() -> bool:
    try:
        import wandb  # noqa: F401
        return True
    except ImportError:
        return False


def log_experiment(name, config=None, metrics=None, table_csv=None, artifacts=None,
                   group=None, project=PROJECT, mode=None, notes=None, tags=None):
    """Log one experiment as a W&B run and return its URL (None if wandb is unavailable).

    name       – run name (e.g. "lidar-only-baseline")
    config     – dict of hyperparameters / settings
    metrics    – dict of headline scalars → run.summary
    table_csv  – path to a results CSV → logged as a wandb.Table (the full model×metric grid)
    artifacts  – iterable of file paths (CSVs, figures, maps) → logged as W&B Artifacts
    group/tags – W&B grouping + tags for filtering the runs
    mode       – "online" (default via login) / "offline"; falls back to env WANDB_MODE
    """
    if not available():
        return None
    import wandb
    import pandas as pd

    run = wandb.init(project=project, name=name, group=group, tags=tags, notes=notes,
                     config=config or {}, mode=mode or os.environ.get("WANDB_MODE", "online"),
                     reinit=True, settings=wandb.Settings(silent=True))
    try:
        for k, v in (metrics or {}).items():
            run.summary[k] = v
        if table_csv and Path(table_csv).exists():
            run.log({"results": wandb.Table(dataframe=pd.read_csv(table_csv))})
        for a in (artifacts or []):
            p = Path(a)
            if p.exists():
                art = wandb.Artifact(p.stem.replace(" ", "_"), type="result")
                art.add_file(str(p))
                run.log_artifact(art)
        return run.url
    finally:
        run.finish()

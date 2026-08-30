"""Load the per-tree volume ground truth (volumes.csv)."""
from pathlib import Path
import pandas as pd
from greenvista import config

REQUIRED = {"arv", "talhao", "V"}

def load_volumes(path: Path | str | None = None) -> pd.DataFrame:
    """Return cubed-tree volumes. Columns include arv, talhao, V (m³, likely over-bark), D_cm, H_m."""
    path = Path(path) if path is not None else config.VOLUMES_CSV
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"volumes file missing columns: {missing}")
    return df

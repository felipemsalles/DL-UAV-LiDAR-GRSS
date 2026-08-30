"""Load geolocated cubed-tree points and link them to volumes.

Only ~23 of the 26 geoloc points map to a cubed volume: the match assumes Name == the `arv` id of
the destructive-scaling table, and geoloc Names 29/30/29_A are absent from that table while arv
9/15/17/26 were never geolocated.
"""
from pathlib import Path
import pandas as pd
from greenvista import config

def load_geoloc(path: Path | str | None = None) -> pd.DataFrame:
    """Return geoloc points with a numeric `arv` derived from `Name` (non-numeric Names dropped)."""
    path = Path(path) if path is not None else config.GEOLOC_CSV
    df = pd.read_csv(path)
    df = df[df["Name"].astype(str).str.fullmatch(r"\d+")].copy()
    df["arv"] = df["Name"].astype(int)
    return df[["arv", "Easting", "Northing", "Elevation"]].reset_index(drop=True)

def link_geoloc_volumes(geoloc: pd.DataFrame, volumes: pd.DataFrame, talhao: int = 1) -> pd.DataFrame:
    """Inner-join geoloc points to cubed volumes on arv (within one stand). Returns the usable set."""
    v = volumes[volumes["talhao"] == talhao]
    linked = geoloc.merge(v, on="arv", how="inner")
    return linked.reset_index(drop=True)

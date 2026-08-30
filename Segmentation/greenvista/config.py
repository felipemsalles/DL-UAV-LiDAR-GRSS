"""Project paths and constants.

All paths resolve relative to the repo or the system temp dir — nothing is tied to a
specific machine. Staging/output locations can be overridden with environment variables
(GREENVISTA_*), so the project runs wherever it is checked out without code edits.
"""
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _env_path(var: str, default) -> Path:
    """Return Path from env var `var` if set, else `default` (as a Path)."""
    val = os.environ.get(var)
    return Path(val) if val else Path(default)


REPO = Path(__file__).resolve().parents[1]
# The São Manuel field data and point clouds are not shipped with the repository: they are
# covered by a confidentiality agreement. Point GREENVISTA_DATA_DIR at the local copy, or
# leave the folder next to the repository. Without it, the tests that depend on real data
# are skipped and the rest of the suite still passes.
DATA = _env_path("GREENVISTA_DATA_DIR", REPO.parent / "Dados_SãoManuel")

# Field / ground truth
VOLUMES_CSV = DATA / "5-dados_campo" / "volumes.csv"
CUBAGEM_CSV = DATA / "5-dados_campo" / "Cubagem_SaoManuel.csv"
GEOLOC_CSV = DATA / "2-shapes" / "Cubagem_T001" / "arvores_cubadas.csv"

# LiDAR
LIDAR_ZIP = DATA / "Talhões LiDAR" / "SaoManuael_LIDAR.zip"
DELIVERED_TALHOES = ("001", "002", "004", "005", "006", "007")

CRS_EPSG = 31982  # SIRGAS 2000 / UTM zone 22S
CRS = f"EPSG:{CRS_EPSG}"  # string form used by geopandas
GROUND_NOISE_CLASSES = (2, 18)  # everything else = canopy/veg (no wood/leaf split delivered)

# FF3D semantic codes (panoptic PLY output)
SEM_GROUND, SEM_WOOD, SEM_LEAF = 0, 1, 2

# --- Canonical geometry / windowing constants (single source of truth) ------------------------
PLOT_RADIUS_M = 12.0          # lidR clip_roi / plot_metrics radius (metricas_LiDAR.R)
Z_MIN_M = 2.0                 # lidR height window lower bound (excludes ground/understory)
Z_MAX_M = 45.0                # lidR height window upper bound
PLOT_AREA_HA = math.pi * PLOT_RADIUS_M ** 2 / 1e4   # 12 m radius → 0.04524 ha
GROUP_COL = "talhao"          # stand id — the group for leave-one-stand-out CV


@dataclass(frozen=True)
class ABAConfig:
    """Area-based experiment hyperparameters — declarative, so a run is a config and not literals."""
    rf_n_estimators: int = 500
    rf_min_samples_leaf: int = 2
    rf_max_features: float = 0.6
    knn_neighbors: int = 3
    map_cell_m: float = 20.0
    map_min_pts: int = 20
    seed: int = 0


@dataclass(frozen=True)
class CNNConfig:
    """CNN experiment hyperparameters — used by image_cnn.train / spikes."""
    raster_size: int = 128
    n_aug: int = 64
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    seeds: tuple = (0, 1, 2, 3, 4)   # multi-seed reporting
    tta: int = 8                     # test-time-augmentation rotations


ABA = ABAConfig()
CNN = CNNConfig()

# --- Working / staging dirs for intermediate artifacts ---------------------------------
# These hold things produced *outside* the repo during a run (unzipped LiDAR tiles, FF3D
# output PLYs) plus generated outputs. Defaults live under the system temp dir / the repo
# and every one is overridable via an environment variable — no machine-specific literals.
_TMP = Path(tempfile.gettempdir())
LAZ_DIR = _env_path("GREENVISTA_LAZ_DIR", _TMP / "lazall")          # unzipped per-stand clouds
FF3D_OUT_DIR = _env_path("GREENVISTA_FF3D_OUT_DIR", _TMP / "ff3d_out")  # FF3D panoptic PLYs
OUT_DIR = _env_path("GREENVISTA_OUT_DIR", REPO / "manual_match")    # generated viz/maps/csvs
# One-off deliverables that leave the project (Portuguese-language figures, a shapefile for
# third parties). They live under figs_en/ because that is where deliverables are looked for,
# and in a subfolder because they are not numbered paper figures. Script-regenerable, so the
# folder is git-ignored.
ENTREGA_DIR = _env_path("GREENVISTA_ENTREGA_DIR", REPO / "figs_en" / "entrega_cristiano")

# Terrestrial point cloud of stand 001, supplied by the provider of the stem map.
# Prefers .laz and falls back to .las: the cloud was delivered as an 872 MB LAS and the equivalent
# LAZ is 448 MB, lossless (checked coordinate by coordinate). Hard-coding the extension would
# force keeping both copies on disk.
# The file is kept out of git: 448 MB blows GitHub's LFS quota and the data is under NDA.
# Anyone cloning the repository has to request the file.
def _tls_padrao():
    base = DATA / "TLS_Cristiano" / "FLORESTA_SLAM"
    for ext in (".laz", ".las"):
        if (base.with_suffix(ext)).exists():
            return base.with_suffix(ext)
    return base.with_suffix(".laz")


TLS_LAS = _env_path("GREENVISTA_TLS_LAS", _tls_padrao())

# Canonical single-tile artifacts the spikes consume (stand 001).
TILE_LAZ = _env_path("GREENVISTA_TILE_LAZ", LAZ_DIR / "SaoManuelTotal_001.laz")
FF3D_TILE_PLY = _env_path("GREENVISTA_FF3D_PLY", FF3D_OUT_DIR / "SaoManuel_T001_tile_round2.ply")

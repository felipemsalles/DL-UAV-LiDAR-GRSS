"""Plot-raster dataset: cached base rasters, train-fold-only augmentation, and LOPO splits.

Augmentation is label-preserving (a nadir circular canopy raster's plot-mean Vol_ha/DAP is invariant
to rotation/flip) and is applied only inside the training fold. A train dataset is constructed from
the train plot_ids alone, so a held-out plot — and all of its augmented views — cannot leak into
training. `test_image_cnn.py` enforces both properties.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.ndimage import rotate as nd_rotate

from greenvista import config
from greenvista.features.structure import STRUCTURE_KEYS
from . import footprint, render

SCALAR_KEYS = ("zmax", "zmean", "zsd")      # absolute-height side-features (hybrid input)
# Extended side-features: height + structural proxies. Opt-in — use with PlotCNN(n_scalars=7).
SCALAR_KEYS_EXT = SCALAR_KEYS + STRUCTURE_KEYS
TARGET_KEYS = ("Vol_ha", "D_cm")            # multi-task: volume + DAP


# --- circular mask (rotation can bleed interpolation into the corners; re-zero outside) --------
def _circle_mask(size):
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    return (xx - c) ** 2 + (yy - c) ** 2 <= (size / 2.0) ** 2


# --- real-data builders (unit tests use in-memory fake samples instead) ------------------------
def build_plot_table():
    """Merge the 13-plot metric table with parcelas.shp centres → keys + x,y + Vol_ha + D_cm."""
    import geopandas as gpd
    import warnings
    pm = pd.read_csv(config.DATA / "4-resultados/SaoManuelTotal_08022025_plot_metric.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parc = gpd.read_file(config.DATA / "2-shapes/Parcelas/parcelas.shp")
    parc["x"], parc["y"] = parc.geometry.x, parc.geometry.y
    for c in ("talhao", "parcela"):
        parc[c] = pd.to_numeric(parc[c], errors="coerce").astype("Int64")
        pm[c] = pd.to_numeric(pm[c], errors="coerce").astype("Int64")
    df = pm.merge(parc[["talhao", "parcela", "x", "y"]], on=["talhao", "parcela"], how="inner")
    df["plot_id"] = df.talhao.astype(str) + "_" + df.parcela.astype(str)
    keep = ["plot_id", "talhao", "parcela", "x", "y", *SCALAR_KEYS, *TARGET_KEYS]
    return df[keep].sort_values(["talhao", "parcela"]).reset_index(drop=True)


def render_all_plots(df, radius=footprint.DEFAULT_RADIUS_M, size=128, scalar_keys=SCALAR_KEYS,
                     with_rumple=False):
    """Render every plot once → {plot_id: {raster (C,size,size), scalars (S,), labels (T,)}}.

    Clouds are read once per stand and reused. Uses the rendered scalars (footprint-consistent),
    falling back to the CSV metrics only if a footprint comes back empty. Pass
    `scalar_keys=SCALAR_KEYS_EXT` to feed the structural proxies to the CNN scalar branch;
    `with_rumple=True` adds the 4th slope channel.
    """
    import laspy
    samples, cloud = {}, {}
    for _, r in df.iterrows():
        t = int(r.talhao)
        if t not in cloud:
            las = laspy.read(str(config.LAZ_DIR / f"SaoManuelTotal_{t:03d}.laz"))
            cloud[t] = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
        pts = footprint.extract_footprint(cloud[t], (r.x, r.y), radius)
        raster, scal = render.to_raster(pts, (r.x, r.y), radius, size=size, with_rumple=with_rumple)
        # CSV fallback only exists for the base height metrics; structural keys fall back to 0.0
        scal_arr = np.array([scal[k] if scal[k] > 0 else float(r[k]) if k in r else scal[k]
                             for k in scalar_keys], np.float32)
        samples[r.plot_id] = {
            "raster": raster.astype(np.float32),
            "scalars": scal_arr,
            "labels": np.array([float(r[k]) for k in TARGET_KEYS], np.float32),
        }
    return samples


# --- LOPO --------------------------------------------------------------------------------------
def lopo_splits(plot_ids):
    """Yield (test_id, train_ids) for each plot — one whole plot held out per fold."""
    ids = list(plot_ids)
    for i, test_id in enumerate(ids):
        yield test_id, [p for p in ids if p != test_id]


# --- dataset -----------------------------------------------------------------------------------
class PlotRasterDataset(Dataset):
    """Rasters + scalars + labels for a set of plot_ids; augments (rotation/flip) only when train.

    Augmentation is applied on the fly, per access, with a fold-seeded generator, so the net sees a
    fresh rotation each epoch rather than a fixed set of views baked at construction time. `n_aug`
    sets the per-plot virtual multiplier (epoch size); eval (`train=False`) returns the base raster
    verbatim. Reproducible run-to-run because the generator is seeded once; num_workers must stay 0
    (the default).
    """

    def __init__(self, samples, plot_ids, train=False, n_aug=16, seed=0):
        self.train = train
        self.plot_ids = list(plot_ids)
        self.n_aug = n_aug if train else 1
        size = next(iter(samples.values()))["raster"].shape[1]
        self._mask = _circle_mask(size)
        self._rng = np.random.default_rng(seed)   # advances per augmented access → fresh views/epoch
        self._base = [(np.ascontiguousarray(samples[p]["raster"], np.float32),
                       np.asarray(samples[p]["scalars"], np.float32),
                       np.asarray(samples[p]["labels"], np.float32), p)
                      for p in self.plot_ids]

    def _augment(self, raster):
        out = nd_rotate(raster, float(self._rng.uniform(0, 360)), axes=(1, 2), reshape=False,
                        order=1, mode="constant", cval=0.0)
        if self._rng.random() < 0.5:
            out = out[:, ::-1, :]
        if self._rng.random() < 0.5:
            out = out[:, :, ::-1]
        out = np.where(self._mask, out, 0.0)
        return np.ascontiguousarray(out, dtype=np.float32)

    def __len__(self):
        return len(self._base) * self.n_aug

    def __getitem__(self, i):
        raster, scalars, labels, pid = self._base[i // self.n_aug]
        r = self._augment(raster) if self.train else raster
        return (
            torch.from_numpy(r),
            torch.from_numpy(scalars.astype(np.float32)),
            torch.from_numpy(labels.astype(np.float32)),
            pid,
        )

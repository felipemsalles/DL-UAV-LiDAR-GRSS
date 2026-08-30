"""Self-supervised pretraining on the project's own wall-to-wall clouds.

Small-N deep-learning lever from Seely 2026: the pretext task regresses ALS metrics from unlabelled
clouds and the backbone is then fine-tuned (~4-pt rRMSE gain at 20–50 labelled plots). Used instead
of the synthetic pretraining, which did not survive the sim-to-real gap:

  pretext: render thousands of unlabelled 12 m windows sampled across the 6 talhões; the CNN predicts
  each window's held-out canopy metrics (quantiles + shape, not the zmax/zmean/zsd the scalar branch
  already sees, so the conv backbone has to learn structure). The conv backbone then transfers into
  the labelled fine-tune (run_lopo init_state).

Windows within `exclude_radius` of any of the 13 field plots are dropped, so the backbone never sees
a plot it will later be tested on, and no Vol_ha label is used in pretraining.
"""
import numpy as np
import laspy
from scipy.spatial import cKDTree

from greenvista import config
from . import render, footprint, dataset as _ds

# Pretext targets: quantiles + shape (exclude zmax/zmean/zsd — those are the scalar inputs).
PRETEXT_METRICS = ["zq10", "zq25", "zq50", "zq75", "zq90", "zq95", "zskew", "zkurt", "pzabovezmean"]


def _metric_vector(z):
    z = np.asarray(z, float)
    m, sd = z.mean(), z.std(ddof=1) if len(z) > 1 else 1.0
    sd = sd if sd > 1e-6 else 1.0
    return np.array([
        np.quantile(z, 0.10), np.quantile(z, 0.25), np.quantile(z, 0.50),
        np.quantile(z, 0.75), np.quantile(z, 0.90), np.quantile(z, 0.95),
        float(((z - m) ** 3).mean() / sd ** 3),         # skew
        float(((z - m) ** 4).mean() / sd ** 4),         # kurt
        100.0 * float((z > m).mean()),                  # pzabovezmean
    ], np.float32)


def build_ssl_samples(n_windows=3000, radius=footprint.DEFAULT_RADIUS_M, size=128,
                      min_pts=1500, exclude_radius=24.0, zmin=2.0, zmax=45.0, seed=0):
    """Sample `n_windows` unlabelled windows across the 6 clouds → {id: {raster, scalars, labels}}.

    labels = the PRETEXT_METRICS vector (the self-supervised target). Windows near the 13 field plots
    are excluded.
    """
    rng = np.random.default_rng(seed)
    centers = _ds.build_plot_table()[["x", "y"]].to_numpy()    # plots to avoid
    per = max(1, n_windows // len(config.DELIVERED_TALHOES))
    out = {}
    for t in config.DELIVERED_TALHOES:
        las = laspy.read(str(config.LAZ_DIR / f"SaoManuelTotal_{int(t):03d}.laz"))
        xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
        tree = cKDTree(xyz[:, :2])
        x0, y0, x1, y1 = xyz[:, 0].min(), xyz[:, 1].min(), xyz[:, 0].max(), xyz[:, 1].max()
        made, tries = 0, 0
        while made < per and tries < per * 40:
            tries += 1
            cx = rng.uniform(x0 + radius, x1 - radius); cy = rng.uniform(y0 + radius, y1 - radius)
            if centers.size and np.min((centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2) < exclude_radius ** 2:
                continue
            idx = tree.query_ball_point([cx, cy], radius)
            if len(idx) < min_pts:
                continue
            pts = xyz[idx]; z = pts[:, 2]
            inwin = (z >= zmin) & (z <= zmax)
            if inwin.sum() < min_pts:
                continue
            raster, scal = render.to_raster(pts, (cx, cy), radius, size=size, zmin=zmin, zmax=zmax)
            out[f"w_{t}_{made}"] = {
                "raster": raster.astype(np.float32),
                "scalars": np.array([scal["zmax"], scal["zmean"], scal["zsd"]], np.float32),
                "labels": _metric_vector(z[inwin]),
            }
            made += 1
    return out


def pretrain_backbone(ssl_samples, n_aug=1, epochs=20, lr=1e-3, batch=64, seed=0, dev=None):
    """Pretrain a PlotCNN to predict PRETEXT_METRICS from rasters. Returns a CPU state_dict whose conv
    backbone is loaded (shape-compatibly) by run_lopo(init_state=...) for the labelled fine-tune."""
    import torch
    from torch.utils.data import DataLoader
    from greenvista.repro import seed_everything
    from .train import _Norm, device
    from .model import PlotCNN
    dev = dev or device()
    ids = list(ssl_samples)
    n_metrics = len(ssl_samples[ids[0]]["labels"])
    nrm = _Norm().fit(ssl_samples, ids)
    ds = _ds.PlotRasterDataset(ssl_samples, ids, train=(n_aug > 1), n_aug=n_aug, seed=seed)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    seed_everything(seed)   # pins python/numpy/torch + cuDNN determinism
    net = PlotCNN(n_targets=n_metrics).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = torch.nn.MSELoss()
    net.train()
    for _ in range(epochs):
        for r, s, y, _pid in dl:
            r, s, y = r.to(dev), s.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossf(net(nrm.nr(r), nrm.ns(s)), nrm.ny(y))
            loss.backward(); opt.step()
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}

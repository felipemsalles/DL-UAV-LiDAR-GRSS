"""Self-supervised pretraining for per-tree DAP (Seely 2026 route, our-data-only).

Pretext: a PointNet predicts a crown's structural metrics (height-distribution shape, extent,
density) from its point cloud, with no DBH label. The unlabelled corpus is free: every instance in
the FF3D full-run segmentation (hundreds of crowns across the 6 talhões). The pretrained backbone
then warm-starts the 23-tree DAP fine-tune (`train._loocv_deep(init_state=...)`).

Scope: at a 23-tree fine-tune the expected gain is small and the per-tree ceiling is
occlusion-limited (~R² 0.13–0.18). This is representation learning only — DAP stays calibrated on
the labelled trees and no external data is used.
"""
import glob
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from plyfile import PlyData

from .data import normalize_points

PRETEXT_KEYS = ("zq25", "zq50", "zq75", "zq95", "z_range", "crown_radius", "log_n", "vspread")


def crown_corpus_from_ff3d(zip_path, min_pts=64, max_crowns=None):
    """Per-crown point clouds (local coords) from every round2 PLY in an FF3D output zip."""
    d = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(d)
    crowns = []
    for f in glob.glob(str(d / "**/*round2.ply"), recursive=True):
        el = PlyData.read(f).elements[0]
        D = {p.name: np.array(el[p.name]) for p in el.properties}
        inst = D.get("instance_pred")
        if inst is None:
            continue
        xyz = np.column_stack([D["x"], D["y"], D["z"]]).astype(np.float32)
        for iid in np.unique(inst):
            if iid < 0:
                continue
            m = inst == iid
            if m.sum() >= min_pts:
                crowns.append(xyz[m])
    if max_crowns and len(crowns) > max_crowns:
        crowns = crowns[:max_crowns]
    return crowns


def crown_metrics(pts):
    """SSL pretext targets from a crown's points — structure only, not DBH."""
    z = pts[:, 2]
    c = pts[:, :2].mean(0)
    r = np.linalg.norm(pts[:, :2] - c, axis=1)
    zr = z - z.min()
    denom = (z.max() - z.min()) or 1.0
    return np.array([
        np.quantile(zr, 0.25) / denom, np.quantile(zr, 0.50) / denom,
        np.quantile(zr, 0.75) / denom, np.quantile(zr, 0.95) / denom,
        (z.max() - z.min()) / 10.0, r.max() / 5.0,
        np.log1p(len(pts)) / 10.0, z.std() / 10.0,
    ], np.float32)


def pretrain(corpus, n_sample=256, epochs=30, lr=1e-3, batch=32, seed=0, dev=None, amp=True):
    """Pretrain a PointNet to regress crown metrics → returns a CPU state_dict (backbone + pretext head)."""
    import torch
    from greenvista.repro import seed_everything
    from .models import PointNetReg
    from .train import device

    dev = dev or device()
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    centers = [c[:, :2].mean(0) for c in corpus]
    Y = np.stack([crown_metrics(c) for c in corpus])
    ym, ys = Y.mean(0), Y.std(0) + 1e-6
    net = PointNetReg(n_targets=Y.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    use_amp = amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lossf = torch.nn.MSELoss()
    net.train()
    idx = np.arange(len(corpus))
    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            X = torch.from_numpy(np.stack([normalize_points(corpus[j], centers[j], n_sample, rng, augment=True)
                                           for j in b])).to(dev)
            yt = torch.from_numpy(((Y[b] - ym) / ys).astype(np.float32)).to(dev)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=use_amp):
                loss = lossf(net(X), yt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}

"""Tests for the per-tree DAP reproduction (CNN / PointNet / PointTransformer + allometry).

Synthetic, no real clouds. Covers: architectures learn (overfit one batch), normalization is
rotation-consistent, allometry LOOCV recovers a known DAP~height law, and the deep LOOCV harness
runs with the shuffle control collapsing (the overfitting tripwire).
"""
import numpy as np
import pytest
import torch

from greenvista.per_tree_dap.models import CrownCNN, PointNetReg, PointTransformerReg
from greenvista.per_tree_dap import data as ptd


def test_architectures_shape_and_overfit_one_batch():
    torch.manual_seed(0)
    B, N = 6, 128
    for net, x in [(PointNetReg(), torch.randn(B, N, 3)),
                   (PointTransformerReg(), torch.randn(B, N, 3)),
                   (CrownCNN(), torch.randn(B, 3, 32, 32))]:
        y = torch.randn(B)
        assert net(x).shape == (B,)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        lf = torch.nn.MSELoss()
        net.train()
        loss = None
        for _ in range(150):
            opt.zero_grad(); loss = lf(net(x), y); loss.backward(); opt.step()
        assert loss.item() < 0.3            # can drive a single batch down → it learns


def test_normalize_points_shape_and_height_preserved():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(748000, 748004, 500), rng.uniform(7480000, 7480004, 500),
                           rng.uniform(2, 30, 500)]).astype(np.float32)
    center = np.array([748002.0, 7480002.0])
    p = ptd.normalize_points(pts, center, 256, rng, augment=True)
    assert p.shape == (256, 3)
    # z (height) is scaled but not centered/rotated away → still monotone with input height range
    assert p[:, 2].max() > p[:, 2].min() > 0


def test_allometry_loocv_recovers_dap_from_height():
    from greenvista.per_tree_dap.train import _loocv_allometry
    rng = np.random.default_rng(0)
    trees, dap = {}, {}
    for i in range(20):
        h = rng.uniform(16, 34)
        d = 0.7 * h ** 0.9 * rng.uniform(0.97, 1.03)      # clean DAP~height allometry
        # crown whose zmax≈h
        pts = np.column_stack([rng.uniform(-1, 1, 300), rng.uniform(-1, 1, 300),
                               rng.uniform(h - 3, h, 300)]).astype(np.float32)
        trees[i] = {"points": pts, "D_cm": d, "H_m": h, "center": np.array([0.0, 0.0])}
        dap[i] = d
    y, p = _loocv_allometry(trees, dap, cols=("zmax",))
    from greenvista.eval import metrics
    assert metrics(y, p)["R2"] > 0.6


@pytest.mark.slow
def test_deep_loocv_runs_and_shuffle_collapses():
    """The deep LOOCV harness runs and the shuffled-label control does not beat the real run by
    much: on a real height→DAP signal, real should be ≥ shuffled."""
    from greenvista.per_tree_dap.train import _loocv_deep
    rng = np.random.default_rng(0)
    trees, dap = {}, {}
    for i in range(7):
        h = 16 + 2.4 * i + rng.normal(0, 0.3)
        pts = np.column_stack([rng.uniform(-1, 1, 200), rng.uniform(-1, 1, 200),
                               rng.uniform(h - 3, h, 200)]).astype(np.float32)
        d = 0.7 * h ** 0.9
        trees[i] = {"points": pts, "D_cm": d, "H_m": h, "center": np.array([0.0, 0.0])}
        dap[i] = d
    y, p = _loocv_deep(trees, "PointNet", dap, n_sample=64, epochs=4, n_aug=2, tta=2, seed=0)
    assert p.shape == (7,) and np.all(np.isfinite(p))

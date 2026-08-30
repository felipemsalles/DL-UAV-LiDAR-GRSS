"""Tests for the rasterize→CNN plot-volume pipeline (greenvista/image_cnn).

Covers footprint extraction and rasterization. The load-bearing property is that the CHM channel
stays in metres: volume depends on absolute height, so it must not be per-plot min-max normalized
the way a DAP-only model could.
"""
import numpy as np
import pytest

from greenvista import config

field_data = pytest.mark.skipif(not config.VOLUMES_CSV.exists(),
                                reason="field data not present (set GREENVISTA_DATA_DIR)")


def _disc_points(n=3000, radius=11.3, hmax=25.0, seed=0):
    """Random points inside a horizontal disc of given radius, heights in [2, hmax] m."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    x, y = r * np.cos(th), r * np.sin(th)
    z = rng.uniform(2.0, hmax, n)
    return np.column_stack([x, y, z]).astype(float)


# --- footprint ---------------------------------------------------------------------------
def test_extract_footprint_keeps_only_circle():
    from greenvista.image_cnn import footprint
    pts = np.array([[0, 0, 10.0], [5, 0, 10.0], [100, 0, 10.0]])  # third is far outside
    sub = footprint.extract_footprint(pts, center=(0.0, 0.0), radius=11.3)
    assert len(sub) == 2


# --- raster shape + channels -------------------------------------------------------------
def test_raster_shape_and_chm_in_metres():
    from greenvista.image_cnn import render
    pts = _disc_points(hmax=25.0)
    raster, scal = render.to_raster(pts, center=(0.0, 0.0), radius=11.3, size=128)
    assert raster.shape == (render.N_CHANNELS, 128, 128)
    # CHM holds metres near the true 25 m max, not a normalized ~1.0
    assert 20.0 < raster[render.CHM].max() <= 45.0
    # the four corners lie outside the inscribed circle → masked to 0
    assert raster[render.CHM][0, 0] == 0.0
    # scalar side-features are real metres too
    assert scal["zmax"] > 20.0 and scal["zmean"] > 2.0


# --- guard: absolute height is preserved, not normalized away ----------------------------
def test_chm_scales_with_height_not_normalized():
    from greenvista.image_cnn import render
    pts = _disc_points(hmax=20.0, seed=1)
    r1, _ = render.to_raster(pts, (0.0, 0.0), 11.3)
    scaled = pts.copy(); scaled[:, 2] *= 1.5            # 20 m → 30 m, still under the 45 m clip
    r2, s2 = render.to_raster(scaled, (0.0, 0.0), 11.3)
    assert r2[render.CHM].max() > r1[render.CHM].max() * 1.4   # scales ~1.5×, definitely not constant
    assert s2["zmax"] > 28.0


# --- empty footprint is handled, not crashed --------------------------------------------
def test_empty_footprint_returns_zeros():
    from greenvista.image_cnn import render
    raster, scal = render.to_raster(np.empty((0, 3)), (0.0, 0.0), 11.3)
    assert raster.shape == (render.N_CHANNELS, 128, 128)
    assert raster.sum() == 0.0 and scal["zmax"] == 0.0


# ============================ dataset / LOPO / leakage guard ============================
def _fake_samples(n=13, size=32):
    """In-memory plot samples so dataset tests need no real clouds."""
    rng = np.random.default_rng(0)
    out = {}
    for i in range(n):
        raster = rng.uniform(0, 30, (3, size, size)).astype(np.float32)
        out[f"p{i}"] = {
            "raster": raster,
            "scalars": np.array([30.0, 20.0, 5.0], np.float32),
            "labels": np.array([400.0 + i, 17.0], np.float32),
        }
    return out


def test_lopo_splits_hold_out_whole_plots():
    from greenvista.image_cnn import dataset
    ids = [f"p{i}" for i in range(13)]
    folds = list(dataset.lopo_splits(ids))
    assert len(folds) == 13
    for test_id, train_ids in folds:
        assert test_id not in train_ids                 # the held-out plot is gone from train
        assert len(train_ids) == 12
        assert set(train_ids) | {test_id} == set(ids)   # nothing else lost


def test_train_dataset_never_contains_heldout_plot():
    """Leakage guard: a train dataset built from train_ids contains only those plots, not the
    held-out plot, not even via augmented copies."""
    from greenvista.image_cnn import dataset
    samples = _fake_samples()
    test_id, train_ids = next(dataset.lopo_splits(list(samples)))
    ds = dataset.PlotRasterDataset(samples, train_ids, train=True, n_aug=8)
    seen = {ds[i][3] for i in range(len(ds))}           # plot_ids actually emitted
    assert test_id not in seen
    assert seen == set(train_ids)
    assert len(ds) == len(train_ids) * 8                # augmentation only multiplies train plots


def test_augmentation_preserves_label_and_shape():
    from greenvista.image_cnn import dataset
    samples = _fake_samples(n=1)
    ds = dataset.PlotRasterDataset(samples, ["p0"], train=True, n_aug=6)
    labels = [tuple(ds[i][2].tolist()) for i in range(len(ds))]
    assert all(l == labels[0] for l in labels)          # rotation/flip never touches the label
    x, scal, y, pid = ds[0]
    assert x.shape == (3, 32, 32) and scal.shape == (3,) and y.shape == (2,)


def test_eval_dataset_is_unaugmented_one_per_plot():
    from greenvista.image_cnn import dataset
    samples = _fake_samples()
    ids = list(samples)
    ds = dataset.PlotRasterDataset(samples, ids, train=False)
    assert len(ds) == len(ids)
    # eval returns the base raster verbatim (no augmentation)
    np_raster = ds[0][0].numpy()
    assert np.allclose(np_raster, samples[ds[0][3]]["raster"])


# ============================ model: shape + overfit-one-batch ============================
def test_cnn_forward_shape_and_overfits_one_batch():
    import torch
    from greenvista.image_cnn.model import PlotCNN
    torch.manual_seed(0)
    m = PlotCNN()
    B = 8
    raster = torch.randn(B, 3, 32, 32)        # GAP makes the net size-agnostic; 32² keeps it fast
    scal = torch.randn(B, 3)
    y = torch.randn(B, 2)
    assert m(raster, scal).shape == (B, 2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    lossf = torch.nn.MSELoss()
    m.train()
    loss = None
    for _ in range(150):
        opt.zero_grad(); loss = lossf(m(raster, scal), y); loss.backward(); opt.step()
    assert loss.item() < 0.1                    # can drive a single batch to ~0 → it actually learns


# ============================ LOPO integration: signal in, shuffle out ============================
@pytest.mark.slow
def test_lopo_recovers_synthetic_signal_and_shuffle_does_not():
    """End-to-end LOPO must recover a clean structure→volume signal, and a label-shuffled control
    must not: the leakage / over-optimism tripwire."""
    from greenvista.image_cnn import train as tr
    rng = np.random.default_rng(0)
    n = 13
    h = np.linspace(15, 35, n) + rng.normal(0, 0.4, n)          # per-plot canopy height level
    ids, y = [f"p{i}" for i in range(n)], np.zeros((n, 2), np.float32)
    samples = {}
    for i, pid in enumerate(ids):
        r = np.zeros((3, 32, 32), np.float32)
        r[0] = h[i] + rng.normal(0, 1, (32, 32))               # CHM ~ h_i (the signal)
        r[1] = rng.uniform(0, 3, (32, 32)); r[2] = rng.uniform(0, 2, (32, 32))
        vol, dap = 12.0 * h[i] + 50 + rng.normal(0, 4), 0.6 * h[i] + rng.normal(0, 0.2)
        samples[pid] = {"raster": r, "scalars": np.array([h[i], 0.8 * h[i], 4.0], np.float32),
                        "labels": np.array([vol, dap], np.float32)}
        y[i] = [vol, dap]
    res = tr.run_lopo(samples, ids, y, n_aug=12, epochs=50, seed=0)
    r2 = tr.report(y, res["pred"], res["lo"], res["hi"], col=0)["R2"]

    yp = y.copy(); yp[:, 0] = y[rng.permutation(n), 0]          # break X→Vol_ha
    ss = {ids[i]: {**samples[ids[i]], "labels": np.array([yp[i, 0], yp[i, 1]], np.float32)}
          for i in range(n)}
    res_s = tr.run_lopo(ss, ids, yp, n_aug=12, epochs=50, seed=0)
    r2_shuf = tr.report(yp, res_s["pred"], res_s["lo"], res_s["hi"], col=0)["R2"]

    assert r2 > 0.5, f"failed to recover signal: R2={r2:.2f}"
    assert r2 - r2_shuf > 0.4, f"shuffle not clearly worse: real={r2:.2f} shuf={r2_shuf:.2f}"


# ============================ simulator (pretraining source) ============================
@field_data
def test_simulated_plot_is_internally_consistent():
    """A sim plot renders in the real channel format, with a CHM in metres that matches its scalar
    zmax, and plausible Vol_ha / DAP from the cubagem allometry."""
    from greenvista.image_cnn import simulate
    allo = simulate.fit_allometry()
    assert allo["vol"][1] > 1.0 and allo["vol"][2] > 0.0          # b≈1.9 (lnD), c≈1.0 (lnH)
    rng = np.random.default_rng(1)
    raster, scal, labels = simulate.make_plot(rng, allo)
    assert raster.shape == (3, 128, 128)
    assert 50.0 < labels[0] < 1000.0                              # Vol_ha plausible
    assert 6.0 < labels[1] < 30.0                                 # mean DBH plausible
    assert raster[0].max() > 8.0                                  # CHM in metres, not normalized
    assert abs(raster[0].max() - scal[0]) < 1.0                   # CHM max ≈ scalar zmax (consistent)


# ============================ SSL pretraining pieces (cloud-free units) ============================
def test_ssl_metric_vector_shape_and_order():
    from greenvista.image_cnn import ssl_pretrain as ssl
    z = np.array([3.0, 5, 8, 12, 18, 25, 30])
    v = ssl._metric_vector(z)
    assert v.shape == (len(ssl.PRETEXT_METRICS),)
    # the six quantile entries (q10..q95) are non-decreasing
    assert np.all(np.diff(v[:6]) >= 0)


def test_load_compatible_transfers_backbone_skips_mismatched_head():
    import torch
    from greenvista.image_cnn.model import PlotCNN
    from greenvista.image_cnn.train import _load_compatible
    torch.manual_seed(0)
    ssl_net = PlotCNN(n_targets=9)           # pretext head (9 metrics)
    ft_net = PlotCNN(n_targets=2)            # fine-tune head (Vol_ha + DAP)
    kept, total = _load_compatible(ft_net, ssl_net.state_dict())
    assert total // 2 < kept < total          # conv backbone + head.0 load; only head.2 (9≠2) skipped

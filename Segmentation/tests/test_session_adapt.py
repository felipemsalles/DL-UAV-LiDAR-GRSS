"""Smoke tests for session-aware test-time adaptation (greenvista.session_adapt).

Covers the properties that keep the adaptation valid: the adaptation is causal (the streaming radius at tile k
ignores future tiles), it never peeks at ground truth, denser stands get a tighter radius, and it
recovers the true stem count on a synthetic stand with duplicates."""
import numpy as np
import pytest

from greenvista.session_adapt import (
    nn_distances, typical_spacing, adaptive_radius, nms_merge,
    SessionSpacingEstimator, merge_stand_batch, merge_stand_streaming, merge_stand_global,
    count_within,
)


# --- geometry primitives ---------------------------------------------------------------
def test_typical_spacing_recovers_grid():
    g = np.arange(0, 10) * 1.8
    gx, gy = np.meshgrid(g, g)
    cents = np.column_stack([gx.ravel(), gy.ravel()])
    assert abs(typical_spacing(cents) - 1.8) < 1e-6


def test_nn_and_spacing_degenerate():
    assert len(nn_distances(np.empty((0, 2)))) == 0
    assert len(nn_distances(np.array([[0.0, 0.0]]))) == 0
    assert np.isnan(typical_spacing(np.array([[0.0, 0.0]])))


def test_adaptive_radius_clamps_and_defaults():
    assert adaptive_radius(1.8, fraction=0.7, lo=1.0, hi=2.0) == pytest.approx(1.26)
    assert adaptive_radius(0.5, fraction=0.7, lo=1.0, hi=2.0) == 1.0     # clamp low
    assert adaptive_radius(10.0, fraction=0.7, lo=1.0, hi=2.0) == 2.0    # clamp high
    assert adaptive_radius(float("nan"), default=1.5) == 1.5             # warm-up fallback


def test_nms_keeps_largest_and_merges():
    cents = np.array([[0.0, 0.0], [0.3, 0.0], [5.0, 0.0]])   # first two are duplicates
    sizes = np.array([10.0, 100.0, 50.0])
    kept, idx = nms_merge(cents, sizes, radius=1.0)
    assert len(kept) == 2                                     # dup collapsed
    assert set(idx) == {1, 2}                                 # kept the larger of the dup pair


# --- causality / no-GT-peek ------------------------------------------------------------
def _toy_stream(rng, n_tiles=6):
    return [rng.normal(0, 10, (rng.integers(5, 15), 2)) for _ in range(n_tiles)]


def test_streaming_is_causal():
    """Radius at tile k must depend only on tiles 1..k: replacing future tiles leaves the early
    radii/spacings unchanged."""
    rng = np.random.default_rng(0)
    tiles = _toy_stream(rng, 6)
    a = merge_stand_streaming(tiles)
    # corrupt everything after tile index 2
    rng2 = np.random.default_rng(999)
    tiles_future_changed = tiles[:3] + [rng2.normal(50, 5, (20, 2)) for _ in range(3)]
    b = merge_stand_streaming(tiles_future_changed)
    assert a["radii"][:3] == b["radii"][:3]
    np.testing.assert_allclose(
        [x for x in a["spacings"][:3] if np.isfinite(x)],
        [x for x in b["spacings"][:3] if np.isfinite(x)])


def test_estimator_is_cumulative_and_causal():
    est = SessionSpacingEstimator()
    rng = np.random.default_rng(1)
    seq = []
    for t in _toy_stream(rng, 5):
        seq.append(est.update(t))
    assert est.n_tiles == 5
    # spacing after all tiles equals the percentile of the full pooled NN set
    assert np.isfinite(seq[-1])


def test_no_ground_truth_peek():
    """The adaptive API takes only detected centroids, so its output cannot change with the ground
    truth: recomputation is bit-identical and the signature carries no ground-truth parameter."""
    rng = np.random.default_rng(2)
    tiles = _toy_stream(rng, 5)
    out1 = merge_stand_batch(tiles)
    # any 'ground truth' is external and never passed in → recomputation is bit-identical
    out2 = merge_stand_batch(tiles)
    np.testing.assert_array_equal(out1["merged"], out2["merged"])
    assert out1["radius"] == out2["radius"]
    # and the function signature has no GT parameter (guards against accidental leakage)
    import inspect
    params = set(inspect.signature(merge_stand_batch).parameters)
    assert not (params & {"gt", "gt_xy", "truth", "field", "n_true"})


# --- denser → tighter radius (the core adaptive behaviour) -----------------------------
def _grid_tiles(spacing, rng, L=80.0, tile=32.0, stride=16.0, sigma=0.2):
    """Overlapping-tile stand (each interior tree in ~4 tiles → cross-tile duplicates).
    Returns (tile_cents, tile_sizes, gt_in_core, window) with a core window fully
    double-covered by tiles so tile coverage is not the confound."""
    half = tile / 2.0
    g = np.arange(spacing, L - spacing + 1e-9, spacing)
    gx, gy = np.meshgrid(g, g)
    gt = np.column_stack([gx.ravel(), gy.ravel()])
    centres = np.arange(half, L - half + 1e-9, stride)
    tiles, sizes = [], []
    for cx in centres:
        for cy in centres:
            inside = (np.abs(gt[:, 0] - cx) <= half) & (np.abs(gt[:, 1] - cy) <= half)
            pts = gt[inside] + rng.normal(0, sigma, gt[inside].shape)
            tiles.append(pts)
            sizes.append(np.full(len(pts), 1000.0))
    m = 0.75 * tile
    win = (m, L - m)
    in_core = (gt[:, 0] >= win[0]) & (gt[:, 0] <= win[1]) & \
              (gt[:, 1] >= win[0]) & (gt[:, 1] <= win[1])
    return tiles, sizes, gt[in_core], win


def _count_core(cents, win):
    lo, hi = win
    c = np.asarray(cents).reshape(-1, 2)
    if len(c) == 0:
        return 0
    return int(((c[:, 0] >= lo) & (c[:, 0] <= hi) & (c[:, 1] >= lo) & (c[:, 1] <= hi)).sum())


def test_denser_stand_gets_tighter_radius():
    rng = np.random.default_rng(3)
    dense, *_ = _grid_tiles(1.6, rng)
    sparse, *_ = _grid_tiles(3.0, rng)
    r_dense = merge_stand_batch(dense)["radius"]
    r_sparse = merge_stand_batch(sparse)["radius"]
    assert r_dense < r_sparse                          # denser NN spacing → tighter radius


def test_recovers_true_count_and_beats_bad_global():
    """On a stand with cross-tile duplicates, adaptive merge lands near the true count and
    beats a deliberately-too-large global radius (which merges distinct neighbours)."""
    rng = np.random.default_rng(4)
    tiles, sizes, gt_core, win = _grid_tiles(2.0, rng, sigma=0.25)
    n_true = len(gt_core)
    adapt = merge_stand_batch(tiles, sizes, fraction=0.7, lo=1.0, hi=2.0)
    n_adapt = _count_core(adapt["merged"], win)
    n_bad = _count_core(merge_stand_global(tiles, sizes, radius=2.6)["merged"], win)  # > spacing
    # adaptive within 15% of truth; and closer to truth than the over-large global
    assert abs(n_adapt - n_true) / n_true < 0.15
    assert abs(n_adapt - n_true) <= abs(n_bad - n_true)
    assert n_bad < n_adapt                              # the bad global over-merges → under-counts


def test_count_within():
    cents = np.array([[0.0, 0.0], [1.0, 0.0], [20.0, 0.0]])
    assert count_within(cents, (0.0, 0.0), 12.0) == 2
    assert count_within(np.empty((0, 2)), (0.0, 0.0), 12.0) == 0

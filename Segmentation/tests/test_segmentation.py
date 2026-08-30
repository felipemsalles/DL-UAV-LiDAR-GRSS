import numpy as np
from greenvista.segmentation import (build_chm, detect_treetops, match_detections,
                                      load_panoptic_ply, assign_trees_to_instances)
from tests.conftest import make_tree

def test_assign_trees_to_instances_low_band_picks_correct_tree():
    # tree A (suppressed, 8-16 m) at (0,0); tree B (dominant, 8-30 m) at (1.5,0) overtopping A's column.
    rng = np.random.default_rng(0)
    za = rng.uniform(8, 16, 300); xa = rng.normal(0, 0.4, 300); ya = rng.normal(0, 0.4, 300)
    zb = rng.uniform(8, 30, 300); xb = rng.normal(1.5, 0.4, 300); yb = rng.normal(0, 0.4, 300)
    xy = np.column_stack([np.r_[xa, xb], np.r_[ya, yb]])
    z = np.r_[za, zb]; inst = np.r_[np.zeros(300, int), np.ones(300, int)]
    # tree A at (0,0): low-band vote should pick instance 0, not the overtopping instance 1
    out = assign_trees_to_instances(xy, z, inst, np.array([[0.0, 0.0]]), radius=1.2, z_lo=2, z_hi=12)
    assert out[0] == 0

def test_detect_treetops_finds_known_peaks():
    # two trees 10 m apart, heights 25 and 18
    t1 = make_tree(height=25, crown_radius=2.0, center=(0, 0), seed=1)
    t2 = make_tree(height=18, crown_radius=2.0, center=(10, 0), seed=2)
    pts = np.vstack([t1, t2])
    chm, x0, y0, res = build_chm(pts[:, :2], pts[:, 2], res=0.5)
    tops = detect_treetops(chm, x0, y0, res, window=5, hmin=5.0)
    res_match = match_detections(tops[:, :2], np.array([[0, 0], [10, 0]]), radius=2.0)
    assert res_match["detected"] == 2

def test_match_detections_empty():
    r = match_detections(np.empty((0, 2)), np.array([[0, 0]]), radius=1.5)
    assert r["detected"] == 0 and r["detection_rate"] == 0.0

def test_load_panoptic_ply(tmp_path):
    from plyfile import PlyData, PlyElement
    verts = np.array(
        [(0., 0., 1., 0, 1), (0.1, 0., 2., 0, 2), (5., 5., 1., 1, 1), (5.1, 5., 3., 1, 2)],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("instance_pred", "i4"), ("semantic_pred", "i4")])
    p = tmp_path / "panoptic.ply"
    PlyData([PlyElement.describe(verts, "vertex")]).write(str(p))
    trees = load_panoptic_ply(p)
    assert set(trees.keys()) == {0, 1}
    assert trees[0]["points"].shape == (2, 3)

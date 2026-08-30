from greenvista.features import crown_features, CROWN_FEATURE_KEYS
from tests.conftest import make_tree

def test_crown_features_keys_and_sanity(tree):
    f = crown_features(tree)
    assert set(CROWN_FEATURE_KEYS).issubset(f)
    assert 18 < f["H_max"] <= 20.5            # tree height ~20 m
    assert f["crown_area"] > 0 and f["crown_volume"] > 0
    assert f["n_points"] == len(tree)

def test_crown_features_scale_monotonic():
    """Bigger crown -> bigger area/volume (scale must be preserved, not normalized away)."""
    small = crown_features(make_tree(height=20, crown_radius=1.5, seed=2))
    big = crown_features(make_tree(height=20, crown_radius=3.0, seed=2))
    assert big["crown_area"] > small["crown_area"]
    assert big["crown_volume"] > small["crown_volume"]
    assert big["crown_diameter"] > small["crown_diameter"]

def test_crown_features_height_tracks_input():
    tall = crown_features(make_tree(height=30, crown_radius=2.0, seed=3))
    short = crown_features(make_tree(height=15, crown_radius=2.0, seed=3))
    assert tall["H_max"] > short["H_max"] + 10

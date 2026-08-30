"""Shared synthetic fixtures: everything is generated in-process, so no real data is required."""
import numpy as np
import pytest

def make_tree(height=20.0, crown_radius=2.0, n_points=600, center=(0.0, 0.0),
              crown_base_frac=0.4, seed=0):
    """A cone-shaped crown point cloud (radius shrinks to ~0 at the top). Returns (N,3) XYZ in metres."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(crown_base_frac * height, height, n_points)
    frac_up = (z - crown_base_frac * height) / ((1 - crown_base_frac) * height)
    rmax = crown_radius * (1 - frac_up)            # cone taper
    r = rmax * np.sqrt(rng.uniform(0, 1, n_points))
    th = rng.uniform(0, 2 * np.pi, n_points)
    x = center[0] + r * np.cos(th)
    y = center[1] + r * np.sin(th)
    return np.column_stack([x, y, z])

@pytest.fixture
def tree():
    return make_tree()

@pytest.fixture
def synthetic_dataset():
    """N trees with a known volume law V = k * crown_radius^2 * height (∝ crown_area * H).
    Used by the integration harness to check that feature->model->eval recovers the relationship."""
    rng = np.random.default_rng(1)
    rows, clouds = [], []
    k = 0.03
    for i in range(30):
        h = rng.uniform(12, 32)
        rad = rng.uniform(1.2, 3.5)
        V = k * (rad ** 2) * h
        clouds.append(make_tree(height=h, crown_radius=rad, seed=i))
        rows.append({"tree": i, "height": h, "radius": rad, "V": V})
    return rows, clouds

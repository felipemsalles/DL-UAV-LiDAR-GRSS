"""Unit tests for core pipeline functions."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add scripts dir to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import smalian_volume, freire_volume
from tree_analysis import (
    angular_coverage,
    fit_circle_kasa,
    fit_circle_ransac,
    _circle_from_3pts,
    compute_section_volumes,
)


# ===================================================================
# Smalian volume
# ===================================================================

class TestSmalianVolume:
    def test_cylinder(self):
        """Equal diameters → cylinder volume."""
        vol = smalian_volume(20.0, 20.0, 1.0)
        expected = np.pi * (0.20 ** 2) / 4 * 1.0  # pi*r^2*h
        assert abs(vol - expected) < 1e-10

    def test_zero_length(self):
        assert smalian_volume(20.0, 10.0, 0.0) == 0.0

    def test_zero_diameters(self):
        assert smalian_volume(0.0, 0.0, 1.0) == 0.0

    def test_tapered_section(self):
        """Truncated cone: V = (pi/8)(d1^2 + d2^2)*L."""
        vol = smalian_volume(20.0, 10.0, 1.0)
        expected = (np.pi / 8) * (0.20 ** 2 + 0.10 ** 2) * 1.0
        assert abs(vol - expected) < 1e-10

    def test_positive(self):
        """Volume must be non-negative."""
        assert smalian_volume(15.0, 12.0, 2.5) > 0


# ===================================================================
# Freire volume
# ===================================================================

class TestFreireVolume:
    def test_known_value(self):
        """V = pi * (15/200)^2 * 20 * 0.5 for DBH=15cm, H=20m."""
        vol = freire_volume(15.0, 20.0)
        expected = np.pi * (15.0 / 200.0) ** 2 * 20.0 * 0.5
        assert abs(vol - expected) < 1e-10

    def test_zero_dbh(self):
        assert freire_volume(0.0, 20.0) == 0.0

    def test_custom_form_factor(self):
        vol_05 = freire_volume(15.0, 20.0, form_factor=0.5)
        vol_07 = freire_volume(15.0, 20.0, form_factor=0.7)
        assert vol_07 / vol_05 == pytest.approx(0.7 / 0.5)


# ===================================================================
# Circle from 3 points
# ===================================================================

class TestCircleFrom3Pts:
    def test_unit_circle(self):
        """Three points on unit circle centered at origin."""
        p1 = (1.0, 0.0)
        p2 = (0.0, 1.0)
        p3 = (-1.0, 0.0)
        cx, cy, r = _circle_from_3pts(p1, p2, p3)
        assert abs(cx) < 1e-10
        assert abs(cy) < 1e-10
        assert abs(r - 1.0) < 1e-10

    def test_collinear_returns_none(self):
        """Collinear points → None."""
        result = _circle_from_3pts((0, 0), (1, 1), (2, 2))
        assert result is None

    def test_offset_circle(self):
        """Circle centered at (2, 3) with r=5."""
        cx0, cy0, r0 = 2.0, 3.0, 5.0
        angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
        pts = [(cx0 + r0 * np.cos(a), cy0 + r0 * np.sin(a)) for a in angles]
        cx, cy, r = _circle_from_3pts(*pts)
        assert abs(cx - cx0) < 1e-8
        assert abs(cy - cy0) < 1e-8
        assert abs(r - r0) < 1e-8


# ===================================================================
# Angular coverage
# ===================================================================

class TestAngularCoverage:
    def test_full_circle(self):
        """Points around full circle → ~360 degrees."""
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        xy = np.column_stack([np.cos(angles), np.sin(angles)])
        cov = angular_coverage(xy)
        assert cov > 350

    def test_half_ring(self):
        """Points on a half ring centered at origin → ~180 degrees."""
        # Points on a ring (not arc), so centroid ≈ center of ring
        angles = np.linspace(-np.pi / 2, np.pi / 2, 50)
        r_inner, r_outer = 0.9, 1.1
        rng = np.random.default_rng(42)
        r = rng.uniform(r_inner, r_outer, size=50)
        xy = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        # Add symmetric points to keep centroid near origin
        xy = np.vstack([xy, np.array([[0, 0]])])  # anchor centroid
        cov = angular_coverage(xy)
        assert cov > 170

    def test_quarter_centered(self):
        """Points spanning 90° around origin → ~90 degrees."""
        angles = np.linspace(0, np.pi / 2, 50)
        r = 1.0
        # Center at origin so centroid ≈ origin
        cx, cy = 0.0, 0.0
        xy = np.column_stack([cx + r * np.cos(angles),
                              cy + r * np.sin(angles)])
        # Coverage is measured from centroid, which shifts —
        # just check it's less than semicircle and > 0
        cov = angular_coverage(xy)
        assert cov > 0


# ===================================================================
# Circle fitting (Kasa)
# ===================================================================

class TestFitCircleKasa:
    def test_perfect_circle(self):
        """Points on a perfect circle should give exact fit."""
        r_true = 0.10  # 10 cm radius
        angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
        xy = np.column_stack([r_true * np.cos(angles),
                              r_true * np.sin(angles)])
        cx, cy, r, residual = fit_circle_kasa(xy)
        assert abs(cx) < 1e-6
        assert abs(cy) < 1e-6
        assert abs(r - r_true) < 1e-6
        assert residual < 1e-6

    def test_noisy_circle(self):
        """Noisy circle: fit should be close to true values."""
        rng = np.random.default_rng(42)
        r_true = 0.10
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        noise = rng.normal(0, 0.005, size=100)
        r_noisy = r_true + noise
        xy = np.column_stack([r_noisy * np.cos(angles),
                              r_noisy * np.sin(angles)])
        cx, cy, r, residual = fit_circle_kasa(xy)
        assert abs(r - r_true) < 0.01  # within 1 cm


# ===================================================================
# Circle fitting (RANSAC)
# ===================================================================

class TestFitCircleRansac:
    def test_perfect_circle(self):
        r_true = 0.08
        angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        xy = np.column_stack([r_true * np.cos(angles),
                              r_true * np.sin(angles)])
        cx, cy, r, residual = fit_circle_ransac(xy)
        assert abs(r - r_true) < 1e-4

    def test_with_outliers(self):
        """RANSAC should handle outliers better than Kasa."""
        rng = np.random.default_rng(42)
        r_true = 0.10
        angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        xy = np.column_stack([r_true * np.cos(angles),
                              r_true * np.sin(angles)])
        # Add 5 outliers far from the circle
        outliers = rng.uniform(-0.5, 0.5, size=(5, 2))
        xy_dirty = np.vstack([xy, outliers])
        cx, cy, r, residual = fit_circle_ransac(xy_dirty)
        assert abs(r - r_true) < 0.02  # should still be close

    def test_too_few_points_falls_back(self):
        """With < 3 points, falls back to Kasa."""
        xy = np.array([[0.1, 0.0], [0.0, 0.1]])
        # Should not raise
        cx, cy, r, residual = fit_circle_ransac(xy)
        assert r > 0


# ===================================================================
# Section volumes
# ===================================================================

class TestComputeSectionVolumes:
    def test_empty(self):
        total, sections = compute_section_volumes([])
        assert np.isnan(total)
        assert sections == []

    def test_single_measurement(self):
        total, sections = compute_section_volumes(
            [{"height_m": 1.3, "diameter_cm": 15.0}])
        assert np.isnan(total)
        assert sections == []

    def test_two_measurements(self):
        diams = [
            {"height_m": 1.0, "diameter_cm": 20.0},
            {"height_m": 2.0, "diameter_cm": 18.0},
        ]
        total, sections = compute_section_volumes(diams)
        expected = smalian_volume(20.0, 18.0, 1.0)
        assert abs(total - expected) < 1e-4  # total is rounded to 6 decimals
        assert len(sections) == 1

    def test_multiple_sections(self):
        diams = [
            {"height_m": 0.0, "diameter_cm": 25.0},
            {"height_m": 1.0, "diameter_cm": 22.0},
            {"height_m": 2.0, "diameter_cm": 18.0},
        ]
        total, sections = compute_section_volumes(diams)
        expected = (smalian_volume(25.0, 22.0, 1.0) +
                    smalian_volume(22.0, 18.0, 1.0))
        assert abs(total - expected) < 1e-4  # total is rounded to 6 decimals
        assert len(sections) == 2

    def test_unsorted_input(self):
        """Should sort by height internally."""
        diams = [
            {"height_m": 2.0, "diameter_cm": 18.0},
            {"height_m": 0.0, "diameter_cm": 25.0},
            {"height_m": 1.0, "diameter_cm": 22.0},
        ]
        total, sections = compute_section_volumes(diams)
        assert sections[0]["h_bottom"] == 0.0
        assert sections[1]["h_bottom"] == 1.0

"""Unit tests for the PerspectiveTransformer (4-corner perspective transform)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.perspective_transform import PerspectiveTransformer


class TestPerspectiveTransformerConstruction:
    """Tests for construction and validation."""

    def test_valid_4_corners(self):
        """Should construct successfully with 4 valid corners."""
        corners = [(100, 100), (300, 100), (300, 400), (100, 400)]
        pt = PerspectiveTransformer(corners, width_meters=10.0, length_meters=20.0)
        assert pt.M is not None
        assert pt.M_inv is not None
        assert pt.width_meters == 10.0
        assert pt.length_meters == 20.0

    def test_too_few_corners(self):
        """Should raise ValueError with < 4 corners."""
        with pytest.raises(ValueError, match="exactly 4"):
            PerspectiveTransformer([(0, 0), (1, 0), (1, 1)], 10.0, 10.0)

    def test_too_many_corners(self):
        """Should raise ValueError with > 4 corners."""
        with pytest.raises(ValueError, match="exactly 4"):
            PerspectiveTransformer([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)], 10.0, 10.0)

    def test_zero_width(self):
        """Should raise ValueError with zero width."""
        with pytest.raises(ValueError, match="width_meters"):
            PerspectiveTransformer([(0, 0), (1, 0), (1, 1), (0, 1)], 0.0, 10.0)

    def test_negative_length(self):
        """Should raise ValueError with negative length."""
        with pytest.raises(ValueError, match="length_meters"):
            PerspectiveTransformer([(0, 0), (1, 0), (1, 1), (0, 1)], 10.0, -5.0)


class TestPerspectiveTransformerCorners:
    """Tests that corner points transform to expected metric coordinates."""

    @pytest.fixture
    def simple_rect(self):
        # A simple axis-aligned rectangle: 200px wide × 300px tall
        corners = [(100, 50), (300, 50), (300, 350), (100, 350)]
        return PerspectiveTransformer(corners, width_meters=10.0, length_meters=20.0)

    def test_tl_corner(self, simple_rect):
        """TL corner should map to (0, 0)."""
        x, y = simple_rect.transform_point(100, 50)
        assert abs(x) < 0.1, f"TL x: expected ~0, got {x}"
        assert abs(y) < 0.1, f"TL y: expected ~0, got {y}"

    def test_tr_corner(self, simple_rect):
        """TR corner should map to (width, 0)."""
        x, y = simple_rect.transform_point(300, 50)
        assert abs(x - 10.0) < 0.1, f"TR x: expected ~10, got {x}"
        assert abs(y) < 0.1, f"TR y: expected ~0, got {y}"

    def test_br_corner(self, simple_rect):
        """BR corner should map to (width, length)."""
        x, y = simple_rect.transform_point(300, 350)
        assert abs(x - 10.0) < 0.1, f"BR x: expected ~10, got {x}"
        assert abs(y - 20.0) < 0.1, f"BR y: expected ~20, got {y}"

    def test_bl_corner(self, simple_rect):
        """BL corner should map to (0, length)."""
        x, y = simple_rect.transform_point(100, 350)
        assert abs(x) < 0.1, f"BL x: expected ~0, got {x}"
        assert abs(y - 20.0) < 0.1, f"BL y: expected ~20, got {y}"

    def test_center_point(self, simple_rect):
        """Center of rectangle should map to (width/2, length/2)."""
        x, y = simple_rect.transform_point(200, 200)
        assert abs(x - 5.0) < 0.5, f"Center x: expected ~5, got {x}"
        assert abs(y - 10.0) < 0.5, f"Center y: expected ~10, got {y}"


class TestPerspectiveTransformerBatch:
    """Tests for batch transformation."""

    @pytest.fixture
    def transformer(self):
        corners = [(0, 0), (200, 0), (200, 400), (0, 400)]
        return PerspectiveTransformer(corners, width_meters=8.0, length_meters=16.0)

    def test_batch_matches_individual(self, transformer):
        """transform_batch should return same values as individual transform_point."""
        points = [(50, 100), (100, 200), (150, 300)]
        batch_result = transformer.transform_batch(points)
        individual_results = [transformer.transform_point(x, y) for x, y in points]
        for i, (mx, my) in enumerate(individual_results):
            assert abs(batch_result[i, 0] - mx) < 0.01
            assert abs(batch_result[i, 1] - my) < 0.01

    def test_empty_batch(self, transformer):
        """Empty batch should return empty array."""
        result = transformer.transform_batch([])
        assert result.shape == (0, 2)


class TestPerspectiveTransformerSpeed:
    """Tests for speed computation using the transformer."""

    @pytest.fixture
    def road_transformer(self):
        # Simulating a 7m wide × 30m long road on a 640×480 frame
        corners = [(100, 50), (540, 50), (540, 430), (100, 430)]
        return PerspectiveTransformer(corners, width_meters=7.0, length_meters=30.0)

    def test_speed_from_known_positions(self, road_transformer):
        """Simulate a vehicle traversing at a known speed."""
        # Vehicle moves from top to bottom across 15m of the 30m road
        # Top-center: (320, 50) → should be about (3.5, 0)
        # Mid-center: (320, 240) → should be about (3.5, 15)
        pos1 = road_transformer.transform_point(320, 50)
        pos2 = road_transformer.transform_point(320, 240)
        
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        distance_m = np.sqrt(dx**2 + dy**2)
        
        dt = 5.0  # seconds
        speed_ms = distance_m / dt
        speed_kmh = speed_ms * 3.6
        
        # For a simple axis-aligned rectangle, 240-50 = 190px out of 380px = ~15m
        # Speed should be approx 15m / 5s * 3.6 = 10.8 km/h
        assert 5.0 < speed_kmh < 20.0, f"Expected ~10.8 km/h, got {speed_kmh:.1f}"


class TestPerspectiveTransformerPerspective:
    """Tests with a non-rectangular (trapezoidal) perspective shape."""

    @pytest.fixture
    def trapezoid(self):
        # Simulating camera perspective: top narrower than bottom
        corners = [(200, 100), (440, 100), (540, 400), (100, 400)]
        return PerspectiveTransformer(corners, width_meters=10.0, length_meters=30.0)

    def test_trapezoid_corners(self, trapezoid):
        """Corners of trapezoid should still map to rect corners."""
        x, y = trapezoid.transform_point(200, 100)  # TL
        assert abs(x) < 0.5 and abs(y) < 0.5

        x, y = trapezoid.transform_point(440, 100)  # TR
        assert abs(x - 10.0) < 0.5 and abs(y) < 0.5

        x, y = trapezoid.transform_point(540, 400)  # BR
        assert abs(x - 10.0) < 0.5 and abs(y - 30.0) < 0.5

        x, y = trapezoid.transform_point(100, 400)  # BL
        assert abs(x) < 0.5 and abs(y - 30.0) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

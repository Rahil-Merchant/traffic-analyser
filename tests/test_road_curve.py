"""Unit tests for the simplified RoadCurve (user-drawn axis)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.road_curve import RoadCurve


class TestRoadCurveStraightAxis:
    """Tests with a simple vertical straight-line axis."""

    @pytest.fixture
    def straight_axis(self):
        # A straight vertical axis from top to bottom (150, 50) → (150, 450)
        axis = [(150, 50), (150, 450)]
        return RoadCurve(axis, axis_length_meters=100.0)

    def test_start_of_road(self, straight_axis):
        """Point at the top should project near 0m."""
        s = straight_axis.project(150, 50)
        assert s < 5.0, f"Start of road should be near 0m, got {s}"

    def test_end_of_road(self, straight_axis):
        """Point at the bottom should project near 100m."""
        s = straight_axis.project(150, 450)
        assert s > 95.0, f"End of road should be near 100m, got {s}"

    def test_midpoint(self, straight_axis):
        """Point at the vertical center should project near 50m."""
        s = straight_axis.project(150, 250)
        assert 45.0 < s < 55.0, f"Midpoint should be near 50m, got {s}"

    def test_monotonic_along_road(self, straight_axis):
        """Points moving down the road should have increasing s values."""
        y_positions = [60, 100, 200, 300, 400, 440]
        s_values = [straight_axis.project(150, y) for y in y_positions]
        for i in range(len(s_values) - 1):
            assert s_values[i] < s_values[i + 1], (
                f"s should increase: s[{i}]={s_values[i]:.2f} >= s[{i+1}]={s_values[i+1]:.2f}"
            )

    def test_lateral_offset_same_s(self, straight_axis):
        """Points at the same y but different x should project to same s."""
        s_left = straight_axis.project(50, 250)
        s_center = straight_axis.project(150, 250)
        s_right = straight_axis.project(250, 250)
        # All should be nearly identical since they project to the same
        # closest centreline point
        assert abs(s_left - s_center) < 1.0
        assert abs(s_right - s_center) < 1.0

    def test_speed_from_fixed_positions(self, straight_axis):
        """Simulate a vehicle traversing at known speed."""
        s1 = straight_axis.project(150, 100)
        s2 = straight_axis.project(150, 300)
        dt = 5.0
        speed_ms = abs(s2 - s1) / dt
        speed_kmh = speed_ms * 3.6
        assert 25.0 < speed_kmh < 50.0, f"Expected ~36 km/h, got {speed_kmh:.1f}"


class TestRoadCurveMultiPointAxis:
    """Tests with a multi-point curved axis."""

    @pytest.fixture
    def curved_axis(self):
        # An L-shaped axis: goes down then turns right
        axis = [(100, 50), (100, 250), (300, 250)]
        return RoadCurve(axis, axis_length_meters=60.0)

    def test_construction(self, curved_axis):
        """Curved axis should construct successfully."""
        assert curved_axis.centerline is not None
        assert len(curved_axis.centerline) > 100
        assert curved_axis.s_meters[-1] == pytest.approx(60.0, abs=0.1)

    def test_monotonic(self, curved_axis):
        """Arc-length values along the centreline should be monotonic."""
        for i in range(len(curved_axis.s_meters) - 1):
            assert curved_axis.s_meters[i] <= curved_axis.s_meters[i + 1]


class TestRoadCurveEdgeCases:
    """Edge cases and error handling."""

    def test_too_few_points(self):
        """Should raise ValueError with < 2 points."""
        with pytest.raises(ValueError, match="needs >= 2"):
            RoadCurve([(0, 0)], axis_length_meters=10.0)

    def test_two_points(self):
        """Two points should work (simplest valid axis)."""
        curve = RoadCurve([(0, 0), (100, 0)], axis_length_meters=50.0)
        assert curve.s_meters[-1] == pytest.approx(50.0, abs=0.01)

    def test_batch_projection(self):
        """project_batch should return same values as individual project."""
        curve = RoadCurve([(50, 0), (50, 400)], axis_length_meters=100.0)
        points = [(50, 100), (50, 200), (50, 300)]
        batch_result = curve.project_batch(points)
        individual_results = [curve.project(x, y) for x, y in points]
        np.testing.assert_array_almost_equal(batch_result, individual_results)

    def test_total_arc_length(self):
        """Total arc-length should equal axis_length_meters."""
        curve = RoadCurve([(0, 0), (0, 400)], axis_length_meters=123.4)
        assert curve.s_meters[-1] == pytest.approx(123.4, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

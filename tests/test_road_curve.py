"""Unit tests for RoadCurve arc-length speed calculation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from core.road_curve import RoadCurve


class TestRoadCurveStraightRoad:
    """Tests with a simple rectangular (straight) road polygon."""

    @pytest.fixture
    def straight_road(self):
        # Rectangle: 100px wide, 400px tall (simulating a straight road)
        # CW order: TL, TR, BR, BL
        polygon = [
            (100, 50),   # TL
            (200, 50),   # TR
            (200, 450),  # BR
            (100, 450),  # BL
        ]
        return RoadCurve(polygon, road_length_meters=100.0, road_width_meters=25.0)

    def test_start_of_road(self, straight_road):
        """Point at the top-center should project near 0m."""
        s = straight_road.project(150, 50)
        assert s < 10.0, f"Start of road should be near 0m, got {s}"

    def test_end_of_road(self, straight_road):
        """Point at the bottom-center should project near 100m."""
        s = straight_road.project(150, 450)
        assert s > 90.0, f"End of road should be near 100m, got {s}"

    def test_midpoint(self, straight_road):
        """Point at the vertical center should project near 50m."""
        s = straight_road.project(150, 250)
        assert 40.0 < s < 60.0, f"Midpoint should be near 50m, got {s}"

    def test_monotonic_along_road(self, straight_road):
        """Points moving down the road should have increasing s values."""
        y_positions = [60, 100, 200, 300, 400, 440]
        s_values = [straight_road.project(150, y) for y in y_positions]
        for i in range(len(s_values) - 1):
            assert s_values[i] < s_values[i + 1], (
                f"s should increase: s[{i}]={s_values[i]:.2f} >= s[{i+1}]={s_values[i+1]:.2f}"
            )

    def test_lateral_offset_same_s(self, straight_road):
        """Points at the same y but different x should project to similar s."""
        s_left = straight_road.project(110, 250)
        s_center = straight_road.project(150, 250)
        s_right = straight_road.project(190, 250)
        # All should be within ~5m of each other
        assert abs(s_left - s_center) < 5.0
        assert abs(s_right - s_center) < 5.0

    def test_speed_from_fixed_positions(self, straight_road):
        """Simulate a vehicle traversing at known speed."""
        # Vehicle starts at y=100, moves to y=300 in 5 seconds
        # That's roughly 50m in 5s = 10 m/s = 36 km/h
        s1 = straight_road.project(150, 100)
        s2 = straight_road.project(150, 300)
        dt = 5.0
        speed_ms = abs(s2 - s1) / dt
        speed_kmh = speed_ms * 3.6
        assert 25.0 < speed_kmh < 50.0, f"Expected ~36 km/h, got {speed_kmh:.1f}"


class TestRoadCurveCurvedRoad:
    """Tests with a curved road polygon."""

    @pytest.fixture
    def curved_road(self):
        # An L-shaped polygon simulating a curve
        polygon = [
            (50, 50),    # TL
            (150, 50),   # TR (top of the L)
            (150, 200),  # right side mid
            (250, 200),  # right side extends
            (250, 300),  # BR
            (50, 300),   # BL (bottom of the L)
            (50, 200),   # left side mid
        ]
        return RoadCurve(polygon, road_length_meters=80.0)

    def test_construction(self, curved_road):
        """Curved road should construct successfully."""
        assert curved_road.centerline is not None
        assert len(curved_road.centerline) > 100
        assert curved_road.s_meters[-1] == pytest.approx(80.0, abs=0.1)

    def test_monotonic(self, curved_road):
        """Arc-length values along the centerline should be monotonically increasing."""
        for i in range(len(curved_road.s_meters) - 1):
            assert curved_road.s_meters[i] <= curved_road.s_meters[i + 1]


class TestRoadCurveEdgeCases:
    """Edge cases and error handling."""

    def test_too_few_points(self):
        """Should raise ValueError with < 4 points."""
        with pytest.raises(ValueError, match="needs >= 4"):
            RoadCurve([(0, 0), (1, 0), (1, 1)], road_length_meters=10.0)

    def test_batch_projection(self):
        """project_batch should return same values as individual project calls."""
        polygon = [(0, 0), (100, 0), (100, 400), (0, 400)]
        curve = RoadCurve(polygon, road_length_meters=100.0)

        points = [(50, 100), (50, 200), (50, 300)]
        batch_result = curve.project_batch(points)
        individual_results = [curve.project(x, y) for x, y in points]

        np.testing.assert_array_almost_equal(batch_result, individual_results)

    def test_total_arc_length(self):
        """Total arc-length should equal road_length_meters."""
        polygon = [(0, 0), (100, 0), (100, 400), (0, 400)]
        curve = RoadCurve(polygon, road_length_meters=123.4)
        assert curve.s_meters[-1] == pytest.approx(123.4, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

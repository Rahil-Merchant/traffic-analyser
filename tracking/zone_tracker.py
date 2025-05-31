# project/tracking/zone_tracker.py
import numpy as np
import cv2
# from config import FRAME_WIDTH, FRAME_HEIGHT # No longer directly needed if not used by methods

class ZoneTracker:
    def __init__(self, polygons_dict): # polygons_dict keys are direction names, values are np.array(pts, dtype=np.int32)
        self.zones = polygons_dict if polygons_dict else {}
        # self.frame_width, self.frame_height are not strictly needed by ZoneTracker itself if polygons are pre-scaled.
        # Cooldowns and vehicle_zones are for stateful transition logic, which is moving to Pass 2/3.
        # For Pass 1, ZoneTracker is mostly a stateless utility for _check_point_in_zone.
        # self.zone_cooldowns = {}
        # self.vehicle_zones = {} 
        if not self.zones:
            print("ZoneTracker __init__: Received empty or invalid polygons_dict.")
        else:
            print(f"ZoneTracker initialized with {len(self.zones)} zones: {list(self.zones.keys())}")

    def _check_point_in_zone(self, point, zone_polygon): # This method is KEY and remains
        if zone_polygon is None or not isinstance(zone_polygon, np.ndarray) or zone_polygon.ndim != 2 or zone_polygon.shape[0] < 3:
            return False
        if zone_polygon.dtype != np.int32: # Ensure correct dtype
            try:
                zone_polygon = np.array(zone_polygon, dtype=np.int32)
            except ValueError:
                print(f"ERROR: Could not convert zone_polygon to int32. Polygon: {zone_polygon}")
                return False
        try:
            point_int = tuple(map(int, point))
            result = cv2.pointPolygonTest(zone_polygon, point_int, False)
            return result >= 0
        except Exception as e:
            print(f"ERROR in _check_point_in_zone: {e}, Point: {point}, Polygon shape: {getattr(zone_polygon, 'shape', 'N/A')}")
            return False
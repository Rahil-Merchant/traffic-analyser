import cv2
import numpy as np

# Define colors (BGR)
COLOR_GREEN = (0, 255, 0); COLOR_RED = (0, 0, 255); COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255); COLOR_MAGENTA = (255, 0, 255); COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0); COLOR_ORANGE = (0, 165, 255); COLOR_CYAN = (255, 255, 0)
ZONE_COLORS = {'north':COLOR_BLUE, 'south':COLOR_GREEN, 'east':COLOR_RED, 'west':COLOR_MAGENTA}
DEFAULT_ZONE_COLOR = COLOR_WHITE

def draw_detection_box(frame, box_coords, vehicle_id, vehicle_type, status="detected"):
    if not (isinstance(box_coords, (tuple, list)) and len(box_coords) == 4):
         return

    x1, y1, x2, y2 = map(int, box_coords)

    # Simple color mapping
    if status == 'active': box_color = COLOR_ORANGE
    elif status == 'exiting': box_color = COLOR_RED
    elif status == 'entering': box_color = COLOR_CYAN
    else: box_color = COLOR_GREEN

    # Single rectangle draw
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Simple single-line label
    label = f"{vehicle_id[-4:]}:{vehicle_type}"
    
    # Single text draw with outline - no background calculations
    text_pos = (x1, max(15, y1 - 5))
    cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLACK, 2, cv2.LINE_AA)
    cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

def draw_zones(frame, zone_tracker):
    if not hasattr(zone_tracker, 'zones') or not zone_tracker.zones:
        return frame
    
    # Skip overlay blending for performance - just draw outlines
    for direction, polygon_points in zone_tracker.zones.items():
        if not isinstance(polygon_points, np.ndarray) or polygon_points.ndim != 2 or polygon_points.shape[0] < 3:
            continue
        if polygon_points.dtype != np.int32:
            polygon_points = polygon_points.astype(np.int32)
        color = ZONE_COLORS.get(direction, DEFAULT_ZONE_COLOR)
        try:
            cv2.polylines(frame, [polygon_points], isClosed=True, color=color, thickness=2)
            # Simple center calculation for zone label
            M = cv2.moments(polygon_points)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"]); center_y = int(M["m01"] / M["m00"])
                cv2.putText(frame, direction[0].upper(), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        except: pass
    return frame

def draw_event_marker(frame, point, event_type):
    point_int = tuple(map(int, point))
    if event_type == "ENTRY": 
        cv2.circle(frame, point_int, 8, COLOR_GREEN, -1)
    elif event_type == "EXIT": 
        cv2.drawMarker(frame, point_int, COLOR_RED, cv2.MARKER_CROSS, 12, 2)

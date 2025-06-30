# project/visualization/overlay.py

import cv2
import numpy as np

# Define colors (BGR)
COLOR_GREEN = (0, 255, 0); COLOR_RED = (0, 0, 255); COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255); COLOR_MAGENTA = (255, 0, 255); COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0); COLOR_ORANGE = (0, 165, 255); COLOR_CYAN = (255, 255, 0)
ZONE_COLORS = {'north':COLOR_BLUE, 'south':COLOR_GREEN, 'east':COLOR_RED, 'west':COLOR_MAGENTA}
DEFAULT_ZONE_COLOR = COLOR_WHITE

def draw_detection_box(frame, box_coords, vehicle_id, vehicle_type, status="detected"):
    # FIX: Removed the overly strict type check. The map() call below handles validation.
    try:
        x1, y1, x2, y2 = map(int, box_coords)
    except (TypeError, ValueError):
        # If box_coords is not a valid iterable of 4, skip drawing.
        return

    # Simple color mapping based on tracking status
    if status == 'active': box_color = COLOR_ORANGE
    elif status == 'exited': box_color = COLOR_RED
    else: box_color = COLOR_GREEN

    # Draw the bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Prepare and draw the label with a black outline for visibility
    label = f"{vehicle_id[-4:]}:{vehicle_type}"
    text_pos = (x1, max(15, y1 - 5))
    cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLACK, 3, cv2.LINE_AA)
    cv2.putText(frame, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

def draw_zones(frame, zone_tracker):
    if not hasattr(zone_tracker, 'zones') or not zone_tracker.zones:
        return frame
    
    overlay = frame.copy()
    for direction, polygon_points in zone_tracker.zones.items():
        if not isinstance(polygon_points, np.ndarray) or polygon_points.ndim != 2 or polygon_points.shape[0] < 3:
            continue
        
        color = ZONE_COLORS.get(direction, DEFAULT_ZONE_COLOR)
        
        # Draw filled polygon with transparency
        cv2.fillPoly(overlay, [polygon_points.astype(np.int32)], color)
        # Draw polygon outline
        cv2.polylines(frame, [polygon_points.astype(np.int32)], isClosed=True, color=color, thickness=2)
        
        # Add zone label
        M = cv2.moments(polygon_points)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            cv2.putText(frame, direction.capitalize(), (center_x - 15, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2, cv2.LINE_AA)
    
    # Blend the overlay with the original frame
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return frame

def draw_event_marker(frame, point, event_type):
    try:
        point_int = tuple(map(int, point))
    except (TypeError, ValueError):
        return

    if event_type == "ENTRY": 
        cv2.circle(frame, point_int, 8, COLOR_GREEN, -1)
        cv2.circle(frame, point_int, 8, COLOR_BLACK, 1)
    elif event_type == "EXIT": 
        cv2.drawMarker(frame, point_int, COLOR_RED, cv2.MARKER_CROSS, 15, 2)
    # FIX: Added handling for RE-ENTRY event type
    elif event_type == "RE-ENTRY":
        cv2.circle(frame, point_int, 8, COLOR_YELLOW, -1)
        cv2.circle(frame, point_int, 8, COLOR_BLACK, 1)
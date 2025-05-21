# project/visualization/overlay.py
import cv2
import numpy as np
from core.utils import get_display_direction, debug_print

# Define colors (BGR)
COLOR_GREEN = (0, 255, 0); COLOR_RED = (0, 0, 255); COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255); COLOR_MAGENTA = (255, 0, 255); COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0); COLOR_ORANGE = (0, 165, 255); COLOR_CYAN = (255, 255, 0)
ZONE_COLORS = {'north':COLOR_BLUE, 'south':COLOR_GREEN, 'east':COLOR_RED, 'west':COLOR_MAGENTA}
DEFAULT_ZONE_COLOR = COLOR_WHITE; TEXT_BG_ALPHA = 0.7

def draw_detection_box(frame, box_coords, vehicle_id, vehicle_type, status="detected", entry_dir=None, time_active=None):
    if not (isinstance(box_coords, (tuple, list)) and len(box_coords) == 4):
         debug_print(f"draw_detection_box: Invalid box_coords: {box_coords}")
         return

    x1, y1, x2, y2 = map(int, box_coords)

    current_box_color = COLOR_GREEN # Default
    if status == 'active': current_box_color = COLOR_ORANGE
    elif status == 'exiting': current_box_color = COLOR_RED
    elif status == 'entering': current_box_color = COLOR_CYAN
    # Ensure current_box_color is always a valid BGR tuple
    if not (isinstance(current_box_color, tuple) and len(current_box_color) == 3):
        debug_print(f"draw_detection_box: Invalid current_box_color derived from status '{status}'. Using default.")
        current_box_color = COLOR_GREEN 

    cv2.rectangle(frame, (x1, y1), (x2, y2), current_box_color, 2)

    label = f"ID:{vehicle_id[-4:]} {vehicle_type}"; info_lines = [label]
    if entry_dir: info_lines.append(f"From:{get_display_direction(entry_dir)}")
    if time_active is not None: info_lines.append(f"T:{time_active:.1f}s")

    text_y_start = y1 - 7 # Adjusted starting y for text background slightly higher
    
    # Calculate text size for background
    text_sizes = []
    total_text_height_for_bg = 0
    max_text_width_for_bg = 0
    font_scale = 0.5
    thickness = 1 # For calculating text size
    
    for i, line in enumerate(info_lines):
        (w, h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_sizes.append({'w': w, 'h': h, 'b': baseline})
        max_text_width_for_bg = max(max_text_width_for_bg, w)
        total_text_height_for_bg += h + (5 if i > 0 else 0) # Add inter-line spacing for bg
    
    if not info_lines: return # No text to draw

    # Define background rectangle coordinates
    bg_x1 = x1
    bg_x2 = x1 + max_text_width_for_bg + 10 # Add some padding
    
    # Adjusted y calculation for background to be more robust
    # Start from text_y_start and go down by total_text_height_for_bg
    # Pad a bit around
    bg_h_padding = 5 
    bg_y2 = max(0, text_y_start) # text_y_start can be negative if y1 is small
    bg_y1 = max(0, bg_y2 - total_text_height_for_bg - (len(info_lines)-1)*2 - 2*bg_h_padding) # Approx height for N lines + padding
                                                                                         # This was complex, simplifying:
    
    # Simpler background box calculation:
    # Let's find the top-most point for text based on the first line's height and baseline
    # and the bottom-most based on the sum of heights.
    
    # If y1 is very small, text_y_start might be negative.
    # We want the text to appear *above* the box.
    # The first line of text baseline will be at y1 - some_offset.
    # The top of the background should be above that.
    
    padding = 3 # Padding around text for the background
    text_block_height = sum(ts['h'] for ts in text_sizes) + (len(text_sizes) - 1) * 5 # 5 is inter-line spacing
    
    bg_y1_calc = y1 - text_block_height - padding - len(text_sizes)*2 # Estimate based on y1 of box
    bg_y2_calc = y1 - padding 

    # Ensure within frame boundaries and valid
    bg_y1_final = max(0, bg_y1_calc)
    bg_y2_final = max(bg_y1_final, min(y1, bg_y2_calc)) # Ensure bg_y2 is above or at bg_y1, and not below y1
                                                    # and does not exceed y1 of the box.
    bg_x1_final = x1
    bg_x2_final = x1 + max_text_width_for_bg + 2 * padding

    # Clamp to frame boundaries
    bg_y1_final = max(0, bg_y1_final)
    bg_y2_final = min(frame.shape[0], bg_y2_final)
    bg_x1_final = max(0, bg_x1_final)
    bg_x2_final = min(frame.shape[1], bg_x2_final)

    if bg_y1_final >= bg_y2_final or bg_x1_final >= bg_x2_final:
        # If background box is invalid, try to at least draw text without background
        # debug_print(f"draw_detection_box: Invalid background slice. y1,y2=({bg_y1_final},{bg_y2_final}), x1,x2=({bg_x1_final},{bg_x2_final})")
        current_text_y = y1 - 7 
        for i, line in enumerate(info_lines):
            text_origin = (x1 + padding, current_text_y)
            cv2.putText(frame, line, text_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_BLACK, 3, cv2.LINE_AA) # Thicker outline
            cv2.putText(frame, line, text_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_WHITE, 1, cv2.LINE_AA)
            current_text_y -= text_sizes[i]['h'] + 5 # Move up for next line
        return

    try:
        sub_img = frame[bg_y1_final:bg_y2_final, bg_x1_final:bg_x2_final]
        
        # Ensure sub_img is not empty before proceeding
        if sub_img.size == 0:
            # debug_print(f"draw_detection_box: sub_img is empty. Slice: y({bg_y1_final}:{bg_y2_final}), x({bg_x1_final}:{bg_x2_final})")
            # Fallback to drawing text without background (same as above)
            current_text_y = y1 - 7 
            for i, line in enumerate(info_lines):
                text_origin = (x1 + padding, current_text_y)
                cv2.putText(frame, line, text_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_BLACK, 3, cv2.LINE_AA)
                cv2.putText(frame, line, text_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_WHITE, 1, cv2.LINE_AA)
                current_text_y -= text_sizes[i]['h'] + 5
            return

        bg_rect_color_np = np.array(current_box_color, dtype=np.uint8)
        background_rect_array = np.full(sub_img.shape, bg_rect_color_np, dtype=np.uint8)
        
        res = cv2.addWeighted(sub_img, 1.0 - TEXT_BG_ALPHA, background_rect_array, TEXT_BG_ALPHA, 0.0)
        
        frame[bg_y1_final:bg_y2_final, bg_x1_final:bg_x2_final] = res
    except Exception as e_bg:
        debug_print(f"Error drawing text background: {e_bg}. Slice y({bg_y1_final}:{bg_y2_final}), x({bg_x1_final}:{bg_x2_final}), sub_img shape {sub_img.shape if 'sub_img' in locals() else 'N/A'}")
        # Fallback to drawing text without background if error occurs
        current_text_y = y1 - 7 
        for i, line in enumerate(info_lines):
            text_origin = (x1 + padding, current_text_y)
            cv2.putText(frame, line, text_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_BLACK, 3, cv2.LINE_AA)
            cv2.putText(frame, line, text_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_WHITE, 1, cv2.LINE_AA)
            current_text_y -= text_sizes[i]['h'] + 5
        return


    # Draw text on top of the blended background
    # Calculate text positions relative to the background box's top-left (bg_x1_final, bg_y1_final)
    # The text should be placed from bottom-up if y1 is low, or top-down if y1 is high.
    # Let's aim for top-down text placement within the bg box for simplicity.
    current_text_y_offset = padding + text_sizes[0]['h'] # Baseline of first line from top of bg
    for i, line in enumerate(info_lines):
        text_origin = (bg_x1_final + padding, bg_y1_final + current_text_y_offset)
        cv2.putText(frame, line, text_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_BLACK, 3, cv2.LINE_AA) # Thicker outline for contrast
        cv2.putText(frame, line, text_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_WHITE, 1, cv2.LINE_AA)
        if i < len(info_lines) - 1:
            current_text_y_offset += text_sizes[i+1]['h'] + 5 # Move down for next line's baseline


# ... (rest of overlay.py: draw_tracking_trail, draw_zones, draw_event_marker, add_status_overlay - UNCHANGED from previous correct version) ...
def draw_tracking_trail(frame, trail_points, vehicle_id): # UNCHANGED
    if len(trail_points) < 2: return
    try:
        color_seed = int(vehicle_id.split('_')[-1]) if '_' in vehicle_id else hash(vehicle_id)
        np.random.seed(color_seed % (2**32 - 1)) # Ensure seed is within uint32 range for np.random.seed
        color = tuple(np.random.randint(50, 200, size=3).tolist()) # BGR
    except ValueError: color = COLOR_YELLOW
    pts = np.array(trail_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)

def draw_zones(frame, zone_tracker): # UNCHANGED
    if not hasattr(zone_tracker, 'zones') or not zone_tracker.zones:
        return frame
    overlay = frame.copy(); alpha = 0.2
    for direction, polygon_points in zone_tracker.zones.items():
        if not isinstance(polygon_points, np.ndarray) or polygon_points.ndim != 2 or polygon_points.shape[0] < 3:
            debug_print(f"draw_zones: Skipping invalid polygon data for direction '{direction}'. Shape: {getattr(polygon_points, 'shape', 'N/A')}")
            continue
        if polygon_points.dtype != np.int32:
            polygon_points = polygon_points.astype(np.int32)
        color = ZONE_COLORS.get(direction, DEFAULT_ZONE_COLOR)
        try:
            cv2.fillPoly(overlay, [polygon_points], color)
            cv2.polylines(frame, [polygon_points], isClosed=True, color=color, thickness=2)
            M = cv2.moments(polygon_points)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"]); center_y = int(M["m01"] / M["m00"])
                cv2.putText(frame, direction.upper(), (center_x - 20, center_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        except Exception as e: print(f"ERROR drawing zone for direction '{direction}': {e}")
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

def draw_event_marker(frame, point, event_type, vehicle_id_short): # UNCHANGED
    point_int = tuple(map(int, point)); text = f"{event_type[0]}:{vehicle_id_short}"
    if event_type == "ENTRY": color = COLOR_GREEN; cv2.circle(frame, point_int, 10, color, -1)
    elif event_type == "EXIT": color = COLOR_RED; cv2.drawMarker(frame, point_int, color, cv2.MARKER_CROSS, 15, 2)
    else: return
    cv2.putText(frame, text, (point_int[0]+12, point_int[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

def add_status_overlay(frame, frame_number, timestamp, tracker_info): # UNCHANGED from previous fix
    h, w, _ = frame.shape; overlay_h = 60;
    sub_img = frame[0:overlay_h, 0:w]; black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 1.0 - TEXT_BG_ALPHA, black_rect, TEXT_BG_ALPHA, 1.0); frame[0:overlay_h, 0:w] = res
    time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    active_count = 0
    if isinstance(tracker_info, dict) and 'get_active_vehicle_count' in tracker_info:
        active_count_func = tracker_info['get_active_vehicle_count']
        if callable(active_count_func): active_count = active_count_func()
    elif hasattr(tracker_info, 'get_active_vehicle_count') and callable(tracker_info.get_active_vehicle_count):
        active_count = tracker_info.get_active_vehicle_count()
    cv2.putText(frame, f"Frame:{frame_number}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Time:{time_str}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Active:{active_count}", (w-180, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
    return frame
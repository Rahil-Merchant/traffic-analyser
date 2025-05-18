# project/core/video_processor.py
import cv2
import numpy as np
import torch
import time
import os
import threading
import queue
import gc
import math
from datetime import datetime, timedelta
import concurrent.futures

# Configuration
from config import (
    TARGET_FPS, FRAME_WIDTH, FRAME_HEIGHT, OPTIMAL_BATCH_SIZE,
    VIDEO_OUTPUT_DIR, ENABLE_DETAILED_PERFORMANCE_METRICS,
    MIXED_PRECISION, PARALLEL_STREAMS, THREAD_COUNT,
    MEMORY_CLEANUP_INTERVAL, MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD, 
    LINE_MODE, LINE_POINTS, MODEL_INPUT_SIZE, ENABLE_VISUALIZATION,
    USE_ADVANCED_VIDEO_READER, PYVIDEOREADER_DECODE_SEGMENT_SIZE, FFMPEG_PATH
)

# Modules
from core.utils import debug_print, format_timestamp, is_valid_movement, cleanup_memory
from core.performance import PerformanceTracker
from models.model_loader import load_model, get_device
from tracking.vehicle_tracker import ByteTrackWrapper 
from tracking.zone_tracker import ZoneTracker
from tracking.cleanup import cleanup_tracking_data 
from visualization import overlay as visual_overlay
from reporting.excel_report import create_excel_report
from core.converters import convert_dav_to_mp4_internal

# Conditional imports
if ENABLE_DETAILED_PERFORMANCE_METRICS:
    from visualization.performance_viz import visualize_performance

PYVIDEOREADER_AVAILABLE = False
PyVideoReader = None
try:
    from video_reader import PyVideoReader
    PYVIDEOREADER_AVAILABLE = True
    print("PyVideoReader (video_reader-rs) found and available.")
except ImportError:
    print("PyVideoReader (video_reader-rs) not found. Advanced video reader will not be used.")


class VideoProcessor:
    def __init__(self):
        self.target_fps = TARGET_FPS
        self.frame_shape = (FRAME_WIDTH, FRAME_HEIGHT) # W, H
        self.model_input_size_config = MODEL_INPUT_SIZE
        self.batch_size = OPTIMAL_BATCH_SIZE
        self.processing_lock = threading.Lock()
        self.line_mode = LINE_MODE
        self.max_workers = THREAD_COUNT

        self.gradio_polygons = {}
        self.current_gradio_direction = None
        self.gradio_temp_points = []
        self.original_frame_for_gradio = None
        self.last_drawn_polygons_frame = None

        self.device = get_device()
        self.model = None 
        self.vehicle_tracker = None 
        self.zone_tracker = None
        self.perf = None
        self.detection_zones_polygons = None
        self.tracker_lock = threading.Lock()

        self.frame_read_queue = None
        self.video_write_queue = None
        self.stop_event = None
        self.reader_thread = None
        self.writer_thread = None
        self.last_cleanup_time = 0
        self.final_output_path = None

        self.streams = None
        if self.device == 'cuda' and PARALLEL_STREAMS > 1:
            self.streams = [torch.cuda.Stream() for _ in range(PARALLEL_STREAMS)]
        self.current_stream_index = 0

        self.pvr_decode_segment_size = PYVIDEOREADER_DECODE_SEGMENT_SIZE
        self.actually_use_advanced_reader = USE_ADVANCED_VIDEO_READER and PYVIDEOREADER_AVAILABLE
        if self.actually_use_advanced_reader: print("Configuration: Advanced video reader (PyVideoReader) will be used.")
        else:
            if USE_ADVANCED_VIDEO_READER and not PYVIDEOREADER_AVAILABLE: print("Configuration: Advanced video reader was requested but is not available. Using OpenCV.")
            else: print("Configuration: Using OpenCV video reader.")
    
    # --- Gradio Polygon Related Methods ---
    def _draw_polygons_for_gradio(self, frame_to_draw_on):
        if frame_to_draw_on is None: return None
        img_copy = frame_to_draw_on.copy()
        colors = {'north': (255,0,0), 'south': (0,255,0), 'east': (0,0,255), 'west': (255,0,255)} 
        default_color = (255, 255, 255); temp_color = (0, 255, 255)
        for direction, points in self.gradio_polygons.items():
            if points and len(points) >= 3:
                pts_np = np.array(points, dtype=np.int32)
                color = colors.get(direction, default_color)
                cv2.polylines(img_copy, [pts_np], isClosed=True, color=color, thickness=2)
                try: # Add label
                    if len(pts_np) > 0: M = cv2.moments(pts_np)
                    if M["m00"] != 0: cX = int(M["m10"] / M["m00"]); cY = int(M["m01"] / M["m00"]); cv2.putText(img_copy, direction.upper(), (cX-15, cY+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except Exception: pass # Ignore label errors
        if self.current_gradio_direction and self.gradio_temp_points:
            current_temp_color = colors.get(self.current_gradio_direction, temp_color)
            for i, pt in enumerate(self.gradio_temp_points):
                cv2.circle(img_copy, pt, 5, current_temp_color, -1)
                if i > 0: cv2.line(img_copy, self.gradio_temp_points[i-1], pt, current_temp_color, 2)
            if len(self.gradio_temp_points) >= 2: # Closing hint line
                last_pt = self.gradio_temp_points[-1]; first_pt = self.gradio_temp_points[0]
                hint_color = tuple(c // 2 for c in current_temp_color)
                cv2.line(img_copy, last_pt, first_pt, hint_color, 1, cv2.LINE_AA)
        self.last_drawn_polygons_frame = img_copy
        return img_copy
    def process_video_upload_for_gradio(self, video_path):
        if self.line_mode != 'interactive': return None, "Zone definition mode is 'hardcoded'. Drawing disabled."
        self.reset_gradio_polygons()
        if video_path is None: return None, "Please upload a video."
        try:
            cap=cv2.VideoCapture(video_path); ret,frame=cap.read(); cap.release()
            if ret:
                frame = cv2.resize(frame, self.frame_shape)
                self.original_frame_for_gradio = frame.copy()
                drawn_frame = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
                return drawn_frame, "Video loaded. Select direction and click image to define zone vertices."
            else: return None, "Error: Could not read first frame."
        except Exception as e: return None, f"Error loading video: {e}"
    def set_gradio_polygon_direction(self, direction):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        direction_lower = direction.lower()
        if direction_lower in self.gradio_polygons: del self.gradio_polygons[direction_lower]; print(f"Cleared existing polygon for {direction.capitalize()} to redraw.")
        self.current_gradio_direction = direction_lower; self.gradio_temp_points = []
        status = f"Selected '{direction.capitalize()}'. Click points on image to define zone polygon."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
        return frame_to_show, status
    def handle_gradio_polygon_click(self, evt: 'gr.SelectData'): # type: ignore
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if self.current_gradio_direction is None: return self.last_drawn_polygons_frame, "Select a direction first."
        if self.original_frame_for_gradio is None: return None, "No video loaded."
        point = (evt.index[0], evt.index[1]); self.gradio_temp_points.append(point)
        status = f"Added point {len(self.gradio_temp_points)} for {self.current_gradio_direction.capitalize()}. Click more points or 'Finish Polygon'."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
        return frame_to_show, status
    def finalize_current_polygon(self):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if not self.current_gradio_direction: return self.last_drawn_polygons_frame, "No direction selected to finalize."
        if len(self.gradio_temp_points) < 3: status = f"Error: Need at least 3 points for a polygon (got {len(self.gradio_temp_points)})."; frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio); return frame_to_show, status
        self.gradio_polygons[self.current_gradio_direction] = self.gradio_temp_points.copy()
        status = f"Polygon for {self.current_gradio_direction.capitalize()} saved. Select next direction or Process."
        self.current_gradio_direction = None; self.gradio_temp_points = []
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
        return frame_to_show, status
    def undo_last_polygon_point(self):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if not self.current_gradio_direction: status = "Select a direction first."
        elif not self.gradio_temp_points: status = f"No points added yet for {self.current_gradio_direction.capitalize()}."
        else: self.gradio_temp_points.pop(); status = f"Removed last point for {self.current_gradio_direction.capitalize()}."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
        return frame_to_show, status
    def reset_gradio_polygons(self):
        self.gradio_polygons = {}; self.current_gradio_direction = None
        self.gradio_temp_points = []; status = "All polygons cleared."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio) if self.original_frame_for_gradio is not None else None
        self.last_drawn_polygons_frame = frame_to_show
        return frame_to_show, status

    # --- Video Reading/Writing Threads ---
    def _get_video_properties(self, video_path):
        total_frames = 0; original_fps = 0.0; temp_reader_rs = None
        if self.actually_use_advanced_reader:
            try:
                temp_reader_rs = PyVideoReader(video_path); info = temp_reader_rs.get_info()
                total_frames = int(info.get('frame_count', 0)); fps_str = info.get('fps', '0.0')
                if fps_str:
                    try: original_fps = float(fps_str)
                    except ValueError: pass
            except Exception as e: debug_print(f"PyVideoReader property fetch failed: {e}. Falling back to OpenCV.")
            finally:
                if temp_reader_rs: del temp_reader_rs
        if total_frames <= 0 or original_fps <= 0:
            cap_check = None
            try:
                cap_check = cv2.VideoCapture(video_path)
                if not cap_check.isOpened(): raise FileNotFoundError(f"OpenCV: Cannot open: {video_path}")
                _total_frames_cv = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT)); _original_fps_cv = cap_check.get(cv2.CAP_PROP_FPS)
                if total_frames <= 0: total_frames = _total_frames_cv
                if original_fps <= 0: original_fps = _original_fps_cv
            except Exception as e: print(f"Error getting video properties with OpenCV: {e}")
            finally:
                if cap_check: cap_check.release()
        if original_fps <= 0: original_fps = 30.0
        if total_frames <= 0: print("WARNING: Could not determine total_frames.")   
        return total_frames, original_fps
    def _frame_reader_task(self, video_path, original_fps):
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): print("ERROR: Reader failed to open video."); self.frame_read_queue.put(None); return
            frame_interval = max(1, round(original_fps / self.target_fps)) if self.target_fps > 0 else 1
            frame_count_raw = 0; frames_yielded = 0
            while not self.stop_event.is_set():
                ret, frame_bgr_orig_size = cap.read()
                if not ret: debug_print("OpenCV Reader: End of video/read error."); break
                if frame_count_raw % frame_interval == 0:
                    frame_time_seconds = frame_count_raw / original_fps if original_fps > 0 else 0
                    # Resize to display resolution here
                    frame_bgr_display_res = cv2.resize(frame_bgr_orig_size, self.frame_shape, interpolation=cv2.INTER_LINEAR)
                    metadata = {'is_rgb': False, 'source': 'opencv', 'display_size_hw': self.frame_shape[::-1]} # H, W
                    try:
                        self.frame_read_queue.put((frame_bgr_display_res, frame_count_raw, frame_time_seconds, metadata), block=True, timeout=5.0); frames_yielded += 1
                    except queue.Full: # ... (rest of queue full handling)
                        debug_print("Reader queue full, waiting..."); time.sleep(0.1)
                        if self.stop_event.is_set(): break
                        try: self.frame_read_queue.put((frame_bgr_display_res, frame_count_raw, frame_time_seconds, metadata), block=True, timeout=5.0); frames_yielded += 1
                        except queue.Full: print("ERROR: Reader queue persistently full."); self.stop_event.set(); break
                    except Exception as q_e: debug_print(f"OpenCV Reader: Queue put error: {q_e}"); self.stop_event.set(); break
                frame_count_raw += 1
            debug_print(f"OpenCV Reader finished. Raw: {frame_count_raw}. Yielded: {frames_yielded}.")
        except Exception as e: print(f"FATAL ERROR in _frame_reader_task (OpenCV): {e}"); import traceback; traceback.print_exc()
        finally: # ... (rest of finally block)
            if cap: cap.release()
            if self.frame_read_queue:
                try: self.frame_read_queue.put(None, block=False, timeout=1.0) 
                except queue.Full: debug_print("OpenCV Reader: Queue full on final None put.")
                except Exception as e_fin: debug_print(f"OpenCV Reader: Error on final None put: {e_fin}")
    def _frame_reader_task_pvr_segmented(self, video_path, original_fps_from_props, total_frames_from_props):
        pvr_instance = None
        try:
            target_h, target_w = self.frame_shape[1], self.frame_shape[0]
            debug_print(f"PVR SegReader: Init: device='cuda', target_resize=({target_w}x{target_h})")
            pvr_instance = PyVideoReader(video_path, device='cuda', resize_shorter_side=min(target_h, target_w), resize_longer_side=max(target_h, target_w))   
            frame_interval = max(1, round(original_fps_from_props / self.target_fps)) if self.target_fps > 0 else 1
            queued_frame_count = 0
            num_segments = math.ceil(total_frames_from_props / self.pvr_decode_segment_size) if total_frames_from_props > 0 and self.pvr_decode_segment_size > 0 else 0
            if total_frames_from_props == 0 : num_segments = 0
            for i in range(num_segments): # ... (segment loop)
                if self.stop_event.is_set(): break
                segment_start_raw_idx = i * self.pvr_decode_segment_size
                segment_end_raw_idx = min((i + 1) * self.pvr_decode_segment_size, total_frames_from_props)
                if segment_start_raw_idx >= segment_end_raw_idx : continue
                segment_frames_np_rgb = pvr_instance.decode(start_frame=segment_start_raw_idx, end_frame=segment_end_raw_idx)
                if segment_frames_np_rgb is None or segment_frames_np_rgb.shape[0] == 0: continue
                for frame_in_segment_idx in range(segment_frames_np_rgb.shape[0]): # ... (frame loop in segment)
                    if self.stop_event.is_set(): break
                    current_overall_raw_frame_idx = segment_start_raw_idx + frame_in_segment_idx
                    if current_overall_raw_frame_idx % frame_interval == 0:
                        frame_content_rgb_display_res = segment_frames_np_rgb[frame_in_segment_idx] 
                        if frame_content_rgb_display_res.shape[0] != target_h or frame_content_rgb_display_res.shape[1] != target_w: # Ensure exact target size
                             frame_content_rgb_display_res = cv2.resize(frame_content_rgb_display_res, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        current_frame_time_sec = current_overall_raw_frame_idx / original_fps_from_props if original_fps_from_props > 0 else 0
                        metadata = {'is_rgb': True, 'source': 'pvr_cuda_segmented', 'display_size_hw': (target_h, target_w)}
                        try: # ... (queue put logic)
                            self.frame_read_queue.put((frame_content_rgb_display_res, current_overall_raw_frame_idx, current_frame_time_sec, metadata), block=True, timeout=5.0); queued_frame_count+=1
                        except queue.Full:
                            debug_print("PVR SegReader: Queue full, retrying..."); time.sleep(0.1);
                            if self.stop_event.is_set(): break
                            try: self.frame_read_queue.put((frame_content_rgb_display_res, current_overall_raw_frame_idx, current_frame_time_sec, metadata), block=True, timeout=5.0); queued_frame_count+=1
                            except queue.Full: print("ERROR: PVR SegReader queue persistently full."); self.stop_event.set(); break
                        except Exception as q_e: debug_print(f"PVR SegReader queue put error: {q_e}"); self.stop_event.set(); break
                del segment_frames_np_rgb; gc.collect()
            debug_print(f"PVR SegReader finished. Queued {queued_frame_count} frames.")
        except Exception as e: print(f"FATAL ERROR in _frame_reader_task_pvr_segmented: {e}"); import traceback; traceback.print_exc()
        finally: # ... (rest of finally)
            if pvr_instance: del pvr_instance
            if self.frame_read_queue:
                try: self.frame_read_queue.put(None, block=False, timeout=1.0)
                except queue.Full: pass
    def _frame_writer_task(self, output_path): # Unchanged
        out = None; writer_initialized = False; frames_written = 0; actual_output_path = output_path
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v'); out = cv2.VideoWriter(actual_output_path, fourcc, self.target_fps, self.frame_shape)
            if not out.isOpened():
                print(f"Warn: mp4v failed. Trying XVID/AVI."); avi_path = os.path.splitext(actual_output_path)[0] + ".avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID'); out = cv2.VideoWriter(avi_path, fourcc, self.target_fps, self.frame_shape)
                if out.isOpened(): actual_output_path = avi_path; self.final_output_path = avi_path
                else: print(f"ERROR: Writer failed to open."); self.final_output_path = "ERROR_WRITER_FAILED"; return
            writer_initialized = True; print(f"Writer initialized for: {actual_output_path}")
            while True:
                try: frame_data = self.video_write_queue.get(timeout=5.0)
                except queue.Empty:
                    if self.stop_event is not None and self.stop_event.is_set() and self.video_write_queue.empty(): break
                    continue
                if frame_data is None: break
                frame = frame_data # This is already BGR, display_resolution
                if frame is not None and isinstance(frame, np.ndarray) and frame.shape[0] > 0:
                    if frame.dtype != np.uint8: frame = frame.astype(np.uint8)
                    # Should already be correct shape and BGR from _process_inference_results
                    out.write(frame); frames_written += 1
                if self.video_write_queue: self.video_write_queue.task_done()
        except Exception as e: print(f"ERROR in writer thread: {e}"); # ... (error handling)
        finally: # ... (release writer)
            if out and writer_initialized: out.release(); debug_print(f"Writer released. Wrote {frames_written} frames to {actual_output_path}")
            if self.video_write_queue:
                try: self.video_write_queue.put(None, timeout=1.0)
                except queue.Full: pass

    # --- Core Processing Logic ---
    @torch.no_grad()
    def _preprocess_batch(self, frames_batch_content_metadata): # Renamed for clarity
        # frames_batch_content_metadata is list of (frame_content_at_display_res, metadata_dict)
        # frame_content_at_display_res can be RGB or BGR based on metadata['is_rgb']
        preprocessed_tensors_list = []
        target_dim_for_model = self.model_input_size_config

        for frame_content_display_res, metadata in frames_batch_content_metadata:
            if frame_content_display_res is None: continue
            
            img_rgb_display_sized = frame_content_display_res
            if not metadata.get('is_rgb', False): # If BGR, convert to RGB
                img_rgb_display_sized = cv2.cvtColor(frame_content_display_res, cv2.COLOR_BGR2RGB)

            # Resize from display resolution to model input size
            img_rgb_model_sized = cv2.resize(img_rgb_display_sized,
                                             (target_dim_for_model, target_dim_for_model),
                                             interpolation=cv2.INTER_LINEAR)
            
            if img_rgb_model_sized.ndim == 2: img_rgb_model_sized = cv2.cvtColor(img_rgb_model_sized, cv2.COLOR_GRAY2RGB)
            elif img_rgb_model_sized.shape[2] == 4: img_rgb_model_sized = cv2.cvtColor(img_rgb_model_sized, cv2.COLOR_RGBA2RGB)
            if img_rgb_model_sized.shape[2] != 3: continue # Should be RGB

            img_chw = img_rgb_model_sized.transpose(2, 0, 1) # HWC to CHW
            img_chw_contiguous = np.ascontiguousarray(img_chw)
            preprocessed_tensors_list.append(img_chw_contiguous)

        if not preprocessed_tensors_list: return None
        try: batch_np_stacked = np.stack(preprocessed_tensors_list)
        except ValueError as e: debug_print(f"Preprocess: Error stacking tensors: {e}"); return None
        
        batch_tensor_from_np = torch.from_numpy(batch_np_stacked).to(self.device)
        if batch_tensor_from_np.dtype != torch.uint8: batch_tensor_from_np = torch.clamp(batch_tensor_from_np, 0, 255).to(torch.uint8)
        dtype_final = torch.float16 if (MIXED_PRECISION and self.device == 'cuda') else torch.float32
        final_batch_tensor = batch_tensor_from_np.to(dtype_final) / 255.0
        return final_batch_tensor

    def _process_inference_results(self, yolo_results_list_for_batch, frames_batch_content_metadata_list, frame_numbers_batch, frame_times_batch, start_datetime_obj):
        # yolo_results_list_for_batch: list of ultralytics.engine.results.Results objects
        # frames_batch_content_metadata_list: list of (original_frame_at_display_res, metadata_dict)
        batch_event_logs = []
        processed_frames_for_viz = [None] * len(yolo_results_list_for_batch)

        for i in range(len(yolo_results_list_for_batch)):
            yolo_result_single_frame = yolo_results_list_for_batch[i] # This is ONE Results object
            original_frame_content_display_res, frame_metadata = frames_batch_content_metadata_list[i]
            frame_num = frame_numbers_batch[i]
            frame_time_video_sec = frame_times_batch[i]
            current_timestamp_dt = start_datetime_obj + timedelta(seconds=frame_time_video_sec)

            # The yolo_result_single_frame.orig_shape is the model input size (e.g., 416x416)
            # The yolo_result_single_frame.boxes.xyxy are also in model input space.
            model_input_shape_hw = yolo_result_single_frame.orig_shape 

            if self.perf: self.perf.start_timer('tracking_update')
            # ByteTrackWrapper.process_detections now takes the YOLO Results object for the frame.
            # It handles calling byte_tracker.update() and scaling track boxes to display resolution.
            online_tracks_display_scaled = self.vehicle_tracker.process_detections(
                yolo_result_single_frame, # Pass the Results object
                model_input_shape_hw,     # Shape of image fed to model
                frame_time_video_sec
            )
            if self.perf: self.perf.end_timer('tracking_update')
            
            # online_tracks_display_scaled is list of dicts:
            # {'id': track_id, 'box_coords_display_scale': (x1,y1,x2,y2), 
            #  'center_display_scale': (cx,cy), 'type_raw_name': str, 'conf': float}
            # All coordinates are now in display_resolution space.

            frame_local_event_log = []
            current_frame_visible_vehicle_map = {} # For drawing and zone logic

            with self.tracker_lock: # Protects self.vehicle_tracker's app-level state
                for track_data_display in online_tracks_display_scaled:
                    track_id = track_data_display['id']
                    consistent_type = self.vehicle_tracker.get_consistent_type(track_id, track_data_display['type_raw_name'])
                    
                    current_frame_visible_vehicle_map[track_id] = {
                        'center': track_data_display['center_display_scale'],
                        'box': track_data_display['box_coords_display_scale'],
                        'type': consistent_type, 
                        'event_marker': None,
                        'status_for_drawing': 'detected' 
                    }

                    # prev_center_pos from vehicle_tracker is already in display scale
                    prev_center_pos_display = self.vehicle_tracker.get_last_position(track_id) 
                    current_bbox_display = track_data_display['box_coords_display_scale']

                    if prev_center_pos_display and self.zone_tracker and self.zone_tracker.zones:
                        if self.perf: self.perf.start_timer('zone_checking')
                        event_type, event_dir = self.zone_tracker.check_zone_transition(
                            prev_center_pos_display, current_bbox_display, track_id, frame_time_video_sec
                        )
                        if self.perf: self.perf.end_timer('zone_checking')

                        if event_type: 
                            vehicle_app_state = self.vehicle_tracker.active_vehicles.get(track_id)
                            if event_type == "ENTRY":
                                if self.vehicle_tracker.register_entry(track_id, event_dir, current_timestamp_dt, track_data_display['center_display_scale'], consistent_type):
                                    if self.perf: self.perf.record_vehicle_entry()
                                    frame_local_event_log.append({
                                        'timestamp_dt': current_timestamp_dt, 'vehicle_id': track_id,
                                        'vehicle_type': consistent_type, 'event_type': 'ENTRY',
                                        'direction_from': event_dir, 'direction_to': None,
                                        'status': 'entry', 'frame_number': frame_num
                                    })
                                    current_frame_visible_vehicle_map[track_id]['event_marker'] = 'ENTRY'
                                    current_frame_visible_vehicle_map[track_id]['status_for_drawing'] = 'entering'
                            elif event_type == "EXIT":
                                if vehicle_app_state and vehicle_app_state['status'] == 'active':
                                    entry_dir_of_vehicle = vehicle_app_state['entry_direction']
                                    active_zone_keys = list(self.zone_tracker.zones.keys())
                                    if is_valid_movement(entry_dir_of_vehicle, event_dir, active_zone_keys):
                                        success_exit, t_in_int = self.vehicle_tracker.register_exit(
                                            track_id, event_dir, current_timestamp_dt, track_data_display['center_display_scale']
                                        )
                                        if success_exit: # ... (log exit event)
                                            if self.perf: self.perf.record_vehicle_exit('exited', t_in_int)
                                            completed_path = next((p for p in reversed(self.vehicle_tracker.completed_paths) if p.get('id') == track_id), None)
                                            log_type = consistent_type
                                            if completed_path and completed_path.get('type'): log_type = completed_path['type']
                                            frame_local_event_log.append({
                                                'timestamp_dt': current_timestamp_dt, 'vehicle_id': track_id,
                                                'vehicle_type': log_type, 'event_type': 'EXIT',
                                                'direction_from': entry_dir_of_vehicle, 'direction_to': event_dir,
                                                'status': 'exit', 'time_in_intersection': t_in_int, 'frame_number': frame_num
                                            })
                                            current_frame_visible_vehicle_map[track_id]['event_marker'] = 'EXIT'
                                            current_frame_visible_vehicle_map[track_id]['status_for_drawing'] = 'exiting'
                
                timed_out_ids = self.vehicle_tracker.check_timeouts(current_timestamp_dt)
                for v_id_to in timed_out_ids: # ... (log timeout event)
                    if self.perf: self.perf.record_vehicle_exit('timed_out')
                    p_data = next((p for p in reversed(self.vehicle_tracker.completed_paths) if p['id'] == v_id_to and p['status'] == 'timed_out'), None)
                    if p_data:
                        frame_local_event_log.append({
                            'timestamp_dt': current_timestamp_dt, 'vehicle_id': v_id_to,
                            'vehicle_type': p_data.get('type', 'UnknownType'), 'event_type': 'TIMEOUT',
                            'direction_from': p_data.get('entry_direction', 'UNKNOWN'), 'direction_to': 'TIMEOUT',
                            'status': 'timeout', 'time_in_intersection': p_data.get('time_in_intersection', 'N/A'),
                            'frame_number': frame_num
                        })
                    if v_id_to in current_frame_visible_vehicle_map: 
                        current_frame_visible_vehicle_map[v_id_to]['status_for_drawing'] = 'timed_out'

            batch_event_logs.extend(frame_local_event_log)

            if ENABLE_VISUALIZATION: # Drawing logic
                if self.perf: self.perf.start_timer('drawing')
                # original_frame_content_display_res is already at display resolution
                output_frame_viz = original_frame_content_display_res.copy()
                if frame_metadata.get('is_rgb', False): # If it was RGB (e.g. from PVR)
                    output_frame_viz = cv2.cvtColor(output_frame_viz, cv2.COLOR_RGB2BGR)
                # Else, it's BGR (e.g. from OpenCV), ready for drawing

                if self.zone_tracker and self.zone_tracker.zones:
                    output_frame_viz = visual_overlay.draw_zones(output_frame_viz, self.zone_tracker)

                for track_id_draw, data_draw in current_frame_visible_vehicle_map.items():
                    if track_id_draw in self.vehicle_tracker.completed_vehicle_ids and \
                       data_draw.get('status_for_drawing') not in ['exiting', 'timed_out', 'entering']:
                        continue 
                    center_pt_draw, box_draw, type_draw = data_draw['center'], data_draw['box'], data_draw['type']
                    draw_status = data_draw.get('status_for_drawing', 'detected')
                    entry_dir_draw = None; time_active_draw = None
                    active_vehicle_app_info = self.vehicle_tracker.active_vehicles.get(track_id_draw)
                    if active_vehicle_app_info and active_vehicle_app_info.get('status') == 'active':
                        if draw_status not in ['entering', 'exiting', 'timed_out']: draw_status = 'active'
                        entry_dir_draw = active_vehicle_app_info.get('entry_direction')
                        if active_vehicle_app_info.get('entry_time'): time_active_draw = (current_timestamp_dt - active_vehicle_app_info['entry_time']).total_seconds()
                    if draw_status != 'timed_out' or track_id_draw in current_frame_visible_vehicle_map:
                         visual_overlay.draw_detection_box(output_frame_viz, box_draw, str(track_id_draw), type_draw, status=draw_status, entry_dir=entry_dir_draw, time_active=time_active_draw)
                    if track_id_draw not in self.vehicle_tracker.completed_vehicle_ids or draw_status in ['exiting', 'timed_out', 'entering']:
                        trail = self.vehicle_tracker.get_tracking_trail(track_id_draw)
                        visual_overlay.draw_tracking_trail(output_frame_viz, trail, str(track_id_draw))
                    event_marker_draw = data_draw.get('event_marker')
                    if event_marker_draw: visual_overlay.draw_event_marker(output_frame_viz, center_pt_draw, event_marker_draw, str(track_id_draw)[-4:])
                output_frame_viz = visual_overlay.add_status_overlay(output_frame_viz, frame_num, current_timestamp_dt, self.vehicle_tracker)
                processed_frames_for_viz[i] = output_frame_viz
                if self.perf: self.perf.end_timer('drawing')
        
        return batch_event_logs, processed_frames_for_viz

    # --- Main Processing Function (process_video) ---
    # The overall structure of process_video remains largely the same.
    # Key changes were already highlighted in the previous response for this method
    # (initializing ByteTrackWrapper, how batches are formed for _preprocess_batch,
    #  and how _process_inference_results is called).
    # The rest of process_video (setup, DAV conversion, zone definition,
    # reader/writer thread management, main loop structure, final batch, reporting)
    # can remain as it was in the previous provided version.
    # For brevity, I'll only show the start and indicate where the main loop is.

    def process_video(self, video_path, start_date_str, start_time_str, primary_direction):
        if not self.processing_lock.acquire(blocking=False):
            return "Processing is already in progress."

        self.perf = PerformanceTracker(enabled=ENABLE_DETAILED_PERFORMANCE_METRICS)
        self.stop_event = threading.Event()
        self.frame_read_queue = queue.Queue(maxsize=max(self.batch_size * 2, 64)) # Adjusted queue
        self.last_cleanup_time = time.monotonic()
        self.detection_zones_polygons = None
        output_path = None; self.final_output_path = None
        self.writer_thread = None; self.video_write_queue = None

        print(f"--- Starting New Video Processing Run (ByteTrack v2) ---")
        # ... (initial prints: video, timestamp, zone mode, etc.)

        actual_video_path_to_process = video_path
        temp_converted_dav_mp4 = None
        if video_path.lower().endswith(".dav"): 
            if not FFMPEG_PATH: raise RuntimeError("FFmpeg not found for .dav conversion.")
            print(f"Input is .dav: {video_path}. Converting with FFmpeg...")
            converted_path = convert_dav_to_mp4_internal(video_path)
            if converted_path and os.path.exists(converted_path):
                actual_video_path_to_process = converted_path; temp_converted_dav_mp4 = converted_path
                print(f".dav converted to: {actual_video_path_to_process}"); time.sleep(0.5)
            else: raise RuntimeError(f"Failed to convert .dav file: {video_path}")
        
        try:
            # Zone Definition (Hardcoded or Interactive)
            if self.line_mode == 'hardcoded': # ... (same as before)
                if not LINE_POINTS: raise ValueError("Hardcoded: LINE_POINTS missing.")
                valid_polygons_hc = {d: p for d, p in LINE_POINTS.items() if isinstance(p, list) and len(p) >= 3}
                if not valid_polygons_hc: raise ValueError("Hardcoded: No valid polygons.")
                self.detection_zones_polygons = {d: np.array(p, dtype=np.int32) for d, p in valid_polygons_hc.items()}
            elif self.line_mode == 'interactive': # ... (same as before)
                if not self.gradio_polygons: raise ValueError("Interactive: No zones defined.")
                valid_polygons_gr = {d: p for d, p in self.gradio_polygons.items() if p and len(p) >= 3}
                if len(valid_polygons_gr) < 2: raise ValueError(f"Interactive: Need >= 2 zones. Found {len(valid_polygons_gr)}.")
                self.detection_zones_polygons = {d: np.array(p, dtype=np.int32) for d, p in valid_polygons_gr.items()}
            else: raise ValueError(f"Invalid LINE_MODE '{self.line_mode}'.")
            if not self.detection_zones_polygons: raise ValueError("Zone polygons missing.")
            self.zone_tracker = ZoneTracker(self.detection_zones_polygons) 
            if not self.zone_tracker or not self.zone_tracker.zones: raise RuntimeError("ZoneTracker init failed.")

            # Model and Tracker Initialization
            if self.model is None: self.model = load_model(MODEL_PATH, self.device, CONF_THRESHOLD, IOU_THRESHOLD)
            if self.model is None: raise RuntimeError(f"Failed to load model {MODEL_PATH}")
            if not hasattr(self.model, 'names') or not self.model.names:
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'): self.model.names = self.model.model.names
                else: raise RuntimeError("Model names (class labels) not found.")
            self.vehicle_tracker = ByteTrackWrapper(model_names_dict=self.model.names)

            # Timestamp Parsing
            try: # ... (same as before)
                start_date_obj=datetime.strptime(start_date_str, "%y%m%d"); time_part,ms_part=(start_time_str[:6], start_time_str[6:]) if len(start_time_str)>6 else (start_time_str,"0")
                base_time_obj=datetime.strptime(time_part,"%H%M%S").time(); microseconds_val=int(ms_part.ljust(3,'0'))*1000
                start_time_obj=base_time_obj.replace(microsecond=microseconds_val); start_datetime_obj=datetime.combine(start_date_obj.date(), start_time_obj)
                print(f"Video Start Timestamp: {start_datetime_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            except ValueError as e: raise ValueError(f"Invalid date/time format: {e}.")

            total_frames, original_fps = self._get_video_properties(actual_video_path_to_process)
            print(f"Video Props: ~{total_frames} frames, {original_fps:.2f} FPS (Original)")
            if total_frames <= 0: raise ValueError("Cannot process video with zero/unreadable total frames.")

            # I/O Threads
            print("Starting I/O threads...") # ... (same logic as before for starting reader/writer)
            reader_args = (actual_video_path_to_process, original_fps)
            reader_task_fn = self._frame_reader_task
            if self.actually_use_advanced_reader:
                reader_args = (actual_video_path_to_process, original_fps, total_frames)
                reader_task_fn = self._frame_reader_task_pvr_segmented
            self.reader_thread = threading.Thread(target=reader_task_fn, args=reader_args, daemon=True); self.reader_thread.start()
            if ENABLE_VISUALIZATION:
                os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True); ts_now=datetime.now().strftime("%Y%m%d_%H%M%S")
                output_fn_base=f"output_{os.path.splitext(os.path.basename(actual_video_path_to_process))[0]}_{ts_now}"
                output_path=os.path.join(VIDEO_OUTPUT_DIR, f"{output_fn_base}.mp4");
                self.final_output_path = output_path 
                self.video_write_queue = queue.Queue(maxsize=max(self.batch_size*2, 32))
                self.writer_thread = threading.Thread(target=self._frame_writer_task, args=(output_path,), daemon=True); self.writer_thread.start(); print("Video Writer thread started.")
            else: self.video_write_queue = None; self.writer_thread = None; self.final_output_path = None
            
            # --- Main Processing Loop ---
            all_event_logs_final = []; frames_processed_count = 0
            frame_interval = max(1, round(original_fps / self.target_fps)) if self.target_fps > 0 and original_fps > 0 else 1
            total_target_frames_for_perf = total_frames // frame_interval if total_frames > 0 and frame_interval > 0 else (total_frames if total_frames > 0 else 0)
            if self.perf: self.perf.start_processing(total_target_frames_for_perf)
            
            current_batch_frames_content_metadata = [] 
            current_batch_frame_numbers = []
            current_batch_frame_times_sec = []
            last_progress_print_time = time.monotonic()
            print(f"Target FPS: {self.target_fps:.1f}, Processing every ~{frame_interval} frame(s).")

            while True: # Main processing loop
                try:
                    frame_data_from_queue = self.frame_read_queue.get(timeout=60)
                    if frame_data_from_queue is None: self.frame_read_queue.task_done(); break 
                    if self.perf: self.perf.start_timer('video_read')
                    frame_content_display_res, frame_num_orig, frame_time_video_sec, metadata_dict = frame_data_from_queue
                    if self.perf: self.perf.end_timer('video_read')
                    if frame_content_display_res is None: self.frame_read_queue.task_done(); continue

                    current_batch_frames_content_metadata.append((frame_content_display_res, metadata_dict))
                    current_batch_frame_numbers.append(frame_num_orig)
                    current_batch_frame_times_sec.append(frame_time_video_sec)

                    if len(current_batch_frames_content_metadata) >= self.batch_size:
                        batch_mono_start_time = time.monotonic()
                        if self.perf: self.perf.start_timer('preprocessing')
                        input_batch_tensor_for_yolo = self._preprocess_batch(current_batch_frames_content_metadata)
                        if self.perf: self.perf.end_timer('preprocessing')

                        if input_batch_tensor_for_yolo is None or input_batch_tensor_for_yolo.nelement() == 0:
                            debug_print("Skipping empty preprocessed batch.")
                        else:
                            current_cuda_stream=None # ... (YOLO Inference)
                            if self.device=='cuda' and self.streams: current_cuda_stream=self.streams[self.current_stream_index%len(self.streams)]; self.current_stream_index+=1
                            with torch.cuda.stream(current_cuda_stream) if current_cuda_stream else torch.no_grad():
                               with torch.amp.autocast(device_type=self.device,enabled=MIXED_PRECISION and self.device=='cuda'):
                                 if self.perf and self.perf.start_event: self.perf.start_event.record(stream=current_cuda_stream)
                                 yolo_results_list_for_batch = self.model.predict(source=input_batch_tensor_for_yolo, verbose=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
                                 if self.perf and self.perf.end_event: self.perf.end_event.record(stream=current_cuda_stream)
                            if current_cuda_stream: current_cuda_stream.synchronize()
                            if self.perf and self.perf.start_event and self.perf.end_event: self.perf.record_inference_time_gpu(self.perf.start_event, self.perf.end_event)
                            
                            # Post-process: Tracking, Zone Checks, Events
                            batch_event_log, processed_viz_frames_list = self._process_inference_results(
                                yolo_results_list_for_batch, current_batch_frames_content_metadata,
                                current_batch_frame_numbers, current_batch_frame_times_sec, start_datetime_obj
                            )
                            all_event_logs_final.extend(batch_event_log)
                            if self.perf: self.perf.record_detection(sum(len(r.boxes) for r in yolo_results_list_for_batch if hasattr(r,'boxes') and r.boxes))

                            if ENABLE_VISUALIZATION and self.video_write_queue: # ... (Write visualized frames)
                                if self.perf: self.perf.start_timer('video_write')
                                for pf_viz in processed_viz_frames_list:
                                    if pf_viz is not None:
                                        try: self.video_write_queue.put(pf_viz, block=True, timeout=2.0)
                                        except queue.Full: print("WARN: Video writer queue full, frame dropped.")
                                if self.perf: self.perf.end_timer('video_write')
                        
                        # Stats & Cleanup for this batch
                        batch_processing_time_sec = time.monotonic() - batch_mono_start_time # ... (rest of batch stats and cleanup)
                        frames_processed_count += len(current_batch_frames_content_metadata)
                        if self.perf: self.perf.record_batch_processed(len(current_batch_frames_content_metadata), batch_processing_time_sec); self.perf.sample_system_metrics()
                        current_mono_time = time.monotonic()
                        if current_mono_time - self.last_cleanup_time > MEMORY_CLEANUP_INTERVAL:
                            if self.perf: self.perf.start_timer('memory_cleanup')
                            last_f_time_sec_in_batch = current_batch_frame_times_sec[-1] if current_batch_frame_times_sec else 0
                            cleanup_tracking_data(self.vehicle_tracker, self.zone_tracker, last_f_time_sec_in_batch)
                            cleanup_memory(); self.last_cleanup_time = current_mono_time
                            if self.perf: self.perf.end_timer('memory_cleanup')
                        if self.perf and (current_mono_time - last_progress_print_time > 1.0): # Progress Print
                            prog=self.perf.get_progress(); print(f"\rProgress:{prog['percent']:.1f}%|FPS:{prog['fps']:.1f}|Active (App):{self.vehicle_tracker.get_active_vehicle_count()}|ETA:{prog['eta']} ", end="")
                            last_progress_print_time = current_mono_time

                        current_batch_frames_content_metadata.clear(); current_batch_frame_numbers.clear(); current_batch_frame_times_sec.clear()
                        if 'input_batch_tensor_for_yolo' in locals(): del input_batch_tensor_for_yolo
                        if 'yolo_results_list_for_batch' in locals(): del yolo_results_list_for_batch
                        cleanup_memory() 

                    self.frame_read_queue.task_done()
                except queue.Empty: 
                    if self.stop_event.is_set() and (self.frame_read_queue is None or self.frame_read_queue.empty()): break
                    continue 
                except Exception as e: print(f"\nERROR during processing loop: {e}"); import traceback; traceback.print_exc(); self.stop_event.set(); break
            
            # --- Process Final Batch ---
            if current_batch_frames_content_metadata: # ... (Same logic as in-loop batch processing)
                print("\nProcessing final batch...")
                batch_mono_start_time = time.monotonic()
                if self.perf: self.perf.start_timer('preprocessing')
                input_batch_tensor_for_yolo = self._preprocess_batch(current_batch_frames_content_metadata)
                if self.perf: self.perf.end_timer('preprocessing')
                if input_batch_tensor_for_yolo is not None and input_batch_tensor_for_yolo.nelement() > 0:
                    current_cuda_stream=None # ... (YOLO Inference for final batch)
                    if self.device=='cuda' and self.streams: current_cuda_stream=self.streams[self.current_stream_index%len(self.streams)]; self.current_stream_index+=1
                    with torch.cuda.stream(current_cuda_stream) if current_cuda_stream else torch.no_grad():
                       with torch.amp.autocast(device_type=self.device,enabled=MIXED_PRECISION and self.device=='cuda'):
                         if self.perf and self.perf.start_event: self.perf.start_event.record(stream=current_cuda_stream)
                         yolo_results_list_for_batch = self.model.predict(source=input_batch_tensor_for_yolo, verbose=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
                         if self.perf and self.perf.end_event: self.perf.end_event.record(stream=current_cuda_stream)
                    if current_cuda_stream: current_cuda_stream.synchronize()
                    if self.perf and self.perf.start_event and self.perf.end_event: self.perf.record_inference_time_gpu(self.perf.start_event, self.perf.end_event)

                    batch_event_log, processed_viz_frames_list = self._process_inference_results(
                        yolo_results_list_for_batch, current_batch_frames_content_metadata,
                        current_batch_frame_numbers, current_batch_frame_times_sec, start_datetime_obj
                    )
                    all_event_logs_final.extend(batch_event_log)
                    if ENABLE_VISUALIZATION and self.video_write_queue: # ... (Write final visualized frames)
                        if self.perf: self.perf.start_timer('video_write');
                        for pf_viz in processed_viz_frames_list:
                            if pf_viz is not None:
                                try: self.video_write_queue.put(pf_viz, block=True, timeout=2.0)
                                except queue.Full: print("WARN: Video writer queue full during final batch, frame dropped.")
                        if self.perf: self.perf.end_timer('video_write')
                    batch_processing_time_sec = time.monotonic() - batch_mono_start_time
                    frames_processed_count += len(current_batch_frames_content_metadata)
                    if self.perf: self.perf.record_batch_processed(len(current_batch_frames_content_metadata), batch_processing_time_sec)
                    del input_batch_tensor_for_yolo, yolo_results_list_for_batch; cleanup_memory()
                else: debug_print("Skipping empty final batch.")
            print("\nVideo processing loop finished.")

            # --- Finalize & Report ---
            if self.vehicle_tracker and self.vehicle_tracker.active_vehicles: # ... (Force exit remaining)
                print(f"Forcing exit for {len(self.vehicle_tracker.active_vehicles)} remaining app-level active vehicles...")
                last_ts_dt = start_datetime_obj + timedelta(seconds=current_batch_frame_times_sec[-1]) if current_batch_frame_times_sec else start_datetime_obj
                for track_id_force_exit in list(self.vehicle_tracker.active_vehicles.keys()):
                    if self.vehicle_tracker.force_exit_vehicle(track_id_force_exit, last_ts_dt):
                        if self.perf: self.perf.record_vehicle_exit('forced_exit')
            
            print("Signaling threads to stop..."); self.stop_event.set() # ... (Signal & join threads)
            if ENABLE_VISUALIZATION and self.video_write_queue:
                try: self.video_write_queue.put(None, timeout=5.0) 
                except queue.Full: print("WARN: Could not signal writer queue (full).")
            if self.frame_read_queue: 
                try: self.frame_read_queue.put_nowait(None)
                except queue.Full: pass 
            print("Waiting for I/O threads..."); t_wait_start=time.monotonic()
            if self.writer_thread and self.writer_thread.is_alive(): self.writer_thread.join(timeout=30)
            if self.reader_thread and self.reader_thread.is_alive(): self.reader_thread.join(timeout=5)
            print(f"I/O Threads joined/timed out in {time.monotonic()-t_wait_start:.2f}s")

            # Performance Summary & Reporting
            total_time_taken=0.0; final_fps_val=0.0; completed_paths_count_valid=0 # ... (Calculate stats)
            if self.perf: self.perf.end_processing(); self.perf.print_summary()
            final_completed_paths_list = self.vehicle_tracker.get_completed_paths() if self.vehicle_tracker else []
            for path_info in final_completed_paths_list: # Count valid paths
                is_valid_status = path_info.get('status') == 'exited'
                is_valid_entry = path_info.get('entry_direction') and path_info['entry_direction'] not in ['UNKNOWN', None]
                is_valid_exit = path_info.get('exit_direction') and path_info['exit_direction'] not in ['UNKNOWN', 'TIMEOUT', 'FORCED', None]
                if is_valid_status and is_valid_entry and is_valid_exit: completed_paths_count_valid += 1

            excel_created_path = create_excel_report( # ... (Create Excel report)
                completed_paths_data=final_completed_paths_list, start_datetime=start_datetime_obj,
                primary_direction=primary_direction, video_path=actual_video_path_to_process
            )
            if ENABLE_DETAILED_PERFORMANCE_METRICS and self.perf: visualize_performance(self.perf)

            result_msg=f"✅ Processing completed.\n" # ... (Prepare result message)
            if ENABLE_VISUALIZATION:
                if self.final_output_path and "ERROR" not in self.final_output_path: result_msg+=f"Output video: '{self.final_output_path}'.\n"
                elif self.final_output_path and "ERROR" in self.final_output_path: result_msg+=f"Output video generation failed: {self.final_output_path}.\n"
            if excel_created_path: result_msg+=f"Excel report: {excel_created_path}\n"
            result_msg+=f"\n--- Summary Stats ---\nSTAT_FRAMES_PROCESSED={frames_processed_count}\n"
            if self.perf and self.perf.total_time > 0: result_msg+=f"STAT_TIME_SECONDS={self.perf.total_time:.2f}\nSTAT_FPS={frames_processed_count/self.perf.total_time:.2f}\n"
            else: result_msg+=f"STAT_TIME_SECONDS=N/A\nSTAT_FPS=N/A\n"
            result_msg+=f"STAT_COMPLETED_PATHS={completed_paths_count_valid}\n--- End Stats ---"
            return result_msg

        except (ValueError, RuntimeError, FileNotFoundError, Exception) as e: # Error Handling
            print(f"\nFATAL ERROR during setup or processing: {e}"); import traceback; traceback.print_exc()
            if hasattr(self, 'stop_event') and self.stop_event: self.stop_event.set()
            return f"❌ Error: {e}"
        finally: # Final Cleanup
            try: 
                if hasattr(self, 'reader_thread') and self.reader_thread and self.reader_thread.is_alive(): self.reader_thread.join(timeout=1)
                if hasattr(self, 'writer_thread') and self.writer_thread and self.writer_thread.is_alive(): self.writer_thread.join(timeout=1)
            except Exception as join_e: print(f"Error joining threads on final cleanup: {join_e}")
            if temp_converted_dav_mp4:
                try: os.remove(temp_converted_dav_mp4); debug_print(f"Cleaned up: {temp_converted_dav_mp4}")
                except OSError as e_del: debug_print(f"Error deleting {temp_converted_dav_mp4}: {e_del}")
            self.model=None; self.vehicle_tracker=None; self.zone_tracker=None; self.perf=None
            self.frame_read_queue=None; self.video_write_queue=None; self.stop_event=None
            self.detection_zones_polygons=None 
            cleanup_memory()
            if self.processing_lock.locked():
                try: self.processing_lock.release(); print("Processing finished, lock released.")
                except RuntimeError as release_err: print(f"Warn: Error releasing lock: {release_err}")
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
from types import SimpleNamespace # For ByteTrack args

# Configuration
from config import (
    TARGET_FPS, FRAME_WIDTH, FRAME_HEIGHT, OPTIMAL_BATCH_SIZE,
    VIDEO_OUTPUT_DIR, ENABLE_DETAILED_PERFORMANCE_METRICS,
    MIXED_PRECISION, PARALLEL_STREAMS, THREAD_COUNT,
    MEMORY_CLEANUP_INTERVAL, MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD,
    LINE_MODE, LINE_POINTS, MODEL_INPUT_SIZE, ENABLE_VISUALIZATION,
    USE_ADVANCED_VIDEO_READER, PYVIDEOREADER_DECODE_SEGMENT_SIZE, FFMPEG_PATH,
    # ByteTrack Args from config
    BYTETRACK_TRACK_HIGH_THRESH, BYTETRACK_TRACK_LOW_THRESH,
    BYTETRACK_NEW_TRACK_THRESH, BYTETRACK_MATCH_THRESH,
    BYTETRACK_TRACK_BUFFER, BYTETRACK_FUSE_SCORE
)

# Modules
from core.utils import debug_print, format_timestamp, is_valid_movement, cleanup_memory
from core.performance import PerformanceTracker
from models.model_loader import load_model, get_device
# from tracking.vehicle_tracker import VehicleTracker # REMOVE OLD TRACKER
from tracking.bytetrack_wrapper import ByteTrackWrapper # NEW
from tracking.path_manager import VehiclePathManager     # NEW
from tracking.zone_tracker import ZoneTracker
# from tracking.cleanup import cleanup_tracking_data # PathManager handles its own, ByteTrack internal
from visualization import overlay as visual_overlay
from reporting.excel_report import create_excel_report
from core.converters import convert_dav_to_mp4_internal

# Conditional imports
if ENABLE_DETAILED_PERFORMANCE_METRICS:
    from visualization.performance_viz import visualize_performance

# PyVideoReader
PYVIDEOREADER_AVAILABLE = False
PyVideoReader = None
try:
    from video_reader import PyVideoReader # Ensure this matches your video_reader-rs installation
    PYVIDEOREADER_AVAILABLE = True
    print("PyVideoReader (video_reader-rs) found and available.")
except ImportError:
    print("PyVideoReader (video_reader-rs) not found. Advanced video reader will not be used.")


class VideoProcessor:
    def __init__(self):
        self.target_fps = TARGET_FPS
        self.frame_shape = (FRAME_WIDTH, FRAME_HEIGHT) # (Display_W, Display_H)
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
        self.byte_tracker_wrapper = None
        self.path_manager = None
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
        # (Print statements for reader status are here in original code)

        self.bytetrack_args = SimpleNamespace(
            track_high_thresh=BYTETRACK_TRACK_HIGH_THRESH,
            track_low_thresh=BYTETRACK_TRACK_LOW_THRESH,
            new_track_thresh=BYTETRACK_NEW_TRACK_THRESH,
            match_thresh=BYTETRACK_MATCH_THRESH, # Cost threshold
            track_buffer=BYTETRACK_TRACK_BUFFER,
            fuse_score=BYTETRACK_FUSE_SCORE,
        )

    # --- Gradio Polygon Methods ---
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
                try:
                    if len(pts_np) > 0: M = cv2.moments(pts_np)
                    if M["m00"] != 0: cX = int(M["m10"] / M["m00"]); cY = int(M["m01"] / M["m00"]); cv2.putText(img_copy, direction.upper(), (cX-15, cY+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except Exception: pass
        if self.current_gradio_direction and self.gradio_temp_points:
            current_temp_color = colors.get(self.current_gradio_direction, temp_color)
            for i, pt in enumerate(self.gradio_temp_points):
                cv2.circle(img_copy, pt, 5, current_temp_color, -1)
                if i > 0: cv2.line(img_copy, self.gradio_temp_points[i-1], pt, current_temp_color, 2)
            if len(self.gradio_temp_points) >= 2: last_pt = self.gradio_temp_points[-1]; first_pt = self.gradio_temp_points[0]; hint_color = tuple(c // 2 for c in current_temp_color); cv2.line(img_copy, last_pt, first_pt, hint_color, 1, cv2.LINE_AA)
        self.last_drawn_polygons_frame = img_copy
        return img_copy
    def process_video_upload_for_gradio(self, video_path):
        if self.line_mode != 'interactive': return None, "Zone definition mode is 'hardcoded'. Drawing disabled."
        self.reset_gradio_polygons()
        if video_path is None: return None, "Please upload a video."
        try:
            cap=cv2.VideoCapture(video_path); ret,frame=cap.read(); cap.release()
            if ret: frame = cv2.resize(frame, self.frame_shape); self.original_frame_for_gradio = frame.copy(); drawn_frame = self._draw_polygons_for_gradio(self.original_frame_for_gradio); return drawn_frame, "Video loaded. Select direction and click image to define zone vertices."
            else: return None, "Error: Could not read first frame."
        except Exception as e: return None, f"Error loading video: {e}"
    def set_gradio_polygon_direction(self, direction):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        direction_lower = direction.lower()
        if direction_lower in self.gradio_polygons: del self.gradio_polygons[direction_lower]; print(f"Cleared existing polygon for {direction.capitalize()} to redraw.")
        self.current_gradio_direction = direction_lower; self.gradio_temp_points = []
        status = f"Selected '{direction.capitalize()}'. Click points on image to define zone polygon."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio); return frame_to_show, status
    def handle_gradio_polygon_click(self, evt: 'gr.SelectData'):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if self.current_gradio_direction is None: return self.last_drawn_polygons_frame, "Select a direction first."
        if self.original_frame_for_gradio is None: return None, "No video loaded."
        point = (evt.index[0], evt.index[1]); self.gradio_temp_points.append(point)
        status = f"Added point {len(self.gradio_temp_points)} for {self.current_gradio_direction.capitalize()}. Click more points or 'Finish Polygon'."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio); return frame_to_show, status
    def finalize_current_polygon(self):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if not self.current_gradio_direction: return self.last_drawn_polygons_frame, "No direction selected to finalize."
        if len(self.gradio_temp_points) < 3: status = f"Error: Need at least 3 points for a polygon (got {len(self.gradio_temp_points)})."; frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio); return frame_to_show, status
        self.gradio_polygons[self.current_gradio_direction] = self.gradio_temp_points.copy()
        status = f"Polygon for {self.current_gradio_direction.capitalize()} saved. Select next direction or Process."
        self.current_gradio_direction = None; self.gradio_temp_points = []
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio); return frame_to_show, status
    def undo_last_polygon_point(self):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if not self.current_gradio_direction: status = "Select a direction first."
        elif not self.gradio_temp_points: status = f"No points added yet for {self.current_gradio_direction.capitalize()}."
        else: removed_point = self.gradio_temp_points.pop(); status = f"Removed last point for {self.current_gradio_direction.capitalize()}."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio); return frame_to_show, status
    def reset_gradio_polygons(self):
        self.gradio_polygons = {}; self.current_gradio_direction = None; self.gradio_temp_points = []; status = "All polygons cleared."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio) if self.original_frame_for_gradio is not None else None
        self.last_drawn_polygons_frame = frame_to_show; return frame_to_show, status

    # --- Video Reading/Writing Threads ---
    def _get_video_properties(self, video_path):
        total_frames = 0
        original_fps = 0.0

        if self.actually_use_advanced_reader:
            temp_reader_rs = None
            try:
                temp_reader_rs = PyVideoReader(video_path)
                info = temp_reader_rs.get_info()
                total_frames = int(info.get('frame_count', 0))
                fps_str = info.get('fps', '0.0')
                # -------- CORRECTED BLOCK START --------
                if fps_str:
                    try:
                        original_fps = float(fps_str)
                    except ValueError:
                        debug_print(f"PyVideoReader: Could not parse FPS string '{fps_str}' to float.")
                        pass # Keep original_fps as 0.0 if parsing fails
                # -------- CORRECTED BLOCK END --------
            except Exception as e:
                debug_print(f"PyVideoReader property fetch failed: {e}. Falling back to OpenCV for props.")
            finally: # Ensure temp_reader_rs is cleaned up
                if temp_reader_rs:
                    del temp_reader_rs

        if total_frames <= 0 or original_fps <= 0: # If PyVideoReader failed or wasn't used
            cap_check = None
            try:
                cap_check = cv2.VideoCapture(video_path)
                if not cap_check.isOpened():
                    raise FileNotFoundError(f"OpenCV: Cannot open video for properties: {video_path}")

                _total_frames_cv = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
                _original_fps_cv = cap_check.get(cv2.CAP_PROP_FPS)

                if total_frames <= 0: total_frames = _total_frames_cv
                if original_fps <= 0: original_fps = _original_fps_cv
            except Exception as e:
                print(f"Error getting video properties with OpenCV: {e}")
            finally:
                if cap_check: cap_check.release()

        if original_fps <= 0: original_fps = 30.0 # Final fallback if all methods fail
        if total_frames <= 0:
            print("WARNING: Could not determine total_frames accurately. Progress calculation might be affected.")
        return total_frames, original_fps

    def _frame_reader_task(self, video_path, original_fps): # No change
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): print("ERROR: Reader failed to open video."); self.frame_read_queue.put(None); return
            frame_interval = max(1, round(original_fps / self.target_fps)) if self.target_fps > 0 and original_fps > 0 else 1
            frame_count_raw = 0; frames_yielded = 0
            while not self.stop_event.is_set():
                ret, frame_bgr_orig_size = cap.read()
                if not ret: debug_print("OpenCV Reader: End of video or read error."); break
                if frame_count_raw  % frame_interval == 0:
                    frame_time_seconds = frame_count_raw  / original_fps
                    metadata = {'is_rgb': False, 'source': 'opencv'} # OpenCV provides BGR
                    try: self.frame_read_queue.put((frame_bgr_orig_size, frame_count_raw , frame_time_seconds, metadata), block=True, timeout=5.0); frames_yielded += 1
                    except queue.Full:
                        debug_print("Reader queue full, waiting..."); time.sleep(0.1)
                        if self.stop_event.is_set(): break
                        try: self.frame_read_queue.put((frame_bgr_orig_size, frame_count_raw , frame_time_seconds, metadata), block=True, timeout=5.0); frames_yielded += 1
                        except queue.Full: print("ERROR: Reader queue persistently full."); self.stop_event.set(); break
                    except Exception as q_e: debug_print(f"OpenCV Reader: Queue put error: {q_e}"); self.stop_event.set(); break
                frame_count_raw  += 1
            debug_print(f"OpenCV Reader finished. Raw frames read: {frame_count_raw}. Frames yielded to queue: {frames_yielded}.")
        except Exception as e: print(f"FATAL ERROR in _frame_reader_task (OpenCV): {e}"); import traceback; traceback.print_exc()
        finally:
            if cap: cap.release()
            if self.frame_read_queue:
                try: self.frame_read_queue.put(None, block=False, timeout=1.0)
                except queue.Full: debug_print("OpenCV Reader: Queue full on final None put.")
                except Exception as e_fin: debug_print(f"OpenCV Reader: Error on final None put: {e_fin}")

    def _frame_reader_task_pvr_segmented(self, video_path, original_fps_from_props, total_frames_from_props): # No change
        pvr_instance = None
        try:
            target_dim_model = self.model_input_size_config
            debug_print(f"PVR SegReader: Init PyVideoReader: device='cuda', target_size=({target_dim_model}x{target_dim_model})")
            pvr_instance = PyVideoReader(video_path, device='cuda', resize_shorter_side=target_dim_model, resize_longer_side=target_dim_model)
            frame_interval = max(1, round(original_fps_from_props / self.target_fps)) if self.target_fps > 0 and original_fps_from_props > 0 else 1
            queued_frame_count = 0
            num_segments = math.ceil(total_frames_from_props / self.pvr_decode_segment_size) if total_frames_from_props > 0 and self.pvr_decode_segment_size > 0 else 1
            if total_frames_from_props == 0 : num_segments = 0
            debug_print(f"PVR SegReader: Video Length={total_frames_from_props} frames, Segment Size={self.pvr_decode_segment_size}, Num Segments={num_segments}")
            for i in range(num_segments):
                if self.stop_event.is_set(): break
                segment_start_raw_idx = i * self.pvr_decode_segment_size
                segment_end_raw_idx = min((i + 1) * self.pvr_decode_segment_size, total_frames_from_props)
                if segment_start_raw_idx >= segment_end_raw_idx : continue
                debug_print(f"PVR SegReader: Decoding segment {i+1}/{num_segments} (raw frames {segment_start_raw_idx} to {segment_end_raw_idx-1})...")
                segment_frames_np = pvr_instance.decode(start_frame=segment_start_raw_idx, end_frame=segment_end_raw_idx) # RGB, (model_input_size, model_input_size, 3)
                if segment_frames_np is None or segment_frames_np.shape[0] == 0: debug_print(f"PVR SegReader: Segment {i+1} returned no frames."); continue
                for frame_in_segment_idx in range(segment_frames_np.shape[0]):
                    if self.stop_event.is_set(): break
                    current_overall_raw_frame_idx = segment_start_raw_idx + frame_in_segment_idx
                    if current_overall_raw_frame_idx % frame_interval == 0:
                        frame_content = segment_frames_np[frame_in_segment_idx]
                        current_frame_time_sec = current_overall_raw_frame_idx / original_fps_from_props
                        metadata = {'is_rgb': True, 'source': 'pvr_cuda_segmented'} # PVR provides RGB
                        try: self.frame_read_queue.put((frame_content, current_overall_raw_frame_idx, current_frame_time_sec, metadata), block=True, timeout=5.0); queued_frame_count += 1
                        except queue.Full:
                            debug_print("PVR SegReader: Queue full, retrying..."); time.sleep(0.1);
                            if self.stop_event.is_set(): break
                            try: self.frame_read_queue.put((frame_content, current_overall_raw_frame_idx, current_frame_time_sec, metadata), block=True, timeout=5.0); queued_frame_count += 1
                            except queue.Full: print("ERROR: PVR SegReader queue persistently full."); self.stop_event.set(); break
                        except Exception as q_e: debug_print(f"PVR SegReader queue put error: {q_e}"); self.stop_event.set(); break
                del segment_frames_np; gc.collect()
            debug_print(f"PVR SegReader finished. Queued {queued_frame_count} frames.")
        except Exception as e: print(f"FATAL ERROR in _frame_reader_task_pvr_segmented: {e}"); import traceback; traceback.print_exc()
        finally:
            if pvr_instance: del pvr_instance
            if self.frame_read_queue:
                try: self.frame_read_queue.put(None, block=False, timeout=1.0)
                except queue.Full: pass

    def _frame_writer_task(self, output_path): # No change
        out = None; writer_initialized = False; frames_written = 0; actual_output_path = output_path
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v'); out = cv2.VideoWriter(actual_output_path, fourcc, self.target_fps, self.frame_shape)
            if not out.isOpened():
                print(f"Warn: mp4v failed for {actual_output_path}. Trying XVID/AVI."); avi_path = os.path.splitext(actual_output_path)[0] + ".avi"; fourcc = cv2.VideoWriter_fourcc(*'XVID'); out = cv2.VideoWriter(avi_path, fourcc, self.target_fps, self.frame_shape)
                if out.isOpened(): actual_output_path = avi_path; self.final_output_path = avi_path
                else: print(f"ERROR: Writer failed to open for both MP4 and AVI."); self.final_output_path = "ERROR_WRITER_FAILED"; return
            writer_initialized = True; print(f"Writer initialized for: {actual_output_path}")
            while True:
                try: frame_data = self.video_write_queue.get(timeout=5.0)
                except queue.Empty:
                    if self.stop_event is not None and self.stop_event.is_set() and self.video_write_queue.empty(): break
                    continue
                if frame_data is None: break
                frame = frame_data
                if frame is not None and isinstance(frame, np.ndarray) and frame.shape[0] > 0:
                    if frame.dtype != np.uint8: frame = frame.astype(np.uint8)
                    if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    if frame.shape[1]!=self.frame_shape[0] or frame.shape[0]!=self.frame_shape[1]: frame = cv2.resize(frame, self.frame_shape)
                    out.write(frame); frames_written += 1
                if self.video_write_queue: self.video_write_queue.task_done()
        except Exception as e: print(f"ERROR in writer thread: {e}");
        finally:
            if out and writer_initialized: out.release(); debug_print(f"Writer released. Wrote {frames_written} frames to {actual_output_path}")
            if self.video_write_queue:
                try: self.video_write_queue.put(None, timeout=1.0)
                except queue.Full: pass

    # --- Core Processing Logic ---
    @torch.no_grad()
    def _preprocess_batch(self, frames_batch_with_metadata): # No change
        preprocessed_tensors_list = []
        target_dim_for_model = self.model_input_size_config
        for frame_np_content, metadata in frames_batch_with_metadata:
            if frame_np_content is None or not isinstance(frame_np_content, np.ndarray): debug_print("Skipping None or invalid frame_np_content in _preprocess_batch."); continue
            is_rgb_input = metadata.get('is_rgb', False); source_reader = metadata.get('source', 'unknown'); img_rgb_model_sized = None
            if source_reader.startswith('pvr_cuda'): # PyVideoReader provides RGB, already model-sized
                img_rgb_model_sized = frame_np_content
                if not is_rgb_input: debug_print(f"Preprocess (PVR path): Warning! Expected RGB from PVR, metadata says otherwise: {metadata}. Assuming RGB.") # PVR should be RGB
                # Sanity check size, PVR should have handled it
                if img_rgb_model_sized.shape[0] != target_dim_for_model or img_rgb_model_sized.shape[1] != target_dim_for_model:
                    debug_print(f"Preprocess (PVR path): Warning! Frame shape {img_rgb_model_sized.shape[:2]} not at target model size ({target_dim_for_model},{target_dim_for_model}). Resizing again.")
                    img_rgb_model_sized = cv2.resize(img_rgb_model_sized, (target_dim_for_model, target_dim_for_model), interpolation=cv2.INTER_LINEAR)
            elif source_reader == 'opencv': # OpenCV provides BGR, original size
                resized_bgr = cv2.resize(frame_np_content, (target_dim_for_model, target_dim_for_model), interpolation=cv2.INTER_LINEAR)
                img_rgb_model_sized = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
            else: # Fallback for unknown, assume BGR and resize
                debug_print(f"Preprocess: Unknown frame source '{source_reader}'. Attempting general processing (BGR resize, to RGB).")
                temp_frame_bgr_resized = cv2.resize(frame_np_content, (target_dim_for_model, target_dim_for_model), interpolation=cv2.INTER_LINEAR)
                img_rgb_model_sized = cv2.cvtColor(temp_frame_bgr_resized, cv2.COLOR_BGR2RGB)

            if img_rgb_model_sized is None: debug_print(f"Preprocess: img_rgb_model_sized is None for a frame. Source: {source_reader}. Skipping."); continue
            # Ensure 3 channels (RGB)
            if img_rgb_model_sized.ndim == 2: img_rgb_model_sized = cv2.cvtColor(img_rgb_model_sized, cv2.COLOR_GRAY2RGB)
            elif img_rgb_model_sized.shape[2] == 4: img_rgb_model_sized = cv2.cvtColor(img_rgb_model_sized, cv2.COLOR_RGBA2RGB)
            if img_rgb_model_sized.shape[2] != 3: debug_print(f"Preprocess: Frame does not have 3 channels! Shape: {img_rgb_model_sized.shape}. Skipping."); continue
            
            img_chw = img_rgb_model_sized.transpose(2, 0, 1); img_chw_contiguous = np.ascontiguousarray(img_chw); preprocessed_tensors_list.append(img_chw_contiguous)
        if not preprocessed_tensors_list: debug_print("Preprocess: No tensors to stack after processing batch."); return None
        try: batch_np_stacked = np.stack(preprocessed_tensors_list)
        except ValueError as e: debug_print(f"Preprocess: Error stacking tensors: {e}. Shapes: {[t.shape for t in preprocessed_tensors_list]}"); return None
        
        batch_tensor_from_np = torch.from_numpy(batch_np_stacked).to(self.device)
        if batch_tensor_from_np.dtype != torch.uint8: debug_print(f"Preprocess: Tensor from numpy is not uint8 ({batch_tensor_from_np.dtype}). Clamping."); batch_tensor_from_np = torch.clamp(batch_tensor_from_np, 0, 255).to(torch.uint8)
        dtype_final = torch.float16 if (MIXED_PRECISION and self.device == 'cuda') else torch.float32
        final_batch_tensor = batch_tensor_from_np.to(dtype_final) / 255.0
        return final_batch_tensor

    @torch.no_grad()
    def _process_inference_results(self,
                                   yolo_results_list_for_batch, # List of YOLO Results objects
                                   frames_batch_content_metadata_list, # List of (original_frame_np_from_reader, metadata_dict)
                                   frame_numbers_batch,
                                   frame_times_batch,
                                   start_datetime_obj):

        batch_event_logs = []
        processed_frames_for_viz_list = [None] * len(yolo_results_list_for_batch)

        for i in range(len(yolo_results_list_for_batch)):
            yolo_result_single_frame = yolo_results_list_for_batch[i]
            original_frame_content_from_reader, frame_metadata = frames_batch_content_metadata_list[i]
            frame_num = frame_numbers_batch[i]
            frame_time_video_sec = frame_times_batch[i]
            current_timestamp_dt = start_datetime_obj + timedelta(seconds=frame_time_video_sec)

            output_frame_viz = None
            if ENABLE_VISUALIZATION and original_frame_content_from_reader is not None:
                temp_viz_frame = original_frame_content_from_reader.copy()
                if frame_metadata.get('is_rgb', False): # PVR is RGB
                    output_frame_viz = cv2.cvtColor(temp_viz_frame, cv2.COLOR_RGB2BGR)
                else: # OpenCV is BGR
                    output_frame_viz = temp_viz_frame
                if output_frame_viz.shape[1] != self.frame_shape[0] or output_frame_viz.shape[0] != self.frame_shape[1]:
                     output_frame_viz = cv2.resize(output_frame_viz, self.frame_shape)


            frame_local_event_log_for_this_iter = []
            if not self.byte_tracker_wrapper or not self.zone_tracker or not self.path_manager:
                processed_frames_for_viz_list[i] = output_frame_viz
                continue

            if self.perf: self.perf.start_timer('tracking_update')
            online_tracks_display_scaled = self.byte_tracker_wrapper.process_frame_detections(
                yolo_result_single_frame
            )
            if self.perf: self.perf.end_timer('tracking_update')

            current_frame_active_byte_ids_seen_by_wrapper = {track['id'] for track in online_tracks_display_scaled}
            current_frame_vehicle_map_for_drawing = {}

            with self.tracker_lock:
                for track_data in online_tracks_display_scaled:
                    byte_track_id = track_data['id']
                    bbox_xyxy_display = track_data['box_xyxy_display']
                    vehicle_type_name_from_yolo = track_data['class_name']
                    x1d, y1d, x2d, y2d = bbox_xyxy_display
                    current_bottom_center_display = ((x1d + x2d) // 2, y2d)

                    prev_pos_for_zone_check_display = self.path_manager.update_track_sighting(
                        byte_track_id, current_bottom_center_display, vehicle_type_name_from_yolo,
                        current_timestamp_dt, frame_time_video_sec
                    )
                    path_data_current = self.path_manager.get_path_data(byte_track_id)
                    consistent_type_for_drawing = path_data_current.get('type', vehicle_type_name_from_yolo) if path_data_current else vehicle_type_name_from_yolo
                    current_frame_vehicle_map_for_drawing[byte_track_id] = {
                        'center': current_bottom_center_display, 'box': bbox_xyxy_display,
                        'type': consistent_type_for_drawing,
                        'event_marker': None, 'status_for_drawing': 'detected'
                    }

                    event_type, event_dir = None, None
                    if prev_pos_for_zone_check_display and self.zone_tracker and self.zone_tracker.zones:
                        if self.perf: self.perf.start_timer('zone_checking')
                        event_type, event_dir = self.zone_tracker.check_zone_transition(
                            prev_pos_for_zone_check_display, bbox_xyxy_display,
                            byte_track_id, frame_time_video_sec
                        )
                        if self.perf: self.perf.end_timer('zone_checking')

                    if event_type == "ENTRY":
                        entry_success, entry_log = self.path_manager.register_entry_event(
                            byte_track_id, event_dir, current_timestamp_dt, current_bottom_center_display
                        )
                        if entry_success and entry_log:
                            entry_log['frame_number'] = frame_num
                            frame_local_event_log_for_this_iter.append(entry_log)
                            if self.perf: self.perf.record_vehicle_entry()
                            current_frame_vehicle_map_for_drawing[byte_track_id].update({'event_marker': 'ENTRY', 'status_for_drawing': 'entering'})
                    elif event_type == "EXIT":
                        active_zones = list(self.zone_tracker.zones.keys()) if self.zone_tracker and self.zone_tracker.zones else []
                        exit_success, exit_log = self.path_manager.register_exit_event(
                            byte_track_id, event_dir, current_timestamp_dt, current_bottom_center_display, active_zones
                        )
                        if exit_success and exit_log:
                            exit_log['frame_number'] = frame_num
                            frame_local_event_log_for_this_iter.append(exit_log)
                            if self.perf: self.perf.record_vehicle_exit('exited', exit_log.get('time_in_intersection'))
                            current_frame_vehicle_map_for_drawing[byte_track_id].update({'event_marker': 'EXIT', 'status_for_drawing': 'exiting'})

                timed_out_event_logs = self.path_manager.check_timeouts(
                    current_timestamp_dt, current_frame_active_byte_ids_seen_by_wrapper
                )
                for log_entry in timed_out_event_logs: # This log_entry is from PathManager
                    log_entry['frame_number'] = frame_num
                    frame_local_event_log_for_this_iter.append(log_entry)
                    if self.perf: self.perf.record_vehicle_exit('timed_out')
                    if log_entry['vehicle_id'] in current_frame_vehicle_map_for_drawing:
                         current_frame_vehicle_map_for_drawing[log_entry['vehicle_id']]['status_for_drawing'] = 'timed_out'
            batch_event_logs.extend(frame_local_event_log_for_this_iter)

            if ENABLE_VISUALIZATION and output_frame_viz is not None:
                if self.perf: self.perf.start_timer('drawing')
                if self.zone_tracker and self.zone_tracker.zones:
                    output_frame_viz = visual_overlay.draw_zones(output_frame_viz, self.zone_tracker)

                for track_id_viz, data_viz_frame in current_frame_vehicle_map_for_drawing.items():
                    path_info_pm = self.path_manager.get_path_data(track_id_viz)
                    draw_status_viz = data_viz_frame.get('status_for_drawing')
                    final_draw_type_viz = data_viz_frame.get('type')
                    entry_dir_viz, time_active_viz = None, None

                    if path_info_pm:
                        if draw_status_viz == 'detected' and path_info_pm.get('status') == 'active': draw_status_viz = 'active'
                        entry_dir_viz = path_info_pm.get('entry_direction')
                        if path_info_pm.get('status') == 'active' and path_info_pm.get('entry_time'):
                            time_active_viz = (current_timestamp_dt - path_info_pm['entry_time']).total_seconds()
                        final_draw_type_viz = path_info_pm.get('type', final_draw_type_viz)

                    should_draw_box_viz = True
                    is_fully_completed_path = path_info_pm and path_info_pm.get('status') in ['exited', 'timed_out', 'forced_exit']
                    is_event_this_frame = draw_status_viz in ['exiting', 'timed_out', 'entering']
                    if is_fully_completed_path and not is_event_this_frame:
                        should_draw_box_viz = False
                    
                    # If PathManager timed it out this frame, ensure status is 'timed_out'
                    # Check if log_entry (from previous loop) corresponds to current track_id_viz
                    # This logic might be tricky if multiple timeouts occur. Rely on status from current_frame_vehicle_map_for_drawing
                    # which was updated if a timeout log was generated for it.

                    if should_draw_box_viz:
                        visual_overlay.draw_detection_box(
                            output_frame_viz, data_viz_frame['box'], str(track_id_viz),
                            final_draw_type_viz, status=draw_status_viz,
                            entry_dir=entry_dir_viz, time_active=time_active_viz
                        )
                    if not is_fully_completed_path or is_event_this_frame:
                        trail = self.path_manager.get_tracking_trail(track_id_viz)
                        if trail: visual_overlay.draw_tracking_trail(output_frame_viz, trail, str(track_id_viz))
                    event_marker_viz = data_viz_frame.get('event_marker')
                    if event_marker_viz:
                        visual_overlay.draw_event_marker(output_frame_viz, data_viz_frame['center'], event_marker_viz, str(track_id_viz)[-4:])
                
                active_count_viz = self.path_manager.get_active_vehicle_count() if self.path_manager else 0
                # Ensure visual_overlay can handle dummy object or just the count
                dummy_tracker_for_overlay = type(
                        'obj', 
                        (object,), 
                        {'get_active_vehicle_count': lambda _: active_count_viz}
                    )() 
                output_frame_viz = visual_overlay.add_status_overlay(
                    output_frame_viz, frame_num, current_timestamp_dt, dummy_tracker_for_overlay
                )
                processed_frames_for_viz_list[i] = output_frame_viz
                if self.perf: self.perf.end_timer('drawing')
        return batch_event_logs, processed_frames_for_viz_list

    # --- Main Processing Function ---
    def process_video(self, video_path, start_date_str, start_time_str, primary_direction):
        if not self.processing_lock.acquire(blocking=False):
            print("Warning: Processing lock already acquired."); return "Processing is already in progress."

        self.byte_tracker_wrapper = None # Initialized after getting video_props
        self.path_manager = VehiclePathManager()
        self.zone_tracker = None # Initialized after getting zone_polys
        self.perf = PerformanceTracker(enabled=ENABLE_DETAILED_PERFORMANCE_METRICS)
        self.stop_event = threading.Event()
        self.frame_read_queue = queue.Queue(maxsize=max(self.batch_size * 4, 64)) # Adjusted queue size
        self.last_cleanup_time = time.monotonic()
        self.detection_zones_polygons = None
        output_path_final_video = None; self.final_output_path = None; self.writer_thread = None; self.video_write_queue = None

        print(f"--- Starting New Video Processing Run (ByteTrack) ---");
        # (Print statements for video, timestamp, zone_mode, viz_enabled)

        actual_video_path_to_process = video_path; temp_converted_dav_mp4 = None
        if video_path.lower().endswith(".dav"):
            if not FFMPEG_PATH or not os.path.exists(FFMPEG_PATH): raise RuntimeError("FFmpeg not found or path invalid for .dav conversion.")
            converted_path = convert_dav_to_mp4_internal(video_path)
            if converted_path and os.path.exists(converted_path):
                actual_video_path_to_process = converted_path; temp_converted_dav_mp4 = converted_path
                print(f".dav converted to: {actual_video_path_to_process}"); time.sleep(0.5) # File system settle
            else: raise RuntimeError(f"Failed to convert .dav file: {video_path}")

        try:
            # --- Determine and Validate Zones ---
            if self.line_mode == 'hardcoded':
                if 'LINE_POINTS' not in globals() or not isinstance(LINE_POINTS, dict) or not LINE_POINTS: raise ValueError("Hardcoded: LINE_POINTS missing/empty in config.")
                valid_polygons_hc = { d: p for d, p in LINE_POINTS.items() if isinstance(p, list) and len(p) >= 3 and all(isinstance(pt, tuple) and len(pt)==2 for pt in p) }
                if not valid_polygons_hc: raise ValueError("Hardcoded: No valid polygons in LINE_POINTS.")
                self.detection_zones_polygons = { d: np.array(p, dtype=np.int32) for d, p in valid_polygons_hc.items() }
            elif self.line_mode == 'interactive':
                if not self.gradio_polygons: raise ValueError("Interactive: No zone polygons defined.")
                valid_polygons_gr = { d: p for d, p in self.gradio_polygons.items() if p and len(p) >= 3 }
                if len(valid_polygons_gr) < 1: raise ValueError(f"Interactive: Need >= 1 valid zone. Found {len(valid_polygons_gr)}.") # At least 1 zone for any processing
                self.detection_zones_polygons = { d: np.array(p, dtype=np.int32) for d, p in valid_polygons_gr.items() }
            else: raise ValueError(f"Invalid LINE_MODE '{self.line_mode}'.")
            if not self.detection_zones_polygons: raise ValueError("Zone polygon data missing or invalid after setup.")
            self.zone_tracker = ZoneTracker(self.detection_zones_polygons)
            if not self.zone_tracker or not self.zone_tracker.zones: raise RuntimeError("CRITICAL: ZoneTracker failed to initialize with zones.")

            # --- Init Model, Time, Video Props ---
            if self.model is None: self.model = load_model(MODEL_PATH, self.device, CONF_THRESHOLD, IOU_THRESHOLD)
            if self.model is None: raise RuntimeError(f"Failed to load model {MODEL_PATH}")
            try:
                start_date_dt=datetime.strptime(start_date_str, "%y%m%d"); time_part,ms_part=(start_time_str[:6], start_time_str[6:]) if len(start_time_str)>6 else (start_time_str,"000")
                base_time_dt=datetime.strptime(time_part,"%H%M%S").time(); microseconds=int(ms_part.ljust(3,'0'))*1000
                start_time_dt=base_time_dt.replace(microsecond=microseconds); start_datetime_obj=datetime.combine(start_date_dt.date(), start_time_dt)
                print(f"Video Start Timestamp: {start_datetime_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            except ValueError as e: raise ValueError(f"Invalid date/time format: {e}.")

            total_frames, original_fps = self._get_video_properties(actual_video_path_to_process)
            print(f"Video Properties: ~{total_frames} frames, {original_fps:.2f} FPS (Original)")
            if total_frames <= 0: raise ValueError("Cannot process video with zero or unreadable total frames.")

            # --- Initialize ByteTrackWrapper ---
            frame_rate_for_bt = original_fps if original_fps > 0 else self.target_fps # Use original if available, else target
            self.byte_tracker_wrapper = ByteTrackWrapper(self.bytetrack_args, frame_rate_for_bt, self.frame_shape) # self.frame_shape is (Display_W, Display_H)

            # --- Setup Output Path and Start I/O Threads ---
            reader_args = (actual_video_path_to_process, original_fps, total_frames) if self.actually_use_advanced_reader else (actual_video_path_to_process, original_fps)
            reader_target = self._frame_reader_task_pvr_segmented if self.actually_use_advanced_reader else self._frame_reader_task
            self.reader_thread = threading.Thread(target=reader_target, args=reader_args, daemon=True); self.reader_thread.start()
            if ENABLE_VISUALIZATION:
                os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True); ts_now=datetime.now().strftime("%Y%m%d_%H%M%S")
                fname_base=f"output_{os.path.splitext(os.path.basename(actual_video_path_to_process))[0]}_{ts_now}"
                output_path_final_video=os.path.join(VIDEO_OUTPUT_DIR, f"{fname_base}.mp4");
                self.final_output_path = output_path_final_video
                self.video_write_queue = queue.Queue(maxsize=max(self.batch_size*2, 32)) # Adjusted queue size
                self.writer_thread = threading.Thread(target=self._frame_writer_task, args=(output_path_final_video,), daemon=True); self.writer_thread.start()

            # --- Main Loop ---
            all_event_logs_collected = []; frames_processed_in_loop = 0
            # Frame interval based on original FPS and target display FPS
            frame_skip_interval = max(1, round(original_fps / self.target_fps)) if self.target_fps > 0 and original_fps > 0 else 1
            total_target_frames_for_perf = total_frames // frame_skip_interval if total_frames > 0 and frame_skip_interval > 0 else total_frames # Estimate for perf
            if self.perf: self.perf.start_processing(total_target_frames_for_perf)
            
            frames_batch_content_metadata, batch_frame_numbers, batch_frame_times = [], [], []
            last_progress_print_time = time.monotonic()
            print(f"Effective Target FPS: {self.target_fps:.1f} (processing every ~{frame_skip_interval} frame(s) from input at {original_fps:.1f} FPS).")

            active_gets_pending_task_done = 0 # Counter for items gotten but not yet task_done'd

            while True:
                try:
                    frame_data_item = self.frame_read_queue.get(timeout=60.0)
                    active_gets_pending_task_done += 1 # Increment for each successful get
                    if frame_data_item is None:
                        self.frame_read_queue.task_done()
                        active_gets_pending_task_done -= 1
                        break
                    
                    if self.perf: self.perf.start_timer('video_read')
                    frame_content_reader, raw_frame_num, video_time_s, reader_meta = frame_data_item
                    if self.perf: self.perf.end_timer('video_read')
                    if frame_content_reader is None:
                        self.frame_read_queue.task_done()
                        active_gets_pending_task_done -= 1
                        continue

                    frames_batch_content_metadata.append((frame_content_reader, reader_meta))
                    batch_frame_numbers.append(raw_frame_num)
                    batch_frame_times.append(video_time_s)

                    if len(frames_batch_content_metadata) >= self.batch_size:
                        batch_proc_start_time = time.monotonic()
                        if self.perf: self.perf.start_timer('preprocessing')
                        yolo_input_tensor = self._preprocess_batch(frames_batch_content_metadata)
                        self.perf.end_timer('preprocessing')

                        if yolo_input_tensor is None or yolo_input_tensor.nelement() == 0:
                            # Mark tasks done for frames in this empty batch
                            for _ in range(len(frames_batch_content_metadata)): self.frame_read_queue.task_done()
                            frames_batch_content_metadata.clear(); batch_frame_numbers.clear(); batch_frame_times.clear()
                            continue

                        # --- YOLO Inference ---
                        active_stream=None
                        if self.device=='cuda' and self.streams: active_stream=self.streams[self.current_stream_index%len(self.streams)]; self.current_stream_index+=1
                        with torch.cuda.stream(active_stream) if active_stream else torch.no_grad():
                           with torch.amp.autocast(device_type=self.device,enabled=MIXED_PRECISION and self.device=='cuda'):
                             if self.perf and self.perf.start_event: self.perf.start_event.record(stream=active_stream)
                             yolo_batch_results = self.model(yolo_input_tensor, verbose=False)
                             if self.perf and self.perf.end_event: self.perf.end_event.record(stream=active_stream)
                        if active_stream: active_stream.synchronize()
                        if self.perf and self.perf.start_event and self.perf.end_event : self.perf.record_inference_time_gpu(self.perf.start_event, self.perf.end_event)
                        
                        current_batch_events, viz_batch_frames = self._process_inference_results(
                            yolo_batch_results, frames_batch_content_metadata,
                            batch_frame_numbers, batch_frame_times, start_datetime_obj
                        )
                        all_event_logs_collected.extend(current_batch_events)
                        if self.perf: self.perf.record_detection(sum(len(r.boxes) for r in yolo_batch_results if hasattr(r,'boxes') and r.boxes))

                        if ENABLE_VISUALIZATION and self.video_write_queue and viz_batch_frames:
                            if self.perf: self.perf.start_timer('video_write')
                            for frame_to_write in viz_batch_frames:
                                if frame_to_write is not None:
                                    try: self.video_write_queue.put(frame_to_write, block=True, timeout=2.0)
                                    except queue.Full: print("WARN: Video writer queue full, frame dropped.")
                            if self.perf: self.perf.end_timer('video_write')

                        current_batch_proc_time = time.monotonic() - batch_proc_start_time
                        frames_processed_in_loop += len(frames_batch_content_metadata)
                        if self.perf: self.perf.record_batch_processed(len(frames_batch_content_metadata), current_batch_proc_time); self.perf.sample_system_metrics()
                        
                        mono_now = time.monotonic()
                        if mono_now - self.last_cleanup_time > MEMORY_CLEANUP_INTERVAL:
                            if self.perf: self.perf.start_timer('memory_cleanup')
                            if self.zone_tracker: self.zone_tracker.cleanup_cooldowns(batch_frame_times[-1] if batch_frame_times else 0)
                            cleanup_memory()
                            self.last_cleanup_time = mono_now
                            if self.perf: self.perf.end_timer('memory_cleanup')

                        if self.perf and (mono_now - last_progress_print_time > 1.0):
                            prog_stats=self.perf.get_progress(); active_pm = self.path_manager.get_active_vehicle_count() if self.path_manager else 0
                            print(f"\rProgress:{prog_stats['percent']:.1f}%|FPS:{prog_stats['fps']:.1f}|Active(PM):{active_pm}|ETA:{prog_stats['eta']}   ", end="")
                            last_progress_print_time = mono_now
                        
                        # Mark tasks done for frames processed in this batch
                        for _ in range(len(frames_batch_content_metadata)):
                            self.frame_read_queue.task_done()
                            active_gets_pending_task_done -= 1

                        frames_batch_content_metadata.clear(); batch_frame_numbers.clear(); batch_frame_times.clear()
                        del yolo_input_tensor, yolo_batch_results; # cleanup_memory()
                    
                    # else: # Frame added to batch, but batch not full yet. Mark this single frame task as done.
                    #     self.frame_read_queue.task_done()

                except queue.Empty:
                    debug_print("Main loop: Frame reader queue timed out. Checking stop event.")
                    if self.stop_event.is_set() and (self.frame_read_queue is None or self.frame_read_queue.empty()): break
                    continue # Continue waiting if not stopped and queue might still get items
                except Exception as e_loop: print(f"\nERROR during processing loop: {e_loop}"); import traceback; traceback.print_exc(); self.stop_event.set(); break
            
            # --- Process Final Batch ---
            if frames_batch_content_metadata: # (Similar logic as main batch processing loop)
                print("\nProcessing final batch...")
                batch_proc_start_time=time.monotonic()
                if self.perf: self.perf.start_timer('preprocessing'); yolo_input_tensor=self._preprocess_batch(frames_batch_content_metadata); self.perf.end_timer('preprocessing')
                if yolo_input_tensor is not None and yolo_input_tensor.nelement() > 0:
                    # ... (YOLO Inference for final batch - same as above) ...
                    active_stream=None
                    if self.device=='cuda' and self.streams: active_stream=self.streams[self.current_stream_index%len(self.streams)]; self.current_stream_index+=1
                    with torch.cuda.stream(active_stream) if active_stream else torch.no_grad():
                        with torch.amp.autocast(device_type=self.device,enabled=MIXED_PRECISION and self.device=='cuda'):
                            if self.perf and self.perf.start_event: self.perf.start_event.record(stream=active_stream)
                            yolo_batch_results = self.model(yolo_input_tensor, verbose=False)
                            if self.perf and self.perf.end_event: self.perf.end_event.record(stream=active_stream)
                        if active_stream: active_stream.synchronize()
                        if self.perf and self.perf.start_event and self.perf.end_event: self.perf.record_inference_time_gpu(self.perf.start_event, self.perf.end_event)

                    current_batch_events, viz_batch_frames = self._process_inference_results(
                        yolo_batch_results, frames_batch_content_metadata,
                        batch_frame_numbers, batch_frame_times, start_datetime_obj
                    )
                    all_event_logs_collected.extend(current_batch_events)
                    if ENABLE_VISUALIZATION and self.video_write_queue and viz_batch_frames: # Write final viz frames
                        if self.perf: self.perf.start_timer('video_write');
                        for frame_to_write in viz_batch_frames:
                            if frame_to_write is not None:
                                try: self.video_write_queue.put(frame_to_write, block=True, timeout=2.0)
                                except queue.Full: print("WARN: Video writer queue full (final batch).")
                        if self.perf: self.perf.end_timer('video_write')
                    current_batch_proc_time=time.monotonic()-batch_proc_start_time; frames_processed_in_loop+=len(frames_batch_content_metadata)
                    if self.perf: self.perf.record_batch_processed(len(frames_batch_content_metadata),current_batch_proc_time)
                    del yolo_input_tensor, yolo_batch_results
                else: debug_print("Skipping empty final batch.")
                # Mark tasks done for frames in this final batch
                for _ in range(len(frames_batch_content_metadata)): self.frame_read_queue.task_done(); active_gets_pending_task_done -= 1
            
            print(f"\nVideo processing loop finished. Active gets pending task_done: {active_gets_pending_task_done}")

            # --- Finalize & Report ---
            if self.path_manager:
                last_ts_for_force = start_datetime_obj + timedelta(seconds=batch_frame_times[-1]) if batch_frame_times else start_datetime_obj
                forced_logs = self.path_manager.force_exit_all_remaining(last_ts_for_force)
                last_fnum_approx = batch_frame_numbers[-1] if batch_frame_numbers else (total_frames - 1 if total_frames > 0 else 0)
                for log_item in forced_logs:
                    log_item['frame_number'] = last_fnum_approx
                    if self.perf: self.perf.record_vehicle_exit('forced_exit')
                all_event_logs_collected.extend(forced_logs)

            self.stop_event.set() # Signal threads
            # (Signal writer and reader queues, wait for threads - same as before)
            if ENABLE_VISUALIZATION and self.video_write_queue:
                try: self.video_write_queue.put(None, block=True, timeout=5.0) # Block to ensure it's received
                except queue.Full: print("WARN: Could not signal writer queue (full) on exit.")
            if self.frame_read_queue and not self.frame_read_queue.empty(): # Drain if reader is stuck
                debug_print(f"Draining reader queue ({self.frame_read_queue.qsize()} items) before join...")
                while not self.frame_read_queue.empty():
                    try: self.frame_read_queue.get_nowait(); self.frame_read_queue.task_done()
                    except queue.Empty: break
            if self.writer_thread and self.writer_thread.is_alive(): self.writer_thread.join(timeout=30.0)
            if self.reader_thread and self.reader_thread.is_alive(): self.reader_thread.join(timeout=10.0)


            # (Performance summary and report generation - same as before using self.path_manager)
            final_paths_for_report = self.path_manager.get_completed_paths_for_report() if self.path_manager else []
            # Calculate completed_paths_count_valid for STATS based on final_paths_for_report
            completed_paths_count_valid = 0
            for path_rep in final_paths_for_report:
                status_rep=path_rep.get('status'); entry_dir_rep=path_rep.get('entry_direction'); exit_dir_rep=path_rep.get('exit_direction')
                is_valid_entry_rep=entry_dir_rep and entry_dir_rep not in ['UNKNOWN',None]
                is_valid_exit_rep=exit_dir_rep and exit_dir_rep not in ['UNKNOWN','TIMEOUT','FORCED', 'FORCED_EXIT',None]
                if status_rep=='exited' and is_valid_entry_rep and is_valid_exit_rep: completed_paths_count_valid+=1
            
            total_time_s = 0.0; final_avg_fps = 0.0
            if self.perf: self.perf.end_processing(); self.perf.print_summary()
            total_time_s=self.perf.total_time if self.perf else 0.0
            final_avg_fps=frames_processed_in_loop/total_time_s if total_time_s > 0 else 0.0

            excel_report_path = create_excel_report(
                completed_paths_data=final_paths_for_report,
                start_datetime=start_datetime_obj, primary_direction=primary_direction,
                video_path=actual_video_path_to_process
            )
            if ENABLE_DETAILED_PERFORMANCE_METRICS and self.perf: visualize_performance(self.perf)
            
            result_message_str=f"✅ Processing completed.\n"
            # (Append video path, excel_report_path, perf charts path to result_message_str - same as before)
            if ENABLE_VISUALIZATION:
                if self.final_output_path and "ERROR" not in self.final_output_path: result_message_str+=f"Output video: '{self.final_output_path}'.\n"
                elif self.final_output_path and "ERROR" in self.final_output_path: result_message_str+=f"Output video generation failed: {self.final_output_path}.\n"
            if excel_report_path: result_message_str+=f"Excel report: {excel_report_path}\n"
            if ENABLE_DETAILED_PERFORMANCE_METRICS and self.perf: result_message_str+=f"Perf charts: 'performance_charts/'\n"
            result_message_str+=f"\n--- Summary Stats ---\nSTAT_FRAMES_PROCESSED={frames_processed_in_loop}\nSTAT_TIME_SECONDS={total_time_s:.2f}\nSTAT_FPS={final_avg_fps:.2f}\nSTAT_COMPLETED_PATHS={completed_paths_count_valid}\n--- End Stats ---"
            return result_message_str

        except (ValueError, RuntimeError, FileNotFoundError, Exception) as e_outer:
            print(f"\nFATAL ERROR during video processing setup or main execution: {e_outer}")
            import traceback; traceback.print_exc()
            if hasattr(self, 'stop_event') and self.stop_event: self.stop_event.set()
            return f"❌ Error: {e_outer}"
        finally: # Cleanup (same as before, ensure instances are reset)
            # (Thread joins, DAV cleanup, instance resets, lock release)
            try:
                if hasattr(self, 'reader_thread') and self.reader_thread and self.reader_thread.is_alive(): self.reader_thread.join(timeout=1.0)
                if hasattr(self, 'writer_thread') and self.writer_thread and self.writer_thread.is_alive(): self.writer_thread.join(timeout=1.0)
            except Exception as join_e_fin: print(f"Error joining threads in final cleanup: {join_e_fin}")

            if hasattr(self, 'frame_read_queue') and self.frame_read_queue is not None:
                # Attempt to gracefully handle any remaining items if reader thread was faster
                # or if an error broke the main loop prematurely.
                # This is more about cleanup than strict task_done balancing for normal operation.
                remaining_in_q = 0
                try:
                    remaining_in_q = self.frame_read_queue.qsize() # Approx
                except (NotImplementedError, AttributeError): # qsize might not be available on all platforms/Pythons for multiprocessing queues
                    pass
                
                if remaining_in_q > 0:
                    debug_print(f"Final cleanup: Draining reader queue ({remaining_in_q} approx. items)...")
                cleaned_in_drain = 0
                while True: # Drain until empty or error
                    try:
                        # Only get, don't task_done here unless you are sure these items
                        # were never accounted for by the main loop's get.
                        # If these are "new" items put by reader after consumer stopped,
                        # then a get + task_done would be okay.
                        # However, to be safe and avoid the ValueError if they were already processed
                        # or were the sentinel 'None' that was already processed:
                        _ = self.frame_read_queue.get_nowait() 
                        # If we successfully get an item here, it means the main loop missed it
                        # or the reader produced it after the main loop exited.
                        # If an item is gotten here, it *does* need a task_done().
                        self.frame_read_queue.task_done()
                        cleaned_in_drain +=1
                    except queue.Empty:
                        break # Queue is empty, normal exit from drain
                    except ValueError as ve_drain: # Catch task_done() too many times specifically
                        print(f"Warning during final queue drain: {ve_drain}. Item might have been processed already.")
                        break # Stop draining if task_done fails
                    except Exception as e_drain:
                        print(f"Error during final queue drain: {e_drain}")
                        break # Stop draining on other errors
                if cleaned_in_drain > 0:
                    debug_print(f"Drained and task_done {cleaned_in_drain} items from reader queue in final cleanup.")

            if temp_converted_dav_mp4 and os.path.exists(temp_converted_dav_mp4):
                try: debug_print(f"Final cleanup: Removing temporary DAV file: {temp_converted_dav_mp4}"); os.remove(temp_converted_dav_mp4)
                except OSError as e_del_fin: debug_print(f"Error deleting temporary DAV file in final cleanup {temp_converted_dav_mp4}: {e_del_fin}")

            self.model=None; self.byte_tracker_wrapper=None; self.path_manager=None; self.zone_tracker=None; self.perf=None
            self.detection_zones_polygons=None; self.frame_read_queue=None; self.video_write_queue=None
            self.reader_thread=None; self.writer_thread=None; self.stop_event=None; self.final_output_path = None
            cleanup_memory()
            if self.processing_lock.locked():
                try: self.processing_lock.release(); print("Processing finished, lock released.")
                except RuntimeError as release_err_fin: print(f"Warn: Error releasing lock in final cleanup: {release_err_fin}")
            # else: print("Processing finished (lock not held or already released in final block).")
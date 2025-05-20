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

# Configuration
from config import (
    TARGET_FPS, FRAME_WIDTH, FRAME_HEIGHT, OPTIMAL_BATCH_SIZE,
    VIDEO_OUTPUT_DIR, ENABLE_DETAILED_PERFORMANCE_METRICS,
    MIXED_PRECISION, PARALLEL_STREAMS, THREAD_COUNT, # THREAD_COUNT less relevant for this part
    MEMORY_CLEANUP_INTERVAL, MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD,
    LINE_MODE, LINE_POINTS, MODEL_INPUT_SIZE, ENABLE_VISUALIZATION,
    USE_ADVANCED_VIDEO_READER, PYVIDEOREADER_DECODE_SEGMENT_SIZE, FFMPEG_PATH,
    BYTETRACK_TRACK_THRESH, BYTETRACK_TRACK_HIGH_THRESH, BYTETRACK_NEW_TRACK_THRESH,
    BYTETRACK_MATCH_THRESH_IOU, BYTETRACK_TRACK_BUFFER_AGE_FRAMES 
)


# Modules
from core.utils import debug_print, format_timestamp, is_valid_movement, cleanup_memory
from core.performance import PerformanceTracker
from models.model_loader import load_model, get_device
from tracking.vehicle_tracker import VehicleTracker # App-level tracker
from tracking.bytetrack_logic import ByteTracker, STrack    # Core ByteTrack
from tracking.zone_tracker import ZoneTracker
from tracking.cleanup import cleanup_tracking_data # Revised cleanup
from visualization import overlay as visual_overlay
from reporting.excel_report import create_excel_report
from core.converters import convert_dav_to_mp4_internal

# Conditional imports
if ENABLE_DETAILED_PERFORMANCE_METRICS:
    from visualization.performance_viz import visualize_performance

# Attempt to import PyVideoReader
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
        self.frame_shape = (FRAME_WIDTH, FRAME_HEIGHT)
        self.model_input_size_config = MODEL_INPUT_SIZE # Square size for model
        self.batch_size = OPTIMAL_BATCH_SIZE
        self.processing_lock = threading.Lock()
        self.line_mode = LINE_MODE # Controls zone definition method
        # self.max_workers = THREAD_COUNT

        # Gradio Polygon State
        self.gradio_polygons = {} # Stores {direction: List[tuple(x,y)]}
        self.current_gradio_direction = None
        self.gradio_temp_points = []
        self.original_frame_for_gradio = None
        self.last_drawn_polygons_frame = None

        # Core components
        self.device = get_device()
        self.model = None
        self.vehicle_tracker = None
        self.byte_tracker = None    # Core ByteTrack instance
        self.zone_tracker = None
        self.perf = None
        self.detection_zones_polygons = None # Holds final polygons (NumPy arrays)
        self.model_input_size_config = MODEL_INPUT_SIZE
        self.tracker_lock = threading.Lock()

        # Threading & Queues
        self.frame_read_queue = None
        self.video_write_queue = None # Will be None if visualization is disabled
        self.stop_event = None
        self.reader_thread = None
        self.writer_thread = None # Will be None if visualization is disabled
        self.last_cleanup_time = 0
        self.final_output_path = None # Store the actual path used by the writer

        # CUDA Streams
        self.streams = None
        if self.device == 'cuda' and PARALLEL_STREAMS > 1:
            self.streams = [torch.cuda.Stream() for _ in range(PARALLEL_STREAMS)]
        self.current_stream_index = 0

        # Determine if advanced reader will be used based on config and availability
        self.pvr_decode_segment_size = PYVIDEOREADER_DECODE_SEGMENT_SIZE
        self.actually_use_advanced_reader = USE_ADVANCED_VIDEO_READER and PYVIDEOREADER_AVAILABLE
        if self.actually_use_advanced_reader:
            print("Configuration: Advanced video reader (PyVideoReader) will be used.")
        else:
            if USE_ADVANCED_VIDEO_READER and not PYVIDEOREADER_AVAILABLE:
                print("Configuration: Advanced video reader was requested but is not available. Using OpenCV.")
            else: # USE_ADVANCED_VIDEO_READER is False
                print("Configuration: Using OpenCV video reader.")

    # --- Gradio Polygon Related Methods ---
    def _draw_polygons_for_gradio(self, frame_to_draw_on):
        if frame_to_draw_on is None: return None
        img_copy = frame_to_draw_on.copy()
        colors = {'north': (255,0,0), 'south': (0,255,0), 'east': (0,0,255), 'west': (255,0,255)} # BGR
        default_color = (255, 255, 255); temp_color = (0, 255, 255)

        # Draw completed polygons
        for direction, points in self.gradio_polygons.items():
            if points and len(points) >= 3:
                pts_np = np.array(points, dtype=np.int32)
                color = colors.get(direction, default_color)
                cv2.polylines(img_copy, [pts_np], isClosed=True, color=color, thickness=2)
                try:
                    if len(pts_np) > 0:
                        M = cv2.moments(pts_np)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"]); cY = int(M["m01"] / M["m00"])
                            cv2.putText(img_copy, direction.upper(), (cX-15, cY+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                except Exception: pass # Ignore label drawing errors

        # Draw temporary points
        if self.current_gradio_direction and self.gradio_temp_points:
            current_temp_color = colors.get(self.current_gradio_direction, temp_color)
            for i, pt in enumerate(self.gradio_temp_points):
                cv2.circle(img_copy, pt, 5, current_temp_color, -1)
                if i > 0: cv2.line(img_copy, self.gradio_temp_points[i-1], pt, current_temp_color, 2)
            if len(self.gradio_temp_points) >= 2: # Closing hint
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
        if direction_lower in self.gradio_polygons:
             del self.gradio_polygons[direction_lower]
             print(f"Cleared existing polygon for {direction.capitalize()} to redraw.")
        self.current_gradio_direction = direction_lower
        self.gradio_temp_points = []
        status = f"Selected '{direction.capitalize()}'. Click points on image to define zone polygon."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
        return frame_to_show, status

    def handle_gradio_polygon_click(self, evt: 'gr.SelectData'):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if self.current_gradio_direction is None: return self.last_drawn_polygons_frame, "Select a direction first."
        if self.original_frame_for_gradio is None: return None, "No video loaded."
        point = (evt.index[0], evt.index[1])
        self.gradio_temp_points.append(point)
        status = f"Added point {len(self.gradio_temp_points)} for {self.current_gradio_direction.capitalize()}. Click more points or 'Finish Polygon'."
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
        return frame_to_show, status

    def finalize_current_polygon(self):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if not self.current_gradio_direction: return self.last_drawn_polygons_frame, "No direction selected to finalize."
        if len(self.gradio_temp_points) < 3:
            status = f"Error: Need at least 3 points for a polygon (got {len(self.gradio_temp_points)})."
            frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
            return frame_to_show, status
        self.gradio_polygons[self.current_gradio_direction] = self.gradio_temp_points.copy()
        status = f"Polygon for {self.current_gradio_direction.capitalize()} saved. Select next direction or Process."
        self.current_gradio_direction = None
        self.gradio_temp_points = []
        frame_to_show = self._draw_polygons_for_gradio(self.original_frame_for_gradio)
        return frame_to_show, status

    def undo_last_polygon_point(self):
        if self.line_mode != 'interactive': return self.last_drawn_polygons_frame, "Drawing disabled."
        if not self.current_gradio_direction: status = "Select a direction first."
        elif not self.gradio_temp_points: status = f"No points added yet for {self.current_gradio_direction.capitalize()}."
        else:
            removed_point = self.gradio_temp_points.pop()
            status = f"Removed last point for {self.current_gradio_direction.capitalize()}."
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
        """Gets total frames and original FPS, trying PyVideoReader first if enabled."""
        total_frames = 0
        original_fps = 0.0

        if self.actually_use_advanced_reader:
            temp_reader_rs = None
            try:
                # For getting properties, no need for device='cuda' or specific resize
                temp_reader_rs = PyVideoReader(video_path)
                info = temp_reader_rs.get_info()
                total_frames = int(info.get('frame_count', 0))
                fps_str = info.get('fps', '0.0')
                if fps_str:
                    try: original_fps = float(fps_str)
                    except ValueError: pass
                del temp_reader_rs 
            except Exception as e:
                debug_print(f"PyVideoReader property fetch failed: {e}. Falling back to OpenCV for props.")
                if temp_reader_rs: del temp_reader_rs
        
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

        if original_fps <= 0: original_fps = 30.0 # Final fallback
        if total_frames <= 0:
            print("WARNING: Could not determine total_frames accurately. Progress calculation might be affected.")   
        return total_frames, original_fps

    def _frame_reader_task(self, video_path, original_fps):
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): print("ERROR: Reader failed to open video."); self.frame_read_queue.put(None); return
            frame_interval = max(1, round(original_fps / self.target_fps))
            frame_count_raw = 0
            frames_yielded = 0
            while not self.stop_event.is_set():
                ret, frame_bgr_orig_size = cap.read() # frame is BGR HWC NumPy array
                if not ret: 
                    debug_print("OpenCV Reader: End of video or read error.")
                    break # End of video or error
                if frame_count_raw  % frame_interval == 0:
                    frame_time_seconds = frame_count_raw  / original_fps
                    # Metadata for OpenCV frames: not RGB, source is opencv
                    metadata = {'is_rgb': False, 'source': 'opencv'}
                    try:
                        # Add frame to queue for processing
                        self.frame_read_queue.put((frame_bgr_orig_size, frame_count_raw , frame_time_seconds, metadata), block=True, timeout=5.0)
                        frames_yielded += 1
                    except queue.Full:
                        debug_print("Reader queue full, waiting...")
                        time.sleep(0.1)
                        if self.stop_event.is_set(): break
                        try: # Retry put
                            self.frame_read_queue.put((frame_bgr_orig_size, frame_count_raw , frame_time_seconds, metadata), block=True, timeout=5.0)
                            frames_yielded += 1
                        except queue.Full: print("ERROR: Reader queue persistently full."); self.stop_event.set(); break
                    except Exception as q_e: 
                        debug_print(f"OpenCV Reader: Queue put error: {q_e}")
                        self.stop_event.set() # Signal main loop
                        break
                frame_count_raw  += 1
            debug_print(f"OpenCV Reader finished. Raw frames read: {frame_count_raw}. Frames yielded to queue: {frames_yielded}.")
        except Exception as e: 
            print(f"FATAL ERROR in _frame_reader_task (OpenCV): {e}")
            import traceback
            traceback.print_exc()
        finally:
            if cap: cap.release()
            if self.frame_read_queue:
                try:
                    # Ensure None is put to signal end, even if loop broke due to error or stop_event
                    self.frame_read_queue.put(None, block=False, timeout=1.0) 
                except queue.Full:
                    debug_print("OpenCV Reader: Queue full on final None put.")
                except Exception as e_fin:
                    debug_print(f"OpenCV Reader: Error on final None put: {e_fin}")

    def _frame_reader_task_pvr_segmented(self, video_path, original_fps_from_props, total_frames_from_props):
        """Frame reader using PyVideoReader, decoding in segments, then selecting frames."""
        pvr_instance = None
        try:
            target_dim_model = self.model_input_size_config
            debug_print(f"PVR SegReader: Init PyVideoReader: device='cuda', target_size=({target_dim_model}x{target_dim_model})")
            
            pvr_instance = PyVideoReader(video_path, device='cuda', 
                                         resize_shorter_side=target_dim_model,
                                         resize_longer_side=target_dim_model)
            
            frame_interval = max(1, round(original_fps_from_props / self.target_fps))
            queued_frame_count = 0
            
            num_segments = math.ceil(total_frames_from_props / self.pvr_decode_segment_size) if total_frames_from_props > 0 else 1
            if total_frames_from_props == 0 : num_segments = 0 # No frames to read

            debug_print(f"PVR SegReader: Video Length={total_frames_from_props} frames, "
                        f"Segment Size={self.pvr_decode_segment_size}, Num Segments={num_segments}")

            for i in range(num_segments):
                if self.stop_event.is_set(): break

                segment_start_raw_idx = i * self.pvr_decode_segment_size
                # For PyVideoReader.decode, end_frame is exclusive
                segment_end_raw_idx = min((i + 1) * self.pvr_decode_segment_size, total_frames_from_props)

                if segment_start_raw_idx >= segment_end_raw_idx : continue # Segment is empty or invalid

                debug_print(f"PVR SegReader: Decoding segment {i+1}/{num_segments} "
                            f"(raw frames {segment_start_raw_idx} to {segment_end_raw_idx-1})...")
                
                # Frames are RGB, (model_input_size, model_input_size, 3)
                segment_frames_np = pvr_instance.decode(start_frame=segment_start_raw_idx, 
                                                        end_frame=segment_end_raw_idx)

                if segment_frames_np is None or segment_frames_np.shape[0] == 0:
                    debug_print(f"PVR SegReader: Segment {i+1} returned no frames. Video might have ended or error.")
                    continue # Try next segment if any, or loop will end.

                # Select frames from this decoded segment based on frame_interval
                for frame_in_segment_idx in range(segment_frames_np.shape[0]):
                    if self.stop_event.is_set(): break

                    # This is the original, absolute frame index in the video
                    current_overall_raw_frame_idx = segment_start_raw_idx + frame_in_segment_idx
                    
                    if current_overall_raw_frame_idx % frame_interval == 0:
                        frame_content = segment_frames_np[frame_in_segment_idx]
                        current_frame_time_sec = current_overall_raw_frame_idx / original_fps_from_props
                        metadata = {'is_rgb': True, 'source': 'pvr_cuda_segmented'}
                        
                        try:
                            self.frame_read_queue.put((frame_content, current_overall_raw_frame_idx, current_frame_time_sec, metadata),
                                                      block=True, timeout=5.0)
                            queued_frame_count += 1
                        except queue.Full:
                            debug_print("PVR SegReader: Queue full, retrying...")
                            time.sleep(0.1);
                            if self.stop_event.is_set(): break
                            try:
                                self.frame_read_queue.put((frame_content, current_overall_raw_frame_idx, current_frame_time_sec, metadata),
                                                           block=True, timeout=5.0)
                                queued_frame_count += 1
                            except queue.Full: print("ERROR: PVR SegReader queue persistently full."); self.stop_event.set(); break # Stop reader
                        except Exception as q_e: debug_print(f"PVR SegReader queue put error: {q_e}"); self.stop_event.set(); break
                
                # Release segment memory if possible (NumPy array will be dereferenced when loop iterates)
                del segment_frames_np 
                gc.collect() # Suggest garbage collection

            debug_print(f"PVR SegReader finished. Queued {queued_frame_count} frames.")

        except Exception as e:
            print(f"FATAL ERROR in _frame_reader_task_pvr_segmented: {e}")
            import traceback; traceback.print_exc()
        finally:
            if pvr_instance: del pvr_instance # PyVideoReader typically doesn't have explicit close
            if self.frame_read_queue:
                try: self.frame_read_queue.put(None, block=False, timeout=1.0)
                except queue.Full: pass

    def _frame_writer_task(self, output_path):
        # This task will only run if ENABLE_VISUALIZATION is True
        out = None; writer_initialized = False; frames_written = 0
        actual_output_path = output_path
        try:
            # Attempt to initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v');
            out = cv2.VideoWriter(actual_output_path, fourcc, self.target_fps, self.frame_shape)
            if not out.isOpened():
                print(f"Warn: mp4v failed for {actual_output_path}. Trying XVID/AVI.");
                avi_path = os.path.splitext(actual_output_path)[0] + ".avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID');
                out = cv2.VideoWriter(avi_path, fourcc, self.target_fps, self.frame_shape)
                if out.isOpened():
                    actual_output_path = avi_path;
                    # Important: Update the class variable if the path changed
                    self.final_output_path = avi_path
                else:
                    print(f"ERROR: Writer failed to open for both MP4 and AVI.");
                    # Signal failure (e.g., by setting final_output_path to None or an error string)
                    self.final_output_path = "ERROR_WRITER_FAILED"
                    return # Exit thread if writer cannot be opened

            writer_initialized = True; print(f"Writer initialized for: {actual_output_path}")

            while True:
                try: frame_data = self.video_write_queue.get(timeout=5.0) # Wait for frames
                except queue.Empty:
                    # Check if processing is done and queue is empty
                    if self.stop_event is not None and self.stop_event.is_set() and self.video_write_queue.empty(): break
                    continue # Continue waiting if not stopped

                if frame_data is None: break # Check for sentinel value

                frame = frame_data
                # Perform checks and write frame
                if frame is not None and isinstance(frame, np.ndarray) and frame.shape[0] > 0:
                    if frame.dtype != np.uint8: frame = frame.astype(np.uint8)
                    if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    # Ensure frame matches output shape before writing
                    if frame.shape[1]!=self.frame_shape[0] or frame.shape[0]!=self.frame_shape[1]:
                        frame = cv2.resize(frame, self.frame_shape)
                    out.write(frame); frames_written += 1

                # Mark task as done only if queue exists
                if self.video_write_queue: self.video_write_queue.task_done()

        except Exception as e:
            print(f"ERROR in writer thread: {e}")
            # Signal failure
            if self.final_output_path != "ERROR_WRITER_FAILED": # Avoid overwriting specific error
                 self.final_output_path = f"ERROR_WRITER_RUNTIME: {e}"
        finally:
            # Release writer if it was initialized
            if out and writer_initialized:
                 out.release();
                 debug_print(f"Writer released. Wrote {frames_written} frames to {actual_output_path}")
            # Signal writer queue end only if it exists (ensures main thread doesn't block indefinitely)
            if self.video_write_queue:
                try: self.video_write_queue.put(None, timeout=1.0)
                except queue.Full: pass


    # --- Core Processing Logic ---
    @torch.no_grad()
    def _preprocess_batch(self, frames_batch_with_metadata): # Expects list of (frame_np, metadata_dict)
        preprocessed_tensors_list = []
        target_dim_for_model = self.model_input_size_config

        for frame_np_content, metadata in frames_batch_with_metadata:
            if frame_np_content is None or not isinstance(frame_np_content, np.ndarray):
                debug_print("Skipping None or invalid frame_np_content in _preprocess_batch.")
                continue

            is_rgb_input = metadata.get('is_rgb', False) # Default to False if metadata is malformed
            source_reader = metadata.get('source', 'unknown')
            
            # Current state of frame_np_content:
            # If source is 'pvr_cuda_segmented':
            #   - It's RGB (is_rgb_input should be True)
            # If source is 'opencv':
            #   - It's BGR (is_rgb_input should be False)
            #   - It's the original frame size from cap.read()

            img_rgb_model_sized = None

            if source_reader.startswith('pvr_cuda'): # Catches 'pvr_cuda_segmented' or future PVR sources
                if not is_rgb_input:
                    # This would be unexpected from PVR, but handle it.
                    debug_print(f"Preprocess (PVR path): Warning! Expected RGB, got BGR? metadata: {metadata}. Converting.")
                    frame_np_content = cv2.cvtColor(frame_np_content, cv2.COLOR_BGR2RGB)

                if frame_np_content.shape[0] != target_dim_for_model or \
                   frame_np_content.shape[1] != target_dim_for_model:
                    # This is a sanity check; PyVideoReader should have resized it.
                    debug_print(f"Preprocess (PVR path): Warning! Frame shape {frame_np_content.shape[:2]} "
                                f"not at target model size ({target_dim_for_model},{target_dim_for_model}). Resizing again.")
                    img_rgb_model_sized = cv2.resize(frame_np_content, 
                                                     (target_dim_for_model, target_dim_for_model), 
                                                     interpolation=cv2.INTER_LINEAR)
                else:
                    img_rgb_model_sized = frame_np_content # Already RGB and correct size
            
            elif source_reader == 'opencv':
                resized_bgr = cv2.resize(frame_np_content, 
                                         (target_dim_for_model, target_dim_for_model), 
                                         interpolation=cv2.INTER_LINEAR)
                img_rgb_model_sized = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
            
            else: # Fallback for unknown source
                debug_print(f"Preprocess: Unknown frame source '{source_reader}'. Attempting general processing.")
                temp_frame = frame_np_content.copy()
                if not is_rgb_input: # If not RGB, assume BGR and convert
                    temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
                
                # Resize if necessary
                if temp_frame.shape[0] != target_dim_for_model or \
                   temp_frame.shape[1] != target_dim_for_model:
                    img_rgb_model_sized = cv2.resize(temp_frame, 
                                                     (target_dim_for_model, target_dim_for_model), 
                                                     interpolation=cv2.INTER_LINEAR)
                else:
                    img_rgb_model_sized = temp_frame
            
            if img_rgb_model_sized is None: 
                debug_print(f"Preprocess: img_rgb_model_sized is None for a frame. Source: {source_reader}. Skipping.")
                continue
            
            # Common processing: HWC uint8 RGB -> CHW float32 [0,1] RGB for PyTorch
            # Ensure it's 3 channels (e.g. not grayscale by mistake)
            if img_rgb_model_sized.ndim == 2: # Grayscale
                debug_print("Preprocess: Grayscale frame detected, converting to RGB for model.")
                img_rgb_model_sized = cv2.cvtColor(img_rgb_model_sized, cv2.COLOR_GRAY2RGB)
            elif img_rgb_model_sized.shape[2] == 4: # RGBA
                debug_print("Preprocess: RGBA frame detected, converting to RGB for model.")
                img_rgb_model_sized = cv2.cvtColor(img_rgb_model_sized, cv2.COLOR_RGBA2RGB)
            
            if img_rgb_model_sized.shape[2] != 3:
                debug_print(f"Preprocess: Frame does not have 3 channels after processing! Shape: {img_rgb_model_sized.shape}. Skipping.")
                continue

            img_chw = img_rgb_model_sized.transpose(2, 0, 1) # HWC to CHW
            img_chw_contiguous = np.ascontiguousarray(img_chw) # Ensure memory is contiguous
            preprocessed_tensors_list.append(img_chw_contiguous)

        if not preprocessed_tensors_list:
            debug_print("Preprocess: No tensors to stack after processing batch.")
            return None

        try:
            batch_np_stacked = np.stack(preprocessed_tensors_list)
        except ValueError as e:
            debug_print(f"Preprocess: Error stacking tensors: {e}. Individual tensor shapes:")
            for i, t in enumerate(preprocessed_tensors_list):
                debug_print(f"  Tensor {i}: shape={t.shape}, dtype={t.dtype}")
            return None

        # Convert to PyTorch tensor, ensure correct dtype and normalization
        # Readers should provide uint8 [0-255] frames.
        batch_tensor_from_np = torch.from_numpy(batch_np_stacked).to(self.device)
        
        if batch_tensor_from_np.dtype != torch.uint8:
             debug_print(f"Preprocess: Warning! Tensor from numpy is not uint8, it's {batch_tensor_from_np.dtype}. Clamping and converting.")
             # Handle potential float inputs if a reader behaves unexpectedly
             batch_tensor_from_np = torch.clamp(batch_tensor_from_np, 0, 255).to(torch.uint8)

        dtype_final = torch.float16 if (MIXED_PRECISION and self.device == 'cuda') else torch.float32
        final_batch_tensor = batch_tensor_from_np.to(dtype_final) / 255.0 # Normalize to [0,1]
        
        return final_batch_tensor

    def _process_batch_with_bytetrack(self, yolo_results_batch,
                                      frames_raw_info_batch,
                                      frame_numbers_batch, frame_times_sec_batch, start_datetime):
        """
        Processes a batch of frames using ByteTrack after YOLO inference.
        This method iterates PER FRAME within the batch for ByteTrack updates.
        """
        batch_app_event_logs_final = [] # Will store lists of app event dicts for the whole batch
        processed_visualization_frames_final = [None] * len(yolo_results_batch)
        active_bt_ids_from_last_frame_in_batch = set()

        model_h_inference, model_w_inference = -1,-1
        if yolo_results_batch and hasattr(yolo_results_batch[0], 'orig_shape') and yolo_results_batch[0].orig_shape is not None:
             model_h_inference, model_w_inference = yolo_results_batch[0].orig_shape
        else:
             model_h_inference, model_w_inference = self.model_input_size_config, self.model_input_size_config
             if yolo_results_batch and (yolo_results_batch[0].orig_shape is None):
                 debug_print(f"Warning: yolo_results_batch[0].orig_shape is None. Falling back to model_input_size_config ({model_w_inference}x{model_h_inference}) for scaling.")


        for frame_idx_in_batch in range(len(yolo_results_batch)):
            yolo_result_this_frame = yolo_results_batch[frame_idx_in_batch]
            original_frame_content, _ = frames_raw_info_batch[frame_idx_in_batch]
            
            current_frame_num = frame_numbers_batch[frame_idx_in_batch]
            current_frame_time_video_sec = frame_times_sec_batch[frame_idx_in_batch]
            current_frame_datetime_obj = start_datetime + timedelta(seconds=current_frame_time_video_sec)

            detections_for_bt_update_this_frame = []
            if hasattr(yolo_result_this_frame, 'boxes') and yolo_result_this_frame.boxes is not None:
                for box in yolo_result_this_frame.boxes:
                    x1_m, y1_m, x2_m, y2_m = box.xyxy[0].cpu().numpy()
                    score_m = float(box.conf[0])
                    class_id_m = int(box.cls[0])
                    class_name_m = yolo_result_this_frame.names.get(class_id_m, f"CLS_{class_id_m}")
                    detections_for_bt_update_this_frame.append({
                        'bbox_xyxy': (x1_m, y1_m, x2_m, y2_m), 'score': score_m,
                        'class_id': class_id_m, 'class_name': class_name_m
                    })
            
            if self.perf: self.perf.start_timer('bytetrack_update')
            online_stracks_this_frame = self.byte_tracker.update(detections_for_bt_update_this_frame)
            if self.perf: self.perf.end_timer('bytetrack_update')

            frame_local_app_event_log = []
            current_active_bt_ids_str_set_this_iter = set()

            output_viz_frame_this_iter = None
            if ENABLE_VISUALIZATION and original_frame_content is not None:
                output_viz_frame_this_iter = cv2.resize(original_frame_content.copy(), self.frame_shape)

            for s_track in online_stracks_this_frame:
                bt_id_str = f"bt_{s_track.track_id}"
                current_active_bt_ids_str_set_this_iter.add(bt_id_str)
                x1_m, y1_m, x2_m, y2_m = s_track.xyxy
                disp_w, disp_h = self.frame_shape[0], self.frame_shape[1]
                
                x1_s = max(0, int(x1_m * disp_w / model_w_inference))
                y1_s = max(0, int(y1_m * disp_h / model_h_inference))
                x2_s = min(disp_w - 1, int(x2_m * disp_w / model_w_inference))
                y2_s = min(disp_h - 1, int(y2_m * disp_h / model_h_inference))
                scaled_bbox_xyxy_disp = (x1_s, y1_s, x2_s, y2_s)
                scaled_center_pt_disp = ((x1_s + x2_s) // 2, (y1_s + y2_s) // 2)
                scaled_bottom_center_disp = ((x1_s + x2_s) // 2, y2_s)

                self.vehicle_tracker.update_vehicle_history(
                    bt_id_str, scaled_center_pt_disp, current_frame_datetime_obj
                )
                prev_pos_scaled_disp = None
                if bt_id_str in self.vehicle_tracker.tracking_history and \
                   len(self.vehicle_tracker.tracking_history[bt_id_str]) > 1:
                    prev_pos_scaled_disp = self.vehicle_tracker.tracking_history[bt_id_str][-2]

                if prev_pos_scaled_disp and self.zone_tracker and self.zone_tracker.zones:
                    if self.perf: self.perf.start_timer('zone_checking')
                    event_type, event_dir = self.zone_tracker.check_zone_transition(
                        prev_pos_scaled_disp, scaled_bbox_xyxy_disp,
                        bt_id_str, current_frame_time_video_sec
                    )
                    if self.perf: self.perf.end_timer('zone_checking')

                    if event_type == "ENTRY":
                        app_vehicle_state = self.vehicle_tracker.active_vehicles.get(bt_id_str)
                        is_app_active = app_vehicle_state and app_vehicle_state.get('status') == 'active'
                        stored_entry_dir = app_vehicle_state.get('entry_direction') if app_vehicle_state else None

                        if is_app_active and stored_entry_dir and stored_entry_dir != event_dir:
                            active_zone_keys = list(self.zone_tracker.zones.keys())
                            if is_valid_movement(stored_entry_dir, event_dir, active_zone_keys):
                                success_inf, t_in_int_inf = self.vehicle_tracker.register_exit(
                                    bt_id_str, event_dir, current_frame_datetime_obj, scaled_bottom_center_disp
                                )
                                if success_inf:
                                    if self.perf: self.perf.record_vehicle_exit('exited', t_in_int_inf)
                                    completed_path = next((p for p in reversed(self.vehicle_tracker.completed_paths) if p.get('id') == bt_id_str and p.get('exit_direction') == event_dir and p.get('status') == 'exited'), None)
                                    if completed_path:
                                        frame_local_app_event_log.append({
                                            'timestamp_dt': current_frame_datetime_obj, 'vehicle_id': bt_id_str,
                                            'vehicle_type': completed_path.get('type', s_track.class_name_consistent),
                                            'event_type': 'EXIT', 'direction_from': stored_entry_dir,
                                            'direction_to': event_dir, 'status': 'exited_inferred',
                                            'time_in_intersection': t_in_int_inf, 'frame_number': current_frame_num
                                        })
                        if self.vehicle_tracker.register_entry(bt_id_str, event_dir, current_frame_datetime_obj,
                                                               scaled_bottom_center_disp, s_track.class_name_consistent):
                            if self.perf: self.perf.record_vehicle_entry()
                            frame_local_app_event_log.append({
                                'timestamp_dt': current_frame_datetime_obj, 'vehicle_id': bt_id_str,
                                'vehicle_type': s_track.class_name_consistent, 'event_type': 'ENTRY',
                                'direction_from': event_dir, 'direction_to': None,
                                'status': 'entry', 'frame_number': current_frame_num
                            })
                    elif event_type == "EXIT":
                        entry_dir_trk = self.vehicle_tracker.active_vehicles.get(bt_id_str, {}).get('entry_direction')
                        active_zone_keys = list(self.zone_tracker.zones.keys()) if self.zone_tracker and self.zone_tracker.zones else []
                        if entry_dir_trk and is_valid_movement(entry_dir_trk, event_dir, active_zone_keys):
                            success_ex, t_in_int_ex = self.vehicle_tracker.register_exit(
                                bt_id_str, event_dir, current_frame_datetime_obj, scaled_bottom_center_disp
                            )
                            if success_ex:
                                if self.perf: self.perf.record_vehicle_exit('exited', t_in_int_ex)
                                completed_path = next((p for p in reversed(self.vehicle_tracker.completed_paths) if p.get('id') == bt_id_str and p.get('exit_direction') == event_dir and p.get('status') == 'exited'), None)
                                if completed_path:
                                    frame_local_app_event_log.append({
                                        'timestamp_dt': current_frame_datetime_obj, 'vehicle_id': bt_id_str,
                                        'vehicle_type': completed_path.get('type', s_track.class_name_consistent),
                                        'event_type': 'EXIT', 'direction_from': entry_dir_trk,
                                        'direction_to': event_dir, 'status': 'exit',
                                        'time_in_intersection': t_in_int_ex, 'frame_number': current_frame_num
                                    })
                if ENABLE_VISUALIZATION and output_viz_frame_this_iter is not None:
                    app_state = self.vehicle_tracker.active_vehicles.get(bt_id_str)
                    draw_status = "tracked_bt"; entry_dir_draw = None; time_active_draw = None
                    if app_state and app_state['status'] == 'active':
                        draw_status = "active_app"; entry_dir_draw = app_state.get('entry_direction')
                        time_active_draw = (current_frame_datetime_obj - app_state['entry_time']).total_seconds()
                    elif any(evt['vehicle_id'] == bt_id_str and evt['event_type'] == 'EXIT' for evt in frame_local_app_event_log):
                        draw_status = "exiting_app"
                    visual_overlay.draw_detection_box(
                        output_viz_frame_this_iter, scaled_bbox_xyxy_disp, bt_id_str,
                        s_track.class_name_consistent, status=draw_status,
                        entry_dir=entry_dir_draw, time_active=time_active_draw
                    )
                    trail = self.vehicle_tracker.get_tracking_trail(bt_id_str)
                    visual_overlay.draw_tracking_trail(output_viz_frame_this_iter, trail, bt_id_str)

            timed_out_app_ids = self.vehicle_tracker.check_timeouts(current_frame_datetime_obj)
            for v_id_to_app in timed_out_app_ids:
                if self.perf: self.perf.record_vehicle_exit('timed_out')
                p_data = next((p for p in reversed(self.vehicle_tracker.completed_paths) if p['id'] == v_id_to_app and p['status'] == 'timed_out'), None)
                if p_data:
                    frame_local_app_event_log.append({
                        'timestamp_dt': current_frame_datetime_obj, 'vehicle_id': v_id_to_app,
                        'vehicle_type': p_data.get('type', 'UnknownType'), 'event_type': 'TIMEOUT',
                        'direction_from': p_data.get('entry_direction', 'UNKNOWN'), 'direction_to': 'TIMEOUT',
                        'status': 'timeout', 'time_in_intersection': p_data.get('time_in_intersection', 'N/A'),
                        'frame_number': current_frame_num
                    })
            if ENABLE_VISUALIZATION and output_viz_frame_this_iter is not None:
                if self.perf: self.perf.start_timer('drawing_overlays')
                if self.zone_tracker and self.zone_tracker.zones:
                    output_viz_frame_this_iter = visual_overlay.draw_zones(output_viz_frame_this_iter, self.zone_tracker)
                output_viz_frame_this_iter = visual_overlay.add_status_overlay(
                    output_viz_frame_this_iter, current_frame_num, current_frame_datetime_obj, self.vehicle_tracker
                )
                processed_visualization_frames_final[frame_idx_in_batch] = output_viz_frame_this_iter
                if self.perf: self.perf.end_timer('drawing_overlays')

            batch_app_event_logs_final.extend(frame_local_app_event_log)
            active_bt_ids_from_last_frame_in_batch = current_active_bt_ids_str_set_this_iter
            
        return batch_app_event_logs_final, processed_visualization_frames_final, active_bt_ids_from_last_frame_in_batch
    # --- Main Processing Function ---
    
    def process_video(self, video_path, start_date_str, start_time_str, primary_direction):
        if not self.processing_lock.acquire(blocking=False):
            print("Warning: Processing lock already acquired.")
            return "Processing is already in progress."

        self.perf = PerformanceTracker(enabled=ENABLE_DETAILED_PERFORMANCE_METRICS)
        self.stop_event = threading.Event()
        self.frame_read_queue = queue.Queue(maxsize=max(self.batch_size * 4, 128))
        self.last_cleanup_time = time.monotonic()
        self.detection_zones_polygons = None
        output_path = None
        self.final_output_path = None
        self.writer_thread = None
        self.video_write_queue = None
        STrack._count = 0 # Crucial: Reset global STrack ID counter for each new video

        print(f"--- Starting New Video Processing Run ---"); 
        print(f"Video: {video_path}"); 
        print(f"Timestamp: {start_date_str} {start_time_str}"); 
        print(f"Zone Mode: {self.line_mode.upper()}"); 
        print(f"Visualization Enabled: {ENABLE_VISUALIZATION}")
        print(f"PVR Segment Size (config): {self.pvr_decode_segment_size}")
        print(f"Use Advanced Reader (config): {USE_ADVANCED_VIDEO_READER}")
        print(f"PyVideoReader Available (import status): {PYVIDEOREADER_AVAILABLE}")

        if primary_direction:
             print(f"Primary Direction for Report: {primary_direction.capitalize()}")
        else:
             print("Warning: No Primary Direction provided for RSA report numbering.")

        actual_video_path_to_process = video_path
        temp_converted_dav_mp4 = None

        if video_path.lower().endswith(".dav"):
            if not FFMPEG_PATH:
                error_msg = "ERROR: FFmpeg not found, cannot convert .dav file. Please configure FFMPEG_CUSTOM_PATH in config.py or ensure ffmpeg is in PATH."
                print(error_msg)
                raise RuntimeError(error_msg)

            print(f"Input is .dav: {video_path}. Converting and resizing with FFmpeg...")
            converted_path = convert_dav_to_mp4_internal(video_path)
            if converted_path and os.path.exists(converted_path):
                actual_video_path_to_process = converted_path
                temp_converted_dav_mp4 = converted_path
                print(f".dav converted to: {actual_video_path_to_process}")

                dav_settle_delay = 0.5 # seconds
                print(f"Waiting {dav_settle_delay}s for file system to settle after DAV remux...")
                time.sleep(dav_settle_delay)
                # --- END DELAY ---
            else:
                error_msg = f"Failed to convert .dav file: {video_path}"
                print(error_msg)
                raise RuntimeError(error_msg)

        try:
            # --- Determine and Validate Zones (Polygon Mode) ---
            self.detection_zones_polygons = None
            if self.line_mode == 'hardcoded':
                if 'LINE_POINTS' not in globals() or not isinstance(LINE_POINTS, dict) or not LINE_POINTS: raise ValueError("Hardcoded mode: LINE_POINTS not found/empty in config.py.")
                valid_polygons_hc = { dir: pts for dir, pts in LINE_POINTS.items() if isinstance(pts, list) and len(pts) >= 3 and all(isinstance(pt, tuple) and len(pt)==2 for pt in pts) }
                if not valid_polygons_hc: raise ValueError("Hardcoded mode: No valid polygons found in LINE_POINTS in config.py")
                if len(valid_polygons_hc) < len(LINE_POINTS): print("Warning: Some hardcoded LINE_POINTS ignored (invalid format/points).")
                self.detection_zones_polygons = { dir: np.array(pts, dtype=np.int32) for dir, pts in valid_polygons_hc.items() }
                print(f"Using {len(self.detection_zones_polygons)} hardcoded zones.")
                if len(self.detection_zones_polygons) < 2: print("Warning: Fewer than 2 valid hardcoded zones.")

            elif self.line_mode == 'interactive':
                if not self.gradio_polygons: raise ValueError("Interactive mode: No zone polygons defined.")
                valid_polygons = { dir: pts for dir, pts in self.gradio_polygons.items() if pts and len(pts) >= 3 }
                if len(valid_polygons) < 2: raise ValueError(f"Interactive mode: Need >= 2 valid zones. Found {len(valid_polygons)}.")
                if len(valid_polygons) < len(self.gradio_polygons): print("Warning: Some drawn polygons ignored (< 3 points).")
                self.detection_zones_polygons = { dir: np.array(pts, dtype=np.int32) for dir, pts in valid_polygons.items() }
                print(f"Using {len(self.detection_zones_polygons)} valid zones from Gradio.")
            else: raise ValueError(f"Invalid LINE_MODE '{self.line_mode}'.")

            # --- Initialize ZoneTracker ---
            self.detection_zones_polygons = None
            if self.line_mode == 'hardcoded':
                if 'LINE_POINTS' not in globals() or not isinstance(LINE_POINTS, dict) or not LINE_POINTS: raise ValueError("Hardcoded mode: LINE_POINTS not found/empty.")
                valid_polygons_hc = { dir: pts for dir, pts in LINE_POINTS.items() if isinstance(pts, list) and len(pts) >= 3 and all(isinstance(pt, tuple) and len(pt)==2 for pt in pts) }
                if not valid_polygons_hc: raise ValueError("Hardcoded mode: No valid polygons in LINE_POINTS.")
                self.detection_zones_polygons = { dir: np.array(pts, dtype=np.int32) for dir, pts in valid_polygons_hc.items() }
            elif self.line_mode == 'interactive':
                if not self.gradio_polygons: raise ValueError("Interactive mode: No zone polygons defined.")
                valid_polygons = { dir: pts for dir, pts in self.gradio_polygons.items() if pts and len(pts) >= 3 }
                if len(valid_polygons) < 2: raise ValueError(f"Interactive mode: Need >= 2 valid zones. Found {len(valid_polygons)}.")
                self.detection_zones_polygons = { dir: np.array(pts, dtype=np.int32) for dir, pts in valid_polygons.items() }
            else: raise ValueError(f"Invalid LINE_MODE '{self.line_mode}'.")

            if not self.detection_zones_polygons: raise ValueError("Zone polygon data missing.")
            self.zone_tracker = ZoneTracker(self.detection_zones_polygons)
            if not self.zone_tracker or not self.zone_tracker.zones:
              raise RuntimeError("CRITICAL: ZoneTracker failed to initialize with valid zones.")

            # Get video properties (total_frames, original_fps)
            total_frames, original_fps = self._get_video_properties(actual_video_path_to_process) # Use actual path
            print(f"Video Properties: ~{total_frames} frames, {original_fps:.2f} FPS (Original)")
            if total_frames <= 0: raise ValueError("Cannot process video with zero or unreadable total frames.")

            # --- Initialize Trackers & Model ---
            self.vehicle_tracker = VehicleTracker()
            processing_fps_for_kf_dt_val = 0.0
            if self.target_fps > 0:
                processing_fps_for_kf_dt_val = self.target_fps
            elif original_fps > 0 : 
                processing_fps_for_kf_dt_val = original_fps
            else: 
                processing_fps_for_kf_dt_val = 30.0 

            kalman_dt_val = 1.0 / processing_fps_for_kf_dt_val
            print(f"DEBUG_KF_INIT: original_video_fps={original_fps}, target_fps_config={self.target_fps}, effective_processing_fps_for_kf={processing_fps_for_kf_dt_val}, kalman_dt={kalman_dt_val:.4f}")

            self.byte_tracker = ByteTracker(
                track_thresh=BYTETRACK_TRACK_THRESH,
                track_high_thresh=BYTETRACK_TRACK_HIGH_THRESH,
                new_track_thresh=BYTETRACK_NEW_TRACK_THRESH,
                match_thresh_iou=BYTETRACK_MATCH_THRESH_IOU,
                track_buffer_age_frames=int(BYTETRACK_TRACK_BUFFER_AGE_FRAMES * (1/kalman_dt_val if kalman_dt_val > 0 else 30)), # Convert age from seconds in config to frames
                kalman_filter_dt=kalman_dt_val
            )
            if self.model is None: self.model = load_model(MODEL_PATH, self.device, CONF_THRESHOLD, IOU_THRESHOLD)
            if self.model is None: raise RuntimeError(f"Failed to load model {MODEL_PATH}")

            try:
                start_date_obj=datetime.strptime(start_date_str, "%y%m%d")
                time_part_str,ms_part_str=(start_time_str[:6], start_time_str[6:]) if len(start_time_str)>6 else (start_time_str,"0")
                base_time_obj=datetime.strptime(time_part_str,"%H%M%S").time()
                microseconds_val=int(ms_part_str.ljust(3,'0'))*1000
                start_time_obj=base_time_obj.replace(microsecond=microseconds_val)
                start_datetime=datetime.combine(start_date_obj.date(), start_time_obj)
                print(f"Video Start Timestamp: {start_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            except ValueError as e_time: raise ValueError(f"Invalid date/time format: {e_time}.")

            # --- Setup Output Path and Start Threads (Writer only if ENABLE_VISUALIZATION) ---
            print("Starting I/O threads...")
            if self.actually_use_advanced_reader:
                self.reader_thread = threading.Thread(
                    target=self._frame_reader_task_pvr_segmented,
                    args=(video_path, original_fps, total_frames),
                    daemon=True)
            else: 
                self.reader_thread = threading.Thread(
                    target=self._frame_reader_task, 
                    args=(video_path, original_fps),
                    daemon=True)
            self.reader_thread.start()

            if ENABLE_VISUALIZATION:
                os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True); ts=datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename_base=f"output_{os.path.splitext(os.path.basename(video_path))[0]}_{ts}"
                output_path=os.path.join(VIDEO_OUTPUT_DIR, f"{output_filename_base}.mp4");
                self.final_output_path = output_path # Initial path assumption
                # Initialize writer queue only if needed
                self.video_write_queue = queue.Queue(maxsize=self.batch_size*4)
                self.writer_thread = threading.Thread(target=self._frame_writer_task, args=(output_path,), daemon=True)
                self.writer_thread.start()
                print("Video Writer thread started.")
            else:
                print("Video writing disabled (ENABLE_VISUALIZATION is False).")
                self.video_write_queue = None
                self.writer_thread = None
                self.final_output_path = None


            # --- Main Loop ---
            all_application_events = []
            frames_processed_in_run = 0
            frame_interval = max(1, round(original_fps / self.target_fps)) if self.target_fps > 0 and original_fps > 0 else 1
            if total_frames > 0 and self.target_fps > 0 and frame_interval > 0:
                total_target_frames_for_perf = total_frames // frame_interval
            elif total_frames > 0: # if target_fps is 0 or frame_interval is 0 for some reason, process all frames
                total_target_frames_for_perf = total_frames
            else: # total_frames is 0 or negative
                total_target_frames_for_perf = 0 # This will show 0% progress, which is correct
            if self.perf: self.perf.start_processing(total_target_frames_for_perf)
            frames_batch_with_metadata, frame_numbers_batch, frame_times_batch = [], [], []
            last_progress_print_time = time.monotonic()
            active_bt_ids_for_cleanup = set() # Store active IDs from the last processed frame for cleanup

            print(f"Target FPS: {self.target_fps:.1f}, Frame Interval: ~{frame_interval} frame(s). Approx frames to process: {total_target_frames_for_perf if total_target_frames_for_perf > 0 else 'All'}")

            while True:
                try:
                    frame_data_from_queue = self.frame_read_queue.get(timeout=60)
                    if frame_data_from_queue is None:
                        self.frame_read_queue.task_done(); break 
                    
                    if self.perf: self.perf.start_timer('video_read')
                    frame_content, frame_number_raw, frame_time_video_sec, metadata = frame_data_from_queue
                    if self.perf: self.perf.end_timer('video_read')

                    if frame_content is None:
                        self.frame_read_queue.task_done(); continue

                    frames_batch_with_metadata.append((frame_content, metadata))
                    frame_numbers_batch.append(frame_number_raw)
                    frame_times_batch.append(frame_time_video_sec)

                    if len(frames_batch_with_metadata) >= self.batch_size:
                        batch_proc_start_time = time.monotonic()
                        if self.perf: self.perf.start_timer('preprocessing')
                        input_batch_tensor = self._preprocess_batch(frames_batch_with_metadata)
                        if self.perf: self.perf.end_timer('preprocessing')

                        if input_batch_tensor is None or input_batch_tensor.nelement() == 0:
                            debug_print("Skipping empty preprocessed batch.")
                            for _ in range(len(frames_batch_with_metadata)): self.frame_read_queue.task_done()
                            frames_batch_with_metadata.clear(); frame_numbers_batch.clear(); frame_times_batch.clear()
                            continue

                        # --- YOLO Inference ---
                        yolo_results_batch = None
                        current_stream = None
                        if self.device == 'cuda' and self.streams:
                            current_stream = self.streams[self.current_stream_index % len(self.streams)]
                            self.current_stream_index += 1
                        
                        with torch.cuda.stream(current_stream) if current_stream else torch.no_grad(): # Context manager for stream
                           with torch.amp.autocast(device_type=self.device, enabled=(MIXED_PRECISION and self.device=='cuda')):
                             if self.perf and self.perf.cuda_events_enabled: self.perf.start_event.record(stream=current_stream)
                             yolo_results_batch = self.model(input_batch_tensor, verbose=False) # List of Results objects
                             if self.perf and self.perf.cuda_events_enabled: self.perf.end_event.record(stream=current_stream)
                        
                        if current_stream: current_stream.synchronize()
                        if self.perf and self.perf.cuda_events_enabled:
                            self.perf.record_inference_time_gpu(self.perf.start_event, self.perf.end_event)
                        
                        # --- Process with ByteTrack (iterates per-frame inside) ---
                        if self.perf: self.perf.start_timer('bytetrack_and_app_logic')
                        batch_app_event_log_list, viz_frames_list, active_bt_ids_last_frame = \
                            self._process_batch_with_bytetrack(
                                yolo_results_batch, frames_batch_with_metadata,
                                frame_numbers_batch, frame_times_batch, start_datetime
                            )
                        all_application_events.extend(batch_app_event_log_list)
                        active_bt_ids_for_cleanup = active_bt_ids_last_frame
                        if self.perf: self.perf.end_timer('bytetrack_and_app_logic')
                        
                        if self.perf and yolo_results_batch:
                            self.perf.record_detection(sum(len(r.boxes) for r in yolo_results_batch if hasattr(r,'boxes') and r.boxes))

                        # --- Write Visualized Frames ---
                        if ENABLE_VISUALIZATION and self.video_write_queue:
                            if self.perf: self.perf.start_timer('video_write')
                            for pf_viz in viz_frames_list:
                                if pf_viz is not None:
                                    try: self.video_write_queue.put(pf_viz, block=True, timeout=2.0)
                                    except queue.Full: print("WARN: Video writer queue full, frame dropped.")
                            if self.perf: self.perf.end_timer('video_write')

                        # --- Stats & Cleanup ---
                        batch_processing_duration = time.monotonic() - batch_proc_start_time
                        frames_processed_in_run += len(frames_batch_with_metadata)
                        if self.perf: self.perf.record_batch_processed(len(frames_batch_with_metadata), batch_processing_duration)
                        if self.perf: self.perf.sample_system_metrics()

                        current_mono_time = time.monotonic()
                        if current_mono_time - self.last_cleanup_time > MEMORY_CLEANUP_INTERVAL:
                            if self.perf: self.perf.start_timer('memory_cleanup_app')
                            last_p_frame_time_sec = frame_times_batch[-1] if frame_times_batch else 0
                            last_p_frame_dt = start_datetime + timedelta(seconds=last_p_frame_time_sec)
                            cleanup_tracking_data(
                                self.vehicle_tracker, self.zone_tracker,
                                last_p_frame_time_sec, last_p_frame_dt, active_bt_ids_for_cleanup
                            )
                            cleanup_memory()
                            self.last_cleanup_time = current_mono_time
                            if self.perf: self.perf.end_timer('memory_cleanup_app')

                        if self.perf and (current_mono_time - last_progress_print_time > 1.0):
                            prog = self.perf.get_progress()
                            active_app_count = self.vehicle_tracker.get_active_vehicle_count() if self.vehicle_tracker else 0
                            print(f"\rProgress:{prog['percent']:.1f}%|FPS:{prog['fps']:.1f}|Active(App):{active_app_count}|ETA:{prog['eta']}   ", end="")
                            last_progress_print_time = current_mono_time
                        
                        # Clear Batch related lists and tensors
                        for _ in range(len(frames_batch_with_metadata)): self.frame_read_queue.task_done()
                        frames_batch_with_metadata.clear(); frame_numbers_batch.clear(); frame_times_batch.clear()
                        del input_batch_tensor, yolo_results_batch; # cleanup_memory() # Called in interval

                except queue.Empty:
                    debug_print("Main loop: Frame reader queue timed out.")
                    if self.stop_event.is_set() and (self.frame_read_queue is None or self.frame_read_queue.empty()):
                        debug_print("Main loop: Stop event set and reader queue empty. Ending loop.")
                        break
                    continue
                except Exception as e_loop:
                    print(f"\nERROR during processing loop: {e_loop}")
                    import traceback; traceback.print_exc()
                    self.stop_event.set(); break
            
            print("\nVideo processing loop finished.") # End of while True

            # --- Process Final Batch ---
            if frames_batch_with_metadata:
                print("Processing final batch...")
                # (Identical logic as inside the loop for processing a full batch)
                batch_proc_start_time = time.monotonic()
                if self.perf: self.perf.start_timer('preprocessing')
                input_batch_tensor = self._preprocess_batch(frames_batch_with_metadata)
                if self.perf: self.perf.end_timer('preprocessing')

                if input_batch_tensor is not None and input_batch_tensor.nelement() > 0:
                    yolo_results_batch = None; current_stream = None
                    if self.device == 'cuda' and self.streams:
                        current_stream = self.streams[self.current_stream_index % len(self.streams)]; self.current_stream_index += 1
                    with torch.cuda.stream(current_stream) if current_stream else torch.no_grad():
                       with torch.amp.autocast(device_type=self.device, enabled=(MIXED_PRECISION and self.device=='cuda')):
                         if self.perf and self.perf.cuda_events_enabled: self.perf.start_event.record(stream=current_stream)
                         yolo_results_batch = self.model(input_batch_tensor, verbose=False)
                         if self.perf and self.perf.cuda_events_enabled: self.perf.end_event.record(stream=current_stream)
                    if current_stream: current_stream.synchronize()
                    if self.perf and self.perf.cuda_events_enabled: self.perf.record_inference_time_gpu(self.perf.start_event, self.perf.end_event)

                    if self.perf: self.perf.start_timer('bytetrack_and_app_logic')
                    batch_app_event_log_list, viz_frames_list, active_bt_ids_last_frame = \
                        self._process_batch_with_bytetrack(
                            yolo_results_batch, frames_batch_with_metadata,
                            frame_numbers_batch, frame_times_batch, start_datetime
                        )
                    all_application_events.extend(batch_app_event_log_list)
                    # active_bt_ids_for_cleanup = active_bt_ids_last_frame # Not strictly needed for cleanup after this
                    if self.perf: self.perf.end_timer('bytetrack_and_app_logic')

                    if ENABLE_VISUALIZATION and self.video_write_queue:
                        if self.perf: self.perf.start_timer('video_write')
                        for pf_viz in viz_frames_list:
                            if pf_viz is not None:
                                try: self.video_write_queue.put(pf_viz, block=True, timeout=2.0)
                                except queue.Full: print("WARN: Video writer queue full (final batch), frame dropped.")
                        if self.perf: self.perf.end_timer('video_write')
                    
                    batch_processing_duration = time.monotonic() - batch_proc_start_time
                    frames_processed_in_run += len(frames_batch_with_metadata)
                    if self.perf: self.perf.record_batch_processed(len(frames_batch_with_metadata), batch_processing_duration)
                    del input_batch_tensor, yolo_results_batch
                else: debug_print("Skipping empty final preprocessed batch.")
                for _ in range(len(frames_batch_with_metadata)): self.frame_read_queue.task_done()
                frames_batch_with_metadata.clear(); frame_numbers_batch.clear(); frame_times_batch.clear()

            # --- Finalize Application-Level Tracks ---
            if self.vehicle_tracker and self.vehicle_tracker.active_vehicles:
                print(f"Forcing app-level exit for {len(self.vehicle_tracker.active_vehicles)} remaining in VehicleTracker...")
                # Determine last timestamp for forced exits
                last_known_frame_time_sec = 0
                if hasattr(self, 'byte_tracker') and self.byte_tracker and self.byte_tracker.current_frame_id_internal > 0:
                    # This assumes frame_times_batch was for the very last set of frames processed.
                    # If frame_times_batch is empty (e.g. video ended before full batch), get from ByteTracker's internal frame_id
                    # This needs a reliable way to map ByteTracker's frame_id back to video time.
                    # For simplicity, use the last recorded frame_time_video_sec from the loop.
                    if frame_times_batch: # If last batch had content
                        last_known_frame_time_sec = frame_times_batch[-1]
                    elif all_application_events : # Fallback to last event time
                        last_known_frame_time_sec = (all_application_events[-1]['timestamp_dt'] - start_datetime).total_seconds()

                last_timestamp_for_force_exit = start_datetime + timedelta(seconds=last_known_frame_time_sec)
                
                for v_id_str_force in list(self.vehicle_tracker.active_vehicles.keys()):
                    self.vehicle_tracker.force_exit_vehicle(v_id_str_force, last_timestamp_for_force_exit)
                    if self.perf: self.perf.record_vehicle_exit('forced_exit')
                    # No need to add to all_application_events, as completed_paths is used for report
                    self.vehicle_tracker.remove_all_data_for_id(v_id_str_force) # Clean from VT
                    if self.zone_tracker: self.zone_tracker.remove_vehicle_data(v_id_str_force)

            # --- Signal & Wait ---
            print("Signaling I/O threads to stop..."); self.stop_event.set()
            if ENABLE_VISUALIZATION and self.video_write_queue:
                print("Signaling writer to finish..."); self.video_write_queue.put(None, timeout=5.0)
            if self.frame_read_queue: # Reader might be blocked on put, ensure it gets None if possible
                try: self.frame_read_queue.put_nowait(None) 
                except queue.Full: debug_print("Reader queue full on final signal, relying on stop_event.")
            
            print("Waiting for I/O threads to complete...")
            t_wait_io_start=time.monotonic()

            if self.writer_thread and self.writer_thread.is_alive(): self.writer_thread.join(timeout=30.0)
            if self.reader_thread and self.reader_thread.is_alive(): self.reader_thread.join(timeout=10.0)
            print(f"I/O Threads joined/timed out in {time.monotonic()-t_wait_io_start:.2f}s")

            # --- Performance Summary & Reporting ---
            total_time_taken_final=0.0; final_overall_fps=0.0; completed_paths_count_valid_final=0
            if self.perf:
                self.perf.end_processing(); self.perf.print_summary()
                total_time_taken_final = self.perf.total_time
                final_overall_fps = frames_processed_in_run / total_time_taken_final if total_time_taken_final > 0 else 0
            
            final_completed_paths_for_report = []
            if hasattr(self, 'vehicle_tracker') and self.vehicle_tracker:
                final_completed_paths_for_report = self.vehicle_tracker.get_completed_paths()
                for path in final_completed_paths_for_report: # Count valid for stats
                    status=path.get('status'); entry_dir=path.get('entry_direction'); exit_dir=path.get('exit_direction')
                    is_valid_entry=entry_dir and entry_dir not in ['UNKNOWN',None]
                    is_valid_exit=exit_dir and exit_dir not in ['UNKNOWN','TIMEOUT','FORCED',None,'IN_PROGRESS','N/A']
                    if status=='exited' and is_valid_entry and is_valid_exit: completed_paths_count_valid_final+=1

            excel_report_path = create_excel_report(
                completed_paths_data=final_completed_paths_for_report,
                start_datetime=start_datetime, primary_direction=primary_direction, video_path=video_path
            )
            if ENABLE_DETAILED_PERFORMANCE_METRICS and self.perf:
                visualize_performance(self.perf)

            # --- Prepare result message ---
            result_msg_final=f"✅ Processing completed (ByteTrack Mode).\n"
            if ENABLE_VISUALIZATION:
                if self.final_output_path and "ERROR" not in str(self.final_output_path): # Ensure it's a string for "ERROR" check
                     result_msg_final+=f"Output video: '{self.final_output_path}'.\n"
                elif self.final_output_path and "ERROR" in str(self.final_output_path):
                     result_msg_final+=f"Output video generation FAILED: {self.final_output_path}.\n"
                else: result_msg_final+=f"Output video status unknown.\n"

            if excel_report_path: result_msg_final+=f"Excel report: {excel_report_path}\n"
            if ENABLE_DETAILED_PERFORMANCE_METRICS and self.perf: result_msg_final+=f"Perf charts: 'performance_charts/'\n"
            result_msg_final+=f"\n--- Summary Stats ---\nSTAT_FRAMES_PROCESSED={frames_processed_in_run}\nSTAT_TIME_SECONDS={total_time_taken_final:.2f}\nSTAT_FPS={final_overall_fps:.2f}\nSTAT_COMPLETED_PATHS={completed_paths_count_valid_final}\n--- End Stats ---"
            
            print(result_msg_final)
            return result_msg_final

        except (ValueError, RuntimeError, FileNotFoundError, Exception) as e_main:
            print(f"\nFATAL ERROR during video processing: {e_main}")
            import traceback; traceback.print_exc()
            if hasattr(self, 'stop_event') and self.stop_event: self.stop_event.set()
            # Ensure lock is released on error path if acquired
            if self.processing_lock.locked(): 
                try: self.processing_lock.release()
                except RuntimeError: pass
            return f"❌ Error: {e_main}"
        finally:
            # --- Final Cleanup ---
            try: # Ensure threads are really joined if an error happened mid-way
                if hasattr(self, 'reader_thread') and self.reader_thread and self.reader_thread.is_alive(): self.reader_thread.join(timeout=1)
                if hasattr(self, 'writer_thread') and self.writer_thread and self.writer_thread.is_alive(): self.writer_thread.join(timeout=1)
            except Exception as join_e_fin: print(f"Error joining threads in final: {join_e_fin}")

            if temp_converted_dav_mp4 and os.path.exists(temp_converted_dav_mp4):
                try: os.remove(temp_converted_dav_mp4); debug_print(f"Cleaned temp DAV: {temp_converted_dav_mp4}")
                except OSError as e_del_dav: debug_print(f"Error deleting temp DAV: {e_del_dav}")

            self.model=None; self.vehicle_tracker=None; self.zone_tracker=None; self.perf=None
            self.byte_tracker = None # Reset ByteTracker
            self.detection_zones_polygons=None; self.frame_read_queue=None; self.video_write_queue=None
            self.reader_thread=None; self.writer_thread=None; self.stop_event=None
            self.final_output_path = None; STrack._count = 0 # Reset STrack ID counter
            cleanup_memory()

            if self.processing_lock.locked():
                try: self.processing_lock.release(); print("Processing finished, lock released.")
                except RuntimeError as release_err: print(f"Warn: Error releasing lock: {release_err}")
            else: print("Processing finished (lock not held or already released by error path).")

# --- End of VideoProcessor class ---
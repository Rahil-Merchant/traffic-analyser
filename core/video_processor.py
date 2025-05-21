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
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

# Configuration
from config import (
    TARGET_FPS, FRAME_WIDTH, FRAME_HEIGHT, OPTIMAL_BATCH_SIZE,
    VIDEO_OUTPUT_DIR, ENABLE_DETAILED_PERFORMANCE_METRICS,
    MIXED_PRECISION, PARALLEL_STREAMS, THREAD_COUNT,
    MEMORY_CLEANUP_INTERVAL, MODEL_PATH, CONF_THRESHOLD, IOU_THRESHOLD,
    LINE_MODE, LINE_POINTS, MODEL_INPUT_SIZE, ENABLE_VISUALIZATION, VEHICLE_TIMEOUTS,
    USE_ADVANCED_VIDEO_READER, PYVIDEOREADER_DECODE_SEGMENT_SIZE, FFMPEG_PATH
)

# Modules
from core.utils import debug_print, is_valid_movement, cleanup_memory
from core.performance import PerformanceTracker
from models.model_loader import load_model, get_device
from tracking.zone_tracker import ZoneTracker
from tracking.cleanup import cleanup_tracking_data
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
        self.max_workers = THREAD_COUNT

        # Gradio Polygon State
        self.gradio_polygons = {} # Stores {direction: List[tuple(x,y)]}
        self.current_gradio_direction = None
        self.gradio_temp_points = []
        self.original_frame_for_gradio = None
        self.last_drawn_polygons_frame = None

        # Core components
        self.device = get_device()
        self.model = None
        self.zone_tracker = None
        self.perf = None
        self.detection_zones_polygons = None # Holds final polygons (NumPy arrays)
        self.model_input_size_config = MODEL_INPUT_SIZE
        self.tracker_lock = threading.Lock()

        # --- Initialize ByteTrack ---
        self.bytetracker = None
        self.current_bytetrack_frame_id = 0 # To keep track of frames for ByteTrack

        # Keep active track data for your zone logic
        self.active_track_data = {} # {track_id: {'entry_direction': str, 'entry_time': datetime, 'type': str, 'last_seen_frame': int, ...}}
        self.finalized_paths_for_report = []


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


    @torch.no_grad()
    def _preprocess_batch(self, frames_batch_with_metadata): # UNCHANGED
        preprocessed_tensors_list = []
        target_dim_for_model = self.model_input_size_config
        for frame_np_content, metadata in frames_batch_with_metadata:
            if frame_np_content is None or not isinstance(frame_np_content, np.ndarray):
                debug_print("Skipping None or invalid frame_np_content in _preprocess_batch.")
                continue
            is_rgb_input = metadata.get('is_rgb', False)
            source_reader = metadata.get('source', 'unknown')
            img_rgb_model_sized = None
            if source_reader.startswith('pvr_cuda'):
                if not is_rgb_input:
                    debug_print(f"Preprocess (PVR path): Warning! Expected RGB, got BGR? metadata: {metadata}. Converting.")
                    frame_np_content = cv2.cvtColor(frame_np_content, cv2.COLOR_BGR2RGB)
                if frame_np_content.shape[0] != target_dim_for_model or frame_np_content.shape[1] != target_dim_for_model:
                    debug_print(f"Preprocess (PVR path): Warning! Frame shape {frame_np_content.shape[:2]} not at target model size. Resizing again.")
                    img_rgb_model_sized = cv2.resize(frame_np_content, (target_dim_for_model, target_dim_for_model), interpolation=cv2.INTER_LINEAR)
                else: img_rgb_model_sized = frame_np_content
            elif source_reader == 'opencv':
                resized_bgr = cv2.resize(frame_np_content, (target_dim_for_model, target_dim_for_model), interpolation=cv2.INTER_LINEAR)
                img_rgb_model_sized = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
            else:
                debug_print(f"Preprocess: Unknown frame source '{source_reader}'. Attempting general processing.")
                temp_frame = frame_np_content.copy()
                if not is_rgb_input: temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
                if temp_frame.shape[0] != target_dim_for_model or temp_frame.shape[1] != target_dim_for_model:
                    img_rgb_model_sized = cv2.resize(temp_frame, (target_dim_for_model, target_dim_for_model), interpolation=cv2.INTER_LINEAR)
                else: img_rgb_model_sized = temp_frame
            if img_rgb_model_sized is None: debug_print(f"Preprocess: img_rgb_model_sized is None for a frame. Source: {source_reader}. Skipping."); continue
            if img_rgb_model_sized.ndim == 2: img_rgb_model_sized = cv2.cvtColor(img_rgb_model_sized, cv2.COLOR_GRAY2RGB)
            elif img_rgb_model_sized.shape[2] == 4: img_rgb_model_sized = cv2.cvtColor(img_rgb_model_sized, cv2.COLOR_RGBA2RGB)
            if img_rgb_model_sized.shape[2] != 3: debug_print(f"Preprocess: Frame does not have 3 channels! Shape: {img_rgb_model_sized.shape}. Skipping."); continue
            img_chw = img_rgb_model_sized.transpose(2, 0, 1); img_chw_contiguous = np.ascontiguousarray(img_chw)
            preprocessed_tensors_list.append(img_chw_contiguous)
        if not preprocessed_tensors_list: debug_print("Preprocess: No tensors to stack."); return None
        try: batch_np_stacked = np.stack(preprocessed_tensors_list)
        except ValueError as e:
            debug_print(f"Preprocess: Error stacking tensors: {e}. Shapes: {[t.shape for t in preprocessed_tensors_list]}")
            return None
        batch_tensor_from_np = torch.from_numpy(batch_np_stacked).to(self.device)
        if batch_tensor_from_np.dtype != torch.uint8:
             debug_print(f"Preprocess: Warning! Tensor from numpy is {batch_tensor_from_np.dtype}, not uint8. Clamping.")
             batch_tensor_from_np = torch.clamp(batch_tensor_from_np, 0, 255).to(torch.uint8)
        dtype_final = torch.float16 if (MIXED_PRECISION and self.device == 'cuda') else torch.float32
        final_batch_tensor = batch_tensor_from_np.to(dtype_final) / 255.0
        return final_batch_tensor

    def _add_to_finalized_paths(self, track_id_str, app_data, final_status_str, exit_direction_str, exit_time_dt, time_in_intersection_val):
            """Helper to create and append a finalized path dictionary for reporting."""
            path_report_entry = {
                'id': track_id_str, # From ByteTrack
                'type': app_data.get('current_type', 'UnknownType'), # Consistent type from app_data
                'entry_direction': app_data.get('entry_direction'),
                'exit_direction': exit_direction_str,
                'entry_time': app_data.get('entry_time'),
                'exit_time': exit_time_dt,
                'time_in_intersection': round(time_in_intersection_val, 2) if time_in_intersection_val is not None else 0.0,
                'status': final_status_str, # e.g., 'exited', 'timed_out', 'lost_track', 'forced_exit'
                # Add other keys if your excel_report.py was expecting them from the old VehicleTracker, e.g.:
                # 'entry_point': app_data.get('entry_point'),
                # 'exit_point': app_data.get('exit_point') # This would need to be captured
            }
            self.finalized_paths_for_report.append(path_report_entry)

    @torch.no_grad()
    def _process_inference_results(self, results, frames_batch_with_metadata, frame_numbers, frame_times, start_datetime_obj_global):
        # This list is for local event logging per frame, not the final report data
        # batch_frame_event_logs = [[] for _ in range(len(results))] # If needed for other debugging
        processed_frames_list_viz = [None] * len(results) # For video output

        def process_single_frame(idx_in_batch):
            yolo_result_obj = results[idx_in_batch]
            # original_frame_content_for_detections = frames_batch_with_metadata[idx_in_batch][0] # Not used for dummy
            
            current_video_frame_num = frame_numbers[idx_in_batch] 
            current_frame_time_sec = frame_times[idx_in_batch]
            current_global_timestamp = start_datetime_obj_global + timedelta(seconds=current_frame_time_sec)

            output_frame_for_viz = None
            if ENABLE_VISUALIZATION:
                original_frame_content_for_viz = frames_batch_with_metadata[idx_in_batch][0]
                output_frame_for_viz = cv2.resize(original_frame_content_for_viz.copy(), self.frame_shape)

            # This list is for verbose event logging if needed, distinct from finalized_paths_for_report
            frame_local_event_log_entries = [] 

            # --- 1. Prepare Detections for ByteTrack ---
            detections_for_bytetrack = []
            if hasattr(yolo_result_obj, 'boxes') and yolo_result_obj.boxes is not None and len(yolo_result_obj.boxes.data) > 0:
                for box_data_tensor in yolo_result_obj.boxes.data:
                    box_cpu = box_data_tensor.cpu().numpy()
                    # Format: [x1, y1, x2, y2, conf, cls_id]
                    x1, y1, x2, y2, conf, cls_id = box_cpu[0], box_cpu[1], box_cpu[2], box_cpu[3], box_cpu[4], box_cpu[5]
                    detections_for_bytetrack.append([x1, y1, x2, y2, conf, int(cls_id)])
            
            detections_np_for_bytetrack = np.array(detections_for_bytetrack) if detections_for_bytetrack else np.empty((0, 6))

            # --- 2. Update ByteTrack ---
            if self.perf: self.perf.start_timer('tracking_update') # Timer for ByteTrack
            online_targets = [] # Output: [x1,y1,x2,y2, track_id, conf, cls_id, det_idx]
            
            with self.tracker_lock: # Lock if ByteTrack methods (like STrack.next_id()) aren't thread-safe
                img_h_model, img_w_model = yolo_result_obj.orig_shape
                dummy_img_for_tracker = np.zeros((img_h_model, img_w_model, 3), dtype=np.uint8)

                online_targets = self.bytetracker.update(
                    detections_np_for_bytetrack,
                    dummy_img_for_tracker 
                )
            if self.perf: self.perf.end_timer('tracking_update')

            # --- 3. Process Tracked Objects ---
            current_frame_active_track_ids_bytetrack = set()
            temp_drawing_info_map = {} # To gather info for drawing for this frame

            for target_data in online_targets:
                # Unpack data from ByteTrack's output
                tx1, ty1, tx2, ty2 = target_data[0], target_data[1], target_data[2], target_data[3] # Coords relative to model input
                track_id = str(int(target_data[4]))
                track_cls_id = int(target_data[6])
                
                current_frame_active_track_ids_bytetrack.add(track_id)
                vehicle_type_name = yolo_result_obj.names.get(track_cls_id, f"CLS_{track_cls_id}")

                # Scale bbox for drawing on `output_frame_for_viz` (sized self.frame_shape)
                # and for zone checking (polygons are relative to self.frame_shape)
                model_h_orig, model_w_orig = yolo_result_obj.orig_shape
                draw_h_target, draw_w_target = self.frame_shape[1], self.frame_shape[0]
                
                draw_x1 = int(tx1 * draw_w_target / model_w_orig)
                draw_y1 = int(ty1 * draw_h_target / model_h_orig)
                draw_x2 = int(tx2 * draw_w_target / model_w_orig)
                draw_y2 = int(ty2 * draw_h_target / model_h_orig)
                
                bbox_for_drawing_and_zone = (draw_x1, draw_y1, draw_x2, draw_y2)
                center_for_drawing = ((draw_x1 + draw_x2) // 2, (draw_y1 + draw_y2) // 2)
                bottom_center_for_zone_check = ((draw_x1 + draw_x2) // 2, draw_y2)

                temp_drawing_info_map[track_id] = {
                    'center': center_for_drawing, 'box': bbox_for_drawing_and_zone,
                    'type': vehicle_type_name, 'event': None, 'status': 'tracked'
                }

                # --- Zone Transition & Application Logic (using self.active_track_data) ---
                with self.tracker_lock: # Lock when accessing/modifying shared self.active_track_data
                    if track_id not in self.active_track_data:
                        # New track ID detected by ByteTrack, check if it's entering a zone
                        entry_event_dir_candidate = None
                        if self.zone_tracker and self.zone_tracker.zones:
                            if self.perf: self.perf.start_timer('zone_checking')
                            for zone_dir_name, zone_poly_pts in self.zone_tracker.zones.items():
                                if self.zone_tracker._check_point_in_zone(bottom_center_for_zone_check, zone_poly_pts):
                                    entry_event_dir_candidate = zone_dir_name
                                    break # Found entry zone
                            if self.perf: self.perf.end_timer('zone_checking')
                        
                        if entry_event_dir_candidate: # If it appeared inside a defined zone
                            self.active_track_data[track_id] = {
                                'id': track_id, 
                                'entry_direction': entry_event_dir_candidate,
                                'entry_time': current_global_timestamp,
                                'type_history': {vehicle_type_name: 1}, # For type consistency
                                'current_type': vehicle_type_name,
                                'last_pos_zone': bottom_center_for_zone_check, # For next frame's prev_pos
                                'last_seen_frame': current_video_frame_num, # Actual video frame number
                                'status': 'active',
                                'entry_point': bottom_center_for_zone_check # For report consistency if needed
                            }
                            if self.perf: self.perf.record_vehicle_entry()
                            temp_drawing_info_map[track_id]['event'] = 'ENTRY'
                            # Optional: Add to frame_local_event_log_entries if detailed per-frame event log needed elsewhere
                            frame_local_event_log_entries.append({'event_type': 'ENTRY', 'id': track_id, 'dir': entry_event_dir_candidate, 'time':current_global_timestamp})


                    else: # Existing track in self.active_track_data
                        track_app_data = self.active_track_data[track_id]
                        track_app_data['last_seen_frame'] = current_video_frame_num # Update last seen
                        
                        # Update type consistency
                        track_app_data['type_history'][vehicle_type_name] = track_app_data['type_history'].get(vehicle_type_name, 0) + 1
                        track_app_data['current_type'] = max(track_app_data['type_history'], key=track_app_data['type_history'].get)
                        temp_drawing_info_map[track_id]['type'] = track_app_data['current_type'] # Update for drawing

                        prev_pos_for_zone_check = track_app_data.get('last_pos_zone')
                        track_app_data['last_pos_zone'] = bottom_center_for_zone_check # Update for next frame

                        if prev_pos_for_zone_check and self.zone_tracker and self.zone_tracker.zones and track_app_data['status'] == 'active':
                            if self.perf: self.perf.start_timer('zone_checking')
                            event_type_zone, event_dir_zone = self.zone_tracker.check_zone_transition(
                                prev_pos_for_zone_check, bbox_for_drawing_and_zone, track_id, current_frame_time_sec
                            )
                            if self.perf: self.perf.end_timer('zone_checking')

                            if event_type_zone == "ENTRY" and track_app_data.get('entry_direction') and track_app_data['entry_direction'] != event_dir_zone:
                                # Inferred Exit from old zone, then New Entry into new zone
                                stored_entry_dir = track_app_data['entry_direction']
                                active_zone_keys = list(self.zone_tracker.zones.keys())
                                if is_valid_movement(stored_entry_dir, event_dir_zone, active_zone_keys):
                                    time_in_int_leg1 = (current_global_timestamp - track_app_data['entry_time']).total_seconds()
                                    # Finalize the first leg of the journey
                                    self._add_to_finalized_paths(track_id, track_app_data, 'exited', event_dir_zone, current_global_timestamp, time_in_int_leg1)
                                    if self.perf: self.perf.record_vehicle_exit('exited', time_in_int_leg1)
                                    
                                    # Update app_data for the new leg (re-entry)
                                    track_app_data['entry_direction'] = event_dir_zone
                                    track_app_data['entry_time'] = current_global_timestamp
                                    track_app_data['entry_point'] = bottom_center_for_zone_check
                                    # track_app_data['type_history'] can be reset or kept cumulative. Keeping cumulative for now.
                                    # track_app_data['status'] remains 'active'
                                    if self.perf: self.perf.record_vehicle_entry()
                                    temp_drawing_info_map[track_id]['event'] = 'RE-ENTRY'
                                    frame_local_event_log_entries.append({'event_type': 'RE-ENTRY', 'id': track_id, 'dir': event_dir_zone, 'time':current_global_timestamp})


                            elif event_type_zone == "EXIT":
                                entry_dir_trk = track_app_data.get('entry_direction')
                                entry_time_stored = track_app_data.get('entry_time')
                                active_zone_keys = list(self.zone_tracker.zones.keys())
                                
                                if entry_dir_trk and is_valid_movement(entry_dir_trk, event_dir_zone, active_zone_keys) and \
                                   entry_time_stored and current_global_timestamp >= entry_time_stored:
                                    time_in_int_exit = (current_global_timestamp - entry_time_stored).total_seconds()
                                    # Finalize the path
                                    self._add_to_finalized_paths(track_id, track_app_data, 'exited', event_dir_zone, current_global_timestamp, time_in_int_exit)
                                    if self.perf: self.perf.record_vehicle_exit('exited', time_in_int_exit)
                                    
                                    track_app_data['status'] = 'exited' # Mark in app state
                                    track_app_data['exit_time'] = current_global_timestamp # For record keeping if needed before removal
                                    track_app_data['exit_direction'] = event_dir_zone 
                                    track_app_data['exit_point'] = bottom_center_for_zone_check
                                    temp_drawing_info_map[track_id]['event'] = 'EXIT'
                                    frame_local_event_log_entries.append({'event_type': 'EXIT', 'id': track_id, 'dir': event_dir_zone, 'time':current_global_timestamp})
            
            # --- 4. Handle Lost / Timed Out Tracks (from self.active_track_data perspective) ---
            ids_to_remove_from_active_data = [] # Tracks to delete from self.active_track_data this frame
            with self.tracker_lock: # Lock when iterating/modifying self.active_track_data
                for track_id_app, track_app_data_item in list(self.active_track_data.items()):
                    if track_app_data_item['status'] != 'active': continue # Already handled (exited)

                    # Check application-level timeout (VEHICLE_TIMEOUTS)
                    time_in_intersection_so_far = (current_global_timestamp - track_app_data_item['entry_time']).total_seconds()
                    vehicle_type_for_timeout = track_app_data_item['current_type']
                    timeout_limit_seconds = VEHICLE_TIMEOUTS.get(vehicle_type_for_timeout, VEHICLE_TIMEOUTS['default'])

                    if time_in_intersection_so_far > timeout_limit_seconds:
                        self._add_to_finalized_paths(track_id_app, track_app_data_item, 'timed_out', 'TIMEOUT', current_global_timestamp, time_in_intersection_so_far)
                        if self.perf: self.perf.record_vehicle_exit('timed_out')
                        
                        if track_id_app in temp_drawing_info_map: temp_drawing_info_map[track_id_app]['status'] = 'timed_out'
                        else: # It wasn't in online_targets this frame but timed out
                            temp_drawing_info_map[track_id_app] = {'status': 'timed_out', 'type': track_app_data_item['current_type'], 'box': None, 'center': None}
                        
                        track_app_data_item['status'] = 'timed_out' # Mark in app state
                        ids_to_remove_from_active_data.append(track_id_app) # Will be removed after loop
                        frame_local_event_log_entries.append({'event_type': 'TIMEOUT', 'id': track_id_app, 'time':current_global_timestamp})
                        continue # Skip to next track if timed out

                # Check for tracks truly lost by ByteTrack (not in its output anymore for too long)
                grace_period_frames_after_bytetrack = self.bytetracker.max_time_lost + int(self.target_fps * 0.5) # e.g. bytetrack_buffer + 0.5 sec delay

                for track_id_app, track_app_data_item in list(self.active_track_data.items()):
                    if track_app_data_item['status'] == 'active' and track_id_app not in current_frame_active_track_ids_bytetrack:
                        # It's active in our app state, but ByteTrack didn't report it this frame.
                        frames_since_last_seen_by_app = current_video_frame_num - track_app_data_item.get('last_seen_frame', current_video_frame_num)
                        if frames_since_last_seen_by_app > grace_period_frames_after_bytetrack:
                            debug_print(f"V:{track_id_app} (type {track_app_data_item['current_type']}) no longer in ByteTrack output for {frames_since_last_seen_by_app} frames. Marking as lost.")
                            time_in_int_at_loss = (current_global_timestamp - track_app_data_item['entry_time']).total_seconds()
                            self._add_to_finalized_paths(track_id_app, track_app_data_item, 'lost_track', 'LOST_TRACK', current_global_timestamp, time_in_int_at_loss)
                            if self.perf: self.perf.record_vehicle_exit('forced_exit') # Or a new 'lost' status for perf
                            
                            track_app_data_item['status'] = 'lost_track' # Use specific status
                            ids_to_remove_from_active_data.append(track_id_app)
                            frame_local_event_log_entries.append({'event_type': 'LOST_TRACK', 'id': track_id_app, 'time':current_global_timestamp})
                
                # Perform removal from self.active_track_data
                for v_id_remove in set(ids_to_remove_from_active_data): # Use set to avoid double removal if timed out and lost in same frame
                    if v_id_remove in self.active_track_data:
                        del self.active_track_data[v_id_remove] # Remove from active, already added to finalized
                    if self.zone_tracker: self.zone_tracker.remove_vehicle_data(v_id_remove)


            # --- 5. Drawing Pass ---
            if ENABLE_VISUALIZATION and output_frame_for_viz is not None:
                if self.perf: self.perf.start_timer('drawing')
                if self.zone_tracker and self.zone_tracker.zones:
                    output_frame_for_viz = visual_overlay.draw_zones(output_frame_for_viz, self.zone_tracker)

                for track_id_draw, data_to_draw in temp_drawing_info_map.items():
                    draw_box_coords = data_to_draw.get('box')
                    draw_type = data_to_draw.get('type', "Unknown")
                    draw_status = data_to_draw.get('status', 'tracked') # Default to 'tracked' from map
                    draw_event = data_to_draw.get('event')
                    draw_center = data_to_draw.get('center')
                    
                    entry_dir_for_viz = None; time_active_for_viz = None
                    # Get latest data from active_track_data if available for more accurate viz status
                    if track_id_draw in self.active_track_data:
                        app_data_viz = self.active_track_data[track_id_draw]
                        if app_data_viz['status'] == 'active': # Update status if still active in app_data
                            draw_status = 'active'
                        entry_dir_for_viz = app_data_viz.get('entry_direction')
                        time_active_for_viz = (current_global_timestamp - app_data_viz['entry_time']).total_seconds()
                    
                    # Avoid drawing boxes for tracks that were only determined to be timed_out/lost this frame
                    # and didn't have a detection box from ByteTrack this frame.
                    if draw_box_coords and draw_status not in ['timed_out', 'lost_track']:
                         visual_overlay.draw_detection_box(output_frame_for_viz, draw_box_coords, track_id_draw, draw_type,
                                                           status=draw_status, entry_dir=entry_dir_for_viz, time_active=time_active_for_viz)
                    
                    # Trail drawing would require collecting points per track_id in self.active_track_data
                    # STrack.history_observations is not directly used here. This part needs separate implementation.
                    # visual_overlay.draw_tracking_trail(output_frame_for_viz, trail_points, track_id_draw)

                    if draw_event and draw_center and draw_event != 'RE-ENTRY': # Don't mark re-entry as a new dot perhaps
                         visual_overlay.draw_event_marker(output_frame_for_viz, draw_center, draw_event, track_id_draw[-4:])
                
                active_count_viz = sum(1 for ad in self.active_track_data.values() if ad['status'] == 'active')
                tracker_info_for_overlay = {'get_active_vehicle_count': lambda: active_count_viz} # Pass lambda for overlay
                output_frame_for_viz = visual_overlay.add_status_overlay(
                    output_frame_for_viz, current_video_frame_num, current_global_timestamp, tracker_info_for_overlay
                )
                if self.perf: self.perf.end_timer('drawing')
            
            return frame_local_event_log_entries, output_frame_for_viz
            # --- End process_single_frame inner function ---

        # --- Main execution for the batch using ThreadPoolExecutor ---
        if self.perf: self.perf.start_timer('detection_processing_batch') # Combined timer for tracker+app_logic

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures_map = {
                executor.submit(process_single_frame, i): i
                for i in range(len(results)) # results is the batch of YOLO outputs
            }
            for future in concurrent.futures.as_completed(futures_map):
                original_batch_idx = futures_map[future]
                try:
                    # frame_log is the list of verbose dicts from frame_local_event_log_entries
                    # We might not need to aggregate it if self.finalized_paths_for_report is primary.
                    _frame_verbose_log, processed_viz_frame = future.result() 
                    # batch_frame_event_logs[original_batch_idx] = _frame_verbose_log # If still needed
                    processed_frames_list_viz[original_batch_idx] = processed_viz_frame
                except Exception as exc_future:
                    print(f'\nError processing frame number {frame_numbers[original_batch_idx]} (batch index {original_batch_idx}): {exc_future}')
                    import traceback
                    traceback.print_exc()
                    # Ensure lists have placeholders
                    # batch_frame_event_logs[original_batch_idx] = [] 
                    processed_frames_list_viz[original_batch_idx] = None
        
        # final_aggregated_verbose_logs = [log_entry for frame_logs in batch_frame_event_logs for log_entry in frame_logs] # If using this
        
        if self.perf: self.perf.end_timer('detection_processing_batch')

        # The primary data for the report is now self.finalized_paths_for_report,
        # so this function doesn't need to return the aggregated verbose logs for that purpose.
        return [], processed_frames_list_viz 
    # --- End _process_inference_results ---

    # --- Main Processing Function ---
    def process_video(self, video_path, start_date_str, start_time_str, primary_direction):
        if not self.processing_lock.acquire(blocking=False):
            print("Warning: Processing lock already acquired."); return "Processing is already in progress."

        if ByteTrack is None:
            self.processing_lock.release()
            return "❌ Error: ByteTrack (BoxMOT) is not available. Processing cannot start."

        self.bytetracker = ByteTrack(
            min_conf=0.1, track_thresh=0.35, match_thresh=0.8,
            track_buffer=int(self.target_fps * 5.0), frame_rate=self.target_fps, per_class=False
        )
        self.current_bytetrack_frame_id = 0
        self.active_track_data.clear()
        self.finalized_paths_for_report = [] # Reset for new video

        self.zone_tracker = None
        self.perf = PerformanceTracker(enabled=ENABLE_DETAILED_PERFORMANCE_METRICS)
        self.stop_event = threading.Event()
        self.frame_read_queue = queue.Queue(maxsize=max(self.batch_size * 4, 64))
        self.last_cleanup_time = time.monotonic()
        self.detection_zones_polygons = None
        self.final_output_path = None
        self.writer_thread = None
        self.video_write_queue = None

        print(f"--- Starting New Video Processing Run (with ByteTrack for Report Parity) ---")
        # ... (rest of initial prints and DAV conversion logic UNCHANGED) ...
        actual_video_path_to_process = video_path; temp_converted_dav_mp4 = None
        if video_path.lower().endswith(".dav"):
            if not FFMPEG_PATH:
                error_msg = "ERROR: FFmpeg not found for .dav conversion."
                print(error_msg); self.processing_lock.release(); raise RuntimeError(error_msg)
            print(f"Input is .dav: {video_path}. Converting...")
            converted_path = convert_dav_to_mp4_internal(video_path)
            if converted_path and os.path.exists(converted_path):
                actual_video_path_to_process = converted_path
                temp_converted_dav_mp4 = converted_path
                print(f".dav converted to: {actual_video_path_to_process}")
                time.sleep(0.5) 
            else:
                error_msg = f"Failed to convert .dav file: {video_path}"
                print(error_msg); self.processing_lock.release(); raise RuntimeError(error_msg)

        try:
            # --- Zone Definition (UNCHANGED) ---
            self.detection_zones_polygons = None
            if self.line_mode == 'hardcoded':
                if 'LINE_POINTS' not in globals() or not isinstance(LINE_POINTS, dict) or not LINE_POINTS: raise ValueError("Hardcoded: LINE_POINTS missing/empty.")
                valid_polygons_hc = {dir: pts for dir, pts in LINE_POINTS.items() if isinstance(pts, list) and len(pts) >= 3}
                if not valid_polygons_hc: raise ValueError("Hardcoded: No valid polygons in LINE_POINTS.")
                self.detection_zones_polygons = {dir: np.array(pts, dtype=np.int32) for dir, pts in valid_polygons_hc.items()}
            elif self.line_mode == 'interactive':
                if not self.gradio_polygons: raise ValueError("Interactive: No zone polygons defined.")
                valid_polygons = {dir: pts for dir, pts in self.gradio_polygons.items() if pts and len(pts) >= 3}
                if len(valid_polygons) < 1: raise ValueError(f"Interactive: Need >= 1 valid zone. Found {len(valid_polygons)}.")
                self.detection_zones_polygons = {dir: np.array(pts, dtype=np.int32) for dir, pts in valid_polygons.items()}
            else: raise ValueError(f"Invalid LINE_MODE '{self.line_mode}'.")
            if not self.detection_zones_polygons: raise ValueError("Zone polygon data missing for ZoneTracker.")
            self.zone_tracker = ZoneTracker(self.detection_zones_polygons)
            if not self.zone_tracker or not self.zone_tracker.zones: raise RuntimeError("CRITICAL: ZoneTracker failed to initialize.")

            if self.model is None:
                effective_model_conf_for_detections = 0.05
                self.model = load_model(MODEL_PATH, self.device, effective_model_conf_for_detections, IOU_THRESHOLD)
            if self.model is None: raise RuntimeError(f"Failed to load model {MODEL_PATH}")

            try:
                start_date_dt=datetime.strptime(start_date_str, "%y%m%d"); time_part,ms_part=(start_time_str[:6], start_time_str[6:]) if len(start_time_str)>6 else (start_time_str,"0")
                base_time_dt=datetime.strptime(time_part,"%H%M%S").time(); microseconds_val=int(ms_part.ljust(3,'0'))*1000
                start_time_dt=base_time_dt.replace(microsecond=microseconds_val); start_datetime_obj_global=datetime.combine(start_date_dt.date(), start_time_dt) # Renamed for clarity
                print(f"Video Start Timestamp: {start_datetime_obj_global.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            except ValueError as e_ts: raise ValueError(f"Invalid date/time format: {e_ts}.")

            total_frames, original_fps = self._get_video_properties(actual_video_path_to_process)
            print(f"Video Properties: ~{total_frames} frames, {original_fps:.2f} FPS (Original)")
            if total_frames <= 0: raise ValueError("Cannot process video with zero or unreadable total frames.")

            # --- Setup I/O Threads (UNCHANGED) ---
            print("Starting I/O threads...")
            reader_args = (actual_video_path_to_process, original_fps, total_frames) if self.actually_use_advanced_reader else (actual_video_path_to_process, original_fps)
            reader_target = self._frame_reader_task_pvr_segmented if self.actually_use_advanced_reader else self._frame_reader_task
            self.reader_thread = threading.Thread(target=reader_target, args=reader_args, daemon=True); self.reader_thread.start()
            if ENABLE_VISUALIZATION:
                os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True); ts_now=datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename_base_viz=f"output_{os.path.splitext(os.path.basename(actual_video_path_to_process))[0]}_{ts_now}"
                output_path_viz=os.path.join(VIDEO_OUTPUT_DIR, f"{output_filename_base_viz}.mp4");
                self.final_output_path = output_path_viz
                self.video_write_queue = queue.Queue(maxsize=self.batch_size*4)
                self.writer_thread = threading.Thread(target=self._frame_writer_task, args=(output_path_viz,), daemon=True); self.writer_thread.start()
            else: self.video_write_queue = None; self.writer_thread = None; self.final_output_path = None
            
            # --- Main Loop ---
            # all_event_logs_final is no longer the primary source for excel report data
            frames_processed_count = 0
            frame_interval = max(1, round(original_fps / self.target_fps)) if self.target_fps > 0 and original_fps > 0 else 1
            total_target_frames_for_perf = total_frames // frame_interval if total_frames > 0 else 0
            if self.perf: self.perf.start_processing(total_target_frames_for_perf if total_target_frames_for_perf > 0 else total_frames)
            
            current_batch_frames_metadata, current_batch_frame_numbers, current_batch_frame_times = [], [], []
            last_prog_print_time = time.monotonic();

            while True: # Main processing loop (largely UNCHANGED from previous ByteTrack version)
                try:
                    frame_data_item = self.frame_read_queue.get(timeout=60)
                    if frame_data_item is None: self.frame_read_queue.task_done(); break 
                    if self.perf: self.perf.start_timer('video_read')
                    frame_content_raw, frame_num_raw, frame_time_s, frame_meta = frame_data_item
                    if self.perf: self.perf.end_timer('video_read')
                    if frame_content_raw is None: self.frame_read_queue.task_done(); continue

                    current_batch_frames_metadata.append((frame_content_raw, frame_meta))
                    current_batch_frame_numbers.append(frame_num_raw)
                    current_batch_frame_times.append(frame_time_s)

                    if len(current_batch_frames_metadata) >= self.batch_size:
                        batch_proc_start_time = time.monotonic()
                        if self.perf: self.perf.start_timer('preprocessing')
                        input_batch_tensor_model = self._preprocess_batch(current_batch_frames_metadata)
                        self.perf.end_timer('preprocessing')

                        if input_batch_tensor_model is None or input_batch_tensor_model.nelement() == 0:
                            current_batch_frames_metadata.clear(); current_batch_frame_numbers.clear(); current_batch_frame_times.clear()
                            for _i in range(len(current_batch_frames_metadata)): self.frame_read_queue.task_done() # Mark done if skipped
                            continue
                        
                        yolo_results_batch=None; current_stream=None
                        if self.device=='cuda' and self.streams: current_stream=self.streams[self.current_stream_index%len(self.streams)]; self.current_stream_index+=1
                        with torch.cuda.stream(current_stream) if current_stream else torch.no_grad():
                           with torch.amp.autocast(device_type=self.device,enabled=MIXED_PRECISION and self.device=='cuda'):
                             if self.perf and self.perf.start_event: self.perf.start_event.record(stream=current_stream)
                             yolo_results_batch=self.model(input_batch_tensor_model,verbose=False)
                             if self.perf and self.perf.end_event: self.perf.end_event.record(stream=current_stream)
                        if current_stream: current_stream.synchronize()
                        if self.perf and self.perf.start_event and self.perf.end_event: self.perf.record_inference_time_gpu(self.perf.start_event, self.perf.end_event)

                        # _process_inference_results now appends to self.finalized_paths_for_report internally
                        _, processed_frames_for_viz = self._process_inference_results(
                            yolo_results_batch, current_batch_frames_metadata, current_batch_frame_numbers, 
                            current_batch_frame_times, start_datetime_obj_global
                        )
                        # all_event_logs_final.extend(batch_event_log_output) # Not primary for report
                        if self.perf: self.perf.record_detection(sum(len(r.boxes) for r in yolo_results_batch if hasattr(r,'boxes') and r.boxes))

                        if ENABLE_VISUALIZATION and self.video_write_queue:
                            if self.perf: self.perf.start_timer('video_write')
                            for pf_viz in processed_frames_for_viz:
                                if pf_viz is not None:
                                    try: self.video_write_queue.put(pf_viz, block=True, timeout=2.0)
                                    except queue.Full: print("WARN: Video writer queue full.")
                            if self.perf: self.perf.end_timer('video_write')

                        batch_proc_duration = time.monotonic() - batch_proc_start_time
                        frames_in_this_batch = len(current_batch_frames_metadata)
                        frames_processed_count += frames_in_this_batch
                        if self.perf: self.perf.record_batch_processed(frames_in_this_batch, batch_proc_duration); self.perf.sample_system_metrics()
                        
                        current_monotonic_time = time.monotonic()
                        if current_monotonic_time - self.last_cleanup_time > MEMORY_CLEANUP_INTERVAL:
                            if self.perf: self.perf.start_timer('memory_cleanup')
                            last_frame_time_sec_in_batch = current_batch_frame_times[-1] if current_batch_frame_times else 0
                            cleanup_tracking_data(self, last_frame_time_sec_in_batch); cleanup_memory()
                            self.last_cleanup_time = current_monotonic_time
                            if self.perf: self.perf.end_timer('memory_cleanup')

                        if self.perf and (current_monotonic_time - last_prog_print_time > 1.0):
                            prog_stats=self.perf.get_progress()
                            active_app_tracks = sum(1 for ad in self.active_track_data.values() if ad['status'] == 'active')
                            print(f"\rProgress:{prog_stats['percent']:.1f}%|FPS:{prog_stats['fps']:.1f}|Active(App):{active_app_tracks}|ETA:{prog_stats['eta']} ", end="")
                            last_prog_print_time = current_monotonic_time

                        current_batch_frames_metadata.clear(); current_batch_frame_numbers.clear(); current_batch_frame_times.clear()
                        del input_batch_tensor_model, yolo_results_batch; cleanup_memory()
                    
                    self.frame_read_queue.task_done()
                except queue.Empty: 
                    if self.stop_event.is_set() and (self.frame_read_queue is None or self.frame_read_queue.empty()): break
                    continue 
                except Exception as e_loop: print(f"\nERROR during processing loop: {e_loop}"); import traceback; traceback.print_exc(); self.stop_event.set(); break
            
            if current_batch_frames_metadata: # Process Final Batch (UNCHANGED from previous ByteTrack version)
                print("\nProcessing final batch...")
                batch_proc_start_time_final=time.monotonic()
                if self.perf: self.perf.start_timer('preprocessing'); input_batch_final=self._preprocess_batch(current_batch_frames_metadata); self.perf.end_timer('preprocessing')
                if input_batch_final is not None and input_batch_final.nelement() > 0:
                    yolo_results_final=None; current_stream_final=None
                    if self.device=='cuda' and self.streams: current_stream_final=self.streams[self.current_stream_index%len(self.streams)]; self.current_stream_index+=1
                    with torch.cuda.stream(current_stream_final) if current_stream_final else torch.no_grad():
                        with torch.amp.autocast(device_type=self.device,enabled=MIXED_PRECISION and self.device=='cuda'):
                           if self.perf and self.perf.start_event: self.perf.start_event.record(stream=current_stream_final)
                           yolo_results_final=self.model(input_batch_final,verbose=False)
                           if self.perf and self.perf.end_event: self.perf.end_event.record(stream=current_stream_final)
                        if current_stream_final: current_stream_final.synchronize()
                        if self.perf and self.perf.start_event and self.perf.end_event: self.perf.record_inference_time_gpu(self.perf.start_event, self.perf.end_event)
                    _, processed_frames_final_viz = self._process_inference_results(yolo_results_final, current_batch_frames_metadata, current_batch_frame_numbers, current_batch_frame_times, start_datetime_obj_global)
                    # all_event_logs_final.extend(batch_log_final) # Not primary for report
                    if ENABLE_VISUALIZATION and self.video_write_queue:
                        if self.perf: self.perf.start_timer('video_write');
                        for pf_final_viz in processed_frames_final_viz:
                            if pf_final_viz is not None:
                                try: self.video_write_queue.put(pf_final_viz, block=True, timeout=2.0)
                                except queue.Full: print("WARN: Vid writer Q full (final batch).")
                        if self.perf: self.perf.end_timer('video_write')
                    batch_proc_duration_final=time.monotonic()-batch_proc_start_time_final
                    frames_in_final_batch = len(current_batch_frames_metadata)
                    frames_processed_count+=frames_in_final_batch
                    if self.perf: self.perf.record_batch_processed(frames_in_final_batch, batch_proc_duration_final)
                    del input_batch_final, yolo_results_final; cleanup_memory()
            print("\nVideo processing loop finished.")

            if self.active_track_data: # Force finalize remaining tracks
                print(f"Finalizing {sum(1 for d in self.active_track_data.values() if d['status'] == 'active')} remaining app-level active tracks...")
                last_ts_overall = start_datetime_obj_global + timedelta(seconds=current_batch_frame_times[-1]) if current_batch_frame_times else start_datetime_obj_global
                
                with self.tracker_lock:
                    for track_id_force_end, data_force_end in list(self.active_track_data.items()):
                        if data_force_end['status'] == 'active':
                            time_in_int_at_end = (last_ts_overall - data_force_end['entry_time']).total_seconds()
                            self._add_to_finalized_paths(track_id_force_end, data_force_end, 'forced_exit', 'FORCED_EXIT', last_ts_overall, time_in_int_at_end)
                            if self.perf: self.perf.record_vehicle_exit('forced_exit')
                    self.active_track_data.clear()

            print("Signaling threads..."); self.stop_event.set() # Rest of thread cleanup (UNCHANGED)
            if ENABLE_VISUALIZATION and self.video_write_queue:
                try: self.video_write_queue.put(None, timeout=5.0)
                except queue.Full: print("WARN: Could not signal writer queue (full).")
            if self.frame_read_queue:
                try: self.frame_read_queue.put_nowait(None)
                except queue.Full: pass
            print("Waiting for threads..."); t_wait_start=time.monotonic()
            if self.writer_thread and self.writer_thread.is_alive(): self.writer_thread.join(timeout=30)
            if self.reader_thread and self.reader_thread.is_alive(): self.reader_thread.join(timeout=5)
            print(f"Threads joined/timed out in {time.monotonic()-t_wait_start:.2f}s")

            if self.perf: self.perf.end_processing(); self.perf.print_summary()
            total_time_taken_final=self.perf.total_time if self.perf else 0.0
            final_fps_overall=frames_processed_count/total_time_taken_final if total_time_taken_final>0 else 0.0
            
            completed_paths_count_valid_excel = sum(1 for p in self.finalized_paths_for_report if p['status'] == 'exited')

            excel_path_result = create_excel_report(
                completed_paths_data=self.finalized_paths_for_report, # Use the new list
                start_datetime=start_datetime_obj_global, 
                primary_direction=primary_direction, 
                video_path=actual_video_path_to_process
            )
            if ENABLE_DETAILED_PERFORMANCE_METRICS and self.perf: visualize_performance(self.perf)

            result_msg_final=f"✅ Processing completed (ByteTrack, Report Parity).\n" # Rest of result message (UNCHANGED)
            if ENABLE_VISUALIZATION:
                if self.final_output_path and "ERROR" not in self.final_output_path: result_msg_final+=f"Output video: '{self.final_output_path}'.\n"
                elif self.final_output_path and "ERROR" in self.final_output_path: result_msg_final+=f"Output video generation FAILED: {self.final_output_path}.\n"
                else: result_msg_final+=f"Output video status unknown.\n"
            if excel_path_result: result_msg_final+=f"Excel report: {excel_path_result}\n"
            if ENABLE_DETAILED_PERFORMANCE_METRICS and self.perf: result_msg_final+=f"Perf charts: 'performance_charts/'\n"
            result_msg_final+=f"\n--- Summary Stats ---\nSTAT_FRAMES_PROCESSED={frames_processed_count}\nSTAT_TIME_SECONDS={total_time_taken_final:.2f}\nSTAT_FPS={final_fps_overall:.2f}\nSTAT_COMPLETED_PATHS={completed_paths_count_valid_excel}\n--- End Stats ---"
            return result_msg_final

        except (ValueError, RuntimeError, FileNotFoundError, Exception) as e_proc: # Final error handling (UNCHANGED)
            print(f"\nFATAL ERROR during setup or processing: {e_proc}")
            import traceback; traceback.print_exc()
            if hasattr(self, 'stop_event') and self.stop_event: self.stop_event.set()
            return f"❌ Error: {e_proc}"
        finally: # Final cleanup (UNCHANGED)
            try:
                if hasattr(self, 'reader_thread') and self.reader_thread and self.reader_thread.is_alive(): self.reader_thread.join(timeout=1)
                if hasattr(self, 'writer_thread') and self.writer_thread and self.writer_thread.is_alive(): self.writer_thread.join(timeout=1)
            except Exception as join_e: print(f"Error joining threads: {join_e}")
            if temp_converted_dav_mp4:
                try: debug_print(f"Cleaning up temp DAV: {temp_converted_dav_mp4}"); os.remove(temp_converted_dav_mp4)
                except OSError as e_del_dav: debug_print(f"Error deleting temp DAV: {e_del_dav}")
            self.model=None; self.zone_tracker=None; self.perf=None; self.detection_zones_polygons=None
            self.frame_read_queue=None; self.video_write_queue=None; self.reader_thread=None; self.writer_thread=None; self.stop_event=None
            self.final_output_path = None; self.active_track_data.clear(); self.finalized_paths_for_report.clear()
            cleanup_memory()
            if self.processing_lock.locked():
                try: self.processing_lock.release(); print("Processing finished, lock released.")
                except RuntimeError as release_err: print(f"Warn: Error releasing lock: {release_err}")
            else: print("Processing finished (lock not held or already released).")

# --- End of VideoProcessor class ---

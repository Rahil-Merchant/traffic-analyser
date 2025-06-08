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
import yaml
from datetime import datetime, timedelta
import concurrent.futures
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

# Configuration
from config import (
    TARGET_FPS, FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH_VIZ, FRAME_HEIGHT_VIZ,
    VIDEO_OUTPUT_DIR, ENABLE_DETAILED_PERFORMANCE_METRICS, OPTIMAL_BATCH_SIZE,
    MIXED_PRECISION, PARALLEL_STREAMS, THREAD_COUNT, IOU_THRESHOLD, TTA,
    MEMORY_CLEANUP_INTERVAL, MODEL_PATH, CONF_THRESHOLD, VEHICLE_TIMEOUTS,
    LINE_POINTS, MODEL_INPUT_SIZE, ENABLE_VISUALIZATION, DATA_YAML_PATH,
    USE_ADVANCED_VIDEO_READER, PYVIDEOREADER_DECODE_SEGMENT_SIZE, NUM_INFERENCE_STREAMS,
    BYTETRACK_MATCH_THRESH, BYTETRACK_MIN_CONF, BYTETRACK_TRACK_BUFFER_SECONDS, BYTETRACK_TRACK_THRESH
)

# Modules
from core.utils import debug_print, is_valid_movement, cleanup_memory
from core.image_processing import enhance_frame_for_detection
from core.performance import PerformanceTracker
from models.model_loader import load_model, get_device
from tracking.zone_tracker import ZoneTracker
from tracking.cleanup import cleanup_tracking_data
from visualization import overlay as visual_overlay
from reporting.excel_report import create_excel_report, create_bidirectional_link_count_report
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
    def __init__(self, LINE_MODE):
        self.target_fps = TARGET_FPS
        self.frame_shape = (FRAME_WIDTH, FRAME_HEIGHT)
        self.frame_shape_viz = (FRAME_WIDTH_VIZ, FRAME_HEIGHT_VIZ)
        self.model_input_size_config = MODEL_INPUT_SIZE # Square size for model
        self.batch_size = OPTIMAL_BATCH_SIZE
        self.processing_lock = threading.Lock()
        self.line_mode = LINE_MODE # Controls zone definition method
        self.max_workers = THREAD_COUNT

        # Polygon State
        self.zone_polygons = {} # Stores {direction: List[tuple(x,y)]}

        # Core components
        self.device = get_device()
        self.model = None
        self.models_list = [] # For multiple inference instances
        self.num_inference_streams = max(1, NUM_INFERENCE_STREAMS)
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

        if DATA_YAML_PATH and os.path.exists(DATA_YAML_PATH):
            try:
                with open(DATA_YAML_PATH, 'r', encoding='utf-8') as f: # Added encoding
                    data_yaml = yaml.safe_load(f)
                if data_yaml and 'names' in data_yaml and isinstance(data_yaml['names'], list):
                    self.class_id_to_name_map = {i: str(name) for i, name in enumerate(data_yaml['names'])}
                    debug_print(f"VideoProcessor: Successfully loaded class names from '{DATA_YAML_PATH}': {self.class_id_to_name_map}")
                else:
                    debug_print(f"VideoProcessor: Warning - 'names' key (as a list) not found or invalid in '{DATA_YAML_PATH}'. Check YAML format.")
            except Exception as e:
                debug_print(f"VideoProcessor: Error loading class names from YAML '{DATA_YAML_PATH}': {e}")
        else:
            debug_print(f"VideoProcessor: Warning - DATA_YAML_PATH ('{DATA_YAML_PATH}') not found or not specified. Class names will be default.")

        # Threading & Queues
        self.frame_read_queue = None
        self.video_write_queue = None # Will be None if visualization is disabled
        self.max_queue_size = max(self.batch_size * self.num_inference_streams * 8, 128)
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

    def force_stop_processing(self):
        """Force stop all processing threads and cleanup"""
        print("Force stopping video processor...")
        
        if hasattr(self, 'stop_event') and self.stop_event:
            self.stop_event.set()
        
        # Force close queues
        if hasattr(self, 'frame_read_queue') and self.frame_read_queue:
            try:
                while not self.frame_read_queue.empty():
                    self.frame_read_queue.get_nowait()
            except:
                pass
        
        if hasattr(self, 'video_write_queue') and self.video_write_queue:
            try:
                while not self.video_write_queue.empty():
                    self.video_write_queue.get_nowait()
            except:
                pass
        
        # Release processing lock
        if hasattr(self, 'processing_lock') and self.processing_lock.locked():
            try:
                self.processing_lock.release()
            except:
                pass
        
        print("Video processor force stop complete")


    # --- Video Reading/Writing Threads ---
    def _get_video_properties(self, video_paths):
        """Gets total frames and original FPS, trying PyVideoReader first if enabled."""
        total_frames = 0
        original_fps = 0.0

        if self.actually_use_advanced_reader:
            temp_reader_rs = None
            total_frames = 0
            try:
                # For getting properties, no need for device='cuda' or specific resize
                for video_path in video_paths:
                    temp_reader_rs = PyVideoReader(video_path)
                    info = temp_reader_rs.get_info()
                    total_frames += int(info.get('frame_count', 0))
                    fps_str = info.get('fps', '0.0')
                    if fps_str:
                        try: original_fps = float(fps_str)
                        except ValueError: pass
                    del temp_reader_rs 
            except Exception as e:
                debug_print(f"PyVideoReader property fetch failed: {e}. Falling back to OpenCV for props.")
                if temp_reader_rs: 
                    del temp_reader_rs
        
        if total_frames <= 0 or original_fps <= 0: # If PyVideoReader failed or wasn't used
            _total_frames_cv = 0
            cap_check = None
            try:
                for video_path in video_paths:
                    cap_check = cv2.VideoCapture(video_path)
                    if not cap_check.isOpened():
                        raise FileNotFoundError(f"OpenCV: Cannot open video for properties: {video_path}")
                    
                    _total_frames_cv += int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
                    _original_fps_cv = cap_check.get(cv2.CAP_PROP_FPS)
                    
                    if total_frames <= 0: total_frames = _total_frames_cv
                    if original_fps <= 0: original_fps = _original_fps_cv
            except Exception as e:
                print(f"Error getting video properties with OpenCV: {e}")
            finally:
                if cap_check: 
                    cap_check.release()

        if original_fps <= 0: original_fps = 30.0 # Final fallback
        if total_frames <= 0:
            print("WARNING: Could not determine total_frames accurately. Progress calculation might be affected.")   
        return total_frames, original_fps

    def _scale_polygons_to_processing_coordinates(self, polygons_dict, original_image_width, original_image_height):
        """
        Scale polygon coordinates from original image size to processing frame size.
        
        Args:
            polygons_dict: Dictionary of {direction: [(x,y), ...]} in original image coordinates
            original_image_width: Width of the original preview image
            original_image_height: Height of the original preview image
        
        Returns:
            Dictionary of scaled polygons in processing frame coordinates
        """
        if not polygons_dict:
            return {}
        
        # These are the coordinates used for detection box scaling
        target_width = self.frame_shape[0]   # FRAME_WIDTH (853)
        target_height = self.frame_shape[1]  # FRAME_HEIGHT (640)
        
        print(f"Scaling polygons from ({original_image_width}x{original_image_height}) to ({target_width}x{target_height})")
        
        scaled_polygons = {}
        for direction, points_list in polygons_dict.items():
            if not points_list or len(points_list) < 3:
                continue
                
            scaled_points = []
            for x, y in points_list:
                # Scale to processing frame coordinates
                scaled_x = int(x * target_width / original_image_width)
                scaled_y = int(y * target_height / original_image_height)
                scaled_points.append((scaled_x, scaled_y))
            
            scaled_polygons[direction] = scaled_points
            print(f"  {direction}: {len(points_list)} points scaled")
            print(f"    Original first point: {points_list[0]}")
            print(f"    Scaled first point: {scaled_points[0]}")
        
        return scaled_polygons

    def _frame_reader_task(self, video_paths, original_fps):
        cap = None
        try:
            for video_path in video_paths:
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
            import traceback; traceback.print_exc()
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

    def _frame_reader_task_pvr_segmented(self, video_paths, original_fps_from_props, total_frames_from_props): # total_frames_from_props IS THE AGGREGATE
        """Frame reader using PyVideoReader, decoding in segments, then selecting frames."""
        pvr_instance = None
        try:
            target_dim_model = self.model_input_size_config
            debug_print(f"PVR SegReader: Init PyVideoReader: device='cuda', target_size=({target_dim_model}x{target_dim_model})")

            # num_segments is based on the AGGREGATE total_frames_from_props
            num_segments = math.ceil(total_frames_from_props / self.pvr_decode_segment_size) if total_frames_from_props > 0 else 1
            if total_frames_from_props == 0 : num_segments = 0

            # This variable will keep track of the frame offset as we process multiple videos
            cumulative_frames_processed_so_far = 0

            for video_idx, video_path in enumerate(video_paths): # Iterate through each video
                print(f"PVR SegReader: Processing video {video_idx+1}/{len(video_paths)}: {video_path}")
                # Get properties for the CURRENT video
                current_video_total_frames = 0
                current_video_original_fps = original_fps_from_props # Assume FPS is consistent, or get it per video
                try:
                    temp_pvr = PyVideoReader(video_path)
                    info = temp_pvr.get_info()
                    current_video_total_frames = int(info.get('frame_count', 0))
                    # You might want to get current_video_original_fps here too if it can vary
                    del temp_pvr
                except Exception as e_info:
                    print(f"PVR SegReader: Error getting info for {video_path}: {e_info}. Skipping this video.")
                    continue # Skip to the next video

                if current_video_total_frames == 0:
                    print(f"PVR SegReader: Video {video_path} has 0 frames. Skipping.")
                    continue

                pvr_instance = PyVideoReader(video_path, device='cuda',
                                            resize_shorter_side=target_dim_model,
                                            resize_longer_side=target_dim_model)

                frame_interval = max(1, round(current_video_original_fps / self.target_fps))
                queued_frame_count_for_this_video = 0

                # Calculate segments for the CURRENT video
                num_segments_for_current_video = math.ceil(current_video_total_frames / self.pvr_decode_segment_size)

                debug_print(f"PVR SegReader (Video: {os.path.basename(video_path)}): "
                            f"Length={current_video_total_frames} frames, "
                            f"Segment Size={self.pvr_decode_segment_size}, Num Segments={num_segments_for_current_video}")

                for i in range(num_segments_for_current_video): # Iterate segments for THIS video
                    if self.stop_event.is_set(): break

                    # Segment indices are relative to the CURRENT video
                    segment_start_current_video_idx = i * self.pvr_decode_segment_size
                    segment_end_current_video_idx = min((i + 1) * self.pvr_decode_segment_size, current_video_total_frames)

                    if segment_start_current_video_idx >= segment_end_current_video_idx :
                        continue

                    debug_print(f"PVR SegReader (Video: {os.path.basename(video_path)}): Decoding segment {i+1}/{num_segments_for_current_video} "
                                f"(frames {segment_start_current_video_idx} to {segment_end_current_video_idx-1} of current video)...")

                    segment_frames_np = pvr_instance.decode(start_frame=segment_start_current_video_idx,
                                                            end_frame=segment_end_current_video_idx)

                    if segment_frames_np is None or segment_frames_np.shape[0] == 0:
                        debug_print(f"PVR SegReader: Segment {i+1} (current video) returned no frames. Video might have ended or error.")
                        continue

                    for frame_in_segment_idx in range(segment_frames_np.shape[0]):
                        if self.stop_event.is_set(): break

                        # This is the original, absolute frame index in the *current individual video*
                        frame_idx_in_current_video = segment_start_current_video_idx + frame_in_segment_idx

                        if frame_idx_in_current_video % frame_interval == 0:
                            frame_content = segment_frames_np[frame_in_segment_idx]

                            # Global frame number and time need to account for previous videos
                            current_overall_raw_frame_idx = cumulative_frames_processed_so_far + frame_idx_in_current_video
                            # Assume original_fps_from_props is the FPS for the primary video or a consistent FPS
                            current_frame_time_sec = current_overall_raw_frame_idx / original_fps_from_props

                            metadata = {'is_rgb': True, 'source': 'pvr_cuda_segmented'}

                            try:
                                self.frame_read_queue.put((frame_content, current_overall_raw_frame_idx, current_frame_time_sec, metadata),
                                                        block=True, timeout=5.0)
                                queued_frame_count_for_this_video += 1
                            # ... (rest of queue put logic) ...
                            except queue.Full:
                                debug_print("PVR SegReader: Queue full, retrying...")
                                time.sleep(0.1);
                                if self.stop_event.is_set(): break
                                try:
                                    self.frame_read_queue.put((frame_content, current_overall_raw_frame_idx, current_frame_time_sec, metadata),
                                                            block=True, timeout=5.0)
                                    queued_frame_count_for_this_video += 1
                                except queue.Full: print("ERROR: PVR SegReader queue persistently full."); self.stop_event.set(); break # Stop reader
                            except Exception as q_e: debug_print(f"PVR SegReader queue put error: {q_e}"); self.stop_event.set(); break


                    del segment_frames_np
                    gc.collect()

                debug_print(f"PVR SegReader: Finished video {os.path.basename(video_path)}. Queued {queued_frame_count_for_this_video} frames.")
                if pvr_instance:
                    del pvr_instance # Explicitly delete to release resources for this video
                    pvr_instance = None
                gc.collect()
                cumulative_frames_processed_so_far += current_video_total_frames # Add this video's total frames to the cumulative count

            debug_print(f"PVR SegReader finished all videos.")

        except Exception as e:
            print(f"FATAL ERROR in _frame_reader_task_pvr_segmented: {e}")
            import traceback; traceback.print_exc()
        finally:
            if pvr_instance: del pvr_instance
            if self.frame_read_queue:
                try: self.frame_read_queue.put(None, block=False, timeout=1.0)
                except queue.Full: 
                    if i==num_segments_for_current_video-1:     #setting this to avoid getting stuck in main loop when no other frames are available to add in queue
                        print("Queue is full and it is the last segment: Setting stop event")
                        self.stop_event.set()
                    pass
                
    def _frame_writer_task(self, output_path):
        # This task will only run if ENABLE_VISUALIZATION is True
        out = None; writer_initialized = False; frames_written = 0
        try:
            # Attempt to initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.target_fps, self.frame_shape_viz)

            writer_initialized = True; print(f"Writer initialized for: {output_path}")

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
                    if frame.shape[1]!=self.frame_shape_viz[0] or frame.shape[0]!=self.frame_shape_viz[1]:
                        frame = cv2.resize(frame, self.frame_shape_viz)
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    frames_written += 1

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
                 out.release()
                 debug_print(f"Writer released. Wrote {frames_written} frames to {output_path}")
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

            # img_rgb_model_sized = enhance_frame_for_detection(img_rgb_model_sized, 'medium')
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
    def _process_inference_results(self, 
                                 yolo_results_batch,  # Results from model(batch_tensor)
                                 frames_batch_with_metadata, 
                                 frame_numbers_in_batch, 
                                 frame_times_in_batch, 
                                 start_datetime_obj_global):

        processed_frames_list_viz = [None] * len(yolo_results_batch)
        
        # --- Part 1: Sequential Tracker Update for the entire batch ---
        if self.perf: self.perf.start_timer('tracking_update_batch')
        
        online_targets_for_each_frame_in_batch = [None] * len(yolo_results_batch)

        # This lock ensures that even if ByteTrack itself isn't fully thread-safe
        # for all its internal list manipulations, these sequential calls protect it.
        # The main concern was STrack.next_id() and shared Kalman.
        with self.tracker_lock: # Use the existing tracker_lock or a new one specific to bytetracker
            for i in range(len(yolo_results_batch)):
                yolo_result_obj_single_frame = yolo_results_batch[i]
                
                detections_for_bytetrack_single_frame = []
                if hasattr(yolo_result_obj_single_frame, 'boxes') and \
                   yolo_result_obj_single_frame.boxes is not None and \
                   len(yolo_result_obj_single_frame.boxes.data) > 0:
                    for box_data_tensor in yolo_result_obj_single_frame.boxes.data:
                        box_cpu = box_data_tensor.cpu().numpy()
                        x1,y1,x2,y2,conf,cls_id = box_cpu[0],box_cpu[1],box_cpu[2],box_cpu[3],box_cpu[4],box_cpu[5]
                        detections_for_bytetrack_single_frame.append([x1, y1, x2, y2, conf, int(cls_id)])
                
                detections_np_single_frame = np.array(detections_for_bytetrack_single_frame) \
                                           if detections_for_bytetrack_single_frame \
                                           else np.empty((0, 6))

                img_h_model, img_w_model = yolo_result_obj_single_frame.orig_shape
                dummy_img_for_tracker = np.zeros((img_h_model, img_w_model, 3), dtype=np.uint8)

                # Call ByteTrack update sequentially for each frame in the batch
                
                online_targets_for_each_frame_in_batch[i] = self.bytetracker.update(
                    detections_np_single_frame,
                    dummy_img_for_tracker 
                )
        
        if self.perf: self.perf.end_timer('tracking_update_batch')

        # --- Part 2: Parallel Application Logic for each frame's tracking results ---
        
        # This inner function now processes results for a single frame, 
        # using the already computed 'online_targets' from ByteTrack.
        # It still needs to manage self.active_track_data and self.finalized_paths_for_report,
        # so locks are needed there.
        def process_tracked_frame_logic(idx_in_batch):
            online_targets_current_frame = online_targets_for_each_frame_in_batch[idx_in_batch]
            yolo_result_obj = yolo_results_batch[idx_in_batch] # For class names
            original_frame_content_for_viz_logic = frames_batch_with_metadata[idx_in_batch][0] # For drawing

            current_video_frame_num = frame_numbers_in_batch[idx_in_batch]
            current_frame_time_sec = frame_times_in_batch[idx_in_batch]
            current_global_timestamp = start_datetime_obj_global + timedelta(seconds=current_frame_time_sec)

            output_frame_for_viz = None
            if ENABLE_VISUALIZATION:
                output_frame_for_viz = cv2.resize(original_frame_content_for_viz_logic.copy(), self.frame_shape)

            # frame_local_event_log_entries = [] # If you still want detailed per-frame event logs

            current_frame_active_track_ids_bytetrack = set() # IDs present in ByteTrack's output for *this* frame
            temp_drawing_info_map = {}

            if online_targets_current_frame is not None: # Check if tracking produced any output
                for target_data in online_targets_current_frame:
                    tx1, ty1, tx2, ty2 = target_data[0], target_data[1], target_data[2], target_data[3]
                    track_id = str(int(target_data[4]))
                    track_cls_id = int(target_data[6])
                    current_frame_active_track_ids_bytetrack.add(track_id)
                    vehicle_type_name = self.class_id_to_name_map.get(track_cls_id, 'Light Vehicle')

                    model_h_orig, model_w_orig = yolo_result_obj.orig_shape
                    draw_h_target, draw_w_target = self.frame_shape[1], self.frame_shape[0]
                    draw_x1 = int(tx1 * draw_w_target / model_w_orig); draw_y1 = int(ty1 * draw_h_target / model_h_orig)
                    draw_x2 = int(tx2 * draw_w_target / model_w_orig); draw_y2 = int(ty2 * draw_h_target / model_h_orig)
                    bbox_for_drawing_and_zone = (draw_x1, draw_y1, draw_x2, draw_y2)
                    center_for_drawing = ((draw_x1 + draw_x2) // 2, (draw_y1 + draw_y2) // 2)
                    bottom_center_for_zone_check = ((draw_x1 + draw_x2) // 2, draw_y2)

                    temp_drawing_info_map[track_id] = {
                        'center': center_for_drawing, 'box': bbox_for_drawing_and_zone,
                        'type': vehicle_type_name, 'event': None, 'status': 'tracked'
                    }

                    # --- Zone Transition & App Logic (uses self.active_track_data) ---
                    # This section needs the tracker_lock because self.active_track_data
                    # and self.finalized_paths_for_report are shared across threads.
                    with self.tracker_lock:
                        if track_id not in self.active_track_data:
                            entry_event_dir_candidate = None
                            if self.zone_tracker and self.zone_tracker.zones:
                                # No perf timer here, it's now part of app_logic_batch
                                for zone_dir_name, zone_poly_pts in self.zone_tracker.zones.items():
                                    if self.zone_tracker._check_point_in_zone(bottom_center_for_zone_check, zone_poly_pts):
                                        entry_event_dir_candidate = zone_dir_name; break
                            if entry_event_dir_candidate:
                                self.active_track_data[track_id] = {
                                    'id': track_id, 'entry_direction': entry_event_dir_candidate,
                                    'entry_time': current_global_timestamp, 'type_history': {vehicle_type_name: 1},
                                    'current_type': vehicle_type_name, 'last_pos_zone': bottom_center_for_zone_check,
                                    'last_seen_frame': current_video_frame_num, 'status': 'active',
                                    'entry_point': bottom_center_for_zone_check
                                }
                                if self.perf: self.perf.record_vehicle_entry()
                                temp_drawing_info_map[track_id]['event'] = 'ENTRY'
                        else: # Existing track in self.active_track_data
                            track_app_data = self.active_track_data[track_id]
                            track_app_data['last_seen_frame'] = current_video_frame_num
                            track_app_data['type_history'][vehicle_type_name] = track_app_data['type_history'].get(vehicle_type_name, 0) + 1
                            track_app_data['current_type'] = max(track_app_data['type_history'], key=track_app_data['type_history'].get)
                            temp_drawing_info_map[track_id]['type'] = track_app_data['current_type']
                            prev_pos_for_zone_check = track_app_data.get('last_pos_zone')
                            track_app_data['last_pos_zone'] = bottom_center_for_zone_check
                            if prev_pos_for_zone_check and self.zone_tracker and self.zone_tracker.zones and track_app_data['status'] == 'active':
                                event_type_zone, event_dir_zone = self.zone_tracker.check_zone_transition(
                                    prev_pos_for_zone_check, bbox_for_drawing_and_zone, track_id, current_frame_time_sec
                                )
                                if event_type_zone == "ENTRY" and track_app_data.get('entry_direction') and track_app_data['entry_direction'] != event_dir_zone:
                                    stored_entry_dir = track_app_data['entry_direction']
                                    active_zone_keys = list(self.zone_tracker.zones.keys())
                                    if is_valid_movement(stored_entry_dir, event_dir_zone, active_zone_keys):
                                        time_in_int_leg1 = (current_global_timestamp - track_app_data['entry_time']).total_seconds()
                                        self._add_to_finalized_paths(track_id, track_app_data, 'exited', event_dir_zone, current_global_timestamp, time_in_int_leg1)
                                        if self.perf: self.perf.record_vehicle_exit('exited', time_in_int_leg1)
                                        track_app_data['entry_direction'] = event_dir_zone
                                        track_app_data['entry_time'] = current_global_timestamp
                                        track_app_data['entry_point'] = bottom_center_for_zone_check
                                        if self.perf: self.perf.record_vehicle_entry()
                                        temp_drawing_info_map[track_id]['event'] = 'RE-ENTRY'
                                elif event_type_zone == "EXIT":
                                    entry_dir_trk = track_app_data.get('entry_direction')
                                    entry_time_stored = track_app_data.get('entry_time')
                                    active_zone_keys = list(self.zone_tracker.zones.keys())
                                    if entry_dir_trk and is_valid_movement(entry_dir_trk, event_dir_zone, active_zone_keys) and \
                                       entry_time_stored and current_global_timestamp >= entry_time_stored:
                                        time_in_int_exit = (current_global_timestamp - entry_time_stored).total_seconds()
                                        self._add_to_finalized_paths(track_id, track_app_data, 'exited', event_dir_zone, current_global_timestamp, time_in_int_exit)
                                        if self.perf: self.perf.record_vehicle_exit('exited', time_in_int_exit)
                                        track_app_data['status'] = 'exited'
                                        track_app_data['exit_time'] = current_global_timestamp 
                                        track_app_data['exit_direction'] = event_dir_zone 
                                        track_app_data['exit_point'] = bottom_center_for_zone_check 
                                        temp_drawing_info_map[track_id]['event'] = 'EXIT'
            
            # --- Handle Lost / Timed Out Tracks (application level) ---
            # This also needs to be locked as it reads/modifies self.active_track_data
            # and appends to self.finalized_paths_for_report
            ids_to_remove_from_active_data = []
            with self.tracker_lock:
                for track_id_app, track_app_data_item in list(self.active_track_data.items()):
                    if track_app_data_item['status'] != 'active': continue # Already finalized

                    # Check application-level timeout
                    time_in_intersection_so_far = (current_global_timestamp - track_app_data_item['entry_time']).total_seconds()
                    vehicle_type_for_timeout = track_app_data_item['current_type']
                    timeout_limit_seconds = VEHICLE_TIMEOUTS.get(vehicle_type_for_timeout, VEHICLE_TIMEOUTS['default'])
                    if time_in_intersection_so_far > timeout_limit_seconds:
                        self._add_to_finalized_paths(track_id_app, track_app_data_item, 'timed_out', 'TIMEOUT', current_global_timestamp, time_in_intersection_so_far)
                        if self.perf: self.perf.record_vehicle_exit('timed_out')
                        if track_id_app in temp_drawing_info_map: temp_drawing_info_map[track_id_app]['status'] = 'timed_out'
                        else: temp_drawing_info_map[track_id_app] = {'status': 'timed_out', 'type': track_app_data_item['current_type'], 'box': None, 'center': None}
                        track_app_data_item['status'] = 'timed_out'
                        ids_to_remove_from_active_data.append(track_id_app)
                        continue 

                # Check for tracks no longer reported by ByteTrack for this frame
                # (and potentially for a grace period beyond ByteTrack's own buffer)
                grace_period_frames_after_bytetrack = self.bytetracker.max_time_lost + int(self.target_fps * 0.5)
                for track_id_app, track_app_data_item in list(self.active_track_data.items()): # Iterate again to catch those not timed out
                    if track_app_data_item['status'] == 'active' and track_id_app not in current_frame_active_track_ids_bytetrack:
                        frames_since_last_seen_by_app = current_video_frame_num - track_app_data_item.get('last_seen_frame', current_video_frame_num)
                        if frames_since_last_seen_by_app > grace_period_frames_after_bytetrack:
                            time_in_int_at_loss = (current_global_timestamp - track_app_data_item['entry_time']).total_seconds()
                            self._add_to_finalized_paths(track_id_app, track_app_data_item, 'lost_track', 'LOST_TRACK', current_global_timestamp, time_in_int_at_loss)
                            if self.perf: self.perf.record_vehicle_exit('forced_exit') # Or 'lost' for perf
                            track_app_data_item['status'] = 'lost_track'
                            ids_to_remove_from_active_data.append(track_id_app)
                
                for v_id_remove in set(ids_to_remove_from_active_data):
                    if v_id_remove in self.active_track_data: del self.active_track_data[v_id_remove]
                    if self.zone_tracker: self.zone_tracker.remove_vehicle_data(v_id_remove)

            # --- Drawing Pass (if visualization enabled) ---
            if ENABLE_VISUALIZATION and output_frame_for_viz is not None:
                # No perf timer here, part of app_logic_batch
                if self.zone_tracker and self.zone_tracker.zones:
                    output_frame_for_viz = visual_overlay.draw_zones(output_frame_for_viz, self.zone_tracker)
                for track_id_draw, data_to_draw in temp_drawing_info_map.items():
                    draw_box_coords = data_to_draw.get('box'); draw_type = data_to_draw.get('type', "Unknown")
                    draw_status = data_to_draw.get('status', 'tracked'); draw_event = data_to_draw.get('event')
                    draw_center = data_to_draw.get('center')
                    entry_dir_for_viz = None; time_active_for_viz = None
                    # Read from self.active_track_data requires lock if it could be modified by another thread concurrently
                    # However, at this stage of drawing, self.active_track_data for *this specific track_id* 
                    # should have been finalized for this frame by the logic above within the lock.
                    # A read here might be safe IF the track_id being drawn is guaranteed to not be concurrently modified.
                    # To be absolutely safe, one might re-acquire lock or pass all necessary info through temp_drawing_info_map.
                    # For now, assuming reads are okay if major state changes for this frame are done.
                    if track_id_draw in self.active_track_data: 
                        with self.tracker_lock: # Add lock for safety if reading from shared active_track_data
                            app_data_viz = self.active_track_data.get(track_id_draw) # Use .get for safety
                            if app_data_viz: # Check if still exists
                                if app_data_viz['status'] == 'active': draw_status = 'active'
                                entry_dir_for_viz = app_data_viz.get('entry_direction')
                                time_active_for_viz = (current_global_timestamp - app_data_viz['entry_time']).total_seconds()
                    
                    if draw_box_coords and draw_status not in ['timed_out', 'lost_track']:
                         visual_overlay.draw_detection_box(output_frame_for_viz, draw_box_coords, track_id_draw, draw_type,
                                                           status=draw_status, entry_dir=entry_dir_for_viz, time_active=time_active_for_viz)
                    if draw_event and draw_center and draw_event != 'RE-ENTRY':
                         visual_overlay.draw_event_marker(output_frame_for_viz, draw_center, draw_event, track_id_draw[-4:])
                
                active_count_viz = 0
                with self.tracker_lock: # Lock for reading count from shared dict
                    active_count_viz = sum(1 for ad in self.active_track_data.values() if ad['status'] == 'active')
                tracker_info_for_overlay = {'get_active_vehicle_count': lambda: active_count_viz}
                output_frame_for_viz = visual_overlay.add_status_overlay(output_frame_for_viz, current_video_frame_num, current_global_timestamp, tracker_info_for_overlay)
            
            return output_frame_for_viz # Only return viz frame now
            # --- End process_tracked_frame_logic inner function ---

        if self.perf: self.perf.start_timer('app_logic_batch') # New timer for the parallel part
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures_map = {
                executor.submit(process_tracked_frame_logic, i): i
                for i in range(len(yolo_results_batch)) 
            }
            for future in concurrent.futures.as_completed(futures_map):
                original_batch_idx = futures_map[future]
                try:
                    processed_viz_frame = future.result() 
                    processed_frames_list_viz[original_batch_idx] = processed_viz_frame
                except Exception as exc_future:
                    print(f'\nError processing tracked frame logic for batch index {original_batch_idx}: {exc_future}')
                    import traceback; traceback.print_exc()
                    processed_frames_list_viz[original_batch_idx] = None # Ensure placeholder
        if self.perf: self.perf.end_timer('app_logic_batch')
        
        # The finalized paths are now built directly into self.finalized_paths_for_report
        # So, this function doesn't need to return event logs for that purpose.
        return [], processed_frames_list_viz 
    # --- End _process_inference_results ---

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
        }
        self.finalized_paths_for_report.append(path_report_entry)

    def run_processing_on_all_videos(self, start_date_str, start_time_str, primary_direction):
        if not self.uploaded_video_paths:
            return "No videos uploaded."

        if not self.zone_polygons:
            return "Zones not defined."

        status_messages = []
        for idx, video_path in enumerate(self.uploaded_video_paths):
            msg = f"Processing video {idx+1}/{len(self.uploaded_video_paths)}: {os.path.basename(video_path)}"
            print(msg)
            status_messages.append(msg)

            # Call your frame reading and model processing pipeline here
            try:
                self.process_video(video_path, start_date_str, start_time_str, primary_direction)
            except Exception as e:
                error_msg = f"Error processing {video_path}: {e}"
                print(error_msg)
                status_messages.append(error_msg)

        return "\n".join(status_messages)
    

    def _execute_single_model_inference(self, model_instance, input_tensor_chunk, stream_to_use=None):
        # It should take a chunk of frames (e.g., self.batch_size frames) and return a list of results.
        # Remember the stream_to_use.synchronize() inside it if stream_to_use is not None.
        if stream_to_use and self.device == 'cuda':
            with torch.cuda.stream(stream_to_use):
                with torch.amp.autocast(device_type=self.device, enabled=(MIXED_PRECISION and self.device == 'cuda')):
                    results = model_instance(input_tensor_chunk, verbose=False, imgsz=MODEL_INPUT_SIZE, augment=TTA)
            stream_to_use.synchronize() # Crucial for ThreadPoolExecutor
            return results
        else: 
            with torch.amp.autocast(device_type=self.device, enabled=(MIXED_PRECISION and self.device == 'cuda')):
                results = model_instance(input_tensor_chunk, verbose=False, imgsz=MODEL_INPUT_SIZE)
            return results

    def _format_duration(self, seconds):
        """Format duration in seconds to human-readable string"""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:  # Less than 1 hour
            minutes = int(seconds // 60)
            remaining_seconds = int(seconds % 60)
            if remaining_seconds > 0:
                return f"{minutes} minutes {remaining_seconds} seconds"
            else:
                return f"{minutes} minutes"
        else:  # 1 hour or more
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            remaining_seconds = int(seconds % 60)
            
            duration_parts = [f"{hours} hour{'s' if hours != 1 else ''}"]
            if remaining_minutes > 0:
                duration_parts.append(f"{remaining_minutes} minute{'s' if remaining_minutes != 1 else ''}")
            if remaining_seconds > 0:
                duration_parts.append(f"{remaining_seconds} second{'s' if remaining_seconds != 1 else ''}")
            
            return " ".join(duration_parts)
    
    # --- Main Processing Function ---
    def process_video(self, video_paths, start_date_str, start_time_str, primary_direction, generate_link_count_report_flag=False, progress_queue=None, preview_image_width=640, preview_image_height=480):
        if not self.processing_lock.acquire(blocking=False):
            print("Warning: Processing lock already acquired."); return "Processing is already in progress."

        self.bytetracker = ByteTrack(
            min_conf=BYTETRACK_MIN_CONF, track_thresh=BYTETRACK_TRACK_THRESH, match_thresh=BYTETRACK_MATCH_THRESH,
            track_buffer=BYTETRACK_TRACK_BUFFER_SECONDS, frame_rate=self.target_fps, per_class=False
        )
        self.current_bytetrack_frame_id = 0
        self.active_track_data.clear()
        self.finalized_paths_for_report = [] # Reset for new video
        effective_trigger_batch_size = self.batch_size * self.num_inference_streams
        print(f"Effective trigger batch size for inference: {effective_trigger_batch_size} (batch_size={self.batch_size} * num_streams={self.num_inference_streams})")

        self.zone_tracker = None
        self.perf = PerformanceTracker(enabled=ENABLE_DETAILED_PERFORMANCE_METRICS)
        self.stop_event = threading.Event()
        self.frame_read_queue = queue.Queue(maxsize=self.max_queue_size)
        self.last_cleanup_time = time.monotonic()
        self.detection_zones_polygons = None
        self.preview_image_width = preview_image_width
        self.preview_image_height = preview_image_height
        self.final_output_path = None
        self.writer_thread = None
        self.video_write_queue = None
        self.models_list = [] # Clear from previous runs

        print(f"--- Starting New Video Processing Run (with ByteTrack for Report Parity) ---")

        actual_video_path_to_process = [] 
        temp_converted_dav_mp4 = []
        for video_path in video_paths:
            if video_path.lower().endswith(".dav"):
                print(f"Input is .dav: {video_path}. Converting...")
                converted_path = convert_dav_to_mp4_internal(video_path)
                if converted_path and os.path.exists(converted_path):
                    actual_video_path_to_process.append(converted_path)
                    temp_converted_dav_mp4.append(converted_path)
                    print(f".dav converted to: {actual_video_path_to_process}")
                    time.sleep(0.5) 
                else:
                    error_msg = f"Failed to convert .dav file: {video_path}"
                    print(error_msg); self.processing_lock.release(); raise RuntimeError(error_msg)
            else:
                actual_video_path_to_process.append(video_path)

        try:
            # --- ADD A STARTING MESSAGE ---
            if progress_queue:
                progress_queue.put_nowait({"progress": 2, "status": "Initializing processor..."})

            # --- Zone Definition ---
            self.detection_zones_polygons = None
            if self.line_mode == 'hardcoded':
                if 'LINE_POINTS' not in globals() or not isinstance(LINE_POINTS, dict) or not LINE_POINTS: raise ValueError("Hardcoded: LINE_POINTS missing/empty.")
                valid_polygons_hc = {dir: pts for dir, pts in LINE_POINTS.items() if isinstance(pts, list) and len(pts) >= 3}
                if not valid_polygons_hc: raise ValueError("Hardcoded: No valid polygons in LINE_POINTS.")
                self.detection_zones_polygons = {dir: np.array(pts, dtype=np.int32) for dir, pts in valid_polygons_hc.items()}
            elif self.line_mode == 'interactive':
                if not self.zone_polygons:
                    raise ValueError("Interactive: No zone polygons defined.")
                valid_polygons = {dir: pts for dir, pts in self.zone_polygons.items() if pts and len(pts) >= 3}
                if len(valid_polygons) < 2:
                    raise ValueError(f"Interactive: Need >= 2 valid zone. Found {len(valid_polygons)}.")
                
                # Get preview image dimensions from the frontend
                preview_width = getattr(self, 'preview_image_width', 640)
                preview_height = getattr(self, 'preview_image_height', 480)
                
                print(f"Scaling polygons from preview ({preview_width}x{preview_height}) to processing ({self.frame_shape[0]}x{self.frame_shape[1]})")
                
                # Scale polygons from preview coordinates to processing coordinates
                scaled_polygons = self._scale_polygons_to_processing_coordinates(
                    valid_polygons, preview_width, preview_height
                )
                self.detection_zones_polygons = {dir: np.array(pts, dtype=np.int32) for dir, pts in scaled_polygons.items()}
            else: raise ValueError(f"Invalid LINE_MODE '{self.line_mode}'.")
            if not self.detection_zones_polygons: raise ValueError("Zone polygon data missing for ZoneTracker.")
            self.zone_tracker = ZoneTracker(self.detection_zones_polygons)
            if not self.zone_tracker or not self.zone_tracker.zones: raise RuntimeError("CRITICAL: ZoneTracker failed to initialize.")

            # --- Load Model(s) ---
            print(f"Loading {self.num_inference_streams} model instances...")
            for i in range(self.num_inference_streams):
                try:
                    model_instance = load_model(MODEL_PATH, self.device, CONF_THRESHOLD, IOU_THRESHOLD)
                    if model_instance is None: raise RuntimeError(f"Failed to load model instance {i+1} from {MODEL_PATH}")
                    self.models_list.append(model_instance)
                    print(f"Loaded model instance {i+1}/{self.num_inference_streams}.")
                except Exception as e:
                    error_msg = f"FATAL: Could not load model instance {i+1}: {e}"
                    print(error_msg); self.processing_lock.release(); raise RuntimeError(error_msg)
            if not self.models_list:
                error_msg = "FATAL: No model instances were loaded."
                print(error_msg); self.processing_lock.release(); raise RuntimeError(error_msg)
            print(f"Successfully loaded {len(self.models_list)} model instances.")
            self.model = self.models_list[0] # For any legacy single-model references

            try:
                start_date_dt=datetime.strptime(start_date_str, "%y%m%d"); time_part,ms_part=(start_time_str[:6], start_time_str[6:]) if len(start_time_str)>6 else (start_time_str,"0")
                base_time_dt=datetime.strptime(time_part,"%H%M%S").time(); microseconds_val=int(ms_part.ljust(3,'0'))*1000
                start_time_dt=base_time_dt.replace(microsecond=microseconds_val); start_datetime_obj_global=datetime.combine(start_date_dt.date(), start_time_dt) # Renamed for clarity
                print(f"Video Start Timestamp: {start_datetime_obj_global.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            except ValueError as e_ts: raise ValueError(f"Invalid date/time format: {e_ts}.")

            # --- Video Properties & I/O Threads ---
            total_frames, original_fps = self._get_video_properties(actual_video_path_to_process)
            print(f"Video Properties: ~{total_frames} frames, {original_fps:.2f} FPS (Original)")
            if total_frames <= 0: raise ValueError("Cannot process video with zero or unreadable total frames.")

            video_duration_seconds = total_frames / original_fps
            video_duration_formatted = self._format_duration(video_duration_seconds)
            print(f"Video Duration: {video_duration_formatted} ({video_duration_seconds:.1f} seconds)")

            if progress_queue:
                progress_queue.put_nowait({"progress": 5, "status": "Starting I/O threads..."})

            print("Starting I/O threads...")
            reader_args = (actual_video_path_to_process, original_fps, total_frames) if self.actually_use_advanced_reader else (actual_video_path_to_process, original_fps)
            reader_target = self._frame_reader_task_pvr_segmented if self.actually_use_advanced_reader else self._frame_reader_task
            self.reader_thread = threading.Thread(target=reader_target, args=reader_args, daemon=True); self.reader_thread.start()
            
            if ENABLE_VISUALIZATION:
                os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True); ts_now=datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename_base_viz=f"output_{os.path.splitext(os.path.basename(actual_video_path_to_process[0]))[0]}_{ts_now}"
                output_path_viz=os.path.join(VIDEO_OUTPUT_DIR, f"{output_filename_base_viz}.mp4")
                self.final_output_path = output_path_viz # Store for result message
                self.video_write_queue = queue.Queue(maxsize=self.max_queue_size)
                self.writer_thread = threading.Thread(target=self._frame_writer_task, args=(output_path_viz,), daemon=True); self.writer_thread.start()
            else: self.video_write_queue = None; self.writer_thread = None; self.final_output_path = None # Explicitly None
            
            # --- Main Loop Logic ---
            frames_processed_count = 0
            frame_interval = max(1, round(original_fps / self.target_fps)) if self.target_fps > 0 and original_fps > 0 else 1
            total_target_frames_for_perf = total_frames // frame_interval if total_frames > 0 else 0
            if self.perf: self.perf.start_processing(total_target_frames_for_perf if total_target_frames_for_perf > 0 else total_frames)
            
            current_batch_frames_metadata, current_batch_frame_numbers, current_batch_frame_times = [], [], []
            last_prog_print_time = time.monotonic()
            
            effective_trigger_batch_size = self.batch_size * self.num_inference_streams
            print(f"Effective trigger batch size for inference: {effective_trigger_batch_size} (batch_size={self.batch_size} * num_streams={self.num_inference_streams})")

            # ==============================================================
            # START OF: Main Processing `while True` Loop
            # ==============================================================
            while True:
                try:
                    frame_data_item = self.frame_read_queue.get(timeout=60) # Wait up to 60s for a frame
                    if frame_data_item is None: # Sentinel from reader thread
                        if not self.frame_read_queue.empty(): self.frame_read_queue.task_done()
                        debug_print("Main loop: Reader sent None, exiting loop.")
                        break 
                    
                    if self.perf: self.perf.start_timer('video_read')
                    frame_content_raw, frame_num_raw, frame_time_s, frame_meta = frame_data_item
                    if self.perf: self.perf.end_timer('video_read')

                    if frame_content_raw is None: # Should not happen if reader sends sentinel correctly
                        if not self.frame_read_queue.empty(): self.frame_read_queue.task_done()
                        continue

                    current_batch_frames_metadata.append((frame_content_raw, frame_meta))
                    current_batch_frame_numbers.append(frame_num_raw)
                    current_batch_frame_times.append(frame_time_s)

                    # Check if we have accumulated enough frames for the effective batch
                    if len(current_batch_frames_metadata) >= effective_trigger_batch_size:
                        batch_proc_start_time = time.monotonic()

                        if self.perf: self.perf.start_timer('preprocessing')
                        input_batch_tensor_model = self._preprocess_batch(current_batch_frames_metadata)
                        if self.perf: self.perf.end_timer('preprocessing')

                        if input_batch_tensor_model is None or input_batch_tensor_model.nelement() == 0:
                            debug_print("Skipping batch: Preprocessing returned None or empty tensor.")
                            # Mark all frames from this failed attempt as done in the queue
                            for _i_skip in range(len(current_batch_frames_metadata)):
                                if not self.frame_read_queue.empty():
                                    try: self.frame_read_queue.task_done()
                                    except ValueError: pass # If already marked done due to earlier None
                            current_batch_frames_metadata.clear(); current_batch_frame_numbers.clear(); current_batch_frame_times.clear()
                            continue
                        
                        actual_frames_in_current_effective_batch = input_batch_tensor_model.shape[0]
                        yolo_results_batch_ordered = [None] * actual_frames_in_current_effective_batch
                        
                        if self.perf and self.perf.cuda_events_enabled and self.device == 'cuda':
                            overall_start_event = torch.cuda.Event(enable_timing=True)
                            overall_end_event = torch.cuda.Event(enable_timing=True)
                            overall_start_event.record()

                        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_inference_streams) as executor:
                            futures_to_original_indices_map = {} 

                            for i in range(self.num_inference_streams):
                                start_idx_for_model = i * self.batch_size
                                end_idx_for_model = (i + 1) * self.batch_size

                                if start_idx_for_model >= actual_frames_in_current_effective_batch: continue
                                end_idx_for_model = min(end_idx_for_model, actual_frames_in_current_effective_batch)
                                if start_idx_for_model >= end_idx_for_model: continue

                                chunk_for_model = input_batch_tensor_model[start_idx_for_model:end_idx_for_model]
                                original_global_indices_for_chunk = list(range(start_idx_for_model, end_idx_for_model))
                                
                                model_to_use = self.models_list[i]
                                current_stream_for_model = None
                                if self.device == 'cuda' and self.streams:
                                     current_stream_for_model = self.streams[i % len(self.streams)]

                                future = executor.submit(self._execute_single_model_inference,
                                                         model_to_use, chunk_for_model, current_stream_for_model)
                                futures_to_original_indices_map[future] = original_global_indices_for_chunk
                            
                            for future in concurrent.futures.as_completed(futures_to_original_indices_map):
                                original_global_indices = futures_to_original_indices_map[future]
                                try:
                                    results_from_single_model_call = future.result() 
                                    if results_from_single_model_call:
                                        for res_idx_in_chunk, yolo_res_obj in enumerate(results_from_single_model_call):
                                            global_idx = original_global_indices[res_idx_in_chunk]
                                            yolo_results_batch_ordered[global_idx] = yolo_res_obj
                                except Exception as e_infer:
                                    print(f"\nError during concurrent inference (global_indices {original_global_indices}): {e_infer}")
                                    for global_idx in original_global_indices:
                                        yolo_results_batch_ordered[global_idx] = None
                        
                        if self.perf and self.perf.cuda_events_enabled and self.device == 'cuda':
                            overall_end_event.record()
                            overall_end_event.synchronize()
                            duration_ms = overall_start_event.elapsed_time(overall_end_event)
                            self.perf.timings['inference'] += duration_ms / 1000.0

                        yolo_results_batch = yolo_results_batch_ordered if any(yolo_results_batch_ordered) else []
                        
                        # Application logic processing (tracking, zone checking, etc.)
                        _, processed_frames_for_viz = self._process_inference_results(
                            yolo_results_batch, 
                            current_batch_frames_metadata, 
                            current_batch_frame_numbers, 
                            current_batch_frame_times, 
                            start_datetime_obj_global
                        )
                        
                        # Send frames to writer if visualization is enabled
                        if ENABLE_VISUALIZATION and self.video_write_queue:
                            if self.perf: self.perf.start_timer('video_write')
                            for pf_viz in processed_frames_for_viz:
                                if pf_viz is not None:
                                    try: self.video_write_queue.put(pf_viz, block=True, timeout=2.0)
                                    except queue.Full: print("WARN: Video writer queue full during main loop.")
                            if self.perf: self.perf.end_timer('video_write')

                        # Performance recording for this effective batch
                        batch_proc_duration = time.monotonic() - batch_proc_start_time
                        # actual_frames_in_current_effective_batch is the count for this effective batch
                        frames_processed_count += actual_frames_in_current_effective_batch
                        if self.perf: 
                            self.perf.record_batch_processed(actual_frames_in_current_effective_batch, batch_proc_duration)
                            self.perf.sample_system_metrics() # Sample system metrics after each big batch
                        
                        # Record detections for all frames in the effective batch
                        if self.perf: 
                            num_detections_in_batch = 0
                            for res_obj_list_item in yolo_results_batch:
                                if res_obj_list_item: 
                                     if hasattr(res_obj_list_item, 'boxes') and res_obj_list_item.boxes is not None:
                                        num_detections_in_batch += len(res_obj_list_item.boxes)
                            self.perf.record_detection(num_detections_in_batch)

                        # Memory cleanup check
                        current_monotonic_time_loop = time.monotonic()
                        if current_monotonic_time_loop - self.last_cleanup_time > MEMORY_CLEANUP_INTERVAL:
                            if self.perf: self.perf.start_timer('memory_cleanup')
                            last_frame_time_sec_in_batch_loop = current_batch_frame_times[-1] if current_batch_frame_times else 0
                            cleanup_tracking_data(self, last_frame_time_sec_in_batch_loop) # Pass self
                            cleanup_memory() # Global util
                            self.last_cleanup_time = current_monotonic_time_loop
                            if self.perf: self.perf.end_timer('memory_cleanup')

                        # Progress printing
                        if self.perf and (current_monotonic_time_loop - last_prog_print_time > 1.0): # Print progress every second
                            prog_stats=self.perf.get_progress()
                            if progress_queue:
                                progress_queue.put_nowait({
                                    "progress": round(prog_stats.get('percent', 0)),
                                    "status": f"{prog_stats['fps']:.1f} FPS",
                                    "eta": str(prog_stats.get('eta', '...'))
                                })
                            print(f"\rProgress:{prog_stats['percent']:.1f}%|FPS:{prog_stats['fps']:.1f}|ETA:{prog_stats['eta']}      ", end="")
                            last_prog_print_time = current_monotonic_time_loop
                        
                        # Mark all frames from this effective batch as done in the read queue
                        for _i_done in range(len(current_batch_frames_metadata)): # Iterate for actual number of items taken
                            if not self.frame_read_queue.empty():
                                try: self.frame_read_queue.task_done()
                                except ValueError: pass 
                        
                        current_batch_frames_metadata.clear(); current_batch_frame_numbers.clear(); current_batch_frame_times.clear()
                        del input_batch_tensor_model, yolo_results_batch, yolo_results_batch_ordered; # Cleanup tensors
                        # Don't call cleanup_memory() here too frequently, rely on the interval check

                    else: # Not enough frames for a full effective_trigger_batch_size yet
                        # We still need to mark the single frame_data_item as processed by the queue
                        if not self.frame_read_queue.empty():
                             try: self.frame_read_queue.task_done()
                             except ValueError: pass

                except queue.Empty: # This timeout (60s in get()) means the reader might be finished or stuck
                    # More robust queue timeout handling
                    if self.stop_event.is_set():
                        debug_print("Main loop: Stop event set during queue timeout, breaking.")
                        break
                    
                    # Check if reader thread is still alive
                    if hasattr(self, 'reader_thread') and self.reader_thread and not self.reader_thread.is_alive():
                        debug_print("Main loop: Reader thread died, processing any remaining frames...")
                        # Try to process any remaining frames in the current batch
                        if current_batch_frames_metadata:
                            debug_print(f"Main loop: Processing final {len(current_batch_frames_metadata)} frames before exit...")
                            continue  # This will trigger the batch processing logic
                        debug_print("Main loop: No remaining frames, breaking.")
                        break
                    
                    # Add a small yield to prevent CPU spinning
                    time.sleep(0.01)  # 10ms yield
                    debug_print("Main loop: Queue empty, continuing to wait...")
                    continue
                except Exception as e_loop:
                    print(f"\nERROR during processing loop: {e_loop}")
                    import traceback; traceback.print_exc()
                    self.stop_event.set() # Signal all threads to stop
                    break # Exit loop on critical error
            # ==============================================================
            # END OF: Main Processing `while True` Loop
            # ==============================================================
            print("\nVideo processing loop finished.")

            # ==============================================================
            # START OF: Final Batch Processing
            # ==============================================================
            if current_batch_frames_metadata: # Process any remaining frames
                final_batch_size_actual = len(current_batch_frames_metadata)
                print(f"\nProcessing final batch of {final_batch_size_actual} frames with multiple model instances...")
                batch_proc_start_time_final = time.monotonic()

                if self.perf: self.perf.start_timer('preprocessing')
                input_batch_final = self._preprocess_batch(current_batch_frames_metadata)
                if self.perf: self.perf.end_timer('preprocessing')

                if input_batch_final is not None and input_batch_final.nelement() > 0:
                    actual_frames_in_final_input_batch = input_batch_final.shape[0] # Should match final_batch_size_actual
                    yolo_results_final_ordered = [None] * actual_frames_in_final_input_batch
                    
                    if self.perf and self.perf.cuda_events_enabled and self.device == 'cuda':
                        final_batch_start_event = torch.cuda.Event(enable_timing=True)
                        final_batch_end_event = torch.cuda.Event(enable_timing=True)
                        final_batch_start_event.record()

                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_inference_streams) as executor:
                        futures_to_original_indices_final_map = {}
                        
                        idx_offset_final = 0 # To track how many frames from input_batch_final have been assigned
                        for i in range(self.num_inference_streams):
                            if idx_offset_final >= actual_frames_in_final_input_batch: break

                            num_frames_for_this_stream_final = min(self.batch_size, actual_frames_in_final_input_batch - idx_offset_final)
                            if num_frames_for_this_stream_final <= 0: continue

                            end_idx_for_model_final = idx_offset_final + num_frames_for_this_stream_final
                            chunk_for_model_final = input_batch_final[idx_offset_final:end_idx_for_model_final]
                            original_global_indices_final = list(range(idx_offset_final, end_idx_for_model_final))

                            model_to_use_final = self.models_list[i]
                            current_stream_final_model = None
                            if self.device == 'cuda' and self.streams:
                                 current_stream_final_model = self.streams[i % len(self.streams)]

                            future_final = executor.submit(self._execute_single_model_inference,
                                                           model_to_use_final, chunk_for_model_final, current_stream_final_model)
                            futures_to_original_indices_final_map[future_final] = original_global_indices_final
                            idx_offset_final += num_frames_for_this_stream_final
                        
                        for future_final in concurrent.futures.as_completed(futures_to_original_indices_final_map):
                            original_global_indices_final = futures_to_original_indices_final_map[future_final]
                            try:
                                results_from_single_model_final = future_final.result()
                                if results_from_single_model_final:
                                    for res_idx_in_chunk_final, yolo_res_obj_final in enumerate(results_from_single_model_final):
                                        global_idx_final = original_global_indices_final[res_idx_in_chunk_final]
                                        yolo_results_final_ordered[global_idx_final] = yolo_res_obj_final
                            except Exception as e_infer_final:
                                print(f"\nError during concurrent final batch inference (global_indices {original_global_indices_final}): {e_infer_final}")
                                for global_idx_final in original_global_indices_final:
                                    yolo_results_final_ordered[global_idx_final] = None
                    
                    if self.perf and self.perf.cuda_events_enabled and self.device == 'cuda':
                        final_batch_end_event.record()
                        final_batch_end_event.synchronize()
                        duration_ms_final = final_batch_start_event.elapsed_time(final_batch_end_event)
                        self.perf.timings['inference'] += duration_ms_final / 1000.0

                    yolo_results_final = yolo_results_final_ordered if any(yolo_results_final_ordered) else []
                    
                    # Process final batch results
                    _, processed_frames_final_viz = self._process_inference_results(
                        yolo_results_final, current_batch_frames_metadata, 
                        current_batch_frame_numbers, current_batch_frame_times, start_datetime_obj_global)

                    if ENABLE_VISUALIZATION and self.video_write_queue:
                        if self.perf: self.perf.start_timer('video_write');
                        for pf_final_viz in processed_frames_final_viz:
                            if pf_final_viz is not None:
                                try: self.video_write_queue.put(pf_final_viz, block=True, timeout=2.0)
                                except queue.Full: print("WARN: Vid writer Q full (final batch).")
                        if self.perf: self.perf.end_timer('video_write')
                    
                    # Performance for final batch
                    batch_proc_duration_final = time.monotonic() - batch_proc_start_time_final
                    frames_processed_count += actual_frames_in_final_input_batch # Add frames from final batch
                    if self.perf: 
                        self.perf.record_batch_processed(actual_frames_in_final_input_batch, batch_proc_duration_final)
                        num_detections_final = 0
                        for res_obj_list_f_item in yolo_results_final:
                            if res_obj_list_f_item:
                                 if hasattr(res_obj_list_f_item, 'boxes') and res_obj_list_f_item.boxes is not None:
                                    num_detections_final += len(res_obj_list_f_item.boxes)
                        self.perf.record_detection(num_detections_final)
                    
                    # Mark final batch frames as done in read queue
                    for _i_f_done in range(len(current_batch_frames_metadata)):
                        if not self.frame_read_queue.empty():
                            try: self.frame_read_queue.task_done()
                            except ValueError: pass

                    del input_batch_final, yolo_results_final, yolo_results_final_ordered; cleanup_memory()
            # ==============================================================
            # END OF: Final Batch Processing
            # ==============================================================


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

            print("Signaling threads..."); self.stop_event.set() # Rest of thread cleanup
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
                completed_paths_data=self.finalized_paths_for_report,
                start_datetime=start_datetime_obj_global, 
                primary_direction=primary_direction, 
                video_path=actual_video_path_to_process[0],
                video_duration_formatted=video_duration_formatted
            )

            result_msg_final=f"✅ Processing completed.\n"

            # --- Conditionally create Bi-directional Link Count Report ---
            excel_link_count_report_path = None
            if generate_link_count_report_flag:
                if not self.finalized_paths_for_report:
                    print("Skipping Bi-directional Link Count Report: No completed paths data available.")
                else:
                    excel_link_count_report_path = create_bidirectional_link_count_report(
                        completed_paths_data=self.finalized_paths_for_report,
                        start_datetime_video=start_datetime_obj_global,
                        video_path=actual_video_path_to_process[0],
                        primary_direction=primary_direction,
                        secondary_direction=[direction for direction in self.zone_polygons.keys() if direction!=primary_direction][0],
                        video_duration_formatted=video_duration_formatted
                    )
                    if excel_link_count_report_path:
                        result_msg_final += f"Bi-directional Link Count report: {excel_link_count_report_path}\n"
                    else:
                        result_msg_final += "Bi-directional Link Count report generation failed or produced no data.\n"
        
            if ENABLE_DETAILED_PERFORMANCE_METRICS and self.perf: visualize_performance(self.perf)

            if ENABLE_VISUALIZATION:
                if self.final_output_path and "ERROR" not in self.final_output_path: result_msg_final+=f"Output video: '{self.final_output_path}'.\n"
                elif self.final_output_path and "ERROR" in self.final_output_path: result_msg_final+=f"Output video generation FAILED: {self.final_output_path}.\n"
                else: result_msg_final+=f"Output video status unknown.\n"
            if excel_path_result: result_msg_final+=f"Excel report: {excel_path_result}\n"
            if ENABLE_DETAILED_PERFORMANCE_METRICS and self.perf: result_msg_final+=f"Perf charts: 'performance_charts/'\n"
            result_msg_final+=f"\n--- Summary Stats ---\nSTAT_FRAMES_PROCESSED={frames_processed_count}\nSTAT_TIME_SECONDS={total_time_taken_final:.2f}\nSTAT_FPS={final_fps_overall:.2f}\nSTAT_COMPLETED_PATHS={completed_paths_count_valid_excel}\n--- End Stats ---"
            if progress_queue: 
                progress_queue.put_nowait({ 
                    "progress": 100, 
                    "status": "complete", 
                    "results": { 
                        "video_path": self.final_output_path, 
                        "excel_path": excel_path_result, 
                        "bidirectional_excel_path": excel_link_count_report_path 
                    },
                    "statistics": {
                        "total_valid_paths": completed_paths_count_valid_excel,
                        "processing_time_seconds": round(total_time_taken_final, 2),
                        "processing_fps": round(final_fps_overall, 2),
                        "total_frames_processed": frames_processed_count
                    }
                })
            return result_msg_final

        except (ValueError, RuntimeError, FileNotFoundError, Exception) as e_proc: # Final error handling
            print(f"\nFATAL ERROR during setup or processing: {e_proc}")
            if progress_queue:
                progress_queue.put_nowait({"progress": 100, "status": "error", "message": str(e_proc)})
            import traceback; traceback.print_exc()
            if hasattr(self, 'stop_event') and self.stop_event: self.stop_event.set()
            return f"❌ Error: {e_proc}"
        finally: # Final cleanup
            # This ensures the WebSocket knows the queue is done, even on cleanup.
            if progress_queue:
                progress_queue.put_nowait({"status": "finished"})
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

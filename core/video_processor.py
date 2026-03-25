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
from supervision_modified.tracker.byte_tracker.core import ByteTrack
from datetime import datetime, timedelta
import concurrent.futures
from ultralytics.data.augment import LetterBox
from loguru import logger
from collections import deque
import builtins
if not hasattr(builtins, "profile"):
    def profile(func):
        return func
    builtins.profile = profile
profile = builtins.profile

from config import (
    TARGET_FPS, FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH_VIZ, FRAME_HEIGHT_VIZ,
    VIDEO_OUTPUT_DIR, ENABLE_DETAILED_PERFORMANCE_METRICS, OPTIMAL_BATCH_SIZE,
    MIXED_PRECISION, PARALLEL_STREAMS, IOU_THRESHOLD, TTA,
    MEMORY_CLEANUP_INTERVAL, MODEL_PATH, CONF_THRESHOLD, VEHICLE_TIMEOUTS,
    LINE_POINTS, MODEL_INPUT_SIZE, ENABLE_VISUALIZATION, DATA_YAML_PATH,
    PYVIDEOREADER_DECODE_SEGMENT_SIZE, NUM_INFERENCE_STREAMS, BYTETRACK_MATCH_THRESH,
    BYTETRACK_TRACK_BUFFER_SECONDS, BYTETRACK_TRACK_THRESH, CROP_MODE,
    USE_PATCHWISE_INFERENCE, PATCHWISE_INFERENCE_CONFIG, PATCHWISE_USE_BATCH_PROCESSING,
    MOTION_INFERENCE_MIN_FRAMES, MOTION_INFERENCE_MAX_PROJECTION, MOTION_INFERENCE_MIN_DIST,
    MIN_SPEED_POSITIONS, MIN_SPEED_DURATION_SEC, MAX_PLAUSIBLE_SPEED_KMH,
    JUMP_THRESHOLD_MULTIPLIER, DEBUG_SPEED, DEBUG_MODE, BASE_OUTPUT_DIR
)
from core.patchwise_detector import PatchwiseDetector
from core.utils import is_valid_movement, cleanup_memory
from core.perspective_transform import PerspectiveTransformer
from core.performance import PerformanceTracker
from models.model_loader import load_model, get_device
from tracking.zone_tracker import ZoneTracker
from visualization import overlay as visual_overlay
from reporting.excel_report import create_excel_report, create_bidirectional_link_count_report
from core.converters import convert_dav_to_mp4_internal
from video_reader import PyVideoReader
from core.finalized_paths_store import FinalizedPathsStore

class VideoProcessor:
    @profile
    def __init__(self, line_mode: str, data_yaml_path: str = DATA_YAML_PATH):
        self.line_mode = line_mode
        self.target_fps = TARGET_FPS
        self.frame_shape = (FRAME_WIDTH, FRAME_HEIGHT)
        self.frame_shape_viz = (FRAME_WIDTH_VIZ, FRAME_HEIGHT_VIZ)
        self.model_input_size_config = MODEL_INPUT_SIZE
        self.batch_size = OPTIMAL_BATCH_SIZE
        self.num_inference_streams = max(1, NUM_INFERENCE_STREAMS)
        self.pvr_decode_segment_size = PYVIDEOREADER_DECODE_SEGMENT_SIZE
        self.use_patchwise_inference = USE_PATCHWISE_INFERENCE
        self.patchwise_detector = None

        self.device = get_device()
        self.processing_lock = threading.Lock()
        self.tracker_lock = threading.Lock()

        self.zone_axis = None
        self.zone_pair = None

        self.class_id_to_name_map = {}
        if data_yaml_path and os.path.exists(data_yaml_path):
            try:
                with open(data_yaml_path, 'r', encoding='utf-8') as f:
                    data_yaml = yaml.safe_load(f)
                if data_yaml and 'names' in data_yaml and isinstance(data_yaml['names'], list):
                    self.class_id_to_name_map = {i: str(name) for i, name in enumerate(data_yaml['names'])}
                    logger.debug(f"Successfully loaded {len(self.class_id_to_name_map)} class names.")
            except Exception as e:
                logger.error(f"Error loading class names from YAML '{data_yaml_path}': {e}")
        else:
            logger.warning(f"DATA_YAML_PATH ('{data_yaml_path}') not found. Class names will be default.")

        self.models_list = []
        self.bytetracker = None
        self.zone_tracker = None
        self.perf = None
        self.stop_event = None
        
        # Define all threads
        self.reader_thread = None
        self.preprocessor_thread = None
        self.postprocessor_thread = None
        self.writer_thread = None
        
        # Define all queues for the new pipeline
        self.raw_frame_queue = None
        self.preprocessed_tensor_queue = None
        self.results_queue = None
        self.video_write_queue = None

        self.streams = None
        self.active_track_data = {}
        self.finalized_paths_store = FinalizedPathsStore()
        self.zone_polygons = {}
        self.detection_zones_polygons = None
        self.last_cleanup_time = 0
        self.final_output_path = None
        self.temp_converted_files = []

        # Letterbox inverse-mapping parameters (dw, dh, unpad_w, unpad_h)
        # Computed once in _preprocess_batch from the first frame's dimensions.
        self.letterbox_params = None

        # Speed estimation (perspective transform)
        self.perspective_transformer = None
        self.perspective_corners_raw = None
        self.road_width_meters = None
        self.road_length_meters = None
        self.length_skew_factor = 1.0
        self.width_skew_factor = 1.0
        self.speed_limit_kmh = None
        self.night_hours_start = None
        self.night_hours_end = None
        self.last_vehicle_info = None

    @profile
    def _initialize_run(self, video_paths, start_date_str, start_time_str, primary_direction, station_number_str, generate_link_count_report_flag, preview_image_width, preview_image_height):
        logger.info("--- Initializing New Video Processing Run ---")
        self.bytetracker = ByteTrack(
            track_activation_threshold=BYTETRACK_TRACK_THRESH,
            minimum_matching_threshold=BYTETRACK_MATCH_THRESH,
            lost_track_buffer=int(BYTETRACK_TRACK_BUFFER_SECONDS * self.target_fps),  # Convert seconds to frames
            frame_rate=self.target_fps
        )
        self.active_track_data.clear()
        self.finalized_paths_store.clear()
        self.temp_converted_files.clear()
        self.last_vehicle_info = None
        
        self.perf = PerformanceTracker(enabled=ENABLE_DETAILED_PERFORMANCE_METRICS)
        self.stop_event = threading.Event()
        self.last_cleanup_time = time.monotonic()
        
        # Initialize all queues. Sizes are tuned to prevent any stage from blocking another.
        max_queue_size = max(self.batch_size * 4, 128)
        self.raw_frame_queue = queue.Queue(maxsize=max_queue_size)
        self.results_queue = queue.Queue(maxsize=max_queue_size)
        if not self.use_patchwise_inference:
            self.preprocessed_tensor_queue = queue.Queue(maxsize=8)
        
        if self.device == 'cuda' and PARALLEL_STREAMS > 1:
            self.streams = [torch.cuda.Stream() for _ in range(PARALLEL_STREAMS)]

        actual_video_paths = []
        for path in video_paths:
            if str(path).lower().endswith(".dav"):
                logger.info(f"DAV file detected. Converting: {path}")
                converted_path = convert_dav_to_mp4_internal(path)
                if not converted_path or not os.path.exists(converted_path):
                    raise RuntimeError(f"Failed to convert .dav file: {path}")
                actual_video_paths.append(converted_path)
                self.temp_converted_files.append(converted_path)
            else:
                actual_video_paths.append(path)

        if self.line_mode == 'hardcoded':
            valid_polygons = {k: v for k, v in LINE_POINTS.items() if isinstance(v, list) and len(v) >= 3}
            if not valid_polygons: raise ValueError("Hardcoded mode: No valid polygons in LINE_POINTS.")
            self.detection_zones_polygons = {d: np.array(p, dtype=np.int32) for d, p in valid_polygons.items()}
        elif self.line_mode == 'interactive':
            if not self.zone_polygons: raise ValueError("Interactive mode: No zone polygons received from frontend.")
            valid_polygons = {k: v for k, v in self.zone_polygons.items() if v and len(v) >= 3}
            if len(valid_polygons) < 2: raise ValueError(f"Interactive mode: Need >= 2 valid zones. Found {len(valid_polygons)}.")
            scaled_polygons = self._scale_polygons_to_processing_coordinates(valid_polygons, preview_image_width, preview_image_height)
            self.detection_zones_polygons = {d: np.array(p, dtype=np.int32) for d, p in scaled_polygons.items()}
        else:
            raise ValueError(f"Invalid LINE_MODE '{self.line_mode}'. Must be 'hardcoded' or 'interactive'.")

        self.zone_tracker = ZoneTracker(self.detection_zones_polygons)
        if not self.zone_tracker or not self.zone_tracker.zones:
            raise RuntimeError("CRITICAL: ZoneTracker failed to initialize with valid zones.")
        
        if len(self.zone_tracker.zones) == 2:
            zone_names = list(self.zone_tracker.zones.keys())
            z1, z2 = zone_names[0], zone_names[1]
            self.zone_pair = (z1, z2)

            # Perspective transform from user-drawn 4-corner road rectangle.
            # User marks TL, TR, BR, BL corners on the road surface and
            # provides real-world width & length in meters.
            if (self.perspective_corners_raw and len(self.perspective_corners_raw) == 4
                    and self.road_width_meters and self.road_length_meters):
                scaled_corners = self._scale_polygons_to_processing_coordinates(
                    {'corners': self.perspective_corners_raw}, preview_image_width, preview_image_height
                ).get('corners', [])
                if len(scaled_corners) == 4:
                    try:
                        # Try to capture first frame for debug visualisation
                        debug_frame = None
                        try:
                            cap = cv2.VideoCapture(str(actual_video_paths[0]))
                            ret, first_frame = cap.read()
                            cap.release()
                            if ret and first_frame is not None:
                                debug_frame = cv2.resize(first_frame, self.frame_shape)
                        except Exception as e:
                            logger.warning(f"Could not read first frame for PerspectiveTransformer debug: {e}")

                        self.perspective_transformer = PerspectiveTransformer(
                            scaled_corners,
                            self.road_width_meters,
                            self.road_length_meters,
                            debug_frame=debug_frame
                        )
                        # Zone axis from TL/BL midpoint to TR/BR midpoint
                        corners_arr = np.array(scaled_corners, dtype=np.float32)
                        near_mid = (corners_arr[3] + corners_arr[0]) / 2  # BL-TL midpoint
                        far_mid = (corners_arr[2] + corners_arr[1]) / 2   # BR-TR midpoint
                        axis_vec = far_mid - near_mid
                        self.zone_axis = axis_vec / (np.linalg.norm(axis_vec) + 1e-6)
                        logger.info(f"PerspectiveTransformer built: "
                                    f"width={self.road_width_meters}m, length={self.road_length_meters}m")
                    except Exception as e:
                        logger.warning(f"Failed to build PerspectiveTransformer: {e}")
                        self.perspective_transformer = None
                        self.zone_axis = None
                else:
                    self.perspective_transformer = None
                    self.zone_axis = None
            else:
                logger.warning(f"Perspective transform inputs missing: corners={'set' if self.perspective_corners_raw else 'None'}, "
                               f"width={self.road_width_meters}, length={self.road_length_meters}")
                self.perspective_transformer = None
                # Fallback: centroid-based zone axis
                c1 = np.mean(self.zone_tracker.zones[z1], axis=0)
                c2 = np.mean(self.zone_tracker.zones[z2], axis=0)
                axis = c2 - c1
                self.zone_axis = axis / (np.linalg.norm(axis) + 1e-6)
        else:
            self.zone_axis = None
            self.zone_pair = None
            self.perspective_transformer = None

        # --- MODEL LOADING LOGIC ---
        if self.use_patchwise_inference:
            logger.info("Patchwise inference enabled. Loading one model instance.")
            # For patchwise, conf/iou are handled internally, so pass None
            model_instance = load_model(str(MODEL_PATH), self.device, None, None)
            if not self.class_id_to_name_map:
                self.class_id_to_name_map = model_instance.names
            self.patchwise_detector = PatchwiseDetector(model_instance, PATCHWISE_INFERENCE_CONFIG)
        else:
            logger.info(f"Standard inference enabled. Loading {self.num_inference_streams} model instance(s)...")
            for i in range(self.num_inference_streams):
                model_instance = load_model(str(MODEL_PATH), self.device, CONF_THRESHOLD, IOU_THRESHOLD)
                if not self.class_id_to_name_map:
                    self.class_id_to_name_map = model_instance.names
                self.models_list.append(model_instance)
                logger.debug(f"Loaded model instance {i+1}/{self.num_inference_streams}.")

        try:
            time_part, ms_part = (start_time_str[:6], start_time_str[6:]) if len(start_time_str) > 6 else (start_time_str, "0")
            start_dt = datetime.strptime(f"{start_date_str}{time_part}", "%y%m%d%H%M%S")
            start_dt = start_dt.replace(microsecond=int(ms_part.ljust(3, '0')) * 1000)
            logger.info(f"Video start timestamp set to: {start_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        except ValueError as e:
            raise ValueError(f"Invalid date/time format: {e}. Expected YYMMDD and HHMMSSmmm.")

        total_frames, original_fps = self._get_video_properties(actual_video_paths)
        if total_frames <= 0:
            raise ValueError("Cannot process video with zero or unreadable total frames.")
        
        video_duration_seconds = total_frames / original_fps if original_fps > 0 else 0
        logger.info(f"Video Properties: ~{total_frames} frames, {original_fps:.2f} FPS (Original), Duration: {self._format_duration(video_duration_seconds)}")

        # Start all pipeline threads
        logger.info("Starting all pipeline threads...")
        
        # 1. Reader Thread
        reader_args = (actual_video_paths, original_fps)
        self.reader_thread = threading.Thread(target=self._frame_reader_task_pvr_segmented, args=reader_args, daemon=True)
        self.reader_thread.start()

        # 2. Preprocessor Thread (only for standard inference)
        if not self.use_patchwise_inference:
            self.preprocessor_thread = threading.Thread(target=self._preprocessor_task, daemon=True)
            self.preprocessor_thread.start()

        # 3. Post-processor Thread
        postprocessor_args = (start_dt,)
        self.postprocessor_thread = threading.Thread(target=self._postprocessor_task, args=postprocessor_args, daemon=True)
        self.postprocessor_thread.start()
        
        # 4. Writer Thread (Optional)
        if ENABLE_VISUALIZATION:
            os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
            ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(actual_video_paths[0]))[0]
            output_path_viz = os.path.join(VIDEO_OUTPUT_DIR, f"output_{base_name}_{ts_now}.mp4")
            self.final_output_path = output_path_viz
            self.video_write_queue = queue.Queue(maxsize=max_queue_size)
            self.writer_thread = threading.Thread(target=self._frame_writer_task, args=(output_path_viz,), daemon=True)
            self.writer_thread.start()
        
        frame_interval = max(1, round(original_fps / self.target_fps)) if self.target_fps > 0 and original_fps > 0 else 1
        total_target_frames = total_frames // frame_interval if total_frames > 0 else 0
        if self.perf: self.perf.start_processing(total_target_frames)
        
        return {
            "start_datetime": start_dt,
            "total_frames": total_frames,
            "original_fps": original_fps,
            "primary_direction": primary_direction,
            "station_number": station_number_str,
            "generate_link_count_report": generate_link_count_report_flag,
            "video_paths": actual_video_paths,
            "video_duration_formatted": self._format_duration(video_duration_seconds)
        }

    @profile
    def _preprocessor_task(self):
        logger.info("Preprocessor thread started.")
        while not self.stop_event.is_set():
            try:
                # 1. Get a batch of raw frames from the reader
                batch_data = self._get_batch_from_queue(self.raw_frame_queue)
                if batch_data is None:
                    break # Sentinel received
                
                frames_metadata, frame_numbers, frame_times = batch_data

                # 2. Preprocess the batch into a tensor
                if self.perf: self.perf.start_timer('preprocessing')
                input_batch_tensor = self._preprocess_batch(frames_metadata)
                if self.perf: self.perf.end_timer('preprocessing')

                if input_batch_tensor is None or input_batch_tensor.nelement() == 0:
                    continue

                # 3. Put the tensor and its metadata into the next queue
                self.preprocessed_tensor_queue.put((input_batch_tensor, frames_metadata, frame_numbers, frame_times))
            except queue.Empty:
                continue # This is normal if the queue is temporarily empty
            except Exception as e:
                logger.exception("Error in preprocessor thread.")
                self.stop_event.set()
                break
        
        self.preprocessed_tensor_queue.put(None) # Pass sentinel downstream
        logger.info("Preprocessor thread finished.")

    @profile
    def _postprocessor_task(self, start_datetime):
        logger.info("Post-processor thread started.")
        while not self.stop_event.is_set():
            try:
                # 1. Get a batch of results from the inference stage
                results_data = self.results_queue.get(timeout=1.0)
                if results_data is None:
                    break # Sentinel received

                yolo_results_batch, frames_metadata, frame_numbers, frame_times = results_data
                
                # 2. Run the slow post-processing logic
                _, processed_frames_for_viz = self._process_inference_results(
                    yolo_results_batch, frames_metadata, frame_numbers, frame_times, start_datetime
                )
                
                # 3. If visualization is enabled, put frames into the writer queue
                if ENABLE_VISUALIZATION and self.video_write_queue:
                    if self.perf: self.perf.start_timer('video_write')
                    for frame in processed_frames_for_viz:
                        if frame is not None:
                            try: self.video_write_queue.put(frame, block=False)
                            except queue.Full: logger.warning("Video writer queue is full. Frame dropped.")
                    if self.perf: self.perf.end_timer('video_write')

            except queue.Empty:
                continue # Normal if inference is busy
            except Exception as e:
                logger.exception("Error in post-processor thread.")
                self.stop_event.set()
                break
        
        if self.video_write_queue:
            self.video_write_queue.put(None) # Pass sentinel to writer
        logger.info("Post-processor thread finished.")

    @profile
    def _get_batch_from_queue(self, q):
        # Wait for the first item to start a batch
        first_item = q.get(timeout=60)
        if first_item is None:
            return None # Sentinel received
        
        items = [first_item]
        # Grab other items in the queue without blocking to form a full batch
        while len(items) < self.batch_size:
            try:
                items.append(q.get_nowait())
                if items[-1] is None: # Handle sentinel in the middle of a batch
                    # Put back the sentinel for the next stage to handle
                    q.put(None)
                    items.pop()
                    break
            except queue.Empty:
                break # Queue is empty, process the current batch
        
        # Unzip the batch data
        frames_metadata, frame_numbers, frame_times = [], [], []
        for item in items:
            frame_content_raw, frame_num_raw, frame_time_s, frame_meta = item
            frames_metadata.append((frame_content_raw, frame_meta))
            frame_numbers.append(frame_num_raw)
            frame_times.append(frame_time_s)
        
        return frames_metadata, frame_numbers, frame_times
    
    @profile
    def _main_processing_loop(self, init_data, progress_queue):
        # This method now acts as a router to the correct inference loop
        if self.use_patchwise_inference:
            return self._main_processing_loop_patchwise(init_data, progress_queue)
        else:
            return self._main_processing_loop_standard(init_data, progress_queue)
        
    @profile
    def _main_processing_loop_patchwise(self, init_data, progress_queue):
        logger.info("Starting main processing loop (Inference Worker)...")
        frames_processed_count = 0
        last_prog_print_time = time.monotonic()
        
        while not self.stop_event.is_set():
            try:
                batch_data = self._get_batch_from_queue(self.raw_frame_queue)
                if batch_data is None:
                    break

                frames_metadata, frame_numbers, frame_times = batch_data
                frames_bgr_list = [fm[0] for fm in frames_metadata]
                actual_batch_size = len(frames_bgr_list)
                batch_proc_start_time = time.monotonic()

                if self.perf: self.perf.start_timer('inference')

                # --- LOGIC TO CHOOSE PROCESSING METHOD ---
                if PATCHWISE_USE_BATCH_PROCESSING:
                    # High-throughput mode
                    yolo_results_batch = self.patchwise_detector.process_batch(frames_bgr_list)
                else:
                    # Low-latency mode: process each frame in the batch individually
                    yolo_results_batch = []
                    for frame in frames_bgr_list:
                        result = self.patchwise_detector.process_frame(frame)
                        yolo_results_batch.append(result)
                
                if self.perf: self.perf.end_timer('inference')

                self.results_queue.put((yolo_results_batch, frames_metadata, frame_numbers, frame_times))
                
                # Performance and progress reporting
                if self.perf:
                    duration = time.monotonic() - batch_proc_start_time
                    self.perf.record_batch_processed(actual_batch_size, duration)
                    self.perf.record_detection(sum(len(r.boxes) for r in yolo_results_batch if r and r.boxes))
                    self.perf.sample_system_metrics()
                
                frames_processed_count += actual_batch_size
                if time.monotonic() - last_prog_print_time > 5.0:
                    prog_stats = self.perf.get_progress()
                    if progress_queue:
                        progress_queue.put_nowait({"progress": round(prog_stats.get('percent', 0)), "status": f"{prog_stats['fps']:.1f} FPS", "eta": str(prog_stats.get('eta', '...'))})
                    logger.debug(f"Progress: {prog_stats['percent']:.1f}% | FPS: {prog_stats['fps']:.1f} | ETA: {prog_stats['eta']}")
                    print(f"\rProgress: {prog_stats['percent']:.1f}% | FPS: {prog_stats['fps']:.1f} | ETA: {prog_stats['eta']}      ", end="")
                    last_prog_print_time = time.monotonic()

                # Memory cleanup can stay here
                if time.monotonic() - self.last_cleanup_time > MEMORY_CLEANUP_INTERVAL:
                    if self.perf: self.perf.start_timer('memory_cleanup')
                    cleanup_memory()
                    self.last_cleanup_time = time.monotonic()
                    if self.perf: self.perf.end_timer('memory_cleanup')

            except queue.Empty:
                logger.warning("Preprocessor queue empty for 60s. Assuming processing is complete.")
                break
            except Exception as e:
                logger.exception("Fatal error in main processing loop.")
                self.stop_event.set()
                break
        
        self.results_queue.put(None) # Pass sentinel to post-processor
        print()
        return frames_processed_count

    @profile
    def _main_processing_loop_standard(self, init_data, progress_queue):
        logger.info("Starting main processing loop (Inference Worker)...")
        frames_processed_count = 0
        last_prog_print_time = time.monotonic()
        
        while not self.stop_event.is_set():
            try:
                # 1. Get a preprocessed batch tensor
                tensor_data = self.preprocessed_tensor_queue.get(timeout=60)
                if tensor_data is None:
                    break # Sentinel received, exit loop
                
                input_batch_tensor, frames_metadata, frame_numbers, frame_times = tensor_data
                actual_batch_size = input_batch_tensor.shape[0]

                # 2. Run inference on the GPU
                batch_proc_start_time = time.monotonic()
                if self.perf and self.device == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_inference_streams) as executor:
                    futures_map = {}
                    yolo_results_batch = [None] * actual_batch_size
                    for i in range(self.num_inference_streams):
                        start_idx = i * self.batch_size
                        end_idx = min((i + 1) * self.batch_size, actual_batch_size)
                        if start_idx >= end_idx:
                            continue

                        chunk = input_batch_tensor[start_idx:end_idx]

                        # Pad if chunk is smaller than batch_size
                        if chunk.shape[0] < OPTIMAL_BATCH_SIZE:
                            pad_size = OPTIMAL_BATCH_SIZE - chunk.shape[0]
                            pad_tensor = torch.zeros((pad_size, *chunk.shape[1:]), dtype=chunk.dtype, device=chunk.device)
                            chunk = torch.cat([chunk, pad_tensor], dim=0)

                        stream = self.streams[i % len(self.streams)] if self.streams else None
                        future = executor.submit(
                            self._execute_single_model_inference,
                            self.models_list[i],
                            chunk,
                            stream
                        )

                        futures_map[future] = list(range(start_idx, end_idx))
                    
                    for future in concurrent.futures.as_completed(futures_map):
                        original_indices = futures_map[future]
                        try:
                            results_chunk = future.result()
                            for i, res in enumerate(results_chunk[:len(original_indices)]):
                                yolo_results_batch[original_indices[i]] = res
                        except Exception as e:
                            logger.error(f"Error during concurrent inference for indices {original_indices}: {e}")

                if self.perf and self.device == 'cuda':
                    end_event.record()
                    end_event.synchronize()
                    self.perf.timings['inference'] += start_event.elapsed_time(end_event) / 1000.0
                
                # 3. Put results into the next queue for post-processing
                self.results_queue.put((yolo_results_batch, frames_metadata, frame_numbers, frame_times))
                
                # Performance and progress reporting
                if self.perf:
                    duration = time.monotonic() - batch_proc_start_time
                    self.perf.record_batch_processed(actual_batch_size, duration)
                    self.perf.record_detection(sum(len(r.boxes) for r in yolo_results_batch if r and r.boxes))
                    self.perf.sample_system_metrics()
                
                frames_processed_count += actual_batch_size
                if time.monotonic() - last_prog_print_time > 5.0:
                    prog_stats = self.perf.get_progress()
                    if progress_queue:
                        progress_queue.put_nowait({"progress": round(prog_stats.get('percent', 0)), "status": f"{prog_stats['fps']:.1f} FPS", "eta": str(prog_stats.get('eta', '...'))})
                    logger.debug(f"Progress: {prog_stats['percent']:.1f}% | FPS: {prog_stats['fps']:.1f} | ETA: {prog_stats['eta']}")
                    print(f"\rProgress: {prog_stats['percent']:.1f}% | FPS: {prog_stats['fps']:.1f} | ETA: {prog_stats['eta']}      ", end="")
                    last_prog_print_time = time.monotonic()

                # Memory cleanup can stay here
                if time.monotonic() - self.last_cleanup_time > MEMORY_CLEANUP_INTERVAL:
                    if self.perf: self.perf.start_timer('memory_cleanup')
                    # Note: cleanup_tracking_data is now implicitly handled by the post-processor
                    cleanup_memory()
                    self.last_cleanup_time = time.monotonic()
                    if self.perf: self.perf.end_timer('memory_cleanup')

            except queue.Empty:
                logger.warning("Preprocessor queue empty for 60s. Assuming processing is complete.")
                break
            except Exception as e:
                logger.exception("Fatal error in main processing loop.")
                self.stop_event.set()
                break
        
        self.results_queue.put(None) # Pass sentinel to post-processor
        print()
        return frames_processed_count
    
    @profile
    def _finalize_and_report(self, init_data, frames_processed_count):
        logger.info("Finalizing tracks and generating reports...")
        if self.active_track_data:
            logger.info(f"Forcibly finalizing {len(self.active_track_data)} remaining active tracks.")
            last_ts = init_data["start_datetime"] + timedelta(seconds=(init_data["total_frames"] / init_data["original_fps"]))
            with self.tracker_lock:
                for track_id, data in list(self.active_track_data.items()):
                    if data.get('status') == 'active':
                        duration = (last_ts - data['entry_time']).total_seconds()
                        self._add_to_finalized_paths(track_id, data, 'forced_exit', 'FORCED_EXIT', last_ts, duration)
                        if self.perf: self.perf.record_vehicle_exit('forced_exit')
                self.active_track_data.clear()

        if self.perf:
            self.perf.end_processing()
            self.perf.print_summary()
        
        total_time = self.perf.total_time if self.perf else 0.0
        final_fps = frames_processed_count / total_time if total_time > 0 else 0.0
        completed_paths = self.finalized_paths_store.count_by_status('exited')

        calib_factors = {
            'Base Road Length (m)': getattr(self, 'road_length_meters', 'N/A'),
            'Base Road Width (m)': getattr(self, 'road_width_meters', 'N/A'),
            'Length Skew Factor': getattr(self, 'length_skew_factor', 1.0),
            'Width Skew Factor': getattr(self, 'width_skew_factor', 1.0)
        }

        excel_path = create_excel_report(
            completed_paths_data=self.finalized_paths_store.get_all(),
            start_datetime=init_data["start_datetime"],
            primary_direction=init_data["primary_direction"],
            video_path=init_data["video_paths"][0],
            report_type='standard',
            calibration_factors=calib_factors
        )

        excel_tia_path = create_excel_report(
            completed_paths_data=self.finalized_paths_store.get_all(),
            start_datetime=init_data["start_datetime"],
            primary_direction=init_data["primary_direction"],
            video_path=init_data["video_paths"][0],
            report_type='tia',
            calibration_factors=calib_factors
        )
        
        excel_link_count_path = None
        
        if init_data.get("generate_link_count_report", False):
            secondary_dirs = [d for d in self.zone_polygons.keys() if d != init_data["primary_direction"]]
            if secondary_dirs:
                excel_link_count_path = create_bidirectional_link_count_report(
                    completed_paths_data=self.finalized_paths_store.get_all(),
                    start_datetime_video=init_data["start_datetime"],
                    video_path=init_data["video_paths"][0],
                    primary_direction=init_data["primary_direction"],
                    secondary_direction=secondary_dirs[0],
                    video_duration_formatted=init_data["video_duration_formatted"],
                    station_number_str=init_data["station_number"],
                    speed_limit_kmh=self.speed_limit_kmh,
                    night_hours_start=self.night_hours_start,
                    night_hours_end=self.night_hours_end
                )

        result_msg = "✅ Processing completed."
        if excel_path: result_msg += f"\nExcel report: {excel_path}"
        if excel_link_count_path: result_msg += f"\nBi-directional report: {excel_link_count_path}"
        if self.final_output_path: result_msg += f"\nOutput video: {self.final_output_path}"
        
        results_dict = {
            "video_path": self.final_output_path,
            "excel_path": excel_path,
            "bidirectional_excel_path": excel_link_count_path,
            "tia_excel_path": excel_tia_path
        }
        all_paths = self.finalized_paths_store.get_all()
        speeds = [p['average_speed_kmh'] for p in all_paths if p.get('average_speed_kmh') is not None]
        avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else None

        stats_dict = {
            "total_valid_paths": completed_paths,
            "processing_time_seconds": round(total_time, 2),
            "processing_fps": round(final_fps, 2),
            "total_frames_processed": frames_processed_count,
            "fleet_average_speed_kmh": avg_speed
        }
        return result_msg, results_dict, stats_dict

    def abort(self):
        """Separate logic to gracefully abort processing and unblock threads."""
        logger.info("Abort requested. Setting stop event and flushing queues...")
        if self.stop_event:
            self.stop_event.set()

        def _flush_queues():
            # Send sentinels to unblock `get()` operations waiting on empty queues
            for q in [self.raw_frame_queue, self.preprocessed_tensor_queue, self.results_queue, self.video_write_queue]:
                if q is not None:
                    try: q.put_nowait(None)
                    except queue.Full: pass
            
            threads = [self.reader_thread, self.preprocessor_thread, self.postprocessor_thread, self.writer_thread]
            
            # Continuously empty queues to unblock `put()` operations waiting on full queues
            while any(t and t.is_alive() for t in threads):
                for q in [self.raw_frame_queue, self.preprocessed_tensor_queue, self.results_queue, self.video_write_queue]:
                    if q is not None:
                        while not q.empty():
                            try: q.get_nowait()
                            except queue.Empty: break
                time.sleep(0.05)
            logger.info("All threads exited. Queues flushed.")

        # Start the flushing in a separate daemon thread so abort() returns immediately
        threading.Thread(target=_flush_queues, daemon=True).start()

    @profile
    def _cleanup_run(self):
        logger.info("Cleaning up resources...")
        if self.stop_event: self.stop_event.set()
        
        # Wait for all threads to finish
        if self.reader_thread and self.reader_thread.is_alive(): self.reader_thread.join(timeout=10)
        if self.preprocessor_thread and self.preprocessor_thread.is_alive(): self.preprocessor_thread.join(timeout=10)
        if self.postprocessor_thread and self.postprocessor_thread.is_alive(): self.postprocessor_thread.join(timeout=10)
        if self.writer_thread and self.writer_thread.is_alive(): self.writer_thread.join(timeout=10)
        
        for temp_file in self.temp_converted_files:
            try:
                os.remove(temp_file)
                logger.debug(f"Removed temporary file: {temp_file}")
            except OSError as e:
                logger.error(f"Failed to remove temporary file {temp_file}: {e}")
        
        self.models_list.clear()
        self.active_track_data.clear()
        self.bytetracker = self.zone_tracker = self.perf = self.stop_event = None
        self.reader_thread = self.preprocessor_thread = self.postprocessor_thread = self.writer_thread = None
        self.raw_frame_queue = self.preprocessed_tensor_queue = self.results_queue = self.video_write_queue = self.streams = None
        self.final_output_path = None
        self.letterbox_params = None
        
        cleanup_memory()
        if self.processing_lock.locked():
            self.processing_lock.release()
        logger.info("Cleanup complete. Lock released.")

    @profile
    def process_video(self, video_paths, start_date_str, start_time_str, station_number_str, primary_direction, generate_link_count_report_flag=False, progress_queue=None, preview_image_width=640, preview_image_height=480):
        if not self.processing_lock.acquire(blocking=False):
            logger.warning("Processing lock already held. Aborting.")
            return "Processing is already in progress."

        init_data = None
        try:
            if progress_queue: progress_queue.put_nowait({"progress": 2, "status": "Initializing..."})
            init_data = self._initialize_run(
                video_paths, start_date_str, start_time_str, primary_direction.lower(), station_number_str,
                generate_link_count_report_flag, preview_image_width, preview_image_height
            )
            
            if progress_queue: progress_queue.put_nowait({"progress": 5, "status": "Processing..."})
            frames_processed = self._main_processing_loop(init_data, progress_queue)
            
            # Wait for the post-processor to finish its queue before finalizing
            logger.info("Inference complete. Waiting for post-processing to finish...")
            # Wait for reader
            if self.reader_thread and self.reader_thread.is_alive():
                self.reader_thread.join()

            # Wait for preprocessor
            if self.preprocessor_thread and self.preprocessor_thread.is_alive():
                self.preprocessor_thread.join()

            # Wait for postprocessor
            if self.postprocessor_thread and self.postprocessor_thread.is_alive():
                self.postprocessor_thread.join()

            # Wait for writer
            if self.writer_thread and self.writer_thread.is_alive():
                self.writer_thread.join()

            if progress_queue: progress_queue.put_nowait({"progress": 95, "status": "Finalizing..."})
            result_msg, results_dict, stats_dict = self._finalize_and_report(init_data, frames_processed)

            if progress_queue:
                progress_queue.put_nowait({
                    "progress": 100, "status": "complete",
                    "results": results_dict, "statistics": stats_dict
                })
            return result_msg

        except Exception as e:
            logger.exception("A fatal error occurred during video processing.")
            if progress_queue:
                progress_queue.put_nowait({"progress": 100, "status": "error", "message": str(e)})
            return f"❌ Error: {e}"
        finally:
            if progress_queue:
                progress_queue.put_nowait({"status": "finished"})
            self._cleanup_run()

    @torch.no_grad()
    @profile
    def _preprocess_batch(self, frames_batch_with_metadata):
        if not frames_batch_with_metadata:
            return None

        target_dim = self.model_input_size_config
        
        valid_frames = [f for f, _ in frames_batch_with_metadata if f is not None]
        if not valid_frames:
            return None
        
        count = len(valid_frames)
        # Pre-allocate the batch array. Using uint8 is memory efficient.
        # The gray color (114) is a standard padding value for YOLO models.
        batch_np = np.full((count, 3, target_dim, target_dim), 114, dtype=np.uint8)

        # Compute letterbox inverse-mapping parameters once from the first frame.
        # These are used by _process_inference_results to correctly map model-space
        # coordinates back to processing coordinates, accounting for padding offsets.
        if self.letterbox_params is None:
            frame_h, frame_w = valid_frames[0].shape[:2]
            if CROP_MODE == 'letterbox':
                ratio = min(target_dim / frame_h, target_dim / frame_w)
                new_unpad_w = int(round(frame_w * ratio))
                new_unpad_h = int(round(frame_h * ratio))
                dw = (target_dim - new_unpad_w) / 2.0
                dh = (target_dim - new_unpad_h) / 2.0
                self.letterbox_params = (dw, dh, new_unpad_w, new_unpad_h)
                if dw > 0.5 or dh > 0.5:
                    logger.info(f"Letterbox padding detected: dw={dw:.1f}, dh={dh:.1f}, "
                                f"unpad=({new_unpad_w}x{new_unpad_h}) from frame ({frame_w}x{frame_h})")
            else:
                # Stretch mode: no padding offset, the entire target_dim is used
                self.letterbox_params = (0.0, 0.0, target_dim, target_dim)

        for i, frame_np in enumerate(valid_frames):

            if CROP_MODE == 'letterbox':
                # Letterboxing: Preserves aspect ratio with padding
                transform = LetterBox(new_shape=(target_dim, target_dim))
                frame_np = transform(image=frame_np)

            elif CROP_MODE == 'stretch':
                # Stretching: Distorts aspect ratio to fit the target dimensions
                if frame_np.shape[0] != target_dim or frame_np.shape[1] != target_dim:
                    frame_np = cv2.resize(frame_np, (target_dim, target_dim), interpolation=cv2.INTER_LINEAR)
            
            else:
                raise ValueError(f"Invalid CROP_MODE '{CROP_MODE}'. Must be 'letterbox' or 'stretch'.")
            
            # Transpose from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
            # and place it in the pre-allocated batch array.
            batch_np[i] = frame_np.transpose(2, 0, 1)

        # Convert the entire batch from a NumPy array to a PyTorch tensor
        batch_tensor = torch.from_numpy(batch_np).to(self.device, non_blocking=True)
        
        # Determine the final data type (float16 for mixed precision on CUDA, else float32)
        dtype_final = torch.float16 if (MIXED_PRECISION and self.device == 'cuda') else torch.float32
        
        # Convert to the final float type and normalize pixel values from [0, 255] to [0.0, 1.0]
        return batch_tensor.to(dtype_final) / 255.0

    @torch.no_grad()
    @profile
    def _process_inference_results(self, yolo_results_batch, frames_batch_with_metadata, frame_numbers_in_batch, frame_times_in_batch, start_datetime_obj_global):
        if self.perf: self.perf.start_timer('app_logic_batch')
        processed_frames_list_viz = [None] * len(yolo_results_batch)
        
        all_targets_list = []
        with self.tracker_lock:
            for i, yolo_result in enumerate(yolo_results_batch):
                if yolo_result and hasattr(yolo_result, 'boxes') and len(yolo_result.boxes.data) > 0:
                    # Convert YOLO detections to supervision format [x1, y1, x2, y2, score, class_id]
                    detections_np = yolo_result.boxes.data.cpu().numpy()
                    if detections_np.shape[1] >= 6:  # [x1, y1, x2, y2, conf, cls]
                        # Reorder to supervision format: [x1, y1, x2, y2, score, class_id]
                        supervision_detections = np.column_stack([
                            detections_np[:, :4],  # bbox coordinates
                            detections_np[:, 4],   # confidence score
                            detections_np[:, 5]    # class_id
                        ])
                    else:
                        supervision_detections = np.empty((0, 6))
                else:
                    supervision_detections = np.empty((0, 6))
                
                # Update tracker with tensors
                if supervision_detections.shape[0] > 0:
                    online_stracks = self.bytetracker.update_with_tensors(supervision_detections)
                    
                    if online_stracks:
                        # Convert STrack objects back to numpy format matching boxmot format
                        targets_np = []
                        for strack, det in zip(online_stracks, supervision_detections):
                            # Extract bbox in tlbr format
                            tlbr = strack.tlbr  # [x1, y1, x2, y2]
                            
                            # Use external_track_id (the public track ID)
                            track_id = strack.external_track_id
                            if track_id == strack.external_id_counter.NO_ID:
                                track_id = strack.internal_track_id
                            
                            score = strack.score
                            class_id = strack.class_id

                            target_row = [tlbr[0], tlbr[1], tlbr[2], tlbr[3],
                                        track_id, score, 0, class_id]
                            targets_np.append(target_row)
                        
                        if targets_np:
                            targets_array = np.array(targets_np, dtype=np.float32)
                            # NOW add the batch index column (this was missing!)
                            batch_idx_col = np.full((len(targets_array), 1), i, dtype=np.float32)
                            all_targets_list.append(np.hstack([targets_array, batch_idx_col]))
                else:
                    # Handle empty detections - update tracker anyway to maintain internal state
                    self.bytetracker.update_with_tensors(supervision_detections)


        if not all_targets_list:
            if self.perf: self.perf.end_timer('app_logic_batch')
            return [], []

        all_targets_np = np.vstack(all_targets_list)
        tx1, ty1, tx2, ty2 = all_targets_np[:, 0], all_targets_np[:, 1], all_targets_np[:, 2], all_targets_np[:, 3]
        track_ids = all_targets_np[:, 4].astype(int)
        class_ids = all_targets_np[:, 7].astype(int)
        batch_indices = all_targets_np[:, 8].astype(int)

        model_h, model_w = self.model_input_size_config, self.model_input_size_config
        draw_w, draw_h = self.frame_shape

        # Use letterbox parameters to correctly undo padding before scaling.
        # For stretch mode (or patchwise where letterbox_params is None),
        # dw/dh are 0 and unpad dims equal model dims, so this reduces to the
        # original simple scale.
        if self.letterbox_params is not None:
            dw, dh, unpad_w, unpad_h = self.letterbox_params
        else:
            dw, dh = 0.0, 0.0
            unpad_w, unpad_h = model_w, model_h
        scale_x = draw_w / unpad_w
        scale_y = draw_h / unpad_h
        
        bboxes_scaled = np.vstack([
            (tx1 - dw) * scale_x, (ty1 - dh) * scale_y,
            (tx2 - dw) * scale_x, (ty2 - dh) * scale_y
        ]).T.astype(np.int32)
        bottom_centers = np.vstack([(bboxes_scaled[:, 0] + bboxes_scaled[:, 2]) // 2, bboxes_scaled[:, 3]]).T

        temp_drawing_info_list = [{} for _ in range(len(yolo_results_batch))]
        
        with self.tracker_lock:
            # Pre-fetch zone keys to avoid repeated calls in the loop
            zone_keys = list(self.zone_tracker.zones.keys())

            for i in range(len(all_targets_np)):
                track_id_str = str(track_ids[i])
                cls_id, batch_idx = class_ids[i], batch_indices[i]
                bbox_scaled, bottom_center = bboxes_scaled[i], bottom_centers[i]
                
                current_frame_num = frame_numbers_in_batch[batch_idx]
                current_frame_time_sec = frame_times_in_batch[batch_idx]
                current_global_timestamp = start_datetime_obj_global + timedelta(seconds=current_frame_time_sec)
                vehicle_type = self.class_id_to_name_map.get(cls_id, 'Light Vehicle')
                
                temp_drawing_info_list[batch_idx][track_id_str] = {'box': bbox_scaled, 'type': vehicle_type, 'event': None, 'status': 'tracked', 'center': tuple(bottom_center)}

                if track_id_str not in self.active_track_data:
                    for zone_dir in zone_keys:
                        if self.zone_tracker._check_point_in_zone(tuple(bottom_center), self.zone_tracker.zones[zone_dir]):
                            timeout = VEHICLE_TIMEOUTS.get(vehicle_type, VEHICLE_TIMEOUTS['default'])
                            pos_maxlen = int(timeout * self.target_fps)
                            self.active_track_data[track_id_str] = {
                                                            'id': track_id_str,
                                                            'entry_direction': zone_dir,
                                                            'entry_time': current_global_timestamp,
                                                            'type_history': {vehicle_type: 1},
                                                            'current_type': vehicle_type,
                                                            'last_pos_zone': bottom_center,
                                                            'last_seen_frame': current_frame_num,
                                                            'status': 'active',
                                                            'positions': deque(maxlen=pos_maxlen),
                                                            'frames_seen': 0,
                                                            'confirmed': False
                                                        }
                            if self.perf: self.perf.record_vehicle_entry()
                            temp_drawing_info_list[batch_idx][track_id_str]['event'] = 'ENTRY'
                            break
                else:
                    track_data = self.active_track_data[track_id_str]
                    track_data['last_seen_frame'] = current_frame_num
                    track_data['positions'].append((bottom_center, current_frame_time_sec))
                    track_data['frames_seen'] += 1
                    track_data['type_history'][vehicle_type] = track_data['type_history'].get(vehicle_type, 0) + 1
                    track_data['current_type'] = max(track_data['type_history'], key=track_data['type_history'].get)
                    prev_pos = track_data.get('last_pos_zone')
                    track_data['last_pos_zone'] = bottom_center
                    temp_drawing_info_list[batch_idx][track_id_str]['type'] = track_data['current_type']

                    # Only show speed IF the vehicle has finished crossing the distance
                    if 'final_speed' in track_data and track_data['final_speed'] is not None:
                        temp_drawing_info_list[batch_idx][track_id_str]['speed'] = track_data['final_speed']

                    if prev_pos is not None and track_data['status'] == 'active':
                        event_type, event_dir = self.zone_tracker.check_zone_transition(tuple(prev_pos), tuple(bbox_scaled), track_id_str, current_frame_time_sec)
                        
                        if event_type == "ENTRY" and track_data.get('entry_direction') != event_dir:
                            if is_valid_movement(track_data['entry_direction'], event_dir, zone_keys):
                                duration = (current_global_timestamp - track_data['entry_time']).total_seconds()
                                current_frame_img = frames_batch_with_metadata[batch_idx][0] if frames_batch_with_metadata else None
                                speed = self._compute_average_speed_kmh(list(track_data['positions']), current_frame=current_frame_img, vehicle_type=track_data['current_type'])
                                track_data['final_speed'] = speed
                                self.last_vehicle_info = {'track_id': track_id_str, 'type': track_data['current_type'], 'time_taken': duration, 'speed': speed}
                                self._add_to_finalized_paths(track_id_str, track_data, 'exited', event_dir, current_global_timestamp, duration, average_speed_kmh=speed, current_frame=current_frame_img)
                                if self.perf: self.perf.record_vehicle_exit('exited', duration)
                                track_data['confirmed'] = True
                                track_data['entry_direction'] = event_dir
                                track_data['entry_time'] = current_global_timestamp
                                if self.perf: self.perf.record_vehicle_entry()
                                temp_drawing_info_list[batch_idx][track_id_str]['event'] = 'RE-ENTRY'
                                logger.info(f"[Re-Entry triggered] Track ID - {track_id_str} | Speed - {speed} km/h")

                        elif event_type == "EXIT":
                            if is_valid_movement(track_data['entry_direction'], event_dir, zone_keys):
                                duration = (current_global_timestamp - track_data['entry_time']).total_seconds()
                                current_frame_img = frames_batch_with_metadata[batch_idx][0] if frames_batch_with_metadata else None
                                speed = self._compute_average_speed_kmh(list(track_data['positions']), current_frame=current_frame_img, vehicle_type=track_data['current_type'])
                                track_data['final_speed'] = speed
                                self.last_vehicle_info = {'track_id': track_id_str, 'type': track_data['current_type'], 'time_taken': duration, 'speed': speed}
                                self._add_to_finalized_paths(track_id_str, track_data, 'exited', event_dir, current_global_timestamp, duration, average_speed_kmh=speed, current_frame=current_frame_img)
                                if self.perf: self.perf.record_vehicle_exit('exited', duration)
                                track_data['status'] = 'exited'
                                track_data['confirmed'] = True
                                track_data['exit_time'] = current_global_timestamp
                                temp_drawing_info_list[batch_idx][track_id_str]['event'] = 'EXIT'
                                logger.info(f"[Exit triggered] Track ID - {track_id_str} | Speed - {speed} km/h")

            last_frame_time_in_batch = frame_times_in_batch[-1]
            last_global_timestamp_in_batch = start_datetime_obj_global + timedelta(seconds=last_frame_time_in_batch)
            for track_id, track_data in list(self.active_track_data.items()):
                if track_data['status'] != 'active': continue
                current_frame_img = frames_batch_with_metadata[-1][0] if frames_batch_with_metadata else None
                if self._try_motion_inference(track_id, track_data, current_global_timestamp, current_frame=current_frame_img):
                    if self.perf:
                        self.perf.record_vehicle_exit('motion_inferred')
                duration = (last_global_timestamp_in_batch - track_data['entry_time']).total_seconds()
                timeout_limit = VEHICLE_TIMEOUTS.get(track_data['current_type'], VEHICLE_TIMEOUTS['default'])
                if duration > timeout_limit:
                    self._add_to_finalized_paths(track_id, track_data, 'timed_out', 'TIMEOUT', last_global_timestamp_in_batch, duration)
                    if self.perf: self.perf.record_vehicle_exit('timed_out')
                    track_data['confirmed'] = True
                    track_data['status'] = 'timed_out'
                    track_data['exit_time'] = last_global_timestamp_in_batch

            # Prune exited and timed_out tracks after their timeout duration
            tracks_to_remove = []
            for track_id, track_data in self.active_track_data.items():
                if track_data['status'] in ('exited', 'timed_out'):
                    t_limit = VEHICLE_TIMEOUTS.get(track_data.get('current_type'), VEHICLE_TIMEOUTS['default'])
                    t_since_exit = (last_global_timestamp_in_batch - track_data.get('exit_time', last_global_timestamp_in_batch)).total_seconds()
                    if t_since_exit > t_limit:
                        tracks_to_remove.append(track_id)

            for track_id in tracks_to_remove:
                del self.active_track_data[track_id]
                if self.zone_tracker:
                    self.zone_tracker.remove_vehicle_data(track_id)
            if tracks_to_remove and self.zone_tracker:
                 self.zone_tracker.cleanup_cooldowns(last_frame_time_in_batch)

        if ENABLE_VISUALIZATION:
            for i in range(len(yolo_results_batch)):
                original_frame_content = frames_batch_with_metadata[i][0]
                temp_drawing_info = temp_drawing_info_list[i]
                output_frame_viz = cv2.resize(original_frame_content.copy(), self.frame_shape)
                if self.zone_tracker:
                    output_frame_viz = visual_overlay.draw_zones(output_frame_viz, self.zone_tracker)
                
                for track_id, draw_data in temp_drawing_info.items():
                    final_status = self.active_track_data.get(track_id, {}).get('status', 'tracked')
                    if 'box' in draw_data:
                        visual_overlay.draw_detection_box(output_frame_viz, draw_data['box'], track_id, draw_data['type'], status=final_status)
                    if draw_data.get('event') and 'center' in draw_data:
                        visual_overlay.draw_event_marker(output_frame_viz, draw_data['center'], draw_data['event'])
                    # Draw speed annotation when DEBUG_MODE is enabled
                    if DEBUG_MODE and draw_data.get('speed') is not None and 'box' in draw_data:
                        visual_overlay.draw_speed_annotation(output_frame_viz, draw_data['box'], draw_data['speed'])
                
                if getattr(self, 'last_vehicle_info', None) is not None:
                    visual_overlay.draw_last_vehicle_info(output_frame_viz, self.last_vehicle_info)
                    
                processed_frames_list_viz[i] = output_frame_viz
        
        if self.perf: self.perf.end_timer('app_logic_batch')
        return [], processed_frames_list_viz
    
    def _try_motion_inference(self, track_id, track_data, now_ts, current_frame=None):
        if self.zone_axis is None or track_data['confirmed']:
            return False
        # print(f"[Motion Inference] Number of frames: {track_data['frames_seen']}")
        if track_data['frames_seen'] < MOTION_INFERENCE_MIN_FRAMES:
            return False

        positions_with_times = list(track_data['positions'])
        pts = [p[0] for p in positions_with_times]
        # print(f"[Motion Inference] Points: {pts}")
        if len(pts) < MOTION_INFERENCE_MIN_FRAMES:
            return False

        disp = np.array(pts[-1]) - np.array(pts[0])
        norm = np.linalg.norm(disp)
        # print(f"[Motion Inference] Dist: {norm}")
        if norm < MOTION_INFERENCE_MIN_DIST:
            return False

        motion_dir = disp / norm
        proj = np.dot(motion_dir, self.zone_axis)
        # print(f"[Motion Inference] Proj Angle: {proj}")
        if abs(proj) <MOTION_INFERENCE_MAX_PROJECTION:
            return False

        from_zone, to_zone = self.zone_pair
        inferred_exit = to_zone if proj > 0 else from_zone

        duration = (now_ts - track_data['entry_time']).total_seconds()
        speed = self._compute_average_speed_kmh(positions_with_times, current_frame=current_frame, vehicle_type=track_data['current_type'])
        self.last_vehicle_info = {'track_id': track_id, 'type': track_data['current_type'], 'time_taken': duration, 'speed': speed}
        logger.info(f"[Motion Inference] Track ID - {track_id} | Exit confirmed to: {inferred_exit} | Speed - {speed} km/h")
        self._add_to_finalized_paths(
            track_id,
            track_data,
            'exited',
            inferred_exit,
            now_ts,
            duration,
            average_speed_kmh=speed,
            current_frame=current_frame
        )

        track_data['status'] = 'exited'
        track_data['confirmed'] = True
        track_data['exit_time'] = now_ts
        return True

    
    def _add_to_finalized_paths(self, track_id_str, app_data, final_status_str, exit_direction_str, exit_time_dt, time_in_intersection_val, average_speed_kmh=None, current_frame=None):
        trajectory_data = []
        if 'positions' in app_data:
            for pos, ts in app_data['positions']:
                trajectory_data.append(([float(pos[0]), float(pos[1])], float(ts)))

        path_report_entry = {
            'id': track_id_str,
            'type': app_data.get('current_type', 'UnknownType'),
            'entry_direction': app_data.get('entry_direction'),
            'exit_direction': exit_direction_str,
            'entry_time': app_data.get('entry_time'),
            'exit_time': exit_time_dt,
            'time_in_intersection': round(time_in_intersection_val, 2) if time_in_intersection_val is not None else 0.0,
            'status': final_status_str,
            'average_speed_kmh': average_speed_kmh,
            'trajectory_data': trajectory_data
        }
        self.finalized_paths_store.append(path_report_entry)

    def _compute_average_speed_kmh(self, positions_with_times, current_frame=None, vehicle_type=None):
        """Retroactively compute speed using only points inside the perspective polygon."""
        if vehicle_type and str(vehicle_type).strip().lower() in ['pedestrian', 'person']:
            return None
        if self.perspective_transformer is None:
            return None
            
        # 1. Get the perspective polygon (the reliable road zone) in pixel coordinates
        poly = self.perspective_transformer._raw_corners.astype(np.int32)

        # 2. Filter the vehicle's history: keep ONLY points inside the polygon
        valid_positions = []
        for pos, t in positions_with_times:
            # cv2.pointPolygonTest returns >= 0 if the point is inside or on the edge of the polygon
            if cv2.pointPolygonTest(poly, (float(pos[0]), float(pos[1])), False) >= 0:
                valid_positions.append((pos, t))

        # 3. Ensure we have enough valid points inside the zone to make a calculation
        if len(valid_positions) < 2:
            logger.info("[Speed Calculation] Not enough tracking points inside the reliable zone.")
            return None

        # 4. Get the first and last points INSIDE the zone
        first_pos, first_time = valid_positions[0]
        last_pos, last_time = valid_positions[-1]

        total_time_sec = last_time - first_time
        if total_time_sec < MIN_SPEED_DURATION_SEC:
            logger.info(f"[Speed Calculation] Time in zone too short: {total_time_sec:.2f}s")
            return None

        try:
            # 5. Transform these specific pixel coordinates to metric (meters) coordinates
            x1, y1 = self.perspective_transformer.transform_point(float(first_pos[0]), float(first_pos[1]))
            x2, y2 = self.perspective_transformer.transform_point(float(last_pos[0]), float(last_pos[1]))
        except Exception as e:
            logger.warning(f"[Speed Calculation] Perspective transform failed: {e}")
            return None

        # 6. Calculate the straight-line distance covered in meters
        total_distance_meters = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # Ignore tiny movements (e.g., vehicle clipped the corner of the zone)
        if total_distance_meters < 1.0:
            return None

        # 7. Calculate final speed: v = d / t
        speed_ms = total_distance_meters / total_time_sec
        speed_kmh = speed_ms * 3.6

        if speed_kmh > MAX_PLAUSIBLE_SPEED_KMH:
            logger.info(f"[Speed Calculation] Speed exceeded max plausible limit: {speed_kmh:.1f} km/h")
            return None

        return round(speed_kmh, 1)

    @profile
    def _execute_single_model_inference(self, model_instance, input_tensor_chunk, stream_to_use=None):
        t0 = time.time()
        if stream_to_use and self.device == 'cuda':
            with torch.cuda.stream(stream_to_use):
                with torch.amp.autocast(device_type=self.device, enabled=(MIXED_PRECISION and self.device == 'cuda')):
                    results = model_instance(input_tensor_chunk, verbose=False, imgsz=MODEL_INPUT_SIZE, augment=TTA, half=True, device=self.device)
            stream_to_use.synchronize()
        else: 
            with torch.amp.autocast(device_type=self.device, enabled=(MIXED_PRECISION and self.device == 'cuda')):
                results = model_instance(input_tensor_chunk, verbose=False, imgsz=MODEL_INPUT_SIZE, augment=TTA, half=True, device=self.device)
        return results

    def _format_duration(self, seconds):
        return str(timedelta(seconds=int(seconds)))
    
    def _get_video_properties(self, video_paths):
        total_frames = 0
        original_fps = 0.0
        for path in video_paths:
            try:
                reader = PyVideoReader(str(path))
                info = reader.get_info()
                total_frames += int(info.get('frame_count', 0))
                if original_fps == 0.0:
                    original_fps = float(info.get('fps', '0.0'))
                del reader
            except Exception as e:
                logger.error(f"Could not get properties for video {path}: {e}")
        return total_frames, original_fps

    @profile
    def _scale_polygons_to_processing_coordinates(self, polygons_dict, original_w, original_h):
        if not polygons_dict: return {}
        target_w, target_h = self.frame_shape
        logger.info(f"Scaling polygons from preview ({original_w}x{original_h}) to processing ({target_w}x{target_h})")
        
        scale_x = target_w / original_w
        scale_y = target_h / original_h
        
        scaled_polygons = {}
        for direction, points in polygons_dict.items():
            scaled_polygons[direction] = [(int(x * scale_x), int(y * scale_y)) for x, y in points]
        return scaled_polygons

    @profile
    def _frame_reader_task_pvr_segmented(self, video_paths, original_fps):
        try:
            # If using patchwise, we read frames without resizing.
            # If standard, we resize to model input size.
            resize_dim = self.model_input_size_config
            
            cumulative_frames_offset = 0

            for video_path in video_paths:
                if self.stop_event.is_set(): break
                logger.info(f"Reader starting video: {os.path.basename(str(video_path))}")
                
                pvr = PyVideoReader(str(video_path), device='cuda', resize_shorter_side=resize_dim, resize_longer_side=resize_dim)
                
                current_video_info = pvr.get_info()
                current_video_frames = int(current_video_info.get('frame_count', 0))
                if current_video_frames == 0: 
                    del pvr
                    continue
                
                frame_interval = max(1, round(original_fps / self.target_fps))
                
                num_segments = math.ceil(current_video_frames / self.pvr_decode_segment_size)

                for i in range(num_segments):
                    if self.stop_event.is_set(): break
                    start_idx = i * self.pvr_decode_segment_size
                    end_idx = min((i + 1) * self.pvr_decode_segment_size, current_video_frames)
                    if start_idx >= end_idx: continue

                    try:
                        segment_frames = pvr.decode(start_frame=start_idx, end_frame=end_idx)
                    except Exception as e:
                        logger.warning(f"Skipping corrupted segment {start_idx}-{end_idx}: {e}")
                        continue
                    if segment_frames is None or segment_frames.shape[0] == 0: continue

                    for frame_in_seg_idx, frame_content in enumerate(segment_frames):
                        if self.stop_event.is_set(): break
                        frame_idx_in_video = start_idx + frame_in_seg_idx
                        if frame_idx_in_video % frame_interval == 0:
                            overall_frame_idx = cumulative_frames_offset + frame_idx_in_video
                            frame_time_sec = overall_frame_idx / original_fps
                            metadata = {'is_rgb': True, 'source': 'pvr_cuda_segmented'}
                            # The tuple being put into the queue
                            item = (frame_content, overall_frame_idx, frame_time_sec, metadata)
                            self.raw_frame_queue.put(item)
                
                cumulative_frames_offset += current_video_frames
                del pvr
                gc.collect()

        except Exception as e:
            logger.exception("Fatal error in frame reader task.")
        finally:
            self.raw_frame_queue.put(None) # Put sentinel for preprocessor
            logger.info("Frame reader task finished.")
            
    @profile
    def _frame_writer_task(self, output_path):
        out = None
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.target_fps, self.frame_shape_viz)
            logger.info(f"Video writer initialized for: {output_path}")

            while True:
                try:
                    frame = self.video_write_queue.get(timeout=10.0)
                    if frame is None: break
                    
                    if frame.shape[1] != self.frame_shape_viz[0] or frame.shape[0] != self.frame_shape_viz[1]:
                        frame = cv2.resize(frame, self.frame_shape_viz)
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                except queue.Empty:
                    if self.stop_event.is_set(): break
                    continue
        except Exception as e:
            logger.exception("Error in writer thread.")
            self.final_output_path = f"ERROR_WRITER_RUNTIME: {e}"
        finally:
            if out: out.release()
            logger.info("Video writer released.")
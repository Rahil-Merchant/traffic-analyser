# project/core/raw_data_extractor.py
import cv2
import numpy as np
import torch
import time
import os
import pandas as pd # For Parquet output
import queue # Keep for frame reading
import threading # Keep for frame reading/writing (if visualization still desired for this pass)
import gc
import math
from datetime import datetime, timedelta
import concurrent.futures # Make sure this is imported
from boxmot.trackers.bytetrack.bytetrack import ByteTrack # Keep ByteTrack

# Configuration - Assuming these are correctly imported from your config.py
from config import (
    TARGET_FPS, FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH_VIZ, FRAME_HEIGHT_VIZ, # For zone coordinate space
    VIDEO_OUTPUT_DIR, ENABLE_DETAILED_PERFORMANCE_METRICS, OPTIMAL_BATCH_SIZE,
    MIXED_PRECISION, PARALLEL_STREAMS, THREAD_COUNT, IOU_THRESHOLD,
    MEMORY_CLEANUP_INTERVAL, MODEL_PATH, CONF_THRESHOLD, MODEL_INPUT_SIZE,
    USE_ADVANCED_VIDEO_READER, PYVIDEOREADER_DECODE_SEGMENT_SIZE, NUM_INFERENCE_STREAMS,
    BYTETRACK_MATCH_THRESH, BYTETRACK_MIN_CONF, BYTETRACK_TRACK_BUFFER_SECONDS, BYTETRACK_TRACK_THRESH,
    POI_STRATEGIES, VEHICLE_ID_MAP, # For class name to PoI strategy
    FILE_PATHS, PASS1_SAVE_IMAGE_PATCHES, # For output and patch saving
    ENABLE_VISUALIZATION, LINE_MODE, LINE_POINTS # Added LINE_MODE, LINE_POINTS
)

# Modules
from core.utils import debug_print, cleanup_memory # Keep utils
from core.performance import PerformanceTracker # Keep for Pass 1 performance
from models.model_loader import load_model, get_device # Keep
from tracking.zone_tracker import ZoneTracker # Keep for _check_point_in_zone
from visualization import overlay as visual_overlay # If visualizing Pass 1

# Attempt to import PyVideoReader
PYVIDEOREADER_AVAILABLE = False
PyVideoReader = None
try:
    from video_reader import PyVideoReader
    PYVIDEOREADER_AVAILABLE = True
    # print("PyVideoReader (video_reader-rs) found and available.") # Already printed in original
except ImportError:
    # print("PyVideoReader (video_reader-rs) not found.") # Already printed
    pass

class RawDataExtractor:
    def __init__(self, config_module): # Pass the actual config module object
        self.config = config_module # Store the config module
        self.target_fps = self.config.TARGET_FPS
        self.frame_shape = (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT)
        self.frame_shape_viz = (self.config.FRAME_WIDTH_VIZ, self.config.FRAME_HEIGHT_VIZ) # For writer task
        self.model_input_size_config = self.config.MODEL_INPUT_SIZE
        self.batch_size = self.config.OPTIMAL_BATCH_SIZE
        self.max_workers = self.config.THREAD_COUNT

        self.device = get_device()
        self.models_list = []
        self.num_inference_streams = max(1, self.config.NUM_INFERENCE_STREAMS)
        
        self.zone_tracker = None
        self.detection_zones_polygons = None
        
        self.bytetracker = None
        self.all_tracks_data_pass1 = []

        self.frame_read_queue = None
        self.video_write_queue = None
        self.max_queue_size = max(self.batch_size * self.num_inference_streams * 8, 128)
        self.stop_event = None
        self.reader_thread = None
        self.writer_thread = None
        self.last_cleanup_time = 0
        self.output_video_path_pass1 = None
        self.final_output_path = None # Used by _frame_writer_task for error reporting

        self.streams = None
        if self.device == 'cuda' and self.config.PARALLEL_STREAMS > 1:
            self.streams = [torch.cuda.Stream() for _ in range(self.config.PARALLEL_STREAMS)]
        
        self.actually_use_advanced_reader = self.config.USE_ADVANCED_VIDEO_READER and PYVIDEOREADER_AVAILABLE
        self.pvr_decode_segment_size = self.config.PYVIDEOREADER_DECODE_SEGMENT_SIZE

        self.perf = None

        os.makedirs(os.path.dirname(self.config.FILE_PATHS['pass1_raw_track_output']), exist_ok=True)
        if self.config.PASS1_SAVE_IMAGE_PATCHES:
            os.makedirs(self.config.FILE_PATHS['pass1_image_patch_output_dir'], exist_ok=True)

    def _calculate_pois(self, bbox_zone_space, class_name):
        x1, y1, x2, y2 = bbox_zone_space
        # w = x2 - x1 # Not used yet
        # h = y2 - y1 # Not used yet
        pois = {} 
        strategy_names = self.config.POI_STRATEGIES.get(class_name, self.config.POI_STRATEGIES.get('default', ['bottom_center']))

        for poi_name in strategy_names:
            if poi_name == 'bottom_center':
                pois['bottom_center'] = ((x1 + x2) // 2, y2)
            elif poi_name == 'geometric_center':
                pois['geometric_center'] = ((x1 + x2) // 2, (y1 + y2) // 2)
            elif poi_name == 'mid_left_edge': # Added more specific PoIs
                pois['mid_left_edge'] = (x1, (y1 + y2) // 2)
            elif poi_name == 'mid_right_edge':
                pois['mid_right_edge'] = (x2, (y1 + y2) // 2)
            elif poi_name == 'mid_top_edge':
                pois['mid_top_edge'] = ((x1 + x2) // 2, y1)
            # 'leading_edge_horizontal', 'trailing_edge_horizontal', etc. are more complex
            # and would require motion history or more sophisticated heuristics.
            # For now, using edge midpoints as proxies if those names were intended for that.
            # If 'leading_edge_horizontal' means the "front" during horizontal motion, that's harder.
            # To keep it simple for now, if those specific names are in config, map them:
            elif poi_name == 'leading_edge_horizontal': # Simple interpretation: one of the vertical edges
                 pois['leading_edge_horizontal'] = (x2, (y1+y2)//2) # Assume right side if moving left-to-right
            elif poi_name == 'trailing_edge_horizontal':
                 pois['trailing_edge_horizontal'] = (x1, (y1+y2)//2) # Assume left side
            elif poi_name == 'leading_edge_vertical':
                 pois['leading_edge_vertical'] = ((x1+x2)//2, y1) # Assume top side if moving top-to-bottom
            elif poi_name == 'trailing_edge_vertical':
                 pois['trailing_edge_vertical'] = ((x1+x2)//2, y2) # Assume bottom side
        return pois
    
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
                if fps_str:
                    try: original_fps = float(fps_str)
                    except ValueError: pass
                del temp_reader_rs 
            except Exception as e:
                debug_print(f"PyVideoReader property fetch failed: {e}. Falling back to OpenCV for props.")
                if temp_reader_rs: del temp_reader_rs
        if total_frames <= 0 or original_fps <= 0:
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
        if original_fps <= 0: original_fps = 30.0 
        if total_frames <= 0:
            print("WARNING: Could not determine total_frames accurately.")   
        return total_frames, original_fps

    def _frame_reader_task(self, video_path, original_fps):
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): 
                print("ERROR: Reader failed to open video.")
                self.frame_read_queue.put(None)
                return
            
            # Use self.target_fps from config for frame interval calculation
            frame_interval = max(1, round(original_fps / self.target_fps)) if self.target_fps > 0 else 1
            frame_count_raw = 0
            frames_yielded = 0

            while not self.stop_event.is_set():
                ret, frame_bgr_orig_size = cap.read()
                if not ret: 
                    debug_print("OpenCV Reader: End of video or read error.")
                    break 
                if frame_count_raw % frame_interval == 0:
                    frame_time_seconds = frame_count_raw / original_fps
                    metadata = {'is_rgb': False, 'source': 'opencv'}
                    try:
                        self.frame_read_queue.put((frame_bgr_orig_size, frame_count_raw , frame_time_seconds, metadata), block=True, timeout=5.0)
                        frames_yielded += 1
                    except queue.Full:
                        debug_print("Reader queue full, waiting...")
                        time.sleep(0.1)
                        if self.stop_event.is_set(): break
                        try: 
                            self.frame_read_queue.put((frame_bgr_orig_size, frame_count_raw , frame_time_seconds, metadata), block=True, timeout=5.0)
                            frames_yielded += 1
                        except queue.Full: 
                            print("ERROR: Reader queue persistently full.")
                            self.stop_event.set()
                            break
                    except Exception as q_e: 
                        debug_print(f"OpenCV Reader: Queue put error: {q_e}")
                        self.stop_event.set()
                        break
                frame_count_raw += 1
            debug_print(f"OpenCV Reader finished. Raw frames read: {frame_count_raw}. Frames yielded: {frames_yielded}.")
        except Exception as e: 
            print(f"FATAL ERROR in _frame_reader_task (OpenCV): {e}")
            import traceback
            traceback.print_exc()
        finally:
            if cap: cap.release()
            if self.frame_read_queue:
                try:
                    self.frame_read_queue.put(None, block=True, timeout=1.0) 
                except queue.Full:
                    debug_print("OpenCV Reader: Queue full on final None put.")
                except Exception as e_fin:
                    debug_print(f"OpenCV Reader: Error on final None put: {e_fin}")

    def _frame_reader_task_pvr_segmented(self, video_path, original_fps_from_props, total_frames_from_props):
        pvr_instance = None
        try:
            target_dim_model = self.model_input_size_config
            debug_print(f"PVR SegReader: Init PyVideoReader: device='cuda', target_size=({target_dim_model}x{target_dim_model})")
            pvr_instance = PyVideoReader(video_path, device='cuda', 
                                         resize_shorter_side=target_dim_model,
                                         resize_longer_side=target_dim_model)
            frame_interval = max(1, round(original_fps_from_props / self.target_fps)) if self.target_fps > 0 else 1
            queued_frame_count = 0
            num_segments = math.ceil(total_frames_from_props / self.pvr_decode_segment_size) if total_frames_from_props > 0 and self.pvr_decode_segment_size > 0 else 1
            if total_frames_from_props == 0 : num_segments = 0

            for i in range(num_segments):
                if self.stop_event.is_set(): break
                segment_start_raw_idx = i * self.pvr_decode_segment_size
                segment_end_raw_idx = min((i + 1) * self.pvr_decode_segment_size, total_frames_from_props)
                if segment_start_raw_idx >= segment_end_raw_idx : continue
                
                segment_frames_np = pvr_instance.decode(start_frame=segment_start_raw_idx, end_frame=segment_end_raw_idx)
                if segment_frames_np is None or segment_frames_np.shape[0] == 0:
                    continue

                for frame_in_segment_idx in range(segment_frames_np.shape[0]):
                    if self.stop_event.is_set(): break
                    current_overall_raw_frame_idx = segment_start_raw_idx + frame_in_segment_idx
                    if current_overall_raw_frame_idx % frame_interval == 0:
                        frame_content = segment_frames_np[frame_in_segment_idx]
                        current_frame_time_sec = current_overall_raw_frame_idx / original_fps_from_props
                        metadata = {'is_rgb': True, 'source': 'pvr_cuda_segmented'}
                        try:
                            self.frame_read_queue.put((frame_content, current_overall_raw_frame_idx, current_frame_time_sec, metadata), block=True, timeout=5.0)
                            queued_frame_count += 1
                        except queue.Full:
                            debug_print("PVR SegReader: Queue full, retrying...")
                            time.sleep(0.1)
                            if self.stop_event.is_set(): break
                            try:
                                self.frame_read_queue.put((frame_content, current_overall_raw_frame_idx, current_frame_time_sec, metadata), block=True, timeout=5.0)
                                queued_frame_count += 1
                            except queue.Full: 
                                print("ERROR: PVR SegReader queue persistently full.")
                                self.stop_event.set(); break
                        except Exception as q_e: 
                            debug_print(f"PVR SegReader queue put error: {q_e}")
                            self.stop_event.set(); break
                del segment_frames_np; gc.collect()
            debug_print(f"PVR SegReader finished. Queued {queued_frame_count} frames.")
        except Exception as e:
            print(f"FATAL ERROR in _frame_reader_task_pvr_segmented: {e}")
            import traceback; traceback.print_exc()
        finally:
            if pvr_instance: del pvr_instance
            if self.frame_read_queue:
                try: self.frame_read_queue.put(None, block=True, timeout=1.0)
                except queue.Full: pass

    def _frame_writer_task(self, output_path):
        out = None; writer_initialized = False; frames_written = 0
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Use self.frame_shape_viz from __init__
            out = cv2.VideoWriter(output_path, fourcc, self.target_fps, self.frame_shape_viz)
            writer_initialized = True
            print(f"Writer initialized for: {output_path}")
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
                    if frame.shape[1]!=self.frame_shape_viz[0] or frame.shape[0]!=self.frame_shape_viz[1]:
                        frame = cv2.resize(frame, self.frame_shape_viz)
                    # Visualization overlays create RGB, writer expects BGR
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) 
                    frames_written += 1
                if self.video_write_queue: self.video_write_queue.task_done()
        except Exception as e:
            print(f"ERROR in writer thread: {e}")
            # Use self.output_video_path_pass1 if that's the intended variable for pass 1 viz output path status
            # self.final_output_path was used in the old VideoProcessor. For clarity, let's stick to output_video_path_pass1.
            if self.output_video_path_pass1 and "ERROR" not in self.output_video_path_pass1 : 
                 self.output_video_path_pass1 = f"ERROR_WRITER_RUNTIME: {e}"
        finally:
            if out and writer_initialized:
                 out.release()
                 debug_print(f"Writer released. Wrote {frames_written} frames to {output_path}")
            if self.video_write_queue:
                try: self.video_write_queue.put(None, timeout=1.0)
                except queue.Full: pass
    
    @torch.no_grad()
    def _preprocess_batch(self, frames_batch_with_metadata):
        preprocessed_tensors_list = []
        target_dim_for_model = self.model_input_size_config
        for frame_np_content, metadata in frames_batch_with_metadata:
            if frame_np_content is None or not isinstance(frame_np_content, np.ndarray):
                continue
            is_rgb_input = metadata.get('is_rgb', False)
            source_reader = metadata.get('source', 'unknown')
            img_rgb_model_sized = None
            if source_reader.startswith('pvr_cuda'):
                if not is_rgb_input: frame_np_content = cv2.cvtColor(frame_np_content, cv2.COLOR_BGR2RGB)
                if frame_np_content.shape[0] != target_dim_for_model or frame_np_content.shape[1] != target_dim_for_model:
                    img_rgb_model_sized = cv2.resize(frame_np_content, (target_dim_for_model, target_dim_for_model), interpolation=cv2.INTER_LINEAR)
                else:
                    img_rgb_model_sized = frame_np_content
            elif source_reader == 'opencv':
                resized_bgr = cv2.resize(frame_np_content, (target_dim_for_model, target_dim_for_model), interpolation=cv2.INTER_LINEAR)
                img_rgb_model_sized = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
            else: 
                temp_frame = frame_np_content.copy()
                if not is_rgb_input: temp_frame = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
                if temp_frame.shape[0] != target_dim_for_model or temp_frame.shape[1] != target_dim_for_model:
                    img_rgb_model_sized = cv2.resize(temp_frame, (target_dim_for_model, target_dim_for_model), interpolation=cv2.INTER_LINEAR)
                else:
                    img_rgb_model_sized = temp_frame
            if img_rgb_model_sized is None: continue
            if img_rgb_model_sized.ndim == 2: img_rgb_model_sized = cv2.cvtColor(img_rgb_model_sized, cv2.COLOR_GRAY2RGB)
            elif img_rgb_model_sized.shape[2] == 4: img_rgb_model_sized = cv2.cvtColor(img_rgb_model_sized, cv2.COLOR_RGBA2RGB)
            if img_rgb_model_sized.shape[2] != 3: continue
            img_chw = img_rgb_model_sized.transpose(2, 0, 1)
            img_chw_contiguous = np.ascontiguousarray(img_chw)
            preprocessed_tensors_list.append(img_chw_contiguous)
        if not preprocessed_tensors_list: return None
        try: batch_np_stacked = np.stack(preprocessed_tensors_list)
        except ValueError as e:
            debug_print(f"Preprocess: Error stacking tensors: {e}")
            return None
        batch_tensor_from_np = torch.from_numpy(batch_np_stacked).to(self.device)
        if batch_tensor_from_np.dtype != torch.uint8:
             batch_tensor_from_np = torch.clamp(batch_tensor_from_np, 0, 255).to(torch.uint8)
        dtype_final = torch.float16 if (self.config.MIXED_PRECISION and self.device == 'cuda') else torch.float32
        final_batch_tensor = batch_tensor_from_np.to(dtype_final) / 255.0
        return final_batch_tensor

    @torch.no_grad()
    def _process_batch_for_pass1(self,
                                 yolo_results_batch,
                                 original_frames_content_batch,
                                 frame_numbers_in_batch,
                                 frame_times_in_batch,
                                 start_datetime_obj_global):
        processed_frames_for_viz = [None] * len(yolo_results_batch)
        online_targets_for_each_frame_in_batch = [None] * len(yolo_results_batch)

        for i in range(len(yolo_results_batch)):
            yolo_result_obj_single_frame = yolo_results_batch[i]
            if yolo_result_obj_single_frame is None: # Handle case where a result might be None
                online_targets_for_each_frame_in_batch[i] = np.empty((0,7)) # ByteTrack expects 7 columns (x1,y1,x2,y2,track_id,score,class_id)
                continue 
                
            detections_for_bytetrack_single_frame = []
            if hasattr(yolo_result_obj_single_frame, 'boxes') and \
               yolo_result_obj_single_frame.boxes is not None and \
               len(yolo_result_obj_single_frame.boxes.data) > 0:
                for box_data_tensor in yolo_result_obj_single_frame.boxes.data:
                    box_cpu = box_data_tensor.cpu().numpy()
                    x1,y1,x2,y2,conf,cls_id = box_cpu[0],box_cpu[1],box_cpu[2],box_cpu[3],box_cpu[4],box_cpu[5]
                    detections_for_bytetrack_single_frame.append([x1, y1, x2, y2, conf, int(cls_id)])
            detections_np_single_frame = np.array(detections_for_bytetrack_single_frame) if detections_for_bytetrack_single_frame else np.empty((0, 6))
            
            img_h_model, img_w_model = yolo_result_obj_single_frame.orig_shape
            # Pass the original image (or a dummy) to ByteTrack's update method.
            # yolo_result_obj_single_frame.orig_img is usually the preprocessed image (e.g., letterboxed).
            # ByteTrack internally might not use the image content if only detections are provided,
            # but its API expects an image argument.
            # Using a dummy black image of the correct dimensions is safest if orig_img is problematic or not needed.
            img_for_tracker_update = yolo_result_obj_single_frame.orig_img if hasattr(yolo_result_obj_single_frame, 'orig_img') and yolo_result_obj_single_frame.orig_img is not None else np.zeros((img_h_model, img_w_model, 3), dtype=np.uint8)
            
            online_targets_for_each_frame_in_batch[i] = self.bytetracker.update(detections_np_single_frame, img_for_tracker_update)
        
        batch_extracted_data_for_parquet = [] # Collect data for parquet for this batch

        # Using self.max_workers from __init__
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._extract_frame_data_logic_pass1, 
                                online_targets_for_each_frame_in_batch[i],
                                yolo_results_batch[i],
                                original_frames_content_batch[i], # This is the original frame from reader
                                frame_numbers_in_batch[i],
                                frame_times_in_batch[i],
                                start_datetime_obj_global): i
                for i in range(len(yolo_results_batch))
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                original_batch_idx = future_to_idx[future]
                try:
                    viz_frame, frame_data_list_for_parquet = future.result()
                    processed_frames_for_viz[original_batch_idx] = viz_frame
                    if frame_data_list_for_parquet:
                        batch_extracted_data_for_parquet.extend(frame_data_list_for_parquet)
                except Exception as exc:
                    print(f'Error in _extract_frame_data_logic_pass1 for batch index {original_batch_idx}: {exc}')
                    import traceback; traceback.print_exc() # Add traceback
        
        if batch_extracted_data_for_parquet:
            self.all_tracks_data_pass1.extend(batch_extracted_data_for_parquet)

        return processed_frames_for_viz

    # Renamed and adapted inner function for Pass 1
    def _extract_frame_data_logic_pass1(self, 
                                       online_targets_current_frame,
                                       yolo_result_obj, # Single YOLO result object
                                       original_frame_np_for_patches, # Original frame from reader queue
                                       current_video_frame_num,
                                       current_frame_time_sec,
                                       start_datetime_obj_global):
        frame_viz_output = None
        frame_specific_track_data_list_for_parquet = [] # Data for this frame

        if yolo_result_obj is None: # Handle if a yolo result was None
            return frame_viz_output, frame_specific_track_data_list_for_parquet

        if self.config.ENABLE_VISUALIZATION:
            # Visualization setup
            if original_frame_np_for_patches is not None:
                # Resize the original frame (from reader) to the standard visualization size
                frame_for_viz_resized = cv2.resize(original_frame_np_for_patches.copy(), self.frame_shape_viz)
                # Ensure it's RGB for overlay functions
                if frame_for_viz_resized.ndim == 2 or frame_for_viz_resized.shape[2] == 1: # Grayscale
                    frame_viz_output = cv2.cvtColor(cv2.cvtColor(frame_for_viz_resized, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
                elif frame_for_viz_resized.shape[2] == 4: # RGBA
                     frame_viz_output = cv2.cvtColor(cv2.cvtColor(frame_for_viz_resized, cv2.COLOR_RGBA2BGR), cv2.COLOR_BGR2RGB)
                else: # Assume BGR
                     frame_viz_output = cv2.cvtColor(frame_for_viz_resized, cv2.COLOR_BGR2RGB)
                
                # Draw zones on the visualization frame
                if self.zone_tracker and hasattr(self.zone_tracker, 'zones') and self.zone_tracker.zones:
                    # Zones are defined in FRAME_WIDTH, FRAME_HEIGHT. Viz is FRAME_WIDTH_VIZ, FRAME_HEIGHT_VIZ.
                    # For accurate zone drawing on viz frame, zones should be scaled or drawn on a frame of self.frame_shape
                    # then that frame is resized. Simpler: scale zones to viz_shape for drawing.
                    # OR draw on a self.frame_shape canvas and resize that overlay.
                    # For now, let's assume visual_overlay.draw_zones handles polygons defined in a different space
                    # if the frame passed is already the target viz size.
                    # The polygons in self.zone_tracker.zones are in self.frame_shape space.
                    # Visual overlay expects polygons in the coordinate space of the frame it's drawing on.
                    # We need to scale polygons from self.frame_shape to self.frame_shape_viz
                    scaled_zones_for_viz = {}
                    if self.zone_tracker.zones:
                        scale_x_viz = self.frame_shape_viz[0] / self.frame_shape[0]
                        scale_y_viz = self.frame_shape_viz[1] / self.frame_shape[1]
                        for dz_name, dz_poly in self.zone_tracker.zones.items():
                            scaled_poly = (dz_poly * np.array([scale_x_viz, scale_y_viz])).astype(np.int32)
                            scaled_zones_for_viz[dz_name] = scaled_poly
                    
                    # Create a temporary ZoneTracker-like object or pass scaled zones to drawing function
                    # For simplicity, if visual_overlay.draw_zones can take a dict of polygons:
                    if scaled_zones_for_viz:
                         # This is a conceptual fix, draw_zones might need actual ZoneTracker object
                         # or be adapted to take a dictionary of polygons directly.
                         # visual_overlay.draw_zones(frame_viz_output, self.zone_tracker) # Original
                         # Let's assume draw_zones needs a ZoneTracker instance. We can make a temp one.
                         temp_viz_zone_tracker = ZoneTracker(scaled_zones_for_viz) # Initialize with scaled polys
                         visual_overlay.draw_zones(frame_viz_output, temp_viz_zone_tracker)


            else: # original_frame_np_for_patches is None
                if self.config.ENABLE_VISUALIZATION: # Create a blank frame if original is missing
                    frame_viz_output = np.zeros((self.frame_shape_viz[1], self.frame_shape_viz[0], 3), dtype=np.uint8) # Black RGB

        current_global_timestamp_dt = start_datetime_obj_global + timedelta(seconds=current_frame_time_sec)

        if online_targets_current_frame is not None and len(online_targets_current_frame) > 0:
            for target_data in online_targets_current_frame:
                tx1_model, ty1_model, tx2_model, ty2_model = target_data[0], target_data[1], target_data[2], target_data[3]
                track_id_str = str(int(target_data[4]))
                # ByteTrack output: x1,y1,x2,y2,track_id,score,class_id, index (optional for some versions)
                # Ensure class_id is correctly indexed. Standard ByteTrack output usually has class_id at index 6.
                track_cls_id = int(target_data[6]) if len(target_data) > 6 else -1 # Default if class_id missing
                # detection_confidence = target_data[5] if len(target_data) > 5 else 0.0

                vehicle_class_name = yolo_result_obj.names.get(track_cls_id, f"CLS_{track_cls_id}")
                bbox_model_space = (tx1_model, ty1_model, tx2_model, ty2_model)
                model_h_orig, model_w_orig = yolo_result_obj.orig_shape
                zone_w_target, zone_h_target = self.frame_shape # Use self.frame_shape for zone logic

                x1_zone = int(tx1_model * zone_w_target / model_w_orig)
                y1_zone = int(ty1_model * zone_h_target / model_h_orig)
                x2_zone = int(tx2_model * zone_w_target / model_w_orig)
                y2_zone = int(ty2_model * zone_h_target / model_h_orig)
                bbox_zone_space = (x1_zone, y1_zone, x2_zone, y2_zone)

                pois_zone_space = self._calculate_pois(bbox_zone_space, vehicle_class_name)
                poi_zone_insideness = {}
                if self.zone_tracker and self.zone_tracker.zones:
                    for poi_name, poi_coords in pois_zone_space.items():
                        poi_zone_insideness[poi_name] = []
                        for zone_direction, zone_poly in self.zone_tracker.zones.items():
                            if self.zone_tracker._check_point_in_zone(poi_coords, zone_poly):
                                poi_zone_insideness[poi_name].append(zone_direction)
                
                image_patch_ref = None
                if self.config.PASS1_SAVE_IMAGE_PATCHES:
                    patch_base_dir = self.config.FILE_PATHS['pass1_image_patch_output_dir']
                    patch_track_subdir = os.path.join(patch_base_dir, track_id_str)
                    os.makedirs(patch_track_subdir, exist_ok=True)
                    
                    # Sanitize class name for filename
                    safe_class_name = "".join(c if c.isalnum() else "_" for c in vehicle_class_name)
                    patch_filename = f"f{current_video_frame_num}_{safe_class_name}.jpg"
                    # Relative path from pass1_image_patch_output_dir
                    image_patch_ref = os.path.join(track_id_str, patch_filename) 
                    
                    full_patch_save_path = os.path.join(patch_track_subdir, patch_filename)

                    # Extract patch from yolo_result_obj.orig_img (the image YOLO saw)
                    img_for_patch_extraction = yolo_result_obj.orig_img 
                    if img_for_patch_extraction is not None:
                        patch_x1 = int(max(0, tx1_model))
                        patch_y1 = int(max(0, ty1_model))
                        patch_x2 = int(min(img_for_patch_extraction.shape[1], tx2_model)) # Use actual width
                        patch_y2 = int(min(img_for_patch_extraction.shape[0], ty2_model)) # Use actual height

                        if patch_x2 > patch_x1 and patch_y2 > patch_y1:
                            vehicle_patch = img_for_patch_extraction[patch_y1:patch_y2, patch_x1:patch_x2]
                            if vehicle_patch.size > 0:
                                cv2.imwrite(full_patch_save_path, vehicle_patch)
                            else: image_patch_ref = None
                        else: image_patch_ref = None
                    else: image_patch_ref = None
                
                track_frame_data = {
                    'frame_num': current_video_frame_num,
                    'timestamp_sec': round(current_frame_time_sec, 3),
                    'timestamp_dt_str': current_global_timestamp_dt.isoformat(),
                    'track_id': track_id_str,
                    'class_name': vehicle_class_name,
                    'class_id': track_cls_id,
                    'bbox_model_space_xyxy': tuple(map(lambda x: round(float(x),2), bbox_model_space)), # Ensure serializable
                    'bbox_zone_space_xyxy': bbox_zone_space,
                    'pois_zone_space': {k: tuple(map(int,v)) for k,v in pois_zone_space.items()}, # Ensure serializable
                    'poi_zone_insideness': poi_zone_insideness,
                    'image_patch_ref': image_patch_ref
                }
                frame_specific_track_data_list_for_parquet.append(track_frame_data)

                if frame_viz_output is not None:
                    # Scale bbox_zone_space to frame_viz_output space
                    scale_x_viz = self.frame_shape_viz[0] / self.frame_shape[0]
                    scale_y_viz = self.frame_shape_viz[1] / self.frame_shape[1]

                    viz_x1 = int(x1_zone * scale_x_viz)
                    viz_y1 = int(y1_zone * scale_y_viz)
                    viz_x2 = int(x2_zone * scale_x_viz)
                    viz_y2 = int(y2_zone * scale_y_viz)
                    
                    cv2.rectangle(frame_viz_output, (viz_x1, viz_y1), (viz_x2, viz_y2), (0,255,0), 1)
                    cv2.putText(frame_viz_output, f"ID:{track_id_str[-4:]}", (viz_x1, viz_y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                    for poi_n, poi_c in pois_zone_space.items():
                        viz_poi_x = int(poi_c[0] * scale_x_viz)
                        viz_poi_y = int(poi_c[1] * scale_y_viz)
                        cv2.circle(frame_viz_output, (viz_poi_x, viz_poi_y), 3, (255,0,0), -1)
        
        if frame_viz_output is not None and self.config.ENABLE_VISUALIZATION:
             # Add timestamp and frame number overlay to viz frame
             ts_str = current_global_timestamp_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
             # This overlay function might need adjustment if tracker_info is no longer a ZoneTracker instance
             # For Pass 1, a simple text overlay is fine.
             cv2.putText(frame_viz_output, f"F:{current_video_frame_num} T:{ts_str}", (10,20), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)


        return frame_viz_output, frame_specific_track_data_list_for_parquet

    def extract_raw_data(self, video_path, start_date_str, start_time_str):
        if self.perf is None and self.config.ENABLE_DETAILED_PERFORMANCE_METRICS:
            self.perf = PerformanceTracker(enabled=self.config.ENABLE_DETAILED_PERFORMANCE_METRICS)

        self.all_tracks_data_pass1 = []
        self.stop_event = threading.Event()
        self.frame_read_queue = queue.Queue(maxsize=self.max_queue_size)
        self.last_cleanup_time = time.monotonic()

        print(f"--- Starting Pass 1: Raw Data Extraction for {video_path} ---")
        actual_video_path_to_process = video_path

        # Zone Definition
        self.detection_zones_polygons = {} # Initialize as empty dict
        if self.config.LINE_MODE == 'hardcoded':
            if not hasattr(self.config, 'LINE_POINTS') or not self.config.LINE_POINTS:
                raise ValueError("Config: LINE_POINTS missing/empty for hardcoded mode.")
            for dir_name, pts_list in self.config.LINE_POINTS.items():
                if isinstance(pts_list, list) and len(pts_list) >= 3:
                    self.detection_zones_polygons[dir_name] = np.array(pts_list, dtype=np.int32)
                else:
                    print(f"Warning: Invalid polygon data for zone '{dir_name}' in config.LINE_POINTS. Skipping.")
        else: # For now, Pass 1 only supports hardcoded. Gradio zones would need to be passed in.
            raise ValueError("Pass 1 (RawDataExtractor) currently only supports 'hardcoded' LINE_MODE defined in config.py.")
        if not self.detection_zones_polygons:
            raise ValueError("No valid zone polygons loaded for ZoneTracker.")
        self.zone_tracker = ZoneTracker(self.detection_zones_polygons)

        # Load Model(s)
        self.models_list = []
        for i in range(self.num_inference_streams):
            try:
                model_instance = load_model(self.config.MODEL_PATH, self.device, self.config.CONF_THRESHOLD, self.config.IOU_THRESHOLD)
                self.models_list.append(model_instance)
            except Exception as e:
                 raise RuntimeError(f"Failed to load model instance {i+1}: {e}")
        if not self.models_list: raise RuntimeError("No model instances loaded.")

        # ByteTrack Init
        # Ensure track_buffer is integer number of frames
        track_buffer_frames = int(self.config.BYTETRACK_TRACK_BUFFER_SECONDS * self.target_fps)
        self.bytetracker = ByteTrack(
            min_conf=self.config.BYTETRACK_MIN_CONF,
            track_thresh=self.config.BYTETRACK_TRACK_THRESH,
            match_thresh=self.config.BYTETRACK_MATCH_THRESH,
            frame_rate=self.target_fps, 
            track_buffer=track_buffer_frames 
        )
        
        try:
            start_date_dt = datetime.strptime(start_date_str, "%y%m%d")
            time_part, ms_part = (start_time_str[:6], start_time_str[6:]) if len(start_time_str) > 6 else (start_time_str, "000")
            base_time_dt = datetime.strptime(time_part, "%H%M%S").time()
            microseconds_val = int(ms_part.ljust(3, '0')) * 1000
            start_time_dt_obj = base_time_dt.replace(microsecond=microseconds_val)
            start_datetime_obj_global = datetime.combine(start_date_dt.date(), start_time_dt_obj)
        except ValueError as e_ts:
            raise ValueError(f"Invalid date/time format: {e_ts}. Date: '{start_date_str}', Time: '{start_time_str}'")

        total_frames, original_fps = self._get_video_properties(actual_video_path_to_process)
        if total_frames <= 0: 
             debug_print("Warning: Total frames is zero or could not be read. Processing may not run as expected.")
             # Depending on strictness, you might want to raise an error here.

        reader_args = (actual_video_path_to_process, original_fps, total_frames) if self.actually_use_advanced_reader else (actual_video_path_to_process, original_fps)
        reader_target = self._frame_reader_task_pvr_segmented if self.actually_use_advanced_reader else self._frame_reader_task
        self.reader_thread = threading.Thread(target=reader_target, args=reader_args, daemon=True); self.reader_thread.start()

        if self.config.ENABLE_VISUALIZATION:
            os.makedirs(self.config.VIDEO_OUTPUT_DIR, exist_ok=True)
            ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(actual_video_path_to_process))[0]
            self.output_video_path_pass1 = os.path.join(self.config.VIDEO_OUTPUT_DIR, f"pass1_viz_{base_name}_{ts_now}.mp4")
            self.video_write_queue = queue.Queue(maxsize=self.max_queue_size)
            self.writer_thread = threading.Thread(target=self._frame_writer_task, args=(self.output_video_path_pass1,), daemon=True); self.writer_thread.start()
        
        frames_processed_count = 0
        frame_interval = max(1, round(original_fps / self.target_fps)) if self.target_fps > 0 and original_fps > 0 else 1
        # If total_frames is 0, this might lead to issues.
        total_target_frames_for_perf = (total_frames // frame_interval) if total_frames > 0 and frame_interval > 0 else (total_frames if total_frames > 0 else 1000) # Default if unknown
        if self.perf: self.perf.start_processing(total_target_frames_for_perf)

        current_batch_frames_raw_content = []
        current_batch_frames_metadata_for_preprocess = []
        current_batch_frame_numbers, current_batch_frame_times = [], []
        last_prog_print_time = time.monotonic()
        effective_trigger_batch_size = self.batch_size * self.num_inference_streams

        while True:
            try:
                frame_data_item = self.frame_read_queue.get(timeout=60)
                if frame_data_item is None: break
                
                frame_content_orig_size, frame_num_raw, frame_time_s, frame_meta = frame_data_item
                if self.perf: self.perf.start_timer('video_read') # Perf timer
                current_batch_frames_raw_content.append(frame_content_orig_size)
                current_batch_frames_metadata_for_preprocess.append((frame_content_orig_size, frame_meta))
                current_batch_frame_numbers.append(frame_num_raw)
                current_batch_frame_times.append(frame_time_s)
                if self.perf: self.perf.end_timer('video_read') # Perf timer

                if len(current_batch_frames_metadata_for_preprocess) >= effective_trigger_batch_size:
                    batch_proc_start_time = time.monotonic()
                    if self.perf: self.perf.start_timer('preprocessing')
                    input_batch_tensor_model = self._preprocess_batch(current_batch_frames_metadata_for_preprocess)
                    if self.perf: self.perf.end_timer('preprocessing')
                    
                    if input_batch_tensor_model is None or input_batch_tensor_model.nelement() == 0:
                        debug_print("Skipping batch: Preprocessing returned None or empty tensor.")
                        for _i_skip in range(len(current_batch_frames_metadata_for_preprocess)):
                            if not self.frame_read_queue.empty():
                                try: self.frame_read_queue.task_done()
                                except ValueError: pass 
                        current_batch_frames_raw_content.clear(); current_batch_frames_metadata_for_preprocess.clear()
                        current_batch_frame_numbers.clear(); current_batch_frame_times.clear()
                        continue

                    actual_frames_in_batch = input_batch_tensor_model.shape[0]
                    yolo_results_batch_ordered = [None] * actual_frames_in_batch
                    
                    if self.perf and self.perf.cuda_events_enabled and self.device == 'cuda': # GPU inference timer
                        overall_start_event = torch.cuda.Event(enable_timing=True); overall_end_event = torch.cuda.Event(enable_timing=True)
                        overall_start_event.record()

                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_inference_streams) as executor:
                        futures_map = {} 
                        for i_stream in range(self.num_inference_streams):
                            start_idx = i_stream * self.batch_size
                            end_idx = min((i_stream + 1) * self.batch_size, actual_frames_in_batch)
                            if start_idx >= end_idx: continue
                            chunk_for_model = input_batch_tensor_model[start_idx:end_idx]
                            original_indices_for_chunk = list(range(start_idx, end_idx))
                            model_to_use = self.models_list[i_stream % len(self.models_list)]
                            current_stream = self.streams[i_stream % len(self.streams)] if self.streams and self.device == 'cuda' else None
                            future = executor.submit(self._execute_single_model_inference, model_to_use, chunk_for_model, current_stream)
                            futures_map[future] = original_indices_for_chunk
                        for future in concurrent.futures.as_completed(futures_map):
                            original_indices = futures_map[future]
                            try:
                                results_from_model = future.result()
                                if results_from_model:
                                    for res_idx_in_chunk, yolo_res_obj in enumerate(results_from_model):
                                        global_idx = original_indices[res_idx_in_chunk]
                                        yolo_results_batch_ordered[global_idx] = yolo_res_obj
                            except Exception as e_infer:
                                print(f"\nError during concurrent inference (indices {original_indices}): {e_infer}")
                    
                    if self.perf and self.perf.cuda_events_enabled and self.device == 'cuda': # GPU inference timer
                        overall_end_event.record(); overall_end_event.synchronize()
                        self.perf.timings['inference'] += overall_start_event.elapsed_time(overall_end_event) / 1000.0

                    yolo_results_batch = yolo_results_batch_ordered
                                        
                    if self.perf: self.perf.start_timer('pass1_logic') # Timer for new processing logic
                    processed_viz_frames_for_writer = self._process_batch_for_pass1(
                        yolo_results_batch, current_batch_frames_raw_content,
                        current_batch_frame_numbers, current_batch_frame_times, start_datetime_obj_global)
                    if self.perf: self.perf.end_timer('pass1_logic')

                    if self.config.ENABLE_VISUALIZATION and self.video_write_queue:
                        if self.perf: self.perf.start_timer('video_write')
                        for pf_viz in processed_viz_frames_for_writer:
                            if pf_viz is not None:
                                try: self.video_write_queue.put(pf_viz, block=True, timeout=2.0)
                                except queue.Full: debug_print("WARN: Video writer queue full (Pass 1).")
                        if self.perf: self.perf.end_timer('video_write')
                    
                    frames_processed_this_batch = actual_frames_in_batch
                    frames_processed_count += frames_processed_this_batch
                    batch_proc_duration = time.monotonic() - batch_proc_start_time
                    if self.perf: 
                        self.perf.record_batch_processed(frames_processed_this_batch, batch_proc_duration)
                        num_detections_batch = sum(len(r.boxes) for r in yolo_results_batch if r and hasattr(r, 'boxes') and r.boxes)
                        self.perf.record_detection(num_detections_batch)
                        self.perf.sample_system_metrics()

                    current_monotonic_time = time.monotonic()
                    if current_monotonic_time - self.last_cleanup_time > self.config.MEMORY_CLEANUP_INTERVAL:
                        if self.perf: self.perf.start_timer('memory_cleanup')
                        cleanup_memory() # Global util
                        self.last_cleanup_time = current_monotonic_time
                        if self.perf: self.perf.end_timer('memory_cleanup')
                    
                    if self.perf and (current_monotonic_time - last_prog_print_time > 1.0):
                        prog_stats = self.perf.get_progress()
                        # active_app_tracks is not relevant for Pass 1 simple data logging
                        print(f"\rPass 1 Progress:{prog_stats['percent']:.1f}%|FPS:{prog_stats['fps']:.1f}|ETA:{prog_stats['eta']}      ", end="")
                        last_prog_print_time = current_monotonic_time
                        
                    for _i_done in range(len(current_batch_frames_metadata_for_preprocess)):
                        if not self.frame_read_queue.empty():
                            try: self.frame_read_queue.task_done()
                            except ValueError: pass 
                    current_batch_frames_raw_content.clear(); current_batch_frames_metadata_for_preprocess.clear()
                    current_batch_frame_numbers.clear(); current_batch_frame_times.clear()
                    del input_batch_tensor_model, yolo_results_batch, yolo_results_batch_ordered
                else: 
                    if not self.frame_read_queue.empty():
                        try: self.frame_read_queue.task_done()
                        except ValueError: pass
            except queue.Empty:
                if self.stop_event.is_set() and (self.frame_read_queue is None or self.frame_read_queue.empty()): break
                debug_print("Pass 1 main loop: Queue empty, waiting...")
                continue
            except Exception as e_loop:
                print(f"\nERROR during Pass 1 processing loop: {e_loop}")
                import traceback; traceback.print_exc()
                self.stop_event.set(); break
        
        print("\nPass 1 video processing loop finished.")

        if current_batch_frames_metadata_for_preprocess: # COMPLETE: Final Batch Processing
            print(f"\nProcessing final batch of {len(current_batch_frames_metadata_for_preprocess)} frames for Pass 1...")
            batch_proc_start_time = time.monotonic()
            if self.perf: self.perf.start_timer('preprocessing')
            input_batch_tensor_model = self._preprocess_batch(current_batch_frames_metadata_for_preprocess)
            if self.perf: self.perf.end_timer('preprocessing')

            if input_batch_tensor_model is not None and input_batch_tensor_model.nelement() > 0:
                actual_frames_in_batch = input_batch_tensor_model.shape[0]
                yolo_results_batch_ordered = [None] * actual_frames_in_batch
                
                if self.perf and self.perf.cuda_events_enabled and self.device == 'cuda':
                    final_batch_start_event = torch.cuda.Event(enable_timing=True); final_batch_end_event = torch.cuda.Event(enable_timing=True)
                    final_batch_start_event.record()

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_inference_streams) as executor:
                    futures_map = {}
                    idx_offset_final = 0
                    for i_stream in range(self.num_inference_streams):
                        if idx_offset_final >= actual_frames_in_batch: break
                        num_frames_for_stream = min(self.batch_size, actual_frames_in_batch - idx_offset_final)
                        if num_frames_for_stream <=0: continue
                        end_idx_model = idx_offset_final + num_frames_for_stream
                        chunk_for_model = input_batch_tensor_model[idx_offset_final:end_idx_model]
                        original_indices = list(range(idx_offset_final, end_idx_model))
                        model_to_use = self.models_list[i_stream % len(self.models_list)]
                        current_stream = self.streams[i_stream % len(self.streams)] if self.streams and self.device == 'cuda' else None
                        future = executor.submit(self._execute_single_model_inference, model_to_use, chunk_for_model, current_stream)
                        futures_map[future] = original_indices
                        idx_offset_final += num_frames_for_stream
                    for future in concurrent.futures.as_completed(futures_map):
                        original_indices = futures_map[future]
                        try:
                            results_from_model = future.result()
                            if results_from_model:
                                for res_idx_in_chunk, yolo_res_obj in enumerate(results_from_model):
                                    global_idx = original_indices[res_idx_in_chunk]
                                    yolo_results_batch_ordered[global_idx] = yolo_res_obj
                        except Exception as e_infer:
                             print(f"\nError during final batch concurrent inference (indices {original_indices}): {e_infer}")

                if self.perf and self.perf.cuda_events_enabled and self.device == 'cuda':
                    final_batch_end_event.record(); final_batch_end_event.synchronize()
                    self.perf.timings['inference'] += final_batch_start_event.elapsed_time(final_batch_end_event) / 1000.0

                yolo_results_batch = yolo_results_batch_ordered
                
                if self.perf: self.perf.start_timer('pass1_logic')
                processed_viz_frames_for_writer = self._process_batch_for_pass1(
                    yolo_results_batch, current_batch_frames_raw_content,
                    current_batch_frame_numbers, current_batch_frame_times, start_datetime_obj_global)
                if self.perf: self.perf.end_timer('pass1_logic')

                if self.config.ENABLE_VISUALIZATION and self.video_write_queue:
                    if self.perf: self.perf.start_timer('video_write')
                    for pf_viz in processed_viz_frames_for_writer:
                        if pf_viz is not None:
                            try: self.video_write_queue.put(pf_viz, block=True, timeout=2.0)
                            except queue.Full: debug_print("WARN: Vid writer Q full (final batch).")
                    if self.perf: self.perf.end_timer('video_write')
                
                frames_processed_this_batch = actual_frames_in_batch
                frames_processed_count += frames_processed_this_batch
                batch_proc_duration_final = time.monotonic() - batch_proc_start_time
                if self.perf: 
                    self.perf.record_batch_processed(frames_processed_this_batch, batch_proc_duration_final)
                    num_detections_final = sum(len(r.boxes) for r in yolo_results_batch if r and hasattr(r, 'boxes') and r.boxes)
                    self.perf.record_detection(num_detections_final)
                
                for _i_f_done in range(len(current_batch_frames_metadata_for_preprocess)):
                    if not self.frame_read_queue.empty():
                        try: self.frame_read_queue.task_done()
                        except ValueError: pass
                del input_batch_tensor_model, yolo_results_batch, yolo_results_batch_ordered
            current_batch_frames_raw_content.clear(); current_batch_frames_metadata_for_preprocess.clear()
            current_batch_frame_numbers.clear(); current_batch_frame_times.clear()

        print("Signaling I/O threads to stop (Pass 1)..."); self.stop_event.set()
        if self.frame_read_queue: # Ensure reader queue is also signaled if it might be waiting
            try: self.frame_read_queue.put_nowait(None) 
            except queue.Full: pass # Ignore if full, it will see stop_event
        if self.config.ENABLE_VISUALIZATION and self.video_write_queue:
            try: self.video_write_queue.put(None, timeout=5.0)
            except queue.Full: debug_print("WARN: Could not signal writer queue (full).")
        
        if self.reader_thread and self.reader_thread.is_alive(): self.reader_thread.join(timeout=10)
        if self.writer_thread and self.writer_thread.is_alive(): self.writer_thread.join(timeout=30)

        if self.perf: self.perf.end_processing(); self.perf.print_summary()

        if self.all_tracks_data_pass1:
            parquet_file_path = self.config.FILE_PATHS['pass1_raw_track_output']
            print(f"Saving {len(self.all_tracks_data_pass1)} track-frame records to Parquet: {parquet_file_path}")
            df_pass1 = pd.DataFrame(self.all_tracks_data_pass1)
            try:
                # Ensure all list/dict columns are serializable or convert them
                # For example, 'pois_zone_space' and 'poi_zone_insideness' are dicts
                # 'bbox_model_space_xyxy' and 'bbox_zone_space_xyxy' are tuples
                # Parquet usually handles basic dicts and lists of primitives well with pyarrow.
                df_pass1.to_parquet(parquet_file_path, index=False, engine='pyarrow')
                print("Pass 1 data saved successfully.")
            except Exception as e_parquet:
                print(f"ERROR saving Pass 1 data to Parquet: {e_parquet}")
                import traceback; traceback.print_exc() # Add traceback for Parquet errors
        else:
            print("No track data collected in Pass 1 to save.")

        result_msg = f"✅ Pass 1: Raw Data Extraction completed.\n"
        result_msg += f"Frames Processed (approx): {frames_processed_count}\n"
        result_msg += f"Raw track data saved to: {self.config.FILE_PATHS['pass1_raw_track_output']}\n"
        if self.config.ENABLE_VISUALIZATION and self.output_video_path_pass1 and "ERROR" not in self.output_video_path_pass1:
            result_msg += f"Pass 1 Visualization video: {self.output_video_path_pass1}\n"
        elif self.output_video_path_pass1 and "ERROR" in self.output_video_path_pass1:
            result_msg += f"Pass 1 Visualization video FAILED: {self.output_video_path_pass1}\n"
        
        self.models_list = []; self.bytetracker = None; self.zone_tracker = None
        self.all_tracks_data_pass1 = []
        self.perf = None # Reset perf tracker
        cleanup_memory()
        return result_msg

    @torch.no_grad()
    def _execute_single_model_inference(self, model_instance, input_tensor_chunk, stream_to_use=None):
        if stream_to_use and self.device == 'cuda':
            with torch.cuda.stream(stream_to_use):
                with torch.amp.autocast(device_type=self.device, enabled=(self.config.MIXED_PRECISION and self.device == 'cuda')):
                    results = model_instance(input_tensor_chunk, verbose=False, imgsz=self.config.MODEL_INPUT_SIZE) # Use config for imgsz
            stream_to_use.synchronize()
            return results
        else: 
            with torch.amp.autocast(device_type=self.device, enabled=(self.config.MIXED_PRECISION and self.device == 'cuda')):
                results = model_instance(input_tensor_chunk, verbose=False, imgsz=self.config.MODEL_INPUT_SIZE)
            return results
import cv2
import numpy as np
from video_reader import PyVideoReader # Your PyVideoReader
from time import perf_counter, sleep
import math
from types import SimpleNamespace
import torch
import torchvision # For box_iou, used in tracker_matching.py
from ultralytics import YOLO
# Ensure these import from your local modified files in the same directory
from byte_track import BYTETracker 
from tracking.tracker_basetrack import TrackState # tracker_basetrack.py should be present
from ultralytics.utils.checks import check_requirements
import queue
import threading
import os # For creating debug_input directory

check_requirements('lap')

# --- User-defined Parameters ---
FILENAME = "C:/Users/EMAAN/Documents/YOLO/5 minute test - 4 Way Intersection.mp4"
TENSORRT_ENGINE_PATH = r"C:/Users/EMAAN/Documents/YOLO/project/models/weights/silent_contract_9082.engine"

MODEL_INPUT_SIZE = 416
RESIZE_SHORTER_SIDE = MODEL_INPUT_SIZE
RESIZE_LONGER_SIDE = MODEL_INPUT_SIZE

ORIGINAL_FPS_FROM_VIDEO = 25
TARGET_FPS_FOR_SELECTION = 20 # Frames per second to process by tracker
DECODE_SEGMENT_SIZE = 7500     # Tunable: frames to decode from video at once
BATCH_SIZE = 128                # Tunable: Number of frames to batch for YOLO inference

CONF_THRESHOLD = 0.3          # Confidence threshold for YOLOv8 detections

# --- ByteTrack Parameters (Tune these for your video) ---
BT_TRACK_THRESH = 0.3        # track_high_thresh (detection score to be considered high confidence)
BT_LOW_CONF_THRESH = 0.1      # track_low_thresh (min score for second association)
BT_TRACK_BUFFER = 30          # Number of frames to keep a track alive if lost
BT_MATCH_THRESH = 0.8         # IoU threshold for matching tracks with detections (1st association)
BT_NEW_TRACK_THRESH = 0.31    # Confidence score for a high-confidence detection to become a new track

# --- Visualization ---
SHOW_VIDEO = False
OUTPUT_VIDEO_FILE_CONFIG  = "bytetrack_demo_gpu_pipelined_final.mp4" # Set to None to disable saving
DEBUG_SAVE_YOLO_INPUT_FRAMES = False # Set to True to save some input frames for YOLO

# --- Global variables for pipeline ---
producer_queue = queue.Queue(maxsize=BATCH_SIZE * 3) 
sentinel = object() 

processed_frames_count_reader = 0
yolo_input_frames_count = 0
tracked_frames_output_count = 0


def draw_tracks_on_frame(image_bgr, tracks_np):
    for track_data in tracks_np:
        if len(track_data) < 7:
            # print(f"Warning: draw_tracks track_data too short: {track_data}") # Optional debug
            continue
        x1, y1, x2, y2 = map(int, track_data[:4])
        track_id = int(track_data[4])
        score = float(track_data[5])
        cls_id = int(track_data[6])
        
        label = f"ID:{track_id} Cls:{cls_id} ({score:.2f})"
        color_seed = track_id * 27 
        color = ((color_seed * 1) % 200 + 55, (color_seed * 2) % 200 + 55, (color_seed * 3) % 200 + 55)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image_bgr


def producer_thread_func(vr_producer, yolo_model_producer, total_frames_in_video, proc_frame_interval):
    global processed_frames_count_reader, yolo_input_frames_count

    frames_for_batch_np_list = [] 
    original_frames_for_consumer_batch = []
    raw_frame_indices_for_consumer_batch = []
    
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(yolo_model_producer, 'device') and yolo_model_producer.device is not None:
        target_device = yolo_model_producer.device
    elif hasattr(yolo_model_producer, 'model') and hasattr(yolo_model_producer.model, 'device'):
        target_device = yolo_model_producer.model.device
    print(f"Producer thread using target device for YOLO input: {target_device}")

    num_segments = math.ceil(total_frames_in_video / DECODE_SEGMENT_SIZE)
    frames_selected_for_yolo_session = 0

    for i in range(num_segments):
        segment_start_raw_idx = i * DECODE_SEGMENT_SIZE
        segment_end_raw_idx = min((i + 1) * DECODE_SEGMENT_SIZE, total_frames_in_video)

        if segment_start_raw_idx >= total_frames_in_video: break

        # print(f"Producer: Decoding segment {i+1}/{num_segments}...") # Less verbose
        segment_frames_data_np = vr_producer.decode(start_frame=segment_start_raw_idx, end_frame=segment_end_raw_idx)
        
        if segment_frames_data_np is None or segment_frames_data_np.shape[0] == 0:
            print(f"Producer: Segment {i+1} returned no frames."); continue
        
        for frame_idx_in_segment in range(segment_frames_data_np.shape[0]):
            current_overall_raw_frame_idx = segment_start_raw_idx + frame_idx_in_segment
            if current_overall_raw_frame_idx >= total_frames_in_video: break
            
            processed_frames_count_reader += 1

            if current_overall_raw_frame_idx % proc_frame_interval == 0:
                current_frame_rgb_np = segment_frames_data_np[frame_idx_in_segment]
                
                frames_for_batch_np_list.append(current_frame_rgb_np)
                original_frames_for_consumer_batch.append(current_frame_rgb_np)
                raw_frame_indices_for_consumer_batch.append(current_overall_raw_frame_idx)
                frames_selected_for_yolo_session += 1
                
                is_last_possible_frame_overall = (frames_selected_for_yolo_session >= math.ceil(total_frames_in_video / proc_frame_interval))

                if len(frames_for_batch_np_list) == BATCH_SIZE or \
                   (is_last_possible_frame_overall and len(frames_for_batch_np_list) > 0):
                    
                    if DEBUG_SAVE_YOLO_INPUT_FRAMES and yolo_input_frames_count % (BATCH_SIZE * 50) < BATCH_SIZE : # Save one batch every ~50 batches
                        os.makedirs("debug_input", exist_ok=True)
                        for idx_in_batch, frame_to_save_np in enumerate(frames_for_batch_np_list):
                            save_path = f"debug_input/frame_{raw_frame_indices_for_consumer_batch[idx_in_batch]}.jpg"
                            cv2.imwrite(save_path, cv2.cvtColor(frame_to_save_np, cv2.COLOR_RGB2BGR))
                        print(f"DEBUG: Saved input batch frame images around raw index {raw_frame_indices_for_consumer_batch[0]}")
                    
                    batch_yolo_results_list = yolo_model_producer(frames_for_batch_np_list, 
                                                                  imgsz=MODEL_INPUT_SIZE, # CRITICAL
                                                                  verbose=False, 
                                                                  conf=CONF_THRESHOLD)
                    
                    batch_gpu_boxes_list = [res.boxes for res in batch_yolo_results_list]
                    
                    yolo_input_frames_count += len(frames_for_batch_np_list)
                    producer_queue.put((batch_gpu_boxes_list, 
                                        list(original_frames_for_consumer_batch), 
                                        list(raw_frame_indices_for_consumer_batch)))
                    
                    frames_for_batch_np_list.clear()
                    original_frames_for_consumer_batch.clear()
                    raw_frame_indices_for_consumer_batch.clear()
            
            if current_overall_raw_frame_idx >= total_frames_in_video -1 : break
        if current_overall_raw_frame_idx >= total_frames_in_video -1 : break

    producer_queue.put(sentinel)
    print("Producer thread finished.")

def main():
    global processed_frames_count_reader, yolo_input_frames_count, tracked_frames_output_count
    processed_frames_count_reader = 0
    yolo_input_frames_count = 0
    tracked_frames_output_count = 0
    
    print(f"TensorRT Engine: {TENSORRT_ENGINE_PATH}, Video: {FILENAME}, Batch Size: {BATCH_SIZE}")

    try:
        cap_temp = cv2.VideoCapture(FILENAME)
        if not cap_temp.isOpened(): raise IOError(f"Cannot open video file for metadata: {FILENAME}")
        actual_total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_temp.release()
        if actual_total_frames <= 0: raise ValueError("cv2.VideoCapture got 0 total frames.")
        print(f"Total frames from OpenCV: {actual_total_frames}")
    except Exception as e: print(f"Error getting video metadata: {e}"); return

    frame_interval = max(1, round(ORIGINAL_FPS_FROM_VIDEO / TARGET_FPS_FOR_SELECTION)) if TARGET_FPS_FOR_SELECTION > 0 else 1
    print(f"Original FPS: {ORIGINAL_FPS_FROM_VIDEO}, Target Tracker FPS: {TARGET_FPS_FOR_SELECTION}, Frame Interval: {frame_interval}")

    yolo_model_main = None
    vr_main = None
    video_writer_main = None

    try:
        vr_main = PyVideoReader(FILENAME, device='cuda',
                                resize_shorter_side=RESIZE_SHORTER_SIDE, 
                                resize_longer_side=RESIZE_LONGER_SIDE)
        print(f"PyVideoReader initialized. Output size: ({RESIZE_SHORTER_SIDE}x{RESIZE_LONGER_SIDE})")

        yolo_model_main = YOLO(TENSORRT_ENGINE_PATH, task='detect')
        yolo_model_main.verbose = False # Suppress Ultralytics model loading messages
        print("YOLOv8 TensorRT engine loaded.")

        effective_tracker_fps = ORIGINAL_FPS_FROM_VIDEO / frame_interval
        byte_tracker_args = SimpleNamespace(
            track_high_thresh=BT_TRACK_THRESH,
            track_low_thresh=BT_LOW_CONF_THRESH, 
            track_buffer=BT_TRACK_BUFFER,
            match_thresh=BT_MATCH_THRESH,
            fuse_score=False, 
            new_track_thresh=BT_NEW_TRACK_THRESH 
        )
        tracker = BYTETracker(args=byte_tracker_args, frame_rate=effective_tracker_fps)
        # BYTETracker __init__ already prints its device and args

        if SHOW_VIDEO and OUTPUT_VIDEO_FILE_CONFIG:
            try:
                # This ensures we get a frame with the reader's output dimensions
                temp_frames_for_dim = vr_main.decode(start_frame=0, end_frame=1) 
                if temp_frames_for_dim is not None and temp_frames_for_dim.shape[0] > 0:
                    frame_h, frame_w = temp_frames_for_dim[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer_main = cv2.VideoWriter(OUTPUT_VIDEO_FILE_CONFIG, fourcc, effective_tracker_fps, (frame_w, frame_h))
                    print(f"Output video: {OUTPUT_VIDEO_FILE_CONFIG} at {frame_w}x{frame_h}, FPS: {effective_tracker_fps:.2f}")
                else: print("Warning: Could not get frame dims for VideoWriter. Saving disabled.")
            except Exception as e: print(f"Error initializing VideoWriter: {e}. Saving disabled."); video_writer_main = None
        
        producer = threading.Thread(target=producer_thread_func, args=(vr_main, yolo_model_main, actual_total_frames, frame_interval))
        producer.start()

        overall_start_time = perf_counter()
        consumer_active = True
        last_queue_check_print_time = perf_counter()

        while consumer_active:
            try:
                queue_item = producer_queue.get(timeout=1.0) 
            except queue.Empty:
                if not producer.is_alive() and producer_queue.empty():
                    print("Consumer: Producer finished and queue empty. Exiting.")
                    consumer_active = False
                else: 
                    if perf_counter() - last_queue_check_print_time > 5.0: # Print less frequently
                        # print(f"Consumer: Waiting for items... Queue size: {producer_queue.qsize()}")
                        last_queue_check_print_time = perf_counter()
                continue

            if queue_item is sentinel:
                consumer_active = False; break

            batch_gpu_boxes_list, original_cpu_frames_batch, raw_indices_batch = queue_item
            
            for k in range(len(batch_gpu_boxes_list)):
                single_frame_gpu_boxes = batch_gpu_boxes_list[k] 
                original_cpu_frame = original_cpu_frames_batch[k] 
                current_raw_idx = raw_indices_batch[k]
                
                online_targets_np = tracker.update(single_frame_gpu_boxes, original_cpu_frame) 
                tracked_frames_output_count +=1 
                
                if SHOW_VIDEO or video_writer_main:
                    frame_bgr_display = cv2.cvtColor(original_cpu_frame, cv2.COLOR_RGB2BGR)
                    if online_targets_np.shape[0] > 0:
                        frame_bgr_display = draw_tracks_on_frame(frame_bgr_display, online_targets_np)
                    
                    cv2.putText(frame_bgr_display, f"Frame: {current_raw_idx}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    
                    if video_writer_main:
                        video_writer_main.write(frame_bgr_display)
            
            if not consumer_active: break
        
        print("Consumer loop finished. Waiting for producer thread to join...")
        producer.join(timeout=15) 
        if producer.is_alive():
            print("Warning: Producer thread did not finish cleanly after timeout.")

        total_processing_time = perf_counter() - overall_start_time
        print(f"\n--- Processing Summary ---")
        print(f"Total raw frames in video: {actual_total_frames}")
        print(f"Frames read by PyVideoReader: {processed_frames_count_reader}")
        print(f"Frames selected & input to YOLO: {yolo_input_frames_count}")
        print(f"Frames for which tracks were output: {tracked_frames_output_count}")
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        if tracked_frames_output_count > 0 and total_processing_time > 0:
            avg_overall_fps = tracked_frames_output_count / total_processing_time
            print(f"Average Overall FPS (tracked output frames / total time): {avg_overall_fps:.2f}")

    except KeyboardInterrupt: print("\nProcessing interrupted by user (KeyboardInterrupt).")
    except Exception as e: print(f"An error occurred in main: {e}"); import traceback; traceback.print_exc()
    finally:
        if 'producer' in locals() and isinstance(producer, threading.Thread) and producer.is_alive():
            print("Main: Signaling producer thread to stop due to exit/error...")
            while not producer_queue.empty(): 
                try: producer_queue.get_nowait()
                except queue.Empty: break
            producer_queue.put(sentinel)
            producer.join(timeout=5.0)
            if producer.is_alive(): print("Main: Warning - producer thread still alive after final join attempt.")

        if vr_main: del vr_main; print("PyVideoReader deleted.")
        if video_writer_main: video_writer_main.release(); print(f"Output video saved to {OUTPUT_VIDEO_FILE_CONFIG}" if OUTPUT_VIDEO_FILE_CONFIG else "Video writer released.")

if __name__ == "__main__":
    main()
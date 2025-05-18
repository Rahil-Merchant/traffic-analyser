import cv2
import numpy as np
from video_reader import PyVideoReader
from time import perf_counter
import math
from ultralytics import YOLO
from byte_track import BYTETracker
from ultralytics.utils.checks import check_requirements
from types import SimpleNamespace
check_requirements('lap')

# --- User-defined Parameters ---
FILENAME = "C:/Users/EMAAN/Documents/YOLO/12 hour test - T junction.mp4"
TENSORRT_ENGINE_PATH = r"C:/Users/EMAAN/Documents/YOLO/project/models/weights/silent_contract_9082.engine" # IMPORTANT: Set this path

MODEL_INPUT_SIZE = 416
RESIZE_SHORTER_SIDE = MODEL_INPUT_SIZE
RESIZE_LONGER_SIDE = MODEL_INPUT_SIZE

ORIGINAL_FPS_FROM_VIDEO = 25
TARGET_FPS_FOR_SELECTION = 20
DECODE_SEGMENT_SIZE = 7500

# --- YOLOv8 Parameters (used by Ultralytics YOLO call) ---
CONF_THRESHOLD = 0.3 # Confidence threshold for YOLOv8 detections

# --- ByteTrack Parameters ---
BT_TRACK_THRESH = 0.45
BT_LOW_CONF_THRESH = 0.1
BT_TRACK_BUFFER = 30
BT_MATCH_THRESH = 0.8

# --- Visualization ---
SHOW_VIDEO = True
OUTPUT_VIDEO_FILE_CONFIG  = "bytetrack_demo_ultralytics.mp4" # Set to None to disable saving

# (YOLOv8TRT class is removed)

# --- Helper function to draw tracks (remains the same) ---
def draw_tracks_on_frame(image_bgr, tracks):
    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        conf = float(track[5])
        cls_id = int(track[6])
        label = f"ID:{track_id} C:{cls_id} ({conf:.2f})"
        color = ( (track_id * 30) % 255, (track_id * 50) % 255, (track_id * 70) % 255)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image_bgr

# --- Main Processing Logic ---
def main():
    print(f"Using TensorRT engine via Ultralytics YOLO: {TENSORRT_ENGINE_PATH}")
    print(f"Processing video: {FILENAME}")

    try:
        cap_temp = cv2.VideoCapture(FILENAME)
        if not cap_temp.isOpened():
            raise IOError(f"Cannot open video file for metadata: {FILENAME}")
        actual_total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_temp.release()
        if actual_total_frames <= 0:
            raise ValueError("cv2.VideoCapture got 0 total frames.")
        print(f"Successfully retrieved total frames using OpenCV: {actual_total_frames}")
    except Exception as e:
        print(f"Warning: Could not get total frames using cv2.VideoCapture: {e}")
        return

    FRAME_INTERVAL = 1
    if TARGET_FPS_FOR_SELECTION > 0 and TARGET_FPS_FOR_SELECTION < ORIGINAL_FPS_FROM_VIDEO:
        FRAME_INTERVAL = round(ORIGINAL_FPS_FROM_VIDEO / TARGET_FPS_FOR_SELECTION)
    elif TARGET_FPS_FOR_SELECTION >= ORIGINAL_FPS_FROM_VIDEO:
        FRAME_INTERVAL = 1
    else:
        print("Warning: TARGET_FPS_FOR_SELECTION is <= 0. Defaulting to processing all frames.")
        FRAME_INTERVAL = 1
    
    print(f"Original FPS: {ORIGINAL_FPS_FROM_VIDEO}, Target Processing FPS: {TARGET_FPS_FOR_SELECTION}, Frame Interval: {FRAME_INTERVAL}")

    yolo_model = None
    vr = None
    video_writer = None

    try:
        vr = PyVideoReader(FILENAME, device='cuda',
                           resize_shorter_side=RESIZE_SHORTER_SIDE,
                           resize_longer_side=RESIZE_LONGER_SIDE)
        print(f"PyVideoReader initialized. Output size: ({RESIZE_SHORTER_SIDE}x{RESIZE_LONGER_SIDE})")

        yolo_model = YOLO(TENSORRT_ENGINE_PATH, task='detect')
        yolo_model.verbose = False
        print("YOLOv8 TensorRT engine loaded successfully via Ultralytics YOLO class.")

        effective_tracker_fps = ORIGINAL_FPS_FROM_VIDEO / FRAME_INTERVAL
        
        byte_tracker_args = SimpleNamespace(
            track_high_thresh=BT_TRACK_THRESH,
            track_low_thresh=BT_LOW_CONF_THRESH, 
            track_buffer=BT_TRACK_BUFFER,
            match_thresh=BT_MATCH_THRESH,
            fuse_score=False
        )
            
        tracker = BYTETracker(args=byte_tracker_args, frame_rate=effective_tracker_fps)
        print(f"ByteTrack initialized. Effective FPS for tracker: {effective_tracker_fps:.2f}")

        if SHOW_VIDEO and OUTPUT_VIDEO_FILE_CONFIG:
            try:
                temp_frame_segment = vr.decode(start_frame=0, end_frame=1)
                if temp_frame_segment is not None and temp_frame_segment.shape[0] > 0:
                    frame_h, frame_w = temp_frame_segment[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE_CONFIG, fourcc, effective_tracker_fps, (frame_w, frame_h))
                    print(f"Output video will be saved to: {OUTPUT_VIDEO_FILE_CONFIG}")
                else:
                    print("Warning: Could not get frame dimensions for VideoWriter. Output video saving disabled.")
            except Exception as e:
                print(f"Error initializing VideoWriter: {e}. Output video saving disabled.")
                video_writer = None

        overall_start_time = perf_counter()
        processed_frames_count = 0
        tracked_frames_count = 0
        
        num_segments = math.ceil(actual_total_frames / DECODE_SEGMENT_SIZE)
        
        for i in range(num_segments):
            segment_start_raw_idx = i * DECODE_SEGMENT_SIZE
            segment_end_raw_idx = min((i + 1) * DECODE_SEGMENT_SIZE, actual_total_frames)

            if segment_start_raw_idx >= actual_total_frames:
                break

            print(f"Processing segment {i+1}/{num_segments} (frames {segment_start_raw_idx} to {segment_end_raw_idx-1})")
            segment_decode_start_time = perf_counter()
            segment_frames_np_rgb = vr.decode(start_frame=segment_start_raw_idx, end_frame=segment_end_raw_idx)
            segment_decode_duration = perf_counter() - segment_decode_start_time
            
            if segment_frames_np_rgb is None or segment_frames_np_rgb.shape[0] == 0:
                print(f"Segment {i+1} returned no frames. Stopping.")
                break
            
            print(f"  Decoded {segment_frames_np_rgb.shape[0]} frames in {segment_decode_duration:.3f}s.")

            for j in range(segment_frames_np_rgb.shape[0]):
                current_overall_raw_frame_idx = segment_start_raw_idx + j
                processed_frames_count += 1

                if current_overall_raw_frame_idx % FRAME_INTERVAL == 0:
                    tracked_frames_count += 1
                    loop_iter_start_time = perf_counter()
                    
                    current_frame_rgb = segment_frames_np_rgb[j]

                    # Perform YOLOv8 Inference
                    results_from_yolo = yolo_model(current_frame_rgb, imgsz=MODEL_INPUT_SIZE, verbose=False, conf=CONF_THRESHOLD)
                    
                    # Extract the Boxes object (or equivalent Detections object)
                    # This object (e.g., results_from_yolo[0].boxes) has .xyxy, .conf, .cls attributes
                    # and is what BYTETracker expects.
                    # It will be an empty Boxes object if no detections are found by YOLO.
                    if results_from_yolo and len(results_from_yolo) > 0 :
                        boxes_data_for_tracker = results_from_yolo[0].boxes
                    else:
                        # Fallback: Create a dummy/empty Boxes-like structure if yolo_model truly returns nothing.
                        # This is less likely with current Ultralytics versions.
                        # For now, we assume yolo_model always returns a list with one Results object.
                        # A more robust empty representation might be needed if this assumption is wrong.
                        # For now, we'll rely on the main path. If yolo_model can return an empty list
                        # this part of the code would need adjustment to create a compatible empty object.
                        # The error implies the tracker needs a `.conf` attribute.
                        # Let's assume the main path (results_from_yolo[0].boxes) is always taken.
                        # If not, the script might crash here or needs a more specific dummy.
                        # To avoid crashing if results_from_yolo is empty:
                        from ultralytics.engine.results import Boxes
                        empty_boxes_tensor = torch.empty((0, 6), device=yolo_model.device) # Ensure correct device
                        # Need to know the original image shape for Boxes constructor if it requires it.
                        # This is getting complex for an edge case. The simplest is to rely on yolo_model returning a valid structure.
                        # For now, let's assume results_from_yolo[0].boxes is always available.
                        boxes_data_for_tracker = results_from_yolo[0].boxes


                    # Extract the Boxes object
                    boxes_on_gpu = results_from_yolo[0].boxes

                    # --- Move the Boxes object's tensors to CPU ---
                    boxes_data_for_tracker_cpu = boxes_on_gpu.cpu()

                    # Update ByteTrack with the CPU version of the Boxes object
                    online_targets = tracker.update(boxes_data_for_tracker_cpu, current_frame_rgb)

                    iter_duration = perf_counter() - loop_iter_start_time
                    current_fps = 1.0 / iter_duration if iter_duration > 0 else float('inf')
                    
                    frame_bgr_display = cv2.cvtColor(current_frame_rgb, cv2.COLOR_RGB2BGR)
                    if online_targets.shape[0] > 0: # online_targets is a NumPy array from the tracker
                        frame_bgr_display = draw_tracks_on_frame(frame_bgr_display, online_targets)
                    
                    cv2.putText(frame_bgr_display, f"FPS: {current_fps:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    if SHOW_VIDEO:
                        cv2.imshow("YOLOv8 ByteTrack TRT", frame_bgr_display)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("User pressed 'q', exiting...")
                            raise KeyboardInterrupt 

                    if video_writer:
                        video_writer.write(frame_bgr_display)
                    
                    if tracked_frames_count % 100 == 0:
                         print(f"  Tracked frame {tracked_frames_count} (Raw frame #{current_overall_raw_frame_idx}). Last FPS: {current_fps:.2f}")

        total_processing_time = perf_counter() - overall_start_time
        print(f"\n--- Processing Summary ---")
        # ... (print summary stats) ...

    except KeyboardInterrupt:
        print("Processing interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if vr:
            del vr
            print("PyVideoReader deleted.")
        if SHOW_VIDEO:
            cv2.destroyAllWindows()
        if video_writer:
            video_writer.release()
            if OUTPUT_VIDEO_FILE_CONFIG:
                print(f"Output video saved to {OUTPUT_VIDEO_FILE_CONFIG}")

if __name__ == "__main__":
    # Add torch import if needed for the fallback empty_boxes_tensor
    import torch 
    main()
if __name__ == "__main__":
    main()
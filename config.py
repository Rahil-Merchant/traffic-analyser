# project/config.py
import os
from datetime import datetime

LINE_MODE = 'hardcoded'
ENABLE_VISUALIZATION = True

_now = datetime.now()
START_DATE = _now.strftime("%y%m%d")
START_TIME = _now.strftime("%H%M%S") + f"{_now.microsecond // 1000:03d}"

DEFAULT_PRIMARY_DIRECTION = 'south'

# --- Core Settings ---
TARGET_FPS = 25
BASE_OUTPUT_DIR = r"C:\Users\swest\Desktop\traffic-capturing\traffic-reporting\output"
VIDEO_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
REPORT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "reports")

PIPELINE_CONTROL = {
    'run_pass1_extraction': True,
    'run_pass2_path_analysis': True, 
    'run_pass3_fragment_stitching': True, # For next step
    'run_pass4_final_report': False,
}

FILE_PATHS = {
    'pass1_raw_track_output': os.path.join(BASE_OUTPUT_DIR, "intermediate", "pass1_raw_tracks.parquet"),
    'pass1_image_patch_output_dir': os.path.join(BASE_OUTPUT_DIR, "intermediate", "image_patches"),
    'pass2_initial_paths_output': os.path.join(BASE_OUTPUT_DIR, "intermediate", "pass2_initial_paths.parquet"),
    'pass2_fragmented_tracks_output': os.path.join(BASE_OUTPUT_DIR, "intermediate", "pass2_fragmented_tracks.parquet"),
    'pass3_stitched_paths_output': os.path.join(BASE_OUTPUT_DIR, "intermediate", "pass3_stitched_paths_and_segments.parquet"),
    # ...
}

os.makedirs(os.path.join(BASE_OUTPUT_DIR, "intermediate"), exist_ok=True)
os.makedirs(FILE_PATHS['pass1_image_patch_output_dir'], exist_ok=True) # Create if saving patches

# --- NEW: Points-of-Interest (PoI) Strategies ---
# Define how PoIs are calculated for different vehicle classes.
# Methods can be: 'bottom_center', 'geometric_center', 'leading_edge_horizontal', 'trailing_edge_horizontal', 'leading_edge_vertical', 'trailing_edge_vertical'
# More complex PoIs might need custom functions later.
POI_STRATEGIES = {
    'Light Vehicle': ['bottom_center', 'geometric_center'],
    'Motorcycle': ['bottom_center', 'geometric_center'],
    'Minibus Taxi': ['bottom_center', 'geometric_center', 'leading_edge_horizontal'], # Assuming primarily horizontal movement relative to camera
    'Short Truck': ['bottom_center', 'geometric_center', 'leading_edge_horizontal', 'trailing_edge_horizontal'],
    'Medium Truck': ['bottom_center', 'geometric_center', 'leading_edge_horizontal', 'trailing_edge_horizontal'],
    'Long Truck': ['bottom_center', 'geometric_center', 'leading_edge_horizontal', 'trailing_edge_horizontal'],
    'Bus': ['bottom_center', 'geometric_center', 'leading_edge_horizontal', 'trailing_edge_horizontal'],
    'Pedestrian': ['bottom_center'], # Or just geometric_center
    'Cyclist': ['bottom_center', 'geometric_center'],
    'Animal drawn vehicle': ['bottom_center', 'geometric_center', 'leading_edge_horizontal'],
    'Person with wheel barrow': ['bottom_center', 'geometric_center'],
    'default': ['bottom_center', 'geometric_center'] # Fallback
}

# --- NEW: Re-ID Related (Placeholders for now, Pass 3 will use these) ---
REID_MODEL_SETTINGS = {
    'model_path': "C:/path/to/your/vehicle_reid_model.pt", # Or .onnx, .engine etc.
    'input_size': (256, 128), # Example: (Height, Width)
    'batch_size': 32,
    'feature_dim': 512, # Example output embedding size
    # 'model_type': 'pytorch', # Optional: if you need different loading logic
    'patch_aggregation_method_reid': 'average_last_k', # 'average_last_k', 'use_best_quality_last_k'
    'k_frames_for_reid_patch_aggregation': 5,
}

FRAGMENT_STITCHING_PARAMS = {
    'max_temporal_gap_seconds_for_reid': 5.0,
    'max_spatial_distance_pixels_for_reid': 250, # Distance between end_bbox of F1 and start_bbox of F2
    'min_reid_similarity_stitch': 0.80,         # e.g., Cosine similarity
    'min_motion_consistency_score_stitch': 0.3, # Optional, if using motion score
    'stitch_score_weights': {'reid': 0.7, 'motion': 0.3}, # If combining
    'final_stitch_score_threshold': 0.75, # If combining
    # Heuristic for motion prediction if needed for spatial distance:
    'assume_max_avg_speed_pixels_per_sec': 100 # For simple spatial gating
}

PASS1_SAVE_IMAGE_PATCHES = True # Set to False if you generate Re-ID features directly in Pass 1 later

PATH_ANALYSIS_PARAMS = {
    'poi_quorum_method': 'min_count',  # or 'percentage'
    'poi_quorum_value_min_count': 1,   # If min_count, e.g., at least 1 PoI must be in zone
                                       # For 'percentage', e.g., 0.51 (51% of PoIs)
    'zone_occupancy_smoothing_window_frames': 5, # e.g., 5 frames smoothing window
    'min_frames_for_stable_zone_stay': 5,      # e.g., must be in a zone for 5 frames to count as a stay
}


# --- Model Configuration ---
MODEL_PATH = r"C:\Users\swest\Desktop\traffic-capturing\traffic-reporting\models\weights\silent_contract_9082_960px_bs192_trt.engine"
NUM_INFERENCE_STREAMS = 1
CONF_THRESHOLD = 0.05 # Keep SLIGHTLY LOWER CONF to potentially keep weaker detections
IOU_THRESHOLD = 0.6   # Keep slightly higher
MODEL_INPUT_SIZE = 960 # Keep smaller size for performance

# --- Tracking Configuration --- 
BYTETRACK_MIN_CONF = 0.1    # Minimum detection confidence for ByteTrack to consider
BYTETRACK_TRACK_THRESH = 0.35   # Detection confidence threshold for high-confidence matching
BYTETRACK_MATCH_THRESH = 0.8    # IoU threshold for matching detections to tracks
BYTETRACK_TRACK_BUFFER_SECONDS = 15 # Seconds to keep a track alive if unmatched after it has stopped being associated with any new detections


#*############## Video 1 #################
# --- Input Settings (primarily for hardcoded mode) ---
INPUT_VIDEO_PATH = r"C:\Users\swest\Desktop\traffic-capturing\sample_videos\5 minute test - 4 Way Intersection.mp4"

# --- Line & Zone Definitions ---
LINE_POINTS = {
    'north': [(282, 339), 
              (422, 312),
              (371, 240),
              (272, 264)],
    'south': [(290, 446), 
              (605, 335),
              (619, 441),
              (470, 466),
              (361, 476)],
    'east': [(512, 306),  
             (596, 322),
             (628, 214),
             (525, 175)],
    'west': [(252, 378),  
             (244, 442),
             ( 57, 466),
             ( 53, 397)]
}
FRAME_HEIGHT = MODEL_INPUT_SIZE
FRAME_WIDTH = round(MODEL_INPUT_SIZE*(4/3))
FRAME_HEIGHT_VIZ = 480
FRAME_WIDTH_VIZ = 640

scale_x = FRAME_WIDTH / 640
scale_y = FRAME_HEIGHT / 480

LINE_POINTS = {
    'north': [(int(x * scale_x), int(y * scale_y)) for (x, y) in LINE_POINTS['north']],
    'south': [(int(x * scale_x), int(y * scale_y)) for (x, y) in LINE_POINTS['south']],
    'east':  [(int(x * scale_x), int(y * scale_y)) for (x, y) in LINE_POINTS['east']],
    'west':  [(int(x * scale_x), int(y * scale_y)) for (x, y) in LINE_POINTS['west']],
}
#######################

# # *############## Video 2 #################
# --- Input Settings (primarily for hardcoded mode) ---
# INPUT_VIDEO_PATH = r"C:\Users\swest\Desktop\traffic-capturing\sample_videos\5 min test - Large FOV.mp4"

# # --- Line & Zone Definitions ---
# LINE_POINTS = {'north': [[699, 529],
#        [451, 486],
#        [455, 422],
#        [610, 455]],
#        'south': [[ 425,  573],
#        [ 881,  603],
#        [1254,  765],
#        [ 398,  819]],
#        'west': [[405, 529],
#        [416, 457],
#        [152, 394],
#        [ 49, 368],
#        [ 51, 472],
#        [285, 507]],
#        'east': [[843, 590],
#        [721, 544],
#        [717, 536],
#        [828, 527],
#        [889, 562],
#        [909, 590]]}

# FRAME_HEIGHT = MODEL_INPUT_SIZE
# FRAME_WIDTH = round(MODEL_INPUT_SIZE*(4/3))
# FRAME_HEIGHT_VIZ = 800
# FRAME_WIDTH_VIZ = 960

# scale_x = 1
# scale_y = 1

# LINE_POINTS = {
#     'north': [(int(x * scale_x), int(y * scale_y)) for (x, y) in LINE_POINTS['north']],
#     'south': [(int(x * scale_x), int(y * scale_y)) for (x, y) in LINE_POINTS['south']],
#     'east':  [(int(x * scale_x), int(y * scale_y)) for (x, y) in LINE_POINTS['east']],
#     'west':  [(int(x * scale_x), int(y * scale_y)) for (x, y) in LINE_POINTS['west']],
# }
######################


VALID_MOVEMENTS = { 'north': ['south','east','west'], 'south': ['north','east','west'], 'east': ['west','north','south'], 'west': ['east','north','south'] }

# --- Hardware Optimizations ---
OPTIMAL_BATCH_SIZE = 128
if MODEL_INPUT_SIZE<=420:
    OPTIMAL_BATCH_SIZE = 128
elif MODEL_INPUT_SIZE<=640:
    OPTIMAL_BATCH_SIZE = 64
elif MODEL_INPUT_SIZE<=960:
    OPTIMAL_BATCH_SIZE = 32
else:
    OPTIMAL_BATCH_SIZE = 16
    
OPTIMAL_BATCH_SIZE = 192
USE_ADVANCED_VIDEO_READER = True 
PYVIDEOREADER_DECODE_SEGMENT_SIZE = 7600

_cpu_count = os.cpu_count()
THREAD_COUNT = min(64, _cpu_count * 2)
MIXED_PRECISION = True
PARALLEL_STREAMS = 32

# Timeout durations - In seconds
VEHICLE_TIMEOUTS = {
    'Light Vehicle': 400, 'Motorcycle': 400, 'Minibus Taxi': 500,
    'Short Truck': 500, 'Medium Truck': 550, 'Long Truck': 600,
    'Bus': 550, 'Pedestrian': 650, 'Cyclist': 650,
    'Animal drawn vehicle': 650, 'Person with wheel barrow': 650,
    'default': 400
}

# --- Reporting Configuration ---
VEHICLE_ID_MAP = { 'Light Vehicle': '1', 'Motorcycle': '11', 'Minibus Taxi': '13', 'Short Truck': '2T,2', 'Medium Truck': '2T,3', 'Long Truck': '2T,4', 'Bus': '2B', 'Pedestrian': '9P', 'Cyclist': '9C', 'Animal drawn vehicle': '95', 'Person with wheel barrow': '100' }
ALL_VEHICLE_TYPES = list(VEHICLE_ID_MAP.keys())
REPORTING_DIRECTIONS = { 'From South': ['To North', 'To West', 'To East'], 'From North': ['To South', 'To West', 'To East'], 'From West': ['To North', 'To South', 'To East'], 'From East': ['To North', 'To South', 'To West'] }

# --- Performance & Debugging ---
ENABLE_DETAILED_PERFORMANCE_METRICS = True
DEBUG_MODE = False # Keep False for performance runs
PERFORMANCE_SAMPLE_INTERVAL = 15.0   # In seconds
MEMORY_CLEANUP_INTERVAL = 300   # In seconds

# --- Gradio ---
GRADIO_SERVER_PORT = 7862
GRADIO_SHARE = False

# --- Print Confirmation ---
print(f"--- Configuration Loaded (Refactoring for Multi-Pass) ---")
print(f"--------------------------------------------------------------------")
print(f"--- Configuration Loaded (Tuned for Exit Accuracy) ---")
print(f"Line Definition Mode: {LINE_MODE.upper()}")
if LINE_MODE == 'hardcoded': print(f"Input Video: {INPUT_VIDEO_PATH}")
print(f"Start Date/Time: {START_DATE} / {START_TIME}")
print(f"Using Model: {MODEL_PATH} (Conf: {CONF_THRESHOLD}, IoU: {IOU_THRESHOLD})")
print(f"Model Input Size: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}")
if LINE_MODE == 'hardcoded': print("Using Hardcoded Lines (config.py)")
else: print("Using Interactive Lines (Gradio)")
print(f"Target FPS: {TARGET_FPS}, Batch Size: {OPTIMAL_BATCH_SIZE}, Threads: {THREAD_COUNT}")
print(f"Mixed Precision: {MIXED_PRECISION}, CUDA Streams: {PARALLEL_STREAMS}")
print(f"Pass 1 Output (Raw Tracks): {FILE_PATHS['pass1_raw_track_output']}")
if PASS1_SAVE_IMAGE_PATCHES:
    print(f"Pass 1 Image Patch Dir: {FILE_PATHS['pass1_image_patch_output_dir']}")
print(f"Detailed Perf Metrics: {ENABLE_DETAILED_PERFORMANCE_METRICS}, Debug Mode: {DEBUG_MODE}")
print(f"--------------------------------------------------------------------")
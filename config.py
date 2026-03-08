# project/config.py
import os
from datetime import datetime

ENABLE_VISUALIZATION = False

_now = datetime.now()
START_DATE = _now.strftime("%y%m%d")
START_TIME = _now.strftime("%H%M%S") + f"{_now.microsecond // 1000:03d}"

DEFAULT_PRIMARY_DIRECTION = 'south'

# --- Core Settings ---
TARGET_FPS = 20
BASE_OUTPUT_DIR = r"C:\Users\EMAAN\Documents\AI_output_videos"
VIDEO_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
REPORT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "reports")

# --- Model Configuration ---
MODEL_PATH = r"C:\Users\EMAAN\Documents\YOLO\project\models\weights\premier_station.engine"
DATA_YAML_PATH = r""
NUM_INFERENCE_STREAMS = 1
CONF_THRESHOLD = 0.05 # Keep SLIGHTLY LOWER CONF to potentially keep weaker detections
IOU_THRESHOLD = 0.6   # Keep slightly higher
MODEL_INPUT_SIZE = 640 # Keep smaller size for performance
TTA = True # Enables test-time augmentation (TTA) for predictions
CROP_MODE = 'letterbox'     # letterbox or stretch

# --- INFERENCE MODE SWITCH ---
USE_PATCHWISE_INFERENCE = False # Set to True to use patch-based, False for standard
PATCHWISE_USE_BATCH_PROCESSING = True
# --- PATCHWISE (SAHI-like) INFERENCE CONFIGURATION ---
# This is only used if USE_PATCHWISE_INFERENCE is True
PATCHWISE_INFERENCE_CONFIG = {
    'model_input_size': 416,
    'crop_width': 416,
    'crop_height': 416,
    'overlap_x': 10,  # Percentage
    'overlap_y': 10,  # Percentage
    'base_model_conf': 0.15, # Confidence for individual crops
    'postprocess_iou_thresh': 0.75, # IOU for patch-level inference
    'nms_thresh': 0.75, # NMS threshold for combining detections
    'match_metric': 'IOS' # 'IOU' or 'IOS' for merging
}

# --- Tracking Configuration --- 
BYTETRACK_MIN_CONF = 0.05    # Minimum detection confidence for ByteTrack to consider
BYTETRACK_TRACK_THRESH = 0.12   # Detection confidence threshold for high-confidence matching
BYTETRACK_MATCH_THRESH = 0.88    # IoU threshold for matching detections to tracks
BYTETRACK_TRACK_BUFFER_SECONDS = 25 # Seconds to keep a track alive if unmatched after it has stopped being associated with any new detections

MOTION_INFERENCE_MIN_FRAMES = 7
MOTION_INFERENCE_MIN_DIST = 15
MOTION_INFERENCE_MAX_PROJECTION = 0.5

# #*############## Video 1 #################
# # --- Input Settings (primarily for hardcoded mode) ---
INPUT_VIDEO_PATH = [r"C:\Users\EMAAN\Documents\YOLO\5 minute test - 4 Way Intersection.mp4"]
# INPUT_VIDEO_PATH = [r"C:\Users\EMAAN\Desktop\M1 - NORTH\M1 19710 6-8.mp4"]
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
# #######################

# # *############## Video 2 #################
# --- Input Settings (primarily for hardcoded mode) ---
# INPUT_VIDEO_PATH = [r"C:\Users\swest\Desktop\traffic-capturing\sample_videos\5 min test - Large FOV.mp4"]

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
BIDIRECTIONAL_REPORT_CATEGORIES = {
    'Light': ['Light Vehicle', 'Motorcycle'],
    'Short Trucks': ['Short Truck'],
    'Medium Trucks': ['Medium Truck'],
    'Long Trucks': ['Long Truck'],
    'Buses': ['Bus'],
    'Mini-buses': ['Minibus Taxi']
}
TIA_REPORT_CATEGORIES = {
    'Light': ['Light Vehicle'],
    'Mini-buses': ['Minibus Taxi'],
    'Trucks': ['Short Truck','Medium Truck','Long Truck'],
    'Buses': ['Bus']
}

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
    
OPTIMAL_BATCH_SIZE = 64
PYVIDEOREADER_DECODE_SEGMENT_SIZE = 7600

_cpu_count = os.cpu_count()
THREAD_COUNT = min(64, _cpu_count * 2)
MIXED_PRECISION = True
PARALLEL_STREAMS = 8

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

SERVER_PORT = 7862

# --- Print Confirmation ---
print(f"--- Configuration Loaded (Tuned for Exit Accuracy) ---")
print(f"Start Date/Time: {START_DATE} / {START_TIME}")
print(f"Using Model: {MODEL_PATH} (Conf: {CONF_THRESHOLD}, IoU: {IOU_THRESHOLD})")
print(f"Model Input Size: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}")
print(f"Target FPS: {TARGET_FPS}, Batch Size: {OPTIMAL_BATCH_SIZE}, Threads: {THREAD_COUNT}")
print(f"Mixed Precision: {MIXED_PRECISION}, CUDA Streams: {PARALLEL_STREAMS}")
print(f"Detailed Perf Metrics: {ENABLE_DETAILED_PERFORMANCE_METRICS}, Debug Mode: {DEBUG_MODE}")
print(f"--------------------------------------------------------------------")
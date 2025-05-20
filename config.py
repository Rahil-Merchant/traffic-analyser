# project/config.py
import os
from datetime import datetime
import torch # Import torch to check GPU capability

PROCESSING_MODE = 'enhanced'
LINE_MODE = 'hardcoded' # 'hardcoded' or 'interactive'
ENABLE_VISUALIZATION = True

# --- Input Settings (primarily for hardcoded mode) ---
INPUT_VIDEO_PATH = "C:/Users/EMAAN/Documents/YOLO/5 minute test - 4 Way Intersection.mp4"

_now = datetime.now()
START_DATE = _now.strftime("%y%m%d")
START_TIME = _now.strftime("%H%M%S") + f"{_now.microsecond // 1000:03d}"

DEFAULT_PRIMARY_DIRECTION = 'south'

# --- Core Settings ---
TARGET_FPS = 20 # Target FPS for processing and output video
BASE_OUTPUT_DIR = "C:/Users/EMAAN/Documents/YOLO/project/output"
VIDEO_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
REPORT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "reports")

# --- Model Configuration ---
MODEL_PATH = r"C:/Users/EMAAN/Documents/YOLO/project/models/weights/silent_contract_9082.engine"
# MODEL_PATH = r"C:/Users/EMAAN/Documents/YOLO/project/models/weights/SGDMed_SGD_LR0p0100_WD0p00050_Cls1p50_Mix0p15_20250418_112036.engine"
MODEL_INPUT_SIZE = 416 # Model input resolution (square)

# YOLO Detection Thresholds
CONF_THRESHOLD = 0.1 # Confidence threshold for detections to be considered primary tracks
IOU_THRESHOLD = 0.45  # IoU threshold for NMS

# --- Line & Zone Definitions (Polygons for 'hardcoded' or 'interactive' mode) ---
LINE_POINTS = {
    'north': [(282, 339), (422, 312), (371, 240), (272, 264)],
    'south': [(290, 446), (605, 335), (619, 441), (470, 466), (361, 476)],
    'east': [(512, 306), (596, 322), (628, 214), (525, 175)],
    'west': [(252, 378), (244, 442), (57, 466), (53, 397)]
}
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
scale_x = FRAME_WIDTH / 640
scale_y = FRAME_HEIGHT / 480

for zone in LINE_POINTS:
    LINE_POINTS[zone] = [
        (x * scale_x, y * scale_y) for (x, y) in LINE_POINTS[zone]
    ]

VALID_MOVEMENTS = {
    'north': ['south', 'east', 'west'],
    'south': ['north', 'east', 'west'],
    'east': ['west', 'north', 'south'],
    'west': ['east', 'north', 'south']
}

# --- Hardware Optimizations ---
if MODEL_INPUT_SIZE >= 640:
    _default_batch = 16 if PROCESSING_MODE == 'standard' else 32
else:
    _default_batch = 32 if PROCESSING_MODE == 'standard' else 64 # Adjusted default batch size
OPTIMAL_BATCH_SIZE = _default_batch
USE_ADVANCED_VIDEO_READER = True
PYVIDEOREADER_DECODE_SEGMENT_SIZE = 7500 # Frames per segment for PyVideoReader
FFMPEG_PATH = r'C:\ffmpeg\ffmpeg.exe'

_cpu_count = os.cpu_count()
THREAD_COUNT = min(64, _cpu_count * 2) if _cpu_count else 24
MIXED_PRECISION = True if PROCESSING_MODE == 'enhanced' and torch.cuda.is_available() else False
PARALLEL_STREAMS = 16 if PROCESSING_MODE == 'enhanced' else 2

# --- Tracking Configuration ---
REIDENTIFICATION_TIMEOUT = 10.0 # May become less relevant with ByteTrack, PathManager handles timeouts
MAX_MATCHING_DISTANCE = 50    # Not directly used by ByteTrack, which uses IoU

# Vehicle Timeouts in Intersection (Application-Level)
VEHICLE_TIMEOUTS = { # In seconds
    'Light Vehicle': 400, 'Motorcycle': 400, 'Minibus Taxi': 500,
    'Short Truck': 500, 'Medium Truck': 550, 'Long Truck': 600,
    'Bus': 550, 'Pedestrian': 650, 'Cyclist': 650,
    'Animal drawn vehicle': 650, 'Person with wheel barrow': 650,
    'default': 400
}
TRACK_HISTORY_LENGTH = 100 # For PathManager's visual trails
# Max consecutive misses for a path in PathManager before being considered for timeout
                                          # Or a fixed higher value like 50 or 60.
                                          # This value is in terms of *frames processed by PathManager*.
                                          # If you process every frame (TARGET_FPS = original_fps), then it's direct.
                                          # If TARGET_FPS is lower, this still means 40 *processed frames*.

# --- ByteTrack Specific Arguments (can also be moved to a dedicated tracker_config.py) ---
BYTETRACK_TRACK_HIGH_THRESH = 0.3
BYTETRACK_TRACK_LOW_THRESH = 0.1
BYTETRACK_NEW_TRACK_THRESH = BYTETRACK_TRACK_HIGH_THRESH + .1
BYTETRACK_MATCH_THRESH = 0.8  # This is cost (1-IoU), so IoU >= 0.2
BYTETRACK_TRACK_BUFFER = 90   # Frames to keep a ByteTrack track in 'lost' state internally
MAX_CONSECUTIVE_MISSES_FOR_PATH_MANAGER = BYTETRACK_TRACK_BUFFER + 10 # e.g., 30 + 10 = 40 frames
BYTETRACK_FUSE_SCORE = True

# --- Reporting Configuration ---
VEHICLE_ID_MAP = {
    'Light Vehicle': '1', 'Motorcycle': '11', 'Minibus Taxi': '13',
    'Short Truck': '2T,2', 'Medium Truck': '2T,3', 'Long Truck': '2T,4',
    'Bus': '2B', 'Pedestrian': '9P', 'Cyclist': '9C',
    'Animal drawn vehicle': '95', 'Person with wheel barrow': '100'
}
ALL_VEHICLE_TYPES = list(VEHICLE_ID_MAP.keys())
REPORTING_DIRECTIONS = { # Used for Excel sheet names, actual movements depend on VALID_MOVEMENTS
    'From South': ['To North', 'To West', 'To East'],
    'From North': ['To South', 'To West', 'To East'],
    'From West': ['To North', 'To South', 'To East'],
    'From East': ['To North', 'To South', 'To West']
}

# --- Performance & Debugging ---
ENABLE_DETAILED_PERFORMANCE_METRICS = True
DEBUG_MODE = True # Set to True for verbose debug prints
PERFORMANCE_SAMPLE_INTERVAL = 5.0 # Seconds
MEMORY_CLEANUP_INTERVAL = 300 # Seconds

# --- Gradio ---
GRADIO_SERVER_PORT = 7862
GRADIO_SHARE = False

# --- Print Confirmation ---
print(f"--- Configuration Loaded (ByteTrack Integration) ---")
print(f"Processing Mode: {PROCESSING_MODE.upper()}")
print(f"Line Definition Mode: {LINE_MODE.upper()}")
if LINE_MODE == 'hardcoded': print(f"Input Video: {INPUT_VIDEO_PATH}")
print(f"Start Date/Time: {START_DATE} / {START_TIME}")
print(f"Using Model: {MODEL_PATH} (Conf: {CONF_THRESHOLD}, IoU: {IOU_THRESHOLD})")
print(f"Model Input Size: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}, Display Size: {FRAME_WIDTH}x{FRAME_HEIGHT}")
if LINE_MODE == 'hardcoded': print("Using Hardcoded Zones (config.py)")
else: print("Using Interactive Zones (Gradio)")
print(f"Target FPS: {TARGET_FPS}, Batch Size: {OPTIMAL_BATCH_SIZE}, Threads: {THREAD_COUNT}")
print(f"Mixed Precision: {MIXED_PRECISION}, CUDA Streams: {PARALLEL_STREAMS}")
print(f"PathManager Misses Timeout: {MAX_CONSECUTIVE_MISSES_FOR_PATH_MANAGER} frames")
print(f"Detailed Perf Metrics: {ENABLE_DETAILED_PERFORMANCE_METRICS}, Debug Mode: {DEBUG_MODE}")
print(f"--------------------------------------------------------------------")
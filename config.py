# project/config.py
import os
from datetime import datetime
import torch # Import torch to check GPU capability

PROCESSING_MODE = 'enhanced'
LINE_MODE = 'hardcoded'
ENABLE_VISUALIZATION = True

# --- Input Settings (primarily for hardcoded mode) ---
INPUT_VIDEO_PATH = "C:/Users/EMAAN/Documents/YOLO/5 minute test - 4 Way Intersection.mp4"

_now = datetime.now()
START_DATE = _now.strftime("%y%m%d")
START_TIME = _now.strftime("%H%M%S") + f"{_now.microsecond // 1000:03d}"

DEFAULT_PRIMARY_DIRECTION = 'south'

# --- Core Settings ---
TARGET_FPS = 20
BASE_OUTPUT_DIR = "C:/Users/EMAAN/Documents/YOLO/project/output"
VIDEO_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
REPORT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "reports")

# --- Model Configuration ---
MODEL_PATH = r"C:/Users/EMAAN/Documents/YOLO/project/models/weights/silent_contract_9082.engine"
# MODEL_PATH = r"C:/Users/EMAAN/Documents/YOLO/project/models/weights/SGDMed_SGD_LR0p0100_WD0p00050_Cls1p50_Mix0p15_20250418_112036.engine"
CONF_THRESHOLD = 0.4 # Keep SLIGHTLY LOWER CONF to potentially keep weaker detections
IOU_THRESHOLD = 0.6   # Keep slightly higher
MODEL_INPUT_SIZE = 416 # Keep smaller size for performance

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
FRAME_WIDTH = 640; FRAME_HEIGHT = 480
VALID_MOVEMENTS = { 'north': ['south','east','west'], 'south': ['north','east','west'], 'east': ['west','north','south'], 'west': ['east','north','south'] }

# --- Hardware Optimizations ---
# Adjust Batch Size based on chosen MODEL_INPUT_SIZE (320) and GPU VRAM
if MODEL_INPUT_SIZE >= 640: _default_batch = 16 if PROCESSING_MODE == 'standard' else 32
else: _default_batch = 32 if PROCESSING_MODE == 'standard' else 128 # Keep larger for 320
OPTIMAL_BATCH_SIZE = _default_batch
USE_ADVANCED_VIDEO_READER = True 
PYVIDEOREADER_DECODE_SEGMENT_SIZE = 7500
FFMPEG_PATH = r'C:\ffmpeg\ffmpeg.exe'

_cpu_count = os.cpu_count(); THREAD_COUNT = min(64, _cpu_count * 2) if _cpu_count else 24
MIXED_PRECISION = True if PROCESSING_MODE == 'enhanced' and torch.cuda.is_available() else False
PARALLEL_STREAMS = 16 if PROCESSING_MODE == 'enhanced' else 2

# --- Tracking Configuration --- 
# --- ByteTrack Configuration ---
# Confidence threshold to consider a detection as a potential candidate for tracking.
# Detections below this are ignored by ByteTrack.
BYTETRACK_TRACK_THRESH = 0.4

# Confidence threshold for a detection to be considered "high-confidence".
# These are matched first with existing active tracks.
BYTETRACK_TRACK_HIGH_THRESH = 0.5

# Confidence threshold to initiate a *new* track from an unmatched high-confidence detection.
# This should generally be >= BYTETRACK_TRACK_HIGH_THRESH.
# Some implementations set it slightly higher to be more conservative about starting new tracks.
BYTETRACK_NEW_TRACK_THRESH = 0.6

# IoU (Intersection over Union) threshold for matching tracks with detections.
# If IoU is below this, the match is not considered valid.
BYTETRACK_MATCH_THRESH_IOU = 0.6 # Can be 0.7, 0.8 etc. Depends on how much overlap you expect.

# Maximum number of frames to keep a track in the "lost" state before it's removed.
# This value is in *frames*. If you prefer to think in seconds, convert it based on your
# processing FPS (e.g., if processing at 10 FPS, 30 frames = 3 seconds).
# Example: if your BYTETRACK_TRACK_BUFFER_AGE_SECONDS = 2.0 (seconds)
# and your processing_fps = 10, then BYTETRACK_TRACK_BUFFER_AGE_FRAMES = 2.0 * 10 = 20 frames.
# For simplicity, let's define it directly in frames in config, or calculate it during ByteTracker init.
BYTETRACK_TRACK_BUFFER_AGE_FRAMES = 100 # e.g., for 30 FPS this is 1 second. Adjust based on your needs.

# --- (End of ByteTrack Configuration) ---

# Timeout durations - Keep Increased slightly more
VEHICLE_TIMEOUTS = {
    'Light Vehicle': 400, 'Motorcycle': 400, 'Minibus Taxi': 500,
    'Short Truck': 500, 'Medium Truck': 550, 'Long Truck': 600,
    'Bus': 550, 'Pedestrian': 650, 'Cyclist': 650,
    'Animal drawn vehicle': 650, 'Person with wheel barrow': 650,
    'default': 400 # Increased default timeout
}
TRACK_HISTORY_LENGTH = 100 # Keep reduced slightly

# Keep configuration for tracker behavior
MAX_CONSECUTIVE_MISSES = 20 # Allow track to persist for X frames without detection

# --- Reporting Configuration ---
VEHICLE_ID_MAP = { 'Light Vehicle': '1', 'Motorcycle': '11', 'Minibus Taxi': '13', 'Short Truck': '2T,2', 'Medium Truck': '2T,3', 'Long Truck': '2T,4', 'Bus': '2B', 'Pedestrian': '9P', 'Cyclist': '9C', 'Animal drawn vehicle': '95', 'Person with wheel barrow': '100' }
ALL_VEHICLE_TYPES = list(VEHICLE_ID_MAP.keys())
REPORTING_DIRECTIONS = { 'From South': ['To North', 'To West', 'To East'], 'From North': ['To South', 'To West', 'To East'], 'From West': ['To North', 'To South', 'To East'], 'From East': ['To North', 'To South', 'To West'] }

# --- Performance & Debugging ---
ENABLE_DETAILED_PERFORMANCE_METRICS = True
DEBUG_MODE = True # Keep False for performance runs
DEBUG_MODE_TRACKING  = True
PERFORMANCE_SAMPLE_INTERVAL = 5.0
MEMORY_CLEANUP_INTERVAL = 300

# --- Gradio ---
GRADIO_SERVER_PORT = 7862
GRADIO_SHARE = False

# --- Print Confirmation ---
print(f"--- Configuration Loaded (Tuned for Exit Accuracy) ---")
print(f"Processing Mode: {PROCESSING_MODE.upper()}")
print(f"Line Definition Mode: {LINE_MODE.upper()}")
if LINE_MODE == 'hardcoded': print(f"Input Video: {INPUT_VIDEO_PATH}")
print(f"Start Date/Time: {START_DATE} / {START_TIME}")
print(f"Using Model: {MODEL_PATH} (Conf: {CONF_THRESHOLD}, IoU: {IOU_THRESHOLD})")
print(f"Model Input Size: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}")
if LINE_MODE == 'hardcoded': print("Using Hardcoded Lines (config.py)")
else: print("Using Interactive Lines (Gradio)")
print(f"Target FPS: {TARGET_FPS}, Batch Size: {OPTIMAL_BATCH_SIZE}, Threads: {THREAD_COUNT}")
print(f"Mixed Precision: {MIXED_PRECISION}, CUDA Streams: {PARALLEL_STREAMS}")
# Adjusted print for tracking params
print(f"Detailed Perf Metrics: {ENABLE_DETAILED_PERFORMANCE_METRICS}, Debug Mode: {DEBUG_MODE}")
print(f"--------------------------------------------------------------------")
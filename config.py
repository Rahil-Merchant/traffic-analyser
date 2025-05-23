# project/config.py
import os
from datetime import datetime
import torch # Import torch to check GPU capability

LINE_MODE = 'hardcoded'
ENABLE_VISUALIZATION = True

# --- Input Settings (primarily for hardcoded mode) ---
INPUT_VIDEO_PATH = r"C:\Users\swest\Desktop\traffic-capturing\sample_videos\5 minute test - 4 Way Intersection.mp4"

_now = datetime.now()
START_DATE = _now.strftime("%y%m%d")
START_TIME = _now.strftime("%H%M%S") + f"{_now.microsecond // 1000:03d}"

DEFAULT_PRIMARY_DIRECTION = 'south'

# --- Core Settings ---
TARGET_FPS = 25
BASE_OUTPUT_DIR = r"C:\Users\swest\Desktop\traffic-capturing\traffic-reporting\output"
VIDEO_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
REPORT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "reports")

# --- Model Configuration ---
MODEL_PATH = r"C:\Users\swest\Desktop\traffic-capturing\traffic-reporting\models\weights\silent_contract_9082_960px_bs32.engine"
CONF_THRESHOLD = 0.25 # Keep SLIGHTLY LOWER CONF to potentially keep weaker detections
IOU_THRESHOLD = 0.6   # Keep slightly higher
MODEL_INPUT_SIZE = 960 # Keep smaller size for performance

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

scale_x = FRAME_WIDTH / FRAME_WIDTH_VIZ
scale_y = FRAME_HEIGHT / FRAME_HEIGHT_VIZ

LINE_POINTS = {
    'north': [(int(x * scale_x), int(y * scale_y)) for (x, y) in [(282, 339), (422, 312), (371, 240), (272, 264)]],
    'south': [(int(x * scale_x), int(y * scale_y)) for (x, y) in [(290, 446), (605, 335), (619, 441), (470, 466), (361, 476)]],
    'east':  [(int(x * scale_x), int(y * scale_y)) for (x, y) in [(512, 306), (596, 322), (628, 214), (525, 175)]],
    'west':  [(int(x * scale_x), int(y * scale_y)) for (x, y) in [(252, 378), (244, 442), (57, 466), (53, 397)]],
}
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

OPTIMAL_BATCH_SIZE = 8
    
USE_ADVANCED_VIDEO_READER = True 
PYVIDEOREADER_DECODE_SEGMENT_SIZE = 36000
FFMPEG_PATH = r'C:\ffmpeg\ffmpeg.exe'

_cpu_count = os.cpu_count()
THREAD_COUNT = min(64, _cpu_count * 2)
MIXED_PRECISION = True
PARALLEL_STREAMS = 16

# --- Tracking Configuration --- 
REIDENTIFICATION_TIMEOUT = 10.0
MAX_MATCHING_DISTANCE = 50

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
DEBUG_MODE = False # Keep False for performance runs
PERFORMANCE_SAMPLE_INTERVAL = 5.0
MEMORY_CLEANUP_INTERVAL = 300

# --- Gradio ---
GRADIO_SERVER_PORT = 7862
GRADIO_SHARE = False

# --- Print Confirmation ---
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
# Adjusted print for tracking params
print(f"Re-ID Timeout: {REIDENTIFICATION_TIMEOUT}s, Max Misses: {MAX_CONSECUTIVE_MISSES}") # Removed Max Match Dist
print(f"Detailed Perf Metrics: {ENABLE_DETAILED_PERFORMANCE_METRICS}, Debug Mode: {DEBUG_MODE}")
print(f"--------------------------------------------------------------------")
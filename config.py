# project/config.py
import os
from datetime import datetime
import torch # Import torch to check GPU capability

LINE_MODE = 'interactive'
ENABLE_VISUALIZATION = True

# --- Input Settings (primarily for hardcoded mode) ---
INPUT_VIDEO_PATH = r"C:\Users\EMAAN\Documents\YOLO\DAV Videos\11.00.00-13.00.00[R][0@0][1].dav"
INPUT_VIDEO_PATH = [r"C:\Users\EMAAN\Documents\YOLO\DAV Videos\part1.mp4",r"C:\Users\EMAAN\Documents\YOLO\DAV Videos\part2.mp4"]
_now = datetime.now()
START_DATE = _now.strftime("%y%m%d")
START_TIME = _now.strftime("%H%M%S") + f"{_now.microsecond // 1000:03d}"

DEFAULT_PRIMARY_DIRECTION = 'south'

# --- Core Settings ---
TARGET_FPS = 25
BASE_OUTPUT_DIR = r"C:\Users\swest\Desktop\traffic-capturing\traffic-reporting\output"
BASE_OUTPUT_DIR = r"C:\Users\EMAAN\Documents\YOLO\project\output"
VIDEO_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
REPORT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "reports")

# --- Model Configuration ---
# MODEL_PATH = r"C:\Users\swest\Desktop\traffic-capturing\traffic-reporting\models\weights\silent_contract_9082_960px.engine"
MODEL_PATH = r"C:\Users\EMAAN\Documents\YOLO\project\models\weights\silent_contract_9082_420px.engine"
CONF_THRESHOLD = 0.25 # Keep SLIGHTLY LOWER CONF to potentially keep weaker detections
IOU_THRESHOLD = 0.6   # Keep slightly higher
MODEL_INPUT_SIZE = 640 # Keep smaller size for performance

# --- Tracking Configuration --- 
BYTETRACK_MIN_CONF = 0.1    # Minimum detection confidence for ByteTrack to consider
BYTETRACK_TRACK_THRESH = 0.35   # Detection confidence threshold for high-confidence matching
BYTETRACK_MATCH_THRESH = 0.8    # IoU threshold for matching detections to tracks
BYTETRACK_TRACK_BUFFER_SECONDS = 15 # Seconds to keep a track alive if unmatched after it has stopped being associated with any new detections

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
LINE_POINTS = {'north': [(217, 341), (531, 226), (635, 267), (363, 414)], 
               'south': [(726, 300), (840, 323), (742, 537), (549, 498)], 
               'east': [(677, 197), (758, 162), (840, 222), (818, 264)], 
               'west': [(258, 453), (443, 510), (360, 592), (137, 546)]}
FRAME_HEIGHT = MODEL_INPUT_SIZE
FRAME_WIDTH = round(MODEL_INPUT_SIZE*(4/3))
FRAME_HEIGHT_VIZ = 480
FRAME_WIDTH_VIZ = 640

scale_x = 1# FRAME_WIDTH / 640
scale_y = 1# FRAME_HEIGHT / 480

LINE_POINTS = {
    'north': [(int(x * scale_x), int(y * scale_y)) for (x, y) in LINE_POINTS['north']],
    'south': [(int(x * scale_x), int(y * scale_y)) for (x, y) in LINE_POINTS['south']],
    'east':  [(int(x * scale_x), int(y * scale_y)) for (x, y) in LINE_POINTS['east']],
    'west':  [(int(x * scale_x), int(y * scale_y)) for (x, y) in LINE_POINTS['west']],
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
    
USE_ADVANCED_VIDEO_READER = True 
# PYVIDEOREADER_DECODE_SEGMENT_SIZE = 7600
PYVIDEOREADER_DECODE_SEGMENT_SIZE = 500

_cpu_count = os.cpu_count()
THREAD_COUNT = min(64, _cpu_count * 2)
MIXED_PRECISION = True
F = 8

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
PARALLEL_STREAMS = 8

# --- Gradio ---
GRADIO_SERVER_PORT = 7863
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
# print(f"Mixed Precision: {MIXED_PRECISION}, CUDA Streams: {PARALLEL_STREAMS}")
print(f"Detailed Perf Metrics: {ENABLE_DETAILED_PERFORMANCE_METRICS}, Debug Mode: {DEBUG_MODE}")
print(f"--------------------------------------------------------------------")
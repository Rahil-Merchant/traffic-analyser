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

# YOLO Detection Thresholds (Used as high-confidence for ByteTrack)
CONF_THRESHOLD = 0.25 # Confidence threshold for detections to be considered primary tracks
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
# ByteTrack specific parameters (values from Ultralytics defaults, can be tuned)
# CONF_THRESHOLD above is used as track_high_thresh
TRACK_LOW_CONF_THRESHOLD = 0.1  # Confidence threshold for low-score detections (ByteTrack second association)
NEW_TRACK_CONF_THRESHOLD = 0.3  # Confidence threshold to initiate a new track (usually slightly higher than track_high_thresh or same)
TRACK_BUFFER_FRAMES = int(TARGET_FPS * 2.0) # Max frames to keep a lost track (e.g., 2 seconds at TARGET_FPS)
TRACK_IOU_THRESHOLD = 0.3       # IoU threshold for matching tracks with detections

# Application-level tracking parameters
TRACK_HISTORY_LENGTH = 100      # Max length of tracking trail for visualization
MAX_APP_CONSECUTIVE_MISSES = int(TARGET_FPS * 1.5) # App-level: Max frames a track is missed before considered for special handling (e.g. warning in timeout)

# Vehicle Timeouts in Intersection (Application-Level)
VEHICLE_TIMEOUTS = { # In seconds
    'Light Vehicle': 40, 'Motorcycle': 40, 'Minibus Taxi': 50,
    'Short Truck': 50, 'Medium Truck': 55, 'Long Truck': 60,
    'Bus': 55, 'Pedestrian': 65, 'Cyclist': 65,
    'Animal drawn vehicle': 65, 'Person with wheel barrow': 65,
    'default': 40
}

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
DEBUG_MODE = False # Set to True for verbose debug prints
PERFORMANCE_SAMPLE_INTERVAL = 5.0 # Seconds
MEMORY_CLEANUP_INTERVAL = 300 # Seconds

# --- Gradio ---
GRADIO_SERVER_PORT = 7862
GRADIO_SHARE = False

# --- Print Confirmation ---
print(f"--- Configuration Loaded (ByteTrack Integrated) ---")
print(f"Processing Mode: {PROCESSING_MODE.upper()}")
print(f"Line Definition Mode: {LINE_MODE.upper()}")
if LINE_MODE == 'hardcoded': print(f"Input Video: {INPUT_VIDEO_PATH}")
print(f"Start Date/Time: {START_DATE} / {START_TIME}")
print(f"Using Model: {MODEL_PATH} (Detection Conf: {CONF_THRESHOLD}, IoU_NMS: {IOU_THRESHOLD})")
print(f"Model Input Size: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}")
if LINE_MODE == 'hardcoded': print("Using Hardcoded Zones (config.py)")
else: print("Using Interactive Zones (Gradio)")
print(f"Target FPS: {TARGET_FPS}, Batch Size: {OPTIMAL_BATCH_SIZE}, Threads: {THREAD_COUNT}")
print(f"Mixed Precision: {MIXED_PRECISION}, CUDA Streams: {PARALLEL_STREAMS}")
print(f"ByteTrack Params: LowConfThr={TRACK_LOW_CONF_THRESHOLD}, NewTrackThr={NEW_TRACK_CONF_THRESHOLD}, Buffer={TRACK_BUFFER_FRAMES}f, MatchIoU={TRACK_IOU_THRESHOLD}")
print(f"App-Level Tracking: MaxMisses={MAX_APP_CONSECUTIVE_MISSES}f")
print(f"Detailed Perf Metrics: {ENABLE_DETAILED_PERFORMANCE_METRICS}, Debug Mode: {DEBUG_MODE}")
print(f"--------------------------------------------------------------------")

# --- TRACKER_CONFIG (for Ultralytics ByteTrack) ---
# This dictionary will be built from the parameters above and passed to the tracker.
# It needs to match the argument names expected by ultralytics.trackers.BYTETracker.
# Based on ultralytics/cfg/trackers/bytetrack.yaml
TRACKER_CONFIG_ARGS = {
    'tracker_type': 'bytetrack', # For reference, not directly used by BYTETracker class
    'track_high_thresh': CONF_THRESHOLD,        # High confidence detection threshold
    'track_low_thresh': TRACK_LOW_CONF_THRESHOLD,   # Low confidence detection threshold
    'new_track_thresh': NEW_TRACK_CONF_THRESHOLD, # Threshold for initiating a new track
    'track_buffer': TRACK_BUFFER_FRAMES,        # Buffer for keeping lost tracks
    'match_thresh': TRACK_IOU_THRESHOLD,        # IoU threshold for matching
    # 'gmc_method': None, # Not using Global Motion Compensation for now
    # 'proximity_thresh': 0.5, # Not typically used in basic ByteTrack
    # 'appearance_thresh': 0.25, # Not typically used in basic ByteTrack
    # 'with_reid': False, # Not using ReID model with ByteTrack for now
}
# project/config/deepstream_config.py

import os
from datetime import datetime

# === Core DeepStream Settings ===
DEEPSTREAM_VERSION = "7.1"
ENABLE_VISUALIZATION = False
TARGET_FPS = 20

# === Hardware Configuration ===
GPU_ID = 0
CUDA_MEMORY_FRACTION = 0.9
MAX_BATCH_SIZE = 64
NVDEC_ENABLE_MAX_PERFORMANCE = True

# === Model and Inference ===
TRITON_MODEL_NAME = "yolo_traffic"
TRITON_MODEL_VERSION = "1"
TRITON_URL = "localhost:8001"
MODEL_INPUT_SIZE = 640

# === Tracking Configuration ===
TRACKER_CONFIG = {
    "ll-lib-file": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
    "ll-config-file": "config_tracker_NvDCF_accuracy.yml",
    "tracker-width": 640,
    "tracker-height": 640,
    "gpu-id": GPU_ID,
    "display-tracking-id": True
}

# === Analytics Configuration ===
ANALYTICS_CONFIG_TEMPLATE = """
[property]
enable=1
config-width={width}
config-height={height}
osd-mode=2
display-font-size=12

{roi_sections}

{line_sections}
"""

# === Zone Definitions (converted from your LINE_POINTS) ===
ZONE_DEFINITIONS = {
    'north': [(282, 339), (422, 312), (371, 240), (272, 264)],
    'south': [(290, 446), (605, 335), (619, 441), (470, 466), (361, 476)],
    'east': [(512, 306), (596, 322), (628, 214), (525, 175)],
    'west': [(252, 378), (244, 442), (57, 466), (53, 397)]
}

# === Output Settings ===
BASE_OUTPUT_DIR = r"/app/output"
VIDEO_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "videos")
REPORT_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "reports")

# === Vehicle Configuration (preserved from original) ===
VEHICLE_ID_MAP = {
    'Light Vehicle': '1', 'Motorcycle': '11', 'Minibus Taxi': '13',
    'Short Truck': '2T,2', 'Medium Truck': '2T,3', 'Long Truck': '2T,4',
    'Bus': '2B', 'Pedestrian': '9P', 'Cyclist': '9C',
    'Animal drawn vehicle': '95', 'Person with wheel barrow': '100'
}

VALID_MOVEMENTS = {
    'north': ['south', 'east', 'west'],
    'south': ['north', 'east', 'west'],  
    'east': ['west', 'north', 'south'],
    'west': ['east', 'north', 'south']
}

VEHICLE_TIMEOUTS = {
    'Light Vehicle': 400, 'Motorcycle': 400, 'Minibus Taxi': 500,
    'Short Truck': 500, 'Medium Truck': 550, 'Long Truck': 600,
    'Bus': 550, 'Pedestrian': 650, 'Cyclist': 650,
    'Animal drawn vehicle': 650, 'Person with wheel barrow': 650,
    'default': 400
}

BIDIRECTIONAL_REPORT_CATEGORIES = {
    'Light': ['Light Vehicle', 'Motorcycle'],
    'Short Trucks': ['Short Truck'],
    'Medium Trucks': ['Medium Truck'], 
    'Long Trucks': ['Long Truck'],
    'Buses': ['Bus'],
    'Mini-buses': ['Minibus Taxi']
}

# === Performance Settings ===
PERFORMANCE_SAMPLE_INTERVAL = 15.0
ENABLE_DETAILED_PERFORMANCE_METRICS = True

print(f"=== DeepStream 7.1 Configuration Loaded ===")
print(f"Target FPS: {TARGET_FPS}, Max Batch: {MAX_BATCH_SIZE}")
print(f"Triton Model: {TRITON_MODEL_NAME} at {TRITON_URL}")
print(f"Tracker: NvDCF Accuracy Mode")
print("=" * 50)

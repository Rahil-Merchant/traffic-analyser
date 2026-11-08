<p align="center">
  <h1 align="center">Urban Reconnaissance</h1>
  <p align="center">
    <strong>AI-Powered Traffic Counting, Classification & Speed Estimation Platform</strong>
  </p>
  <p align="center">
    <em>Real-time vehicle detection • Multi-object tracking • Perspective speed calculation • RSA-standard traffic reporting</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python Version" />
    <img src="https://img.shields.io/badge/PyTorch-2.7%2Bcu126-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/TensorRT-10.10-76B900?logo=nvidia&logoColor=white" alt="TensorRT" />
    <img src="https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white" alt="CUDA" />
    <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
    <img src="https://img.shields.io/badge/Tauri-v2-FFC131?logo=tauri&logoColor=black" alt="Tauri" />
    <img src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black" alt="React" />
    <img src="https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript&logoColor=white" alt="TypeScript" />
  </p>
  <p align="center">
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#usage">Usage</a> •
    <a href="#deep-dive-documentation">Documentation</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#project-structure">Structure</a> •
    <a href="#reports">Reports</a>
  </p>
</p>

---

**Urban Reconnaissance** is a production-grade traffic analysis system that processes intersection and road-link video footage to produce South African RSA-standard traffic reports. It combines GPU-accelerated YOLO object detection with custom multi-object tracking, perspective-corrected speed estimation, and automated Excel report generation, all orchestrated through a native desktop application.

## Features

### Detection & Classification
- **11 vehicle classes**: Light Vehicle, Motorcycle, Minibus Taxi, Short/Medium/Long Truck, Bus, Pedestrian, Cyclist, Animal-drawn Vehicle, Person with Wheelbarrow
- **YOLOv8 + TensorRT** acceleration for real-time inference on NVIDIA GPUs
- **Patchwise (SAHI-style) inference** for high-resolution or distant-camera footage
- **Batched GPU processing**: up to 128 frames per batch with configurable batch sizes

### Multi-Object Tracking
- **OC-SORT** (Observation-Centric SORT) tracker with Kalman filtering and velocity-direction consistency
- **ByteTrack** support with configurable confidence thresholds
- **Vehicle stitching**: fragments broken tracks into unified vehicle identities, preventing ID-switch count inflation
- **Buffered IoU matching** (C-BIoU) for fast-moving vehicles whose consecutive detections do not overlap

### Speed Estimation
- **Perspective transform**: user marks 4 road corners plus real-world dimensions to generate homography-based metric coordinates
- **Median-filtered, smoothed trajectories** eliminate detection jitter
- **Per-class speed ceilings** and low-confidence clamping for robust measurements
- **Live recalibration**: adjust road dimensions and see updated speeds without re-processing

### Automated Reporting
- **RSA-standard Excel reports** with 15-minute interval breakdowns
- **Bidirectional link count reports** with speed distribution analysis
- **TIA (Transport Impact Assessment) format** reports
- **Day/night hour splitting** and configurable speed limit thresholds

### Desktop Application
- **Tauri + React + TypeScript** native desktop UI
- **Interactive zone drawing**: click to define detection polygons on the video preview
- **Drag-and-drop file management** with multi-video queuing
- **Real-time progress tracking** via WebSocket with ETA estimation
- **Speed recalibration panel**: adjust parameters post-analysis and regenerate reports

### Frame Selector (Labelling Tool)
- **Active-learning style** frame selection for building training datasets
- **GPU-accelerated detection** with batch processing and parallel video decoding
- **Web-based review UI** for inspecting and exporting selected frames
- **Roboflow integration** for uploading frames directly to annotation projects

## Architecture

```
+-----------------------------------------------------------------+
|                    Desktop Application UI                       |
|              (Tauri + React + TypeScript + Tailwind)            |
+--------------------------------+--------------------------------+
                                 | HTTP / WebSocket
+--------------------------------v--------------------------------+
|                     FastAPI Backend (dev_server.py)             |
|         /api/v1/process  |  /api/v1/recalibrate-speed           |
|         /api/v1/video-preview  |  /ws/progress/{job_id}         |
+--------------------------------+--------------------------------+
                                 |
+--------------------------------v--------------------------------+
|                   VideoProcessor Pipeline                       |
|                                                                 |
|  +---------+   +--------------+   +-----------+   +---------+   |
|  |  Reader |-->| Preprocessor |-->| Inference |-->| Post-   |   |
|  | (PyVR)  |   |  (Letterbox) |   | (TensorRT)|   |processor|   |
|  +---------+   +--------------+   +-----------+   +----+----+   |
|                                                        |        |
|  +-----------------------------------------------------v----+   |
|  |                   Tracking & Counting                    |   |
|  |  OC-SORT/ByteTrack -> ZoneTracker -> VehicleCounter      |   |
|  |  (Kalman filter)     (polygon        (stitching,         |   |
|  |                       point-in)       de-duplication)    |   |
|  +-----------------------------------------------------v----+   |
|                                                        |        |
|  +-----------------------------------------------------v----+   |
|  |                  Speed & Reporting                       |   |
|  |  PerspectiveTransformer -> Speed Pipeline -> Excel Report|   |
|  |  (homography)             (filter+clamp)   (RSA format)  |   |
|  +----------------------------------------------------------+   |
+-----------------------------------------------------------------+
```

The pipeline runs across **4 parallel threads**: video decoding, preprocessing (letterbox and tensor conversion), GPU inference, and post-processing (tracking, counting, and visualization). Queues between stages keep the GPU fully saturated.

## Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| **Python** | 3.10+ |
| **CUDA** | 12.x |
| **TensorRT** | 10.x |
| **Node.js** | 18+ (for frontend) |
| **Rust** | Latest stable (for Tauri) |
| **NVIDIA GPU** | Compute capability 7.0+ recommended |

### Installation

```bash
# Clone the repository
git clone git@github.com:Rahil-Work/traffic-reporting.git
cd traffic-reporting

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install the modified supervision library
cd supervision_modified && pip install -e . && cd ..

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### Model Setup

Place your trained YOLO model weights in `models/weights/` and update the path in [`config.py`](config.py):

```python
PC_PATHS = {
    'your_machine': {
        'model_path': r"models/weights/your_model.engine",
        'base_output_dir': r"./output",
        'video_dir': r"./data",
        'data_yaml_path': r"",
    },
}
PC = 'your_machine'
```

To convert a PyTorch model to TensorRT:
```bash
# PyTorch to ONNX
python utility_scripts/pt2onnx_ultralytics.py

# ONNX to TensorRT
python utility_scripts/onnx2trt_trtutils.py
```

## Usage

### Desktop Application (Recommended)

Start the backend server and the Tauri desktop app:

```bash
# Terminal 1: Start the FastAPI backend
python dev_server.py

# Terminal 2: Start the Tauri frontend
cd frontend
npm run tauri dev
```

**Workflow:**
1. **Add videos**: drag & drop or use the file picker to add video files
2. **Set metadata**: enter station number, date/time, and primary direction
3. **Draw zones**: click on the preview to define directional detection polygons (2 to 4 zones)
4. **Calibrate speed** *(optional)*: draw a road rectangle and enter dimensions in meters
5. **Launch**: click Run to start processing; monitor progress via the live overlay
6. **Review results**: view counts, average speed, and report file paths in the results modal
7. **Recalibrate** *(optional)*: adjust road dimensions post-analysis and regenerate reports

### Headless / Scripted Mode

For batch processing without the GUI, configure [`config.py`](config.py) with hardcoded zone polygons and run:

```python
from core.video_processor import VideoProcessor

processor = VideoProcessor('hardcoded')
processor.process_video(
    video_paths=["path/to/video.mp4"],
    start_date_str="261009",      # YYMMDD
    start_time_str="0800000",     # HHMMSSmmm
    station_number_str="ST001",
    primary_direction="south"
)
```

### Frame Selector (Labelling Tool)

Select the best frames from surveillance footage for model training:

```bash
# Web-based GUI (recommended)
python -m frame_selector gui

# CLI mode
python -m frame_selector video1.mp4 video2.mp4 --out frames_for_annotation/ --budget 100
```

## Deep-Dive Documentation

For comprehensive technical specifications, algorithms, and subsystem guides, explore the dedicated documentation suite:

| Document | Focus Area | Key Topics Covered |
|---|---|---|
| [**Architecture & Concurrency**](docs/ARCHITECTURE.md) | Backend Core | 4-stage pipeline, thread synchronization, queue bounds, CUDA streams, memory cleanup |
| [**Tracking & Vehicle Counting**](docs/TRACKING_AND_COUNTING.md) | Tracking Layer | OC-SORT Kalman filters, C-BIoU buffered matching, vehicle stitching, reversal splits, weighted class voting |
| [**Perspective & Speed Estimation**](docs/SPEED_ESTIMATION.md) | Metrology | Planar homography matrix, ground contact points, median & moving-average filtering, live post-recalibration |
| [**REST & WebSocket API**](docs/API.md) | Integration | FastAPI endpoints, WebSocket progress streaming format, request/response schemas |
| [**Desktop Application UI**](frontend/README.md) | Frontend Client | Tauri v2 architecture, interactive polygon canvas, drag-and-drop playlist, live overlays |
| [**Frame Selector & Active Learning**](frame_selector/README.md) | Dataset Engine | Hardware-accelerated scanning, frame uncertainty scoring, budget selection, Roboflow upload |
| [**Contributing Guide**](CONTRIBUTING.md) | Development | Environment setup, coding conventions, testing guidelines, and PR workflow |

## Configuration

All settings are centralized in [`config.py`](config.py). Key parameters:

### Detection
| Parameter | Default | Description |
|---|---|---|
| `MODEL_INPUT_SIZE` | `640` | YOLO input resolution |
| `CONF_THRESHOLD` | `0.05` | Detection confidence threshold |
| `IOU_THRESHOLD` | `0.6` | NMS IoU threshold |
| `USE_PATCHWISE_INFERENCE` | `False` | Enable SAHI-style tiled inference |
| `TTA` | `True` | Test-time augmentation |

### Tracking
| Parameter | Default | Description |
|---|---|---|
| `TRACKER_TYPE` | `'ocsort'` | Tracker selection (`ocsort` or `bytetrack`) |
| `USE_VEHICLE_STITCHING` | `True` | Enable fragment-to-vehicle stitching |
| `OCSORT_CONFIG.det_thresh` | `0.25` | Minimum score to start a track |
| `OCSORT_CONFIG.max_age_sec` | `1.0` | Seconds before a lost track is dropped |

### Speed Estimation
| Parameter | Default | Description |
|---|---|---|
| `MIN_SPEED_POSITIONS` | `3` | Minimum trajectory points for speed |
| `SPEED_SMOOTHING_WINDOW` | `9` | Moving average window size |
| `MAX_PLAUSIBLE_SPEED_KMH` | `250.0` | Global speed ceiling |
| `VEHICLE_MAX_SPEED_KMH` | *per-class* | Class-specific speed ceilings |

### Performance
| Parameter | Default | Description |
|---|---|---|
| `TARGET_FPS` | `20` | Frame sampling rate |
| `OPTIMAL_BATCH_SIZE` | `64` | Inference batch size |
| `MIXED_PRECISION` | `True` | FP16 inference |
| `PARALLEL_STREAMS` | `8` | CUDA streams for concurrent inference |

## Project Structure

```
traffic-reporting/
├── config.py                    # Central configuration
├── dev_server.py                # FastAPI backend server
├── debug.py                     # Debugging utilities
│
├── core/                        # Core processing engine
│   ├── video_processor.py       #   Main pipeline orchestrator
│   ├── perspective_transform.py #   Homography-based coordinate mapping
│   ├── patchwise_detector.py    #   SAHI-style tiled inference
│   ├── converters.py            #   Video format conversion (.dav -> .mp4)
│   ├── finalized_paths_store.py #   SQLite-backed trajectory storage
│   ├── performance.py           #   Performance metrics & profiling
│   └── utils.py                 #   Shared utilities
│
├── tracking/                    # Multi-object tracking
│   ├── ocsort.py                #   OC-SORT tracker implementation
│   ├── vehicle_counter.py       #   Vehicle stitching & counting logic
│   ├── zone_tracker.py          #   Polygon zone entry/exit detection
│   └── cleanup.py               #   Track lifecycle management
│
├── models/                      # Model management
│   ├── model_loader.py          #   YOLO/TensorRT model loading
│   └── weights/                 #   Model weight files (git-ignored)
│
├── visualization/               # Overlay & charting
│   ├── overlay.py               #   Annotated video frame rendering
│   └── performance_viz.py       #   Performance chart generation
│
├── reporting/                   # Report generation
│   └── excel_report.py          #   RSA-format Excel reports
│
├── frame_selector/              # Frame selection for labelling
│   ├── __main__.py              #   CLI entry point
│   ├── gui.py                   #   Gradio web interface
│   ├── detect.py                #   Detection pipeline
│   ├── select.py                #   Frame scoring & selection
│   ├── export.py                #   Image + label export
│   ├── review.py                #   HTML review page generator
│   ├── pipeline.py              #   End-to-end orchestration
│   └── signals.py               #   Inter-process communication
│
├── frontend/                    # Tauri desktop application
│   ├── src/
│   │   ├── App.tsx              #   Main application component
│   │   └── components/
│   │       ├── SetupPanel.tsx       # File & metadata input
│   │       ├── ControlsPanel.tsx    # Zone drawing controls
│   │       ├── InteractiveCanvas.tsx # Video preview & polygon editor
│   │       ├── ProcessingOverlay.tsx # Progress indicator
│   │       └── ResultsModal.tsx     # Post-analysis results display
│   ├── src-tauri/               #   Tauri/Rust backend config
│   └── package.json
│
├── patched_yolo_infer_batched/  # Modified SAHI crop+detect nodes
├── supervision_modified/        # Modified supervision library (in-repo)
│
├── utility_scripts/             # Dev & conversion tools
│   ├── pt2onnx_ultralytics.py   #   PyTorch to ONNX export
│   ├── onnx2trt_trtutils.py     #   ONNX to TensorRT conversion
│   ├── replay_detections.py     #   Replay saved detection dumps
│   ├── roboflow_upload.py       #   Upload frames to Roboflow
│   └── ...
│
├── tests/                       # Unit tests
│   ├── test_frame_selector.py
│   ├── test_ocsort.py
│   ├── test_perspective_transform.py
│   └── test_vehicle_counter.py
│
├── requirements.txt             # Python dependencies
└── requirements_py310.txt       # Python 3.10-specific dependencies
```

## Reports

The system generates three types of Excel reports:

### Standard Report
- Vehicle counts per 15-minute interval by type and direction
- RSA movement numbering (left/straight/right relative turns)
- Hourly and total summaries with vehicle ID codes

### Bidirectional Link Count Report
- Two-direction road link analysis
- Speed distribution histograms (10 km/h bins)
- Day/night hour breakdown
- Fleet average speed and percentile statistics

### TIA Report
- Transport Impact Assessment format
- Simplified vehicle categories (Light, Minibus, Truck, Bus)
- Formatted for regulatory submission

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_vehicle_counter.py -v
python -m pytest tests/test_ocsort.py -v
python -m pytest tests/test_perspective_transform.py -v
```

## Utility Scripts

| Script | Purpose |
|---|---|
| `pt2onnx_ultralytics.py` | Export YOLO model from PyTorch to ONNX |
| `onnx2trt_trtutils.py` | Convert ONNX model to TensorRT engine |
| `replay_detections.py` | Replay saved detection dumps for offline counting |
| `roboflow_upload.py` | Upload selected frames to Roboflow for annotation |
| `cut_videos.py` | Extract video segments |
| `train.py` / `new_train.py` | Model training scripts |

## Development

### Adding a New Vehicle Class

1. Train a YOLO model with the new class
2. Add the class to `VEHICLE_ID_MAP` in [`config.py`](config.py)
3. Add speed ceilings to `VEHICLE_MAX_SPEED_KMH`
4. Add timeout durations to `VEHICLE_TIMEOUTS`
5. Update report category mappings in `BIDIRECTIONAL_REPORT_CATEGORIES` / `TIA_REPORT_CATEGORIES`

### Switching Trackers

```python
# In config.py
TRACKER_TYPE = 'ocsort'    # or 'bytetrack'
```

### Enabling Patchwise Inference

For cameras mounted far from the road with small vehicles:

```python
# In config.py
USE_PATCHWISE_INFERENCE = True
PATCHWISE_INFERENCE_CONFIG = {
    'crop_width': 416,
    'crop_height': 416,
    'overlap_x': 10,      # percent
    'overlap_y': 10,
    'base_model_conf': 0.15,
    'nms_thresh': 0.75,
}
```

## License

Proprietary. All rights reserved.

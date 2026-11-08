<p align="center">
  <h1 align="center">Frame Selector</h1>
  <p align="center">
    <strong>Active-Learning Dataset Sampling & Curation Engine</strong>
  </p>
  <p align="center">
    <em>Mine high-value training frames from massive video corpora for continuous model improvement</em>
  </p>
</p>

---

## Overview

When training deep learning models for automated traffic surveillance, annotating entire continuous videos is cost-prohibitive and redundant (95%+ of video frames show empty roads or static traffic).

The **Frame Selector** automates active-learning data curation. It scans hours of raw surveillance video, detects vehicles using TensorRT, tracks trajectories using OC-SORT, flags hard or ambiguous instances, and exports a balanced, diverse subset of frames ready for annotation with pre-generated pseudo-labels.

## Key Capabilities

- **GPU-Accelerated High-Throughput Scan**:
  - Supports NVIDIA GPU-accelerated video decoding (via `video_reader` or PyNvVideoCodec).
  - Batched TensorRT inference (batch size 64) running concurrently with frame post-processing.
  - Subprocess isolation per video to prevent CUDA memory fragmentation and leaks.
- **Intelligent Frame Scoring & Selection**:
  - **False Negative Recovery**: Flags frames where tracker interpolation bridged a gap (unseen detections).
  - **Borderline Confidence**: Samples detections near decision boundaries ($0.15 \le \text{conf} \le 0.45$).
  - **Class Instability / Ambiguity**: Detects trajectories where classification labels oscillate (e.g., Short Truck vs. Medium Truck).
  - **High-Density / Occlusion Scenarios**: Prioritizes frames with dense clusters of overlapping vehicles.
  - **Temporal Diversity Budget**: Enforces minimum time gaps between selected frames so a single event doesn't dominate the budget.
- **Web Review GUI**:
  - Built-in lightweight review server (`python -m frame_selector gui`).
  - Interactive browser interface to browse, filter by flagging reason, enlarge, and inspect selected frames.
  - Folder picker, live progress reporting, and one-click label generation.
- **Direct Roboflow Integration**:
  - Export frames directly into Roboflow annotation projects with automated API key reading and batch uploading (`utility_scripts/roboflow_upload.py`).

## Usage

### 1. Launching the Web GUI (Recommended)

```bash
python -m frame_selector gui --port 8090
```
Open `http://localhost:8090` in your browser. Select your raw video directory, choose your budget (e.g., 100 frames per 2 hours), and start the automated extraction.

### 2. Command-Line Interface (CLI)

```bash
# Basic run with model defined in config.py
python -m frame_selector /path/to/videos/*.mp4 --out frames_for_annotation/ --budget 150

# Reusing a cached detection dump (skips GPU inference, instantly recalculates selection)
python -m frame_selector /path/to/video.mp4 --detections dump.npz --out frames_for_annotation/
```

### 3. Pipeline Output Format

The output directory contains:
```
frames_for_annotation/
├── images/             # Full-resolution JPEG frames
│   ├── frame_001240.jpg
│   └── ...
├── labels/             # YOLO format annotation text files (class_id cx cy w h)
│   ├── frame_001240.txt
│   └── ...
├── classes.txt         # List of class names matching the label IDs
├── manifest.csv        # Detailed table with frame index, timestamp, and selection reasons
└── review.html         # Visual gallery of selected frames with bounding box overlays
```

## Active Learning Pipeline Architecture

```
 Raw Surveillance Videos (.mp4, .dav)
                 |
                 v
 +--------------------------------+
 | 1. Hardware Decode & Detection |  (PyNvVideoCodec + TensorRT FP16)
 +---------------+----------------+
                 |
                 v
 +--------------------------------+
 | 2. Multi-Object Tracking       |  (OC-SORT Kalman Filtering)
 +---------------+----------------+
                 |
                 v
 +--------------------------------+
 | 3. Candidate Flagging          |  (Uncertainty, Misses, Class Flip, Density)
 +---------------+----------------+
                 |
                 v
 +--------------------------------+
 | 4. Budget-Constrained Selection|  (Temporal dispersion & reason balancing)
 +---------------+----------------+
                 |
                 v
 +--------------------------------+
 | 5. Export & Pseudo-Labeling    |  (YOLO .txt, Manifest CSV, Review HTML)
 +---------------+----------------+
                 |
                 v
 Roboflow / Label Studio / CVAT Annotation
```

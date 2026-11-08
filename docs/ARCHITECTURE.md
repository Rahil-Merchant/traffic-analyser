# System Architecture & Technical Design

This document details the internal architecture, thread concurrency model, queue semantics, and execution flow of the **Urban Reconnaissance** processing engine.

---

## High-Level System Overview

The system is designed to achieve maximum hardware saturation (both multi-core CPU and NVIDIA Tensor Core GPUs) while handling multi-hour 4K/1080p surveillance video feeds with minimal memory footprint.

```
                          +---------------------------+
                          |    Desktop GUI (Tauri)    |
                          +-------------+-------------+
                                        | HTTP / WebSocket
                                        v
                          +---------------------------+
                          |     dev_server (FastAPI)  |
                          +-------------+-------------+
                                        | Dispatches Task
                                        v
+------------------------------------------------------------------------------+
|                    VideoProcessor Threading Architecture                     |
|                                                                              |
|   +----------------+   raw_frame_queue   +----------------+                  |
|   |  Reader Thread |-------------------->| Preprocessor   |                  |
|   | (PyVideoReader)|   (capped queue)    | Thread         |                  |
|   +----------------+                     +-------+--------+                  |
|                                                  | preprocessed_tensor_queue |
|                                                  v                           |
|   +----------------+                     +----------------+                  |
|   | Post-processor |<--------------------| Inference      |                  |
|   | Thread         |    results_queue    | Worker (Main)  |                  |
|   +-------+--------+                     +----------------+                  |
|           |                                                                  |
|           | video_write_queue (Optional)                                     |
|           v                                                                  |
|   +----------------+                                                         |
|   | Writer Thread  |                                                         |
|   | (OpenCV MP4)   |                                                         |
|   +----------------+                                                         |
+------------------------------------------------------------------------------+
```

---

## Concurrency & Pipeline Stages

The `VideoProcessor` (`core/video_processor.py`) implements a pipelined producer-consumer pattern spanning 4 primary threads communicating across bounded `queue.Queue` buffers:

### 1. Frame Reader Thread (`_frame_reader_task_pvr_segmented`)
- **Technology**: Utilizes `video_reader.PyVideoReader` (Rust bindings wrapping FFmpeg/GPU NVDEC) with fallback to `cv2.VideoCapture`.
- **Segmented Decoding**: Reads video in segments (`PYVIDEOREADER_DECODE_SEGMENT_SIZE = 7600` frames) to prevent memory allocation spikes on huge files.
- **Sampling & Decimation**: Dynamically decimates frames based on source FPS vs. `TARGET_FPS` (e.g. 30 FPS input -> sampled at 20 FPS).
- **DAV Conversion**: Automatically detects proprietary surveillance `.dav` files and invokes an internal FFmpeg converter (`core/converters.py`) to produce intermediate MP4s.

### 2. Preprocessor Thread (`_preprocessor_task`)
- **Letterboxing**: Scales frames while preserving aspect ratio into the model resolution (`MODEL_INPUT_SIZE = 640`), computing unpadding offsets `(dw, dh)`.
- **Normalization & Layout**: Converts BGR `uint8` images into planar `float32` tensors (`(B, 3, H, W)`), normalized to $[0.0, 1.0]$.
- **Batch Assembly**: Dynamically groups consecutive frames into batches matching `OPTIMAL_BATCH_SIZE` (default 64) before submitting to the tensor queue.

### 3. Inference Engine (Main Thread / CUDA Streams)
- **Engine Backends**: Supports native Ultralytics PyTorch models (`.pt`) and high-performance TensorRT engines (`.engine`).
- **Parallel CUDA Streams**: Leverages `torch.cuda.Stream` with a `ThreadPoolExecutor` across `NUM_INFERENCE_STREAMS` to parallelize tensor transfers and kernel execution.
- **Batched Execution**: Employs FP16 mixed-precision inference. Batches smaller than `OPTIMAL_BATCH_SIZE` are zero-padded to ensure fixed-dimension TensorRT engine execution.
- **Patchwise (SAHI) Mode**: If `USE_PATCHWISE_INFERENCE` is active, invokes `PatchwiseDetector` which tiles the frame into overlapping crops, performs batched inference across all crops, and resolves overlaps via Non-Maximum Suppression (NMS) or Intersection-over-Subdivision (IOS).

### 4. Post-Processor Thread (`_postprocessor_task`)
- **Bounding Box Unpadding**: Maps coordinates from letterbox space back to original video pixel space.
- **Tracker Updates**: Forwards raw detections `[x1, y1, x2, y2, score, class_id]` into the tracking engine (`OCSort` or `ByteTrack`).
- **Vehicle Stitching**: Invokes `VehicleCounter` to attach track fragments to physical vehicles.
- **Zone Polygon Intersections**: Computes vectorised point-in-polygon tests to log entry/exit events.
- **Speed Computation**: Transforms bounding box contact points to metric world coordinates via `PerspectiveTransformer` and computes velocity.
- **Visualization Buffer**: (If enabled) Renders bounding boxes, trajectories, and directional vectors to video frames and queues them for writing.

### 5. Video Writer Thread (`_frame_writer_task`)
- Runs when `ENABLE_VISUALIZATION = True`.
- Writes compressed MP4 output using OpenCV's `VideoWriter` with `mp4v` codec in an isolated thread to prevent disk I/O from stalling the post-processor.

---

## State & Memory Management

### Finalized Paths SQLite Store (`core/finalized_paths_store.py`)
In standard computer vision systems, accumulating trajectories in memory for long videos (e.g. 12-hour video with 50,000+ vehicles) causes high RAM consumption.
- The platform uses `FinalizedPathsStore`, an in-memory or on-disk **SQLite table** indexing every completed vehicle path.
- Trajectory coordinates and speed logs are stored as serialised JSON blobs.
- Enables queries, fast updates during post-analysis recalibration, and zero RAM leaks.

### Graceful Abort & Thread Flushing (`abort()`)
Aborting a multi-threaded GPU pipeline without causing deadlocks requires careful coordination:
1. Sets `stop_event = threading.Event()`.
2. Injects sentinel objects (`None`) into all active queues (`raw_frame_queue`, `preprocessed_tensor_queue`, `results_queue`, `video_write_queue`).
3. Launches a daemon thread that aggressively flushes all items out of queues until all worker threads terminate (`is_alive() == False`).
4. Releases global execution locks and invokes CUDA cache cleanup (`torch.cuda.empty_cache()` + `gc.collect()`).

---

## Telemetry & Metrics (`core/performance.py`)

The `PerformanceTracker` monitors pipeline health and publishes metrics every `PERFORMANCE_SAMPLE_INTERVAL` (15s):
- **Stage Timings**: Millisecond breakdown of `decode`, `preprocessing`, `inference`, `tracking`, and `reporting`.
- **Hardware Telemetry**: Real-time GPU utilization, VRAM usage, and GPU temperature via `pynvml` (NVIDIA Management Library).
- **Throughput Calculation**: Effective processing FPS and dynamic ETA estimation streamed to the frontend via WebSockets.

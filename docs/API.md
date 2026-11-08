# API Reference

The backend exposes a FastAPI server (default port `8080`) that the Tauri desktop application communicates with over HTTP and WebSocket.

## Endpoints

### `GET /api/v1/health`
Health check.

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /api/v1/video-preview`
Generate a JPEG preview from the first frame of a video file.

**Request Body:**
```json
{
  "path": "/absolute/path/to/video.mp4"
}
```

**Response:**
```json
{
  "preview_image_url": "http://127.0.0.1:8080/previews/preview_abc123.jpg",
  "video_width": 1920,
  "video_height": 1080
}
```

---

### `POST /api/v1/process`
Start asynchronous video analysis. Returns immediately with a job ID; connect via WebSocket to receive progress updates.

**Request Body:**
```json
{
  "video_paths": ["/path/to/video1.mp4", "/path/to/video2.mp4"],
  "start_date": "261009",
  "start_time": "0800000",
  "station_number": "ST001",
  "primary_direction": "south",
  "polygons": {
    "south": [[290, 446], [605, 335], [619, 441], [470, 466]],
    "north": [[282, 339], [422, 312], [371, 240], [272, 264]]
  },
  "preview_image_width": 1920,
  "preview_image_height": 1080,
  "perspective_corners": [[100, 200], [500, 200], [550, 400], [50, 400]],
  "road_width_meters": 7.5,
  "road_length_meters": 30.0,
  "length_skew_factor": 1.0,
  "width_skew_factor": 1.0,
  "speedLimitKmh": 60,
  "night_hours_start": "18:00",
  "night_hours_end": "06:00"
}
```

**Response:**
```json
{
  "status": "processing_started",
  "job_id": "job_a1b2c3d4"
}
```

---

### `POST /api/v1/recalibrate-speed`
Recalibrate speed measurements with new road dimensions and regenerate reports without re-processing the video.

**Request Body:** Same as `/api/v1/process` plus `video_duration_formatted` field.

**Response:**
```json
{
  "status": "success",
  "results": {
    "excel_path": "/path/to/report.xlsx",
    "bidirectional_excel_path": "/path/to/link_report.xlsx",
    "tia_excel_path": "/path/to/tia_report.xlsx"
  },
  "statistics": {
    "fleet_average_speed_kmh": 52.3
  }
}
```

---

### `POST /api/v1/abort`
Abort the currently running analysis.

**Response:**
```json
{
  "status": "aborted",
  "message": "Analysis abort signal sent."
}
```

---

### `WebSocket /ws/progress/{job_id}`
Real-time progress updates for a running analysis job.

**Messages (server -> client):**
```json
// Progress update
{ "progress": 45, "status": "23.5 FPS", "eta": "0:02:30" }

// Completion
{
  "status": "complete",
  "results": {
    "excel_path": "...",
    "bidirectional_excel_path": "...",
    "tia_excel_path": "...",
    "video_path": "..."
  },
  "statistics": {
    "total_valid_paths": 1234,
    "processing_time_seconds": 180.5,
    "processing_fps": 45.2,
    "total_frames_processed": 8150,
    "fleet_average_speed_kmh": 52.3
  }
}

// Error
{ "status": "error", "message": "Analysis aborted by user." }
```

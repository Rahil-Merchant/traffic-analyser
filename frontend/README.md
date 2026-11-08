<p align="center">
  <h1 align="center">Urban Reconnaissance Desktop UI</h1>
  <p align="center">
    <strong>Cross-Platform Desktop Application for Traffic Monitoring & Analysis</strong>
  </p>
  <p align="center">
    Built with <strong>Tauri v2</strong>, <strong>React 19</strong>, <strong>TypeScript</strong>, and <strong>Tailwind CSS</strong>
  </p>
</p>

---

## Overview

The frontend is a lightweight, performant desktop client built on Tauri v2 that interacts with the Python backend. It enables operators to configure multi-video processing jobs, draw interactive polygon detection zones, calibrate road perspective for speed calculation, track live analysis via WebSockets, and inspect generated RSA-compliant Excel reports.

## Key Features

- **Interactive Polygon Canvas**:
  - Draw multi-point directional zones (North, South, East, West) directly over a real video preview frame.
  - Draw 4-point perspective calibration quadrilaterals for metric road measurement.
  - Vertex dragging, polygon closing, snapping, and undo/clear capabilities.
- **Video Queue & File Management**:
  - Drag-and-drop file reordering via `@dnd-kit`.
  - Native file picker dialogs through Tauri's dialog API.
  - Automatic extraction of video resolution and aspect ratios.
- **Speed Calibration Controls**:
  - Real-world road width and length configuration (in meters).
  - Skew factor adjustments (length & width skew factors).
  - Night hours interval definition (e.g., 18:00 to 06:00).
  - Speed limit threshold enforcement.
- **Real-Time Job Telemetry**:
  - WebSocket connection (`/ws/progress/{job_id}`) providing live progress percentages, processing FPS, and accurate ETA calculations.
  - Live abort controls to halt running analyses immediately without background zombie processes.
- **Results & Recalibration Modal**:
  - Displays vehicle counts, processing speed, and fleet average speeds.
  - One-click access to generated Excel reports (Standard RSA, Bi-directional Link Count, and TIA formats).
  - **Live Post-Analysis Recalibration**: adjust road measurements post-run and instantly regenerate speed calculations without re-running video inference.

## Component Architecture

```
frontend/src/
├── App.tsx                     # Application state coordinator & API manager
├── index.css                   # Global styles & Tailwind directives
├── main.tsx                    # React root mount
│
├── components/
│   ├── SetupPanel.tsx          # Video selection, metadata & timestamp input
│   ├── ControlsPanel.tsx       # Zone selectors, speed calibration inputs & run button
│   ├── InteractiveCanvas.tsx   # Video frame display & interactive Konva canvas
│   ├── DraggableFileListItem.tsx # Reorderable video playlist item
│   ├── CustomTimeInput.tsx     # Specialized time/date input widget
│   ├── ProcessingOverlay.tsx   # Fullscreen live progress & ETA dialog
│   └── ResultsModal.tsx        # Post-processing summary & report links
│
└── src-tauri/
    ├── tauri.conf.json         # Tauri window config, permissions & build specs
    └── capabilities/           # Security capability grants
```

## Development & Build

### Prerequisites
- **Node.js**: >= 18.x
- **Rust**: Latest stable (`rustup update stable`)
- **System Webview**: WebView2 (Windows), WebKitGTK (Linux), or WebKit (macOS)

### Running in Development

```bash
# Install frontend dependencies
npm install

# Run Vite dev server with Tauri shell
npm run tauri dev
```

### Building Distributable Packages

```bash
# Build production bundle (.exe on Windows, .dmg/.app on macOS, .deb/.AppImage on Linux)
npm run tauri build
```

Built executables are placed in `frontend/src-tauri/target/release/bundle/`.

## API & Communication Protocol

The client communicates with the Python backend running at `http://127.0.0.1:8080`:

| Protocol | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/v1/video-preview` | Extract first frame & video dimensions |
| `POST` | `/api/v1/process` | Initiate asynchronous analysis task |
| `POST` | `/api/v1/recalibrate-speed` | Recalibrate speeds using updated road geometry |
| `POST` | `/api/v1/abort` | Interrupt and cancel current job |
| `WebSocket` | `/ws/progress/{job_id}` | Stream real-time progress, FPS, and ETA |

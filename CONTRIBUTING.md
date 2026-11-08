# Contributing to Urban Reconnaissance

Thank you for contributing to the Urban Reconnaissance project! This guide covers the development setup, conventions, and workflow.

## Development Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.x and TensorRT 10.x
- Node.js 18+ and Rust (latest stable) for the frontend
- Git

### Environment Setup

```bash
# Clone and enter the repo
git clone git@github.com:Rahil-Work/traffic-reporting.git
cd traffic-reporting

# Create a Python virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
cd supervision_modified && pip install -e . && cd ..

# Install frontend dependencies
cd frontend && npm install && cd ..

# Configure your machine in config.py
# Add an entry to PC_PATHS with your machine name and paths
```

### Running in Development

```bash
# Backend (auto-reload enabled)
python dev_server.py

# Frontend (hot-reload via Vite)
cd frontend && npm run tauri dev
```

## Code Organisation

| Directory | Responsibility | Language |
|---|---|---|
| `core/` | Video pipeline, detection, perspective transforms | Python |
| `tracking/` | OC-SORT, ByteTrack, vehicle stitching, zone tracking | Python |
| `models/` | Model loading and device management | Python |
| `reporting/` | Excel report generation (RSA format) | Python |
| `visualization/` | Frame annotation overlays, performance charts | Python |
| `frame_selector/` | Active-learning frame selection tool | Python |
| `frontend/src/` | Desktop UI components | TypeScript/React |
| `frontend/src-tauri/` | Native shell configuration | Rust/JSON |
| `utility_scripts/` | One-off dev tools | Python |
| `tests/` | Unit tests | Python |

## Conventions

### Python
- Use **type hints** for function signatures
- Use **loguru** for logging (`from loguru import logger`)
- Use **numpy** for vectorised operations; avoid Python loops over detections
- Constants go in `config.py`; do not hardcode values in module files

### Frontend
- **React functional components** with hooks
- **TypeScript** for all new code
- **Tailwind CSS** for styling
- State management via React `useState`/`useRef`

### Git
- Branch naming: `feature/<name>`, `fix/<name>`, `r&d`
- Commit messages: short imperative summary; colon-separated prefix for scoping  
  e.g., `Frame selector: GPU decode, batch 64, post-processing in parallel with detection`

## Testing

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_vehicle_counter.py -v

# Run with coverage
python -m pytest tests/ --cov=tracking --cov=core -v
```

### Writing Tests
- Place test files in `tests/` with the `test_` prefix
- Use `pytest` fixtures for shared setup
- Mock heavy dependencies (model loading, GPU operations)

## Performance Profiling

The system has built-in profiling support:

```python
# Enable in config.py
ENABLE_DETAILED_PERFORMANCE_METRICS = True

# For line-level profiling, the @profile decorator is available
# Use viztracer for full trace analysis
```

Output charts are saved to `output/performance_charts/`.

## Architecture Notes

### Pipeline Threading Model

The `VideoProcessor` runs 4 parallel threads connected by sized queues:

1. **Reader**: Decodes video frames using PyVideoReader (GPU-accelerated)
2. **Preprocessor**: Applies letterbox transform and builds CUDA tensors
3. **Inference**: Runs YOLO batched inference on GPU (main thread)
4. **Post-processor**: Applies tracking, zone logic, speed estimation, and optional visualisation

### Vehicle Counting Strategy

The `VehicleCounter` layer sits above the tracker and solves three problems:
- **Stitching**: Re-links broken tracker IDs to the same physical vehicle
- **De-duplication**: Prevents double-counting when tracks overlap
- **Reversal detection**: Splits incorrectly merged vehicles whose motion reverses

### Speed Estimation Pipeline

1. User draws 4 road corners -> `PerspectiveTransformer` computes homography
2. Bounding box bottom-centres are projected to metric 2D coordinates
3. Trajectory is median-filtered (window = 5) then smoothed (window = 9)
4. Cumulative distance / elapsed time -> raw speed
5. Clamped to per-class ceiling and low-confidence multiplier

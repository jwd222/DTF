# Drone Traffic Monitoring System

A multi-camera, real-time traffic monitoring pipeline for aerial drone video streams. The system performs vehicle detection, multi-object tracking, cross-camera fusion, and exposes the results through a REST API with persistent storage.

## Architecture

```
┌────────────┐   ┌────────────┐
│  Drone 1   │   │  Drone 2   │   ... N streams
└─────┬──────┘   └─────┬──────┘
      │                │
  ┌───▼───┐        ┌───▼───┐
  │Detect │        │Detect │       YOLOv26  ·  7 vehicle classes
  │+Track │        │+Track │       BoT-SORT ·  Kalman Filter · CMC · ReID
  └───┬───┘        └───┬───┘
      │   ZMQ Pub       │
      ▼                 ▼
  ┌─────────────────────────┐
  │    Fusion Engine        │      Homography projection
  │  (cross-camera merge)   │      Temporal sync
  └────────────┬────────────┘      Conflict resolution
               │
       ┌───────▼───────┐
       │  Persistence  │            PostgreSQL / SQLite
       │    Worker     │            Async batch writes
       └───────┬───────┘
               │
       ┌───────▼───────┐
       │   FastAPI     │            /api/v1/health
       │  REST Server  │            /api/v1/tracks
       └───────────────┘            /api/v1/stats
```

## Features

- **Vehicle Detection** — YOLOv26n/s backbone with 7 vehicle classes (compact car, SUV, van, truck, bus, rickshaw, motorcycle)
- **Multi-Object Tracking** — BoT-SORT tracker with Kalman filter prediction, Camera Motion Compensation (ORB-based), and OSNet ReID appearance matching
- **Multi-Camera Fusion** — Cross-camera track association via homography-based BEV projection, temporal synchronization, and configurable conflict resolution policies
- **REST API** — FastAPI server with health checks, track history, BEV trajectories, aggregate statistics, and event queries
- **Persistence Layer** — Async PostgreSQL storage with configurable batch flushing; SQLite fallback for development
- **Inter-Process Communication** — ZeroMQ pub/sub telemetry between stream workers and fusion workers
- **Model Export** — ONNX and TensorRT (10.x) export for optimized deployment inference

## Project Structure

```
drone_traffic_demo/
├── config.yaml                  # Main configuration file
├── pyproject.toml               # Package definition and dependencies
├── requirements.txt             # Core dependencies
├── src/drone_traffic/
│   ├── __main__.py              # CLI entrypoint (run / check / api)
│   ├── pipeline.py              # Multiprocess pipeline orchestrator
│   ├── api/                     # FastAPI application and routes
│   ├── core/                    # Configuration, types, registry
│   ├── fusion/                  # Homography fusion, association, temporal sync
│   ├── ingestion/               # Video stream reader (RTSP / file)
│   ├── models/                  # YOLOv26 detector, backbone abstractions
│   ├── persistence/             # SQLAlchemy models, async database
│   ├── tracking/                # BoT-SORT, Kalman filter, CMC, ReID, matching
│   ├── viz/                     # Frame annotation and BEV rendering
│   └── workers/                 # Stream, fusion, persistence worker processes
├── scripts/                     # Training, inference, evaluation, export utilities
├── configs/                     # Training args and dataset configuration
├── tests/
│   ├── unit/                    # Unit tests (config, detector, tracker, CMC, etc.)
│   ├── integration/             # API and fusion integration tests
│   └── benchmarks/              # Inference latency and VRAM benchmarks
├── calib/                       # Homography calibration matrices (.npy)
├── weights/                     # Model weights (.pt, .onnx, .trt)
└── docs/                        # Getting started and advanced guides
```

## Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended: NVIDIA with >= 6 GB VRAM)
- CUDA Toolkit 12.6 + cuDNN 9.x
- PostgreSQL 16+ (or SQLite for development)

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Quick Start

```bash
# Validate configuration
python -m drone_traffic check

# Run single-stream pipeline
python -m drone_traffic run --config config.yaml

# Start the API server
python -m drone_traffic api --host 0.0.0.0 --port 8000
```

## Configuration

All settings are managed through `config.yaml`. Key sections:

| Section | Description |
|---------|-------------|
| `system` | Device (CUDA/CPU), FP16, torch compile, worker count |
| `input` | Resolution, target FPS, letterbox, normalization |
| `streams` | Per-stream video source and homography calibration |
| `models.detector` | YOLOv26 variant, weights, confidence/IoU thresholds |
| `tracking` | BoT-SORT parameters, CMC method, ReID settings |
| `fusion` | Enable/disable, temporal sync tolerance, association metric |
| `persistence` | Database URL, batch size, flush interval |
| `api` | Host and port for the REST server |
| `zmq` | Telemetry and fusion port numbers |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | System status and GPU info |
| GET | `/api/v1/sessions` | List all sessions |
| GET | `/api/v1/sessions/{id}/tracks` | Tracks for a session |
| GET | `/api/v1/tracks/{id}/history` | Observation history (paginated) |
| GET | `/api/v1/tracks/{id}/trajectory` | BEV trajectory as JSON |
| GET | `/api/v1/stats` | Aggregate statistics |
| GET | `/api/v1/events` | Events by time or type |

Interactive documentation is available at `/docs` when the API server is running.

## Testing

```bash
# Unit tests (no GPU or database required)
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# GPU benchmarks
pytest tests/benchmarks/ -v -s

# Coverage report
pytest tests/ --cov=drone_traffic --cov-report=html
```

## Training

```bash
# Train detection head
python scripts/train.py

# Evaluate with MOT metrics
python scripts/evaluate_mot.py

# Prepare ReID training data
python scripts/prepare_reid_data_mot.py
python scripts/train_reid.py

# Export to ONNX / TensorRT
python scripts/export_model.py
```

## Supported Vehicle Classes

| Class ID | Label |
|----------|-------|
| 0 | Compact Car |
| 1 | SUV |
| 2 | Van |
| 3 | Truck |
| 4 | Bus |
| 5 | Rickshaw |
| 6 | Motorcycle |

## Documentation

- [Getting Started Guide](docs/GETTING_STARTED.md) — Full environment setup, installation, and first-run instructions
- [Advanced Guide](docs/ADVANCED_GUIDE.md) — Architecture deep-dive, training, calibration, fusion, optimization, and deployment
- [YOLOv26 Next Steps](docs/YOLOV26_NEXT_STEPS.md) — Migration and training workflow for the YOLOv26 detection backbone

## License

This project is for research and educational purposes.

# Drone Traffic Monitoring System — Getting Started Guide

**Target:** WSL2 (Ubuntu 22.04) · NVIDIA RTX 4050 (6 GB) · Python 3.11

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup (WSL2 + CUDA + Docker)](#2-environment-setup)
3. [Python Virtual Environment](#3-python-virtual-environment)
4. [Install Project Dependencies](#4-install-project-dependencies)
5. [Verify Installation](#5-verify-installation)
6. [Prepare Test Data](#6-prepare-test-data)
7. [Running Tests](#7-running-tests)
8. [Running the Pipeline](#8-running-the-pipeline)
9. [Running the API Server](#9-running-the-api-server)
10. [Configuration Reference](#10-configuration-reference)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

- **Windows 11** with WSL2 support (or native Ubuntu 22.04)
- **NVIDIA RTX 4050 Laptop GPU** (or any CUDA-capable GPU with ≥6 GB VRAM)
- **NVIDIA Driver ≥ 560.x** on Windows host
- **At least 16 GB system RAM**
- **~30 GB free disk space** (CUDA toolkit + PyTorch + dependencies)

---

## 2. Environment Setup

### 2.1 Install WSL2 + Ubuntu 22.04

Open **PowerShell as Administrator**:

```powershell
wsl --install -d Ubuntu-22.04
wsl --status          # Verify: Default Version: 2
```

Launch WSL and verify the kernel:

```bash
uname -r   # Should show 5.15+ (WSL2 kernel)
```

### 2.2 Verify GPU Passthrough

Inside WSL, the GPU should be visible via the **Windows host driver** — do NOT install a Linux driver inside WSL.

```bash
nvidia-smi
# Expected output: shows RTX 4050, driver version, CUDA 12.x
```

### 2.3 Install CUDA Toolkit 12.6

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6

echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvcc --version   # Should show 12.6
```

### 2.4 Install cuDNN 9.x

```bash
sudo apt-get install -y cudnn9-cuda-12
```

### 2.5 Install Docker + NVIDIA Container Toolkit (Optional)

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu jammy stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

---

## 3. Python Virtual Environment

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

---

## 4. Install Project Dependencies

```bash
cd ~/drone_traffic_demo    # or wherever the project lives

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e ".[dev]"
```

To verify PyTorch + CUDA:

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
# Expected: PyTorch 2.x, CUDA available: True
```

---

## 5. Verify Installation

```bash
python -m drone_traffic --help
python -m drone_traffic check
```

Output should show the CLI help and confirm `Config validated successfully.`

---

## 6. Prepare Test Data

### 6.1 PostgreSQL Database (Required for Persistence Tests)

Using Docker:

```bash
docker run -d \
  --name drone-postgres \
  -e POSTGRES_USER=drone \
  -e POSTGRES_PASSWORD=drone \
  -e POSTGRES_DB=drone_traffic \
  -p 5432:5432 \
  postgres:16
```

The unit tests that don't require DB access run without it. Integration tests that touch the DB need it running.

### 6.2 Sample Videos

Place test video files in `data/sample_videos/`:

```bash
data/sample_videos/
├── drone1.mp4    # Aerial traffic video from drone 1
└── drone2.mp4    # Aerial traffic video from drone 2 (overlapping FOV)
```

If you don't have real drone footage, the tests generate synthetic videos automatically (via `cv2.VideoWriter` in `conftest.py`).

### 6.3 Calibration Files

Place homography matrices in `calib/`:

```bash
calib/
├── drone1_homography.npy    # 3×3 numpy array
└── drone2_homography.npy
```

Generate a placeholder for testing:

```python
import numpy as np
np.save("calib/drone1_homography.npy", np.eye(3))
np.save("calib/drone2_homography.npy", np.eye(3))
```

### 6.4 Model Weights

Place backbone weights at `weights/es_ev_l.pt`. Without weights, the system uses random initialization and produces dummy detections — useful for testing the pipeline structure but not for real inference.

---

## 7. Running Tests

### 7.1 All Unit Tests (No GPU/DB required)

```bash
pytest tests/unit/ -v
```

This runs config validation, registry, reader (with synthetic video), backbone, detector (NMS), tracker (Kalman), matching, types, and persistence model tests.

### 7.2 Integration Tests

```bash
# Requires FastAPI (no DB needed for health endpoint)
pytest tests/integration/test_api.py -v

# Fusion temporal sync tests (no DB needed)
pytest tests/integration/test_fusion.py -v

# All integration tests
pytest tests/integration/ -v
```

### 7.3 Benchmarks (Requires GPU)

```bash
# Inference latency benchmark
pytest tests/benchmarks/bench_inference.py -v -s

# VRAM usage benchmark
pytest tests/benchmarks/bench_vram.py -v -s
```

### 7.4 Everything at Once

```bash
# All tests, verbose, show print output
pytest tests/ -v -s

# Skip GPU benchmarks
pytest tests/ -v --ignore=tests/benchmarks/

# Only fast unit tests
pytest tests/unit/ -v -m "not slow"
```

### 7.5 Test Coverage

```bash
pip install pytest-cov
pytest tests/ --cov=drone_traffic --cov-report=html
# Open htmlcov/index.html in browser
```

---

## 8. Running the Pipeline

### 8.1 Single-Stream Mode (Fusion Disabled)

Edit `config.yaml`:

```yaml
streams:
  - name: "drone_1"
    source: "data/sample_videos/drone1.mp4"
    homography: "calib/drone1_homography.npy"

fusion:
  enabled: false

persistence:
  db_url: "sqlite+aiosqlite:///./test.db"   # or PostgreSQL URL
```

Run:

```bash
python -m drone_traffic run --config config.yaml
```

The pipeline processes the video file, runs detection + tracking, and publishes telemetry via ZMQ.

### 8.2 Two-Stream Mode with Fusion

```yaml
streams:
  - name: "drone_1"
    source: "data/sample_videos/drone1.mp4"
    homography: "calib/drone1_homography.npy"
  - name: "drone_2"
    source: "data/sample_videos/drone2.mp4"
    homography: "calib/drone2_homography.npy"

fusion:
  enabled: true
  max_time_sync_diff: 0.05
```

```bash
python -m drone_traffic run --config config.yaml
```

### 8.3 CPU-Only Mode (For Debugging)

```yaml
system:
  device: "cpu"
  fp16: false
  torch_compile: false
```

### 8.4 Stop the Pipeline

Press `Ctrl+C` in the terminal running the pipeline. All worker processes terminate gracefully.

---

## 9. Running the API Server

```bash
python -m drone_traffic api --host 0.0.0.0 --port 8000
```

Endpoints:

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/api/v1/health` | System status + GPU info |
| GET | `/api/v1/sessions` | List all sessions |
| GET | `/api/v1/sessions/{id}/tracks` | Tracks in a session |
| GET | `/api/v1/tracks/{id}/history` | Observation history (paginated) |
| GET | `/api/v1/tracks/{id}/trajectory` | BEV trajectory as JSON |
| GET | `/api/v1/stats` | Aggregate statistics |
| GET | `/api/v1/events` | Events by time/type |

Interactive docs: `http://localhost:8000/docs`

---

## 10. Configuration Reference

All settings live in `config.yaml`. Key sections:

| Section | Purpose |
|---------|---------|
| `system` | Device (cuda/cpu), FP16, torch.compile, worker count |
| `input` | Resolution, target FPS, letterbox, normalization |
| `streams` | Per-stream video source + homography file |
| `models.backbone` | Backbone type + weights path, frozen toggle |
| `models.detector` | Head type, class count, confidence/IoU thresholds |
| `tracking` | Tracker type, max_age, min_hits, CMC settings, ReID |
| `fusion` | Enable/disable, sync tolerance, association, conflict policy |
| `persistence` | Database URL, batch size, flush interval |
| `zmq` | Telemetry and fusion port numbers |
| `api` | FastAPI host + port |
| `logging` | Level, format, file output |
| `output` | Annotated frame saving, BEV canvas size |

Validate without running:

```bash
python -m drone_traffic check --config config.yaml
```

---

## 11. Troubleshooting

| Problem | Solution |
|---------|----------|
| `nvidia-smi` fails in WSL | Update Windows NVIDIA driver to ≥560.x, reboot |
| `torch.cuda.is_available()` returns False | Reinstall PyTorch with `--index-url https://download.pytorch.org/whl/cu126` |
| CUDA OOM (out of memory) | Reduce `input.resolution` to `[640, 640]` or `[320, 320]`; set `system.fp16: true` |
| `cv2.imshow` fails in WSL | Use headless OpenCV (already configured). Save frames to disk instead |
| Slow file I/O | Don't store videos on `/mnt/c/` — keep them inside WSL filesystem (`~/`) |
| ZMQ connection refused | Ensure no other process uses ports 5555/5556; check with `ss -tlnp \| grep -E '5555\|5556'` |
| PostgreSQL connection refused | Ensure Docker container is running: `docker ps`, `docker logs drone-postgres` |
| `torch.compile` errors | Disable: set `system.torch_compile: false` in config.yaml |
| Tests fail with ImportError | Ensure `.venv` is activated and `pip install -e ".[dev]"` was run |

---

## Next Steps After Getting Started

1. **Obtain model weights** for the EfficientSAM3 backbone and place in `weights/`
2. **Prepare annotated aerial training data** (COCO-format) for the detection head
3. **Train the YOLO detection head** (see Advanced Guide)
4. **Calibrate drone cameras** and generate homography matrices (see Advanced Guide)
5. **Run end-to-end with real drone footage**
6. **Run benchmarks** to verify FPS and VRAM targets
7. **Enable fusion** for multi-stream tracking

# Advanced Developer Guide — Train, Test, Deploy

This guide covers the full development lifecycle: training the detection head, calibrating cameras, running benchmarks, multi-stream fusion, production deployment, and debugging.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Training the Detection Head](#2-training-the-detection-head)
3. [Camera Calibration & Homography](#3-camera-calibration--homography)
4. [Single-Stream Pipeline Deep Dive](#4-single-stream-pipeline-deep-dive)
5. [Multi-Stream Fusion](#5-multi-stream-fusion)
6. [Performance Optimization](#6-performance-optimization)
7. [TensorRT Export (Phase 2)](#7-tensorrt-export)
8. [Testing Strategy](#8-testing-strategy)
9. [Production Deployment](#9-production-deployment)
10. [Debugging & Profiling](#10-debugging--profiling)
11. [Extending the System](#11-extending-the-system)
12. [Implementation Roadmap Status](#12-implementation-roadmap-status)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    config.yaml                            │
└──────────┬───────────────────────────────────────────────┘
           │
  ┌────────▼────────┐
  │ PipelineManager │  orchestrator (main process)
  └──┬──────┬──────┘
     │      │
  ┌──▼──┐ ┌▼───────┐  ┌──────────────┐  ┌─────────────┐
  │Wkr 1│ │Wkr 2   │  │Fusion Worker │  │Persistence  │
  │     │ │        │  │              │  │Worker       │
  │Read │ │Read    │  │Sync Buffer   │  │Batch Insert │
  │BB+Det│ │BB+Det  │  │Homography   │  │PostgreSQL   │
  │CMC   │ │CMC     │  │Association  │  │             │
  │Track │ │Track   │  │ConflictRes  │  │             │
  └──┬───┘ └──┬─────┘  └──────┬──────┘  └──────┬──────┘
     │        │               │                 │
     └──┬─────┘               │                 │
        │ ZMQ PUB             │ ZMQ PUB         │ ZMQ SUB
        ▼                     ▼                 │
   port 5555              port 5556 ◄───────────┘
                               │
                          ┌────▼────┐
                          │ FastAPI │  query-only
                          └─────────┘
```

**Key design decisions:**
- **Multiprocessing** (not multithreading): bypasses GIL for CPU-bound tracking/CMC
- **ZMQ pub/sub**: lock-free inter-process IPC with natural backpressure handling
- **MessagePack serialization**: compact binary format, faster than JSON
- **Component registry**: swap any backbone/detector/tracker via config without code changes
- **PostgreSQL async writer**: batch-flush every 0.5s or 100 observations, whichever comes first

---

## 2. Training the Detection Head

The YOLO detection head sits on top of the frozen EfficientSAM3 backbone. Only the head is trained.

### 2.1 Prepare Training Data

Your dataset needs COCO-format annotations for aerial/traffic scenes:

```
data/
└── training/
    ├── images/
    │   ├── train/
    │   │   ├── img_0001.jpg
    │   │   └── ...
    │   └── val/
    │       └── ...
    └── labels/
        ├── train/
        │   ├── img_0001.txt    # YOLO format: class_id cx cy w h
        │   └── ...
        └── val/
            └── ...
```

**Class mapping** (from `configs/data.yaml`, VisDrone MOT):

| Class ID | Label |
|----------|-------|
| 0 | car |
| 1 | van |
| 2 | truck |
| 3 | rickshaw |
| 4 | bus |
| 5 | motorcycle |

If you have COCO JSON annotations, convert to YOLO format:

```python
# scripts/convert_coco_to_yolo.py (conceptual)
import json

with open("annotations.json") as f:
    data = json.load(f)

for img in data["images"]:
    width, height = img["width"], img["height"]
    anns = [a for a in data["annotations"] if a["image_id"] == img["id"]]
    lines = []
    for a in anns:
        cx = a["bbox"][0] / width
        cy = a["bbox"][1] / height
        w  = a["bbox"][2] / width
        h  = a["bbox"][3] / height
        lines.append(f"{a['category_id']} {cx} {cy} {w} {h}")
    with open(f"labels/{img['file_name'].replace('.jpg','.txt')}", "w") as f:
        f.write("\n".join(lines))
```

### 2.2 Data Augmentation

Create a training dataset class:

```python
# scripts/train_detector.py
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AerialTrafficDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=640, num_classes=6):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.num_classes = num_classes
        self.images = sorted(Path(image_dir).glob("*.jpg"))

        self.transform = A.Compose([
            A.RandomSizedBBoxSafeCrop(img_size, img_size, erosion_rate=0.2),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.images[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_path = Path(self.label_dir) / (self.images[idx].stem + ".txt")
        bboxes, class_ids = [], []
        if label_path.exists():
            for line in label_path.read_text().strip().split("\n"):
                parts = line.strip().split()
                class_ids.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:5]])

        if not bboxes:
            bboxes = [[0, 0, 0, 0]]
            class_ids = [0]

        transformed = self.transform(image=img, bboxes=bboxes, class_ids=class_ids)
        return transformed["image"], torch.tensor(transformed["class_ids"]), torch.tensor(transformed["bboxes"])
```

### 2.3 Training Loop

```python
# scripts/train_detector.py (continued)
from drone_traffic.models.backbone_base import DummyBackbone
from drone_traffic.models.efficient_sam3 import EfficientSAM3Backbone
from drone_traffic.models.yolo_head import YOLODetectionHead

device = torch.device("cuda")

backbone = EfficientSAM3Backbone(weights="weights/es_ev_l.pt", frozen=True, device=str(device))

detector = YOLODetectionHead(
    in_channels=backbone.output_channels,
    num_classes=6,
    device=str(device),
)

optimizer = torch.optim.AdamW(detector.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

dataset = AerialTrafficDataset("data/training/images/train", "data/training/labels/train")
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=custom_collate)

for epoch in range(100):
    detector.train()
    for images, class_ids, bboxes in loader:
        images = images.to(device)

        with torch.no_grad():
            features = backbone(images)

        pred = detector._decode_predictions([...])
        loss = compute_yolo_loss(pred, bboxes, class_ids)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(detector.parameters(), 10.0)
        optimizer.step()

    scheduler.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")

torch.save(detector.state_dict(), "weights/yolo_head_trained.pt")
```

### 2.4 YOLO Loss Function

The detection head needs a YOLO-style loss (BCE for objectness + classification, CIoU for bounding box regression). For a complete implementation, refer to ultralytics' loss module or implement:

```python
# Loss = λ_bbox * CIoU_loss + λ_obj * BCE(obj_pred, obj_target) + λ_cls * BCE(cls_pred, cls_target)
# λ_bbox = 7.5, λ_obj = 1.0, λ_cls = 0.5 (default YOLO ratios)
```

### 2.5 Saving and Loading Trained Weights

```python
# Save
torch.save({
    "detector": detector.state_dict(),
    "backbone": backbone.state_dict(),
    "config": config_dict,
}, "weights/full_model.pt")

# Load in pipeline — update config.yaml:
# models.detector.weights: "weights/yolo_head_trained.pt"
```

---

## 3. Camera Calibration & Homography

### 3.1 Ground Control Points (GCP) Method

1. Mark 4+ points visible from each drone camera on the ground plane
2. Measure their real-world (BEV) coordinates in meters from a common origin
3. Collect their pixel coordinates from a reference frame

```python
import numpy as np
import cv2

# Pixel coordinates (from image) — must be in order
src_pts = np.array([
    [320, 120],   # top-left
    [960, 120],   # top-right
    [960, 520],   # bottom-right
    [320, 520],   # bottom-left
], dtype=np.float32)

# Real-world BEV coordinates in meters
dst_pts = np.array([
    [0, 0],
    [50, 0],
    [50, 30],
    [0, 30],
], dtype=np.float32)

H = cv2.getPerspectiveTransform(src_pts, dst_pts)
np.save("calib/drone1_homography.npy", H)
```

### 3.2 Verifying Homography Accuracy

```python
# Test reprojection error should be < 2 pixels
test_pt = np.array([640, 320, 1.0], dtype=np.float64)
projected = H @ test_pt
projected = projected[:2] / projected[2]
print(f"BEV position: {projected} meters")
```

### 3.3 Online Recalibration

For drones without gimbal stabilization, the homography drifts over time. The `ORBCMC` module compensates for frame-to-frame motion. For full recalibration:

```python
# In config.yaml
fusion:
  homography_method: "orb_online"   # Future: use feature matching to update H per frame
```

---

## 4. Single-Stream Pipeline Deep Dive

### 4.1 Per-Frame Processing Order

```
VideoReader.read_frames()
  → FramePacket (preprocessed tensor 3×640×640)
    → Backbone.forward(tensor)  →  features {"c1": B×128×H/8×W/8, "c2": B×256×H/16×W/16, "c3": B×512×H/32×W/32}
      → Detector.forward(features, ...)  →  list[Detection]
        → CMC.compute(prev_frame, curr_frame)  →  3×3 affine matrix
          → Tracker.update(detections, raw_frame)  →  list[TrackState]
            → ZMQ publish TelemetryMessage
```

### 4.2 Running Headless (No Video File)

Write a synthetic data generator:

```python
# scripts/synthetic_pipeline.py
import numpy as np
import torch
from drone_traffic.core.config import load_config
from drone_traffic.core.registry import build_component
from drone_traffic.ingestion.reader import FramePacket

config = load_config("config.yaml")
device = torch.device("cpu")  # for quick testing

backbone = build_component("backbone", "dummy_backbone", channels=[64, 128, 256])
detector = build_component("detector", "yolo_head", in_channels=[64, 128, 256], num_classes=6)
tracker = build_component("tracker", "bot_sort")

for i in range(100):
    tensor = torch.randn(1, 3, 640, 640)
    features = backbone(tensor)
    detections = detector(features, (640, 640, 3), 1.0, (0, 0, 0, 0))
    tracks = tracker.update(detections, timestamp=i/30.0)
    print(f"Frame {i}: {len(detections)} dets, {len(tracks)} tracks")
```

---

## 5. Multi-Stream Fusion

### 5.1 How Fusion Works

1. **Temporal Sync Buffer** collects telemetry from all streams, aligning by timestamp within `max_time_sync_diff` (default 50ms)
2. **Homography Projection** transforms per-camera bboxes to BEV coordinates using pre-loaded matrices
3. **Track-to-Track Association** uses Hungarian matching on Mahalanobis distance in BEV space
4. **Conflict Resolution** merges matching tracks (policy: `merge`) and creates new global IDs for unmatched ones

### 5.2 Enabling Fusion

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
  max_time_sync_diff: 0.05          # 50ms sync window
  association:
    metric: "mahalanobis"
    threshold: 2.0                   # chi-squared 2-DOF threshold
  conflict_resolution:
    policy: "merge"                  # or "keep_separate"
```

### 5.3 Testing Fusion with Synthetic Data

```python
from drone_traffic.fusion.conflict_resolver import ConflictResolver
from drone_traffic.fusion.temporal_sync import TemporalSyncBuffer

sync = TemporalSyncBuffer(max_time_diff=0.05, sources_expected=2)
sync.add("drone_1", {"drone_id": "drone_1", "timestamp": 1.0, "frame_id": 0, "tracks": [
    {"id": 1, "bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}, "confidence": 0.9, "class_id": 0}
]})
sync.add("drone_2", {"drone_id": "drone_2", "timestamp": 1.02, "frame_id": 0, "tracks": [
    {"id": 5, "bbox": {"x1": 150, "y1": 150, "x2": 250, "y2": 250}, "confidence": 0.85, "class_id": 0}
]})

result = sync.try_flush()
assert result is not None

resolver = ConflictResolver(policy="merge")
fused = resolver.resolve(result)
print(f"Global tracks: {fused['global_tracks']}")
```

---

## 6. Performance Optimization

### 6.1 VRAM Budget (RTX 4050 6 GB)

| Component | VRAM |
|-----------|------|
| Backbone (FP16) | ~250 MB |
| Detection Head (FP16) | ~15 MB |
| Frame buffers (2×640×640) | ~9 MB |
| Feature maps + intermediates | ~200 MB |
| CUDA context overhead | ~300 MB |
| **Total** | **~800 MB** |

Leaves ~5.2 GB headroom.

### 6.2 Optimization Levers

**Already enabled by default:**
- FP16 inference (`system.fp16: true`)
- `torch.compile` with `reduce-overhead` mode
- CMC on CPU (no GPU contention)
- Kalman tracker on CPU

**Manual optimizations:**
```python
# 1. Memory pinning for faster CPU→GPU transfer
tensor = tensor.pin_memory().to(device, non_blocking=True)

# 2. CUDA streams for concurrent inference (stream_worker.py already uses multiprocessing)
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    features = backbone(tensor)

# 3. Disable gradient checkpointing (not needed at inference)
torch.set_grad_enabled(False)
```

**If OOM occurs:**
1. Reduce resolution: `input.resolution: [320, 320]`
2. Use smaller backbone: swap `efficient_sam3` → `dummy_backbone` with fewer channels
3. Disable ReID: `tracking.reid.enabled: false`
4. Run streams sequentially: `system.num_workers: 1`

### 6.3 Target Performance

| Metric | Target |
|--------|--------|
| Backbone forward (FP16, 640²) | ≤ 8 ms |
| Detection head forward | ≤ 2 ms |
| BoT-SORT update (30 tracks) | ≤ 1 ms |
| CMC (ORB) | ≤ 5 ms |
| **Total per-frame (1 stream)** | **≤ 20 ms (50 FPS)** |
| **Total per-frame (2 streams)** | **≤ 40 ms (25 FPS)** |
| Peak VRAM (2 streams, 640²) | < 2 GB |

---

## 7. TensorRT Export

After validating the PyTorch pipeline, export to TensorRT for ~2× speedup:

### 7.1 Export to ONNX

```python
import torch
from drone_traffic.models.efficient_sam3 import EfficientSAM3Backbone
from drone_traffic.models.yolo_head import YOLODetectionHead

backbone = EfficientSAM3Backbone(weights="weights/es_ev_l.pt", frozen=True, device="cpu")
detector = YOLODetectionHead(in_channels=backbone.output_channels, num_classes=6, device="cpu")

class CombinedModel(torch.nn.Module):
    def __init__(self, backbone, detector):
        super().__init__()
        self.backbone = backbone
        self.detector = detector
    def forward(self, x):
        feats = self.backbone(x)
        return self.detector._decode_predictions(
            [self.detector.necks[i](feats[k]) for i, k in enumerate(sorted(feats.keys()))]
        )

model = CombinedModel(backbone, detector).eval()
dummy = torch.randn(1, 3, 640, 640)

torch.onnx.export(
    model, dummy, "weights/model.onnx",
    opset_version=17,
    input_names=["images"],
    output_names=["predictions"],
    dynamic_axes={"images": {0: "batch"}, "predictions": {0: "batch"}},
)
```

### 7.2 Build TensorRT Engine

```bash
trtexec --onnx=weights/model.onnx --fp16 --saveEngine=weights/model_fp16.trt --workspace=2048
```

### 7.3 Integrate TensorRT Engine

Create a `TensorRTDetector` that implements `DetectorInterface`:

```python
import tensorrt as trt

@register_detector("tensorrt")
class TensorRTDetector(DetectorInterface):
    def __init__(self, engine_path="weights/model_fp16.trt", **kwargs):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        # Allocate device buffers...
```

---

## 8. Testing Strategy

### 8.1 Test Categories

```
tests/
├── unit/            # Fast, no external deps
│   ├── test_config.py       # Config parsing, validation
│   ├── test_registry.py     # Component registration
│   ├── test_reader.py       # VideoReader (synthetic video)
│   ├── test_backbone.py     # Backbone output shapes
│   ├── test_detector.py     # NMS correctness
│   ├── test_cmc.py          # CMC affine on synthetic motion
│   ├── test_tracker.py      # Kalman predict/update, track lifecycle
│   ├── test_matching.py     # Hungarian + IoU distance
│   ├── test_types.py        # BBox math (IoU, to_cxcyah)
│   └── test_persistence.py  # ORM model construction
├── integration/     # Requires some setup
│   ├── test_api.py          # FastAPI endpoints (no DB)
│   └── test_fusion.py       # Temporal sync, conflict resolution
└── benchmarks/      # Requires GPU
    ├── bench_inference.py   # FPS/latency with torch.cuda.Event
    └── bench_vram.py        # Peak VRAM measurement
```

### 8.2 Running Tests by Category

```bash
pytest tests/unit/ -v                              # All unit tests
pytest tests/unit/test_tracker.py -v -k "kalman"  # Specific test
pytest tests/integration/ -v                       # Integration tests
pytest tests/benchmarks/ -v -s -m gpu              # GPU benchmarks (verbose)
pytest tests/ -v --tb=short                        # Everything, short tracebacks
```

### 8.3 Writing New Tests

Follow the existing pattern in `tests/unit/`:

```python
# tests/unit/test_my_feature.py
import pytest
from drone_traffic.my_module import MyComponent

def test_basic_behavior():
    comp = MyComponent(param=10)
    result = comp.process(input_data)
    assert result is not None
    assert len(result) > 0

def test_edge_case():
    comp = MyComponent(param=0)
    with pytest.raises(ValueError):
        comp.process(None)
```

Fixtures are defined in `tests/conftest.py` — add shared fixtures there.

### 8.4 Accuracy Validation

**Detection (if annotated data available):**
```python
# Compute mAP@0.5 on validation set
from scripts.evaluate_detection import compute_map
map50, map50_95 = compute_map(predictions, ground_truth)
assert map50 > 0.80, f"mAP@0.5 = {map50:.3f}, target > 0.80"
```

**Tracking (MOT metrics):**
```python
# Use motmetrics library
import motmetrics as mm
acc = mm.MOTAccumulator()
# ... fill with frame-by-frame comparisons
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1'])
```

---

## 9. Production Deployment

### 9.1 Docker Deployment

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

COPY config.yaml .
COPY weights/ weights/
COPY calib/ calib/
COPY src/ src/

CMD ["python", "-m", "drone_traffic", "run", "--config", "config.yaml"]
```

```bash
docker build -t drone-traffic .
docker run --gpus all -v ./data:/app/data drone-traffic
```

### 9.2 Docker Compose (Pipeline + PostgreSQL)

```yaml
# docker-compose.yml
version: "3.8"
services:
  drone-traffic:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./config.yaml:/app/config.yaml
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    depends_on:
      - postgres

  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: drone
      POSTGRES_PASSWORD: drone
      POSTGRES_DB: drone_traffic
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

```bash
docker compose up -d
```

### 9.3 Environment Variables

Override any config value via environment variables (create `.env`):

```bash
# .env
DRONE_TRAFFIC_SYSTEM_DEVICE=cuda
DRONE_TRAFFIC_PERSISTENCE_DB_URL=postgresql+asyncpg://user:pass@host:5432/db
DRONE_TRAFFIC_LOGGING_LEVEL=DEBUG
```

### 9.4 Live Drone Feed (Future)

Replace video file source with RTSP stream:

```yaml
streams:
  - name: "drone_1"
    source: "rtsp://192.168.1.100:8554/live"
```

The `VideoReader` uses `cv2.VideoCapture` which supports RTSP natively.

---

## 10. Debugging & Profiling

### 10.1 Enable Debug Logging

```yaml
logging:
  level: "DEBUG"
```

### 10.2 Profile Per-Stage Latency

```python
# In stream_worker.py, already logs every 100 frames
# For fine-grained profiling:
import torch

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
features = backbone(tensor)
end.record()
torch.cuda.synchronize()
print(f"Backbone: {start.elapsed_time(end):.2f} ms")
```

### 10.3 Profile VRAM

```python
torch.cuda.reset_peak_memory_stats()
# ... run pipeline ...
print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
print(f"Current VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
```

### 10.4 PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    features = backbone(tensor)
    detections = detector(features, ...)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

### 10.5 ZMQ Debugging

```bash
# Subscribe to all telemetry messages
python -c "
import zmq, msgpack
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect('tcp://localhost:5555')
sub.setsockopt(zmq.SUBSCRIBE, b'')
while True:
    topic = sub.recv_string()
    msg = msgpack.unpackb(sub.recv(), raw=False)
    print(f'{topic}: {len(msg[\"tracks\"])} tracks')
"
```

---

## 11. Extending the System

### 11.1 Adding a New Backbone

```python
# src/drone_traffic/models/my_backbone.py
from drone_traffic.core.registry import register_backbone
from drone_traffic.models.backbone_base import BackboneInterface

@register_backbone("resnet50")
class ResNet50Backbone(BackboneInterface):
    def __init__(self, weights=None, frozen=True, device="cuda"):
        super().__init__()
        # ... your implementation ...

    def forward(self, x):
        # Return {"c1": feat1, "c2": feat2, "c3": feat3}
        pass

    @property
    def output_channels(self):
        return [256, 512, 1024]

    @property
    def output_strides(self):
        return [8, 16, 32]
```

Then in `config.yaml`:

```yaml
models:
  backbone:
    type: "resnet50"       # <-- changed
    weights: "weights/resnet50.pt"
```

### 11.2 Adding a New Tracker

```python
from drone_traffic.core.registry import register_tracker
from drone_traffic.tracking.tracker_base import TrackingInterface

@register_tracker("deepsort")
class DeepSORTTracker(TrackingInterface):
    def __init__(self, **kwargs):
        super().__init__()
        # ... your implementation ...

    def update(self, detections, frame=None, timestamp=0.0):
        # Return list[TrackState]
        pass

    def get_active_tracks(self):
        pass

    def reset(self):
        pass
```

### 11.3 Adding a New API Endpoint

```python
# In src/drone_traffic/api/routes.py, add:

@router.get("/cameras")
async def list_cameras(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Camera))
    return result.scalars().all()
```

---

## 12. Implementation Roadmap Status

### Phase 1: Single-Stream Baseline (Weeks 1-3) — CODE SKELETON COMPLETE

| Step | Status | Notes |
|------|--------|-------|
| 1.1 Project scaffolding | Done | pyproject.toml, config.yaml, CLI |
| 1.2 core.config | Done | Pydantic models, YAML loader |
| 1.3 core.registry | Done | Decorators + factory |
| 1.4 ingestion.reader | Done | VideoReader with letterbox + normalize |
| 1.5 models.backbone | Done | ABC + DummyBackbone + EfficientSAM3 stub |
| 1.6 models.detector | Done | ABC + YOLODetectionHead + NMS |
| 1.7 tracking.cmc | Done | ORBCMC |
| 1.8 tracking.tracker | Done | BoTSORT + Kalman + matching |
| 1.9 End-to-end pipeline | Done | PipelineManager + stream_worker |
| 1.10 persistence.writer | Done | SQLAlchemy async + ORM models |
| 1.11 Integration test | Needs real video + weights | Unit tests pass |
| **Remaining** | | Obtain real weights, annotate data, train head |

### Phase 2: Multi-Stream + GPU Optimization (Weeks 4-5) — SKELETON COMPLETE

| Step | Status | Notes |
|------|--------|-------|
| 2.1 stream_worker | Done | Multiprocessing + ZMQ pub |
| 2.2 ZMQ message schema | Done | TelemetryMessage + MessagePack |
| 2.3 GPU memory management | Done | FP16 + per-worker CUDA |
| 2.4 fusion engine | Done | HomographyFusionEngine |
| 2.5 temporal_sync | Done | TemporalSyncBuffer |
| 2.6 Performance profiling | Scripts ready | bench_inference.py, bench_vram.py |
| 2.7 TensorRT export | Guide written | See Section 7 |

### Phase 3: Cross-Camera Fusion (Weeks 6-7) — SKELETON COMPLETE

| Step | Status |
|------|--------|
| 3.1 Homography projection | Done (conflict_resolver.py) |
| 3.2 Association (Hungarian) | Done (association.py) |
| 3.3 Conflict resolution | Done (conflict_resolver.py) |
| 3.4 Global tracker | Done (via ConflictResolver) |
| 3.5 End-to-end 2-stream test | Needs real overlapping video |

### Phase 4: API, Viz, Final Integration (Weeks 8-9) — SKELETON COMPLETE

| Step | Status |
|------|--------|
| 4.1 PostgreSQL schema | Done (5 tables + indexes) |
| 4.2 FastAPI endpoints | Done (7 endpoints) |
| 4.3 BEV renderer | Done (renderer.py) |
| 4.4 End-to-end test | Needs full calibration + video |
| 4.5 Documentation | This guide |

### Immediate Next Steps (in priority order):

1. **Obtain EfficientSAM3 weights** and verify loading in `EfficientSAM3Backbone`
2. **Collect/annotate aerial training data** (even 200-500 images is a start)
3. **Train the YOLO detection head** using the training loop in Section 2
4. **Generate synthetic calibration** and verify homography projection
5. **Run single-stream pipeline** on a real video, verify detections + tracks in DB
6. **Benchmark FPS/VRAM** against targets (Section 6.3)
7. **Record 2 overlapping drone videos** and test fusion
8. **Optional: TensorRT export** for production FPS targets

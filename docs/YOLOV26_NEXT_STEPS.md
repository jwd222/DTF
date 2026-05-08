# YOLOv26 Migration — Next Steps Guide

Complete guide from environment setup through training, validation, export, inference, and evaluation.

---

## Phase 1: Environment Setup

### 1.1 Install Dependencies

```bash
conda activate dtf_venv
pip install -r requirements.txt
```

Optional groups (install as needed):

```bash
pip install -e ".[train]"
pip install -e ".[reid]"    # requires torchreid
pip install -e ".[tensorrt]" # requires TensorRT >= 10.0
```

### 1.2 Verify Installation

```bash
python -m pytest tests/unit/ -v
```

All 62 tests should pass.

---

## Phase 2: Dataset Preparation

### 2.1 Organize Detection Dataset

The YOLOv26 detector expects a YOLO-format dataset. Create this directory structure:

```
data/vehicle_dataset/
├── images/
│   ├── train/       # training images (.jpg)
│   ├── val/         # validation images (.jpg)
│   └── test/        # test images (.jpg)
├── labels/
│   ├── train/       # YOLO label files (.txt, one per image)
│   ├── val/
│   └── test/
```

Each label file has one row per object: `class_id cx cy w h` (normalized 0-1).

Class mapping (defined in `configs/data.yaml`):

| ID | Class |
|----|-------|
| 0  | compact_car |
| 1  | suv |
| 2  | van |
| 3  | truck |
| 4  | bus |
| 5  | rickshaw |
| 6  | motorcycle |

### 2.2 Convert VisDrone MOT Annotations

If you have VisDrone MOT-format data (`sequences/` + `annotations/`):

```bash
python scripts/convert_visdrone_mot_to_yolo.py \
    --input-dir /path/to/VisDrone2019-MOT-train \
    --output-dir data/vehicle_dataset \
    --copy-images
```

This converts MOT annotations to YOLO format, mapping VisDrone classes to the 7 vehicle classes:

| VisDrone Class | YOLO ID | YOLO Class |
|----------------|---------|------------|
| car (3) | 0 | compact_car |
| van (4) | 2 | van |
| truck (5) | 3 | truck |
| tricycle (6) | 5 | rickshaw |
| awning-tricycle (7) | 5 | rickshaw |
| bus (8) | 4 | bus |
| motor (9) | 6 | motorcycle |

Ignored: pedestrian (0), people (1), bicycle (2). Use `--val-ratio` to control the train/val split (default 15%). Use `--copy-images` for copies; otherwise symlinks are created.

If you have COCO-format annotations:

```python
from ultralytics.data.converter import convert_coco
convert_coco("annotations.json", use_segments=False)
```

### 2.3 Update Dataset Config

Edit `configs/data.yaml` to point to your dataset root:

```yaml
path: data/vehicle_dataset    # absolute or relative to project root
train: images/train
val: images/val
test: images/test
```

---

## Phase 3: Train YOLOv26 Detector

### 3.1 Review Training Hyperparameters

Edit `configs/train_args.yaml` before training. Key parameters:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `model` | `yolo26s.pt` | Pretrained weights (auto-downloaded on first run) |
| `epochs` | `300` | Total training epochs |
| `imgsz` | `960` | Input resolution — **do not reduce below 960** for drone footage |
| `batch` | `8` | Reduce to 4 or 2 if GPU memory is limited |
| `optimizer` | `MuSGD` | Multi-step SGD |
| `copy_paste` | `0.1` | Copy-paste augmentation for small objects |
| `patience` | `50` | Early stopping patience |

### 3.2 Start Training

```bash
python scripts/train.py --config configs/train_args.yaml
```

To resume from a checkpoint:

```bash
python scripts/train.py --config configs/train_args.yaml --resume runs/train/yolo26_vehicle/weights/last.pt
```

Training outputs are saved to `runs/train/yolo26_vehicle/`.

### 3.3 Monitor Training

Ultralytics logs to TensorBoard by default:

```bash
tensorboard --logdir runs/train
```

Watch for:
- `mAP50` plateauing (should reach >0.80 for decent results)
- `box_loss` decreasing steadily
- Overfitting signs: val loss rising while train loss drops

---

## Phase 4: Validate the Detector

### 4.1 Run Validation

```bash
python scripts/validate.py \
    --weights runs/train/yolo26_vehicle/weights/best.pt \
    --data configs/data.yaml \
    --imgsz 960 \
    --device 0
```

### 4.2 Review Metrics

Key metrics to check:

| Metric | Target | Meaning |
|--------|--------|---------|
| `mAP50` | > 0.80 | Mean AP at IoU 0.5 |
| `mAP50-95` | > 0.50 | Mean AP at IoU 0.5:0.95 |
| Per-class AP | > 0.60 | Each of the 7 vehicle classes |

Per-class output example:

```
mAP50:    0.8543
mAP50-95: 0.5612
  compact_car: AP@50-95 = 0.6234
  suv: AP@50-95 = 0.5891
  van: AP@50-95 = 0.5123
  truck: AP@50-95 = 0.4901
  bus: AP@50-95 = 0.4567
  rickshaw: AP@50-95 = 0.5432
  motorcycle: AP@50-95 = 0.6789
```

### 4.3 Iterate if Needed

If mAP is too low:
- Increase `epochs` to 500
- Add more training data for underperforming classes
- Adjust `copy_paste` and `mixup` augmentation strength
- Verify annotation quality

---

## Phase 5: TensorRT Export (Optional)

### 5.1 Export to TensorRT Engine

Requires TensorRT >= 10.0 installed (`pip install -e ".[tensorrt]"`).

FP16 + INT8 export:

```bash
python scripts/export_model.py \
    --weights runs/train/yolo26_vehicle/weights/best.pt \
    --imgsz 960 \
    --half \
    --int8 \
    --calib-data configs/data.yaml
```

FP16 only (if INT8 calibration data is unavailable):

```bash
python scripts/export_model.py \
    --weights runs/train/yolo26_vehicle/weights/best.pt \
    --imgsz 960 \
    --no-int8
```

The `.engine` file is saved alongside the weights.

### 5.2 Update Config for TensorRT

Edit `config.yaml`:

```yaml
models:
  detector:
    weights: "runs/train/yolo26_vehicle/weights/best.engine"
```

The `YOLOv26Detector` automatically detects the file extension and handles both `.pt` and `.engine` files.

---

## Phase 6: Train Re-ID Model

### 6.1 Prepare Re-ID Training Data from VisDrone MOT

Use the MOT conversion script to crop vehicle patches grouped by track identity:

```bash
python scripts/prepare_reid_data_mot.py \
    --input-dir /path/to/VisDrone2019-MOT-train \
    --output-dir data/reid_crops \
    --min-track-len 5 \
    --max-crops-per-track 30
```

This produces a torchreid-compatible structure:

```
data/reid_crops/
├── train/
│   ├── 000000/
│   │   ├── seq_f000045.jpg
│   │   └── ...
│   └── 000001/
├── gallery/
│   ├── 000000/
│   └── ...
└── query/
    ├── 000000/
    └── ...
```

Each identity folder contains cropped and resized (256×128) patches from one tracked vehicle.

If you have COCO-style video annotations instead:

```bash
python scripts/prepare_reid_data.py \
    --video data/sample_videos/drone1.mp4 \
    --annotations data/annotations/drone1_tracks.json \
    --output-dir data/reid_crops \
    --min-track-len 5 \
    --crop-size 256
```

### 6.2 Train OSNet Re-ID Model

```bash
python scripts/train_reid.py \
    --data-dir data \
    --model osnet_x1_0 \
    --epochs 60 \
    --batch-size 32 \
    --lr 0.0003 \
    --loss triplet \
    --output weights/osnet_reid.onnx
```

The script automatically detects identity count from `data/reid_crops/train/` and registers a custom torchreid dataset. It exports to ONNX when `--output` ends in `.onnx`.

---

## Phase 7: Run Inference

### 7.1 Single-Stream Inference with Visualization

Using the Ultralytics built-in tracker:

```bash
python scripts/inference.py \
    --source data/sample_videos/drone1.mp4 \
    --weights runs/train/yolo26_vehicle/weights/best.pt \
    --imgsz 960 \
    --conf 0.25 \
    --device 0 \
    --output output/tracked_drone1.mp4 \
    --show
```

Press `q` to quit the display window.

### 7.2 Using the Pipeline (Stream Worker)

Update `config.yaml` with your paths and run:

```yaml
models:
  detector:
    weights: "runs/train/yolo26_vehicle/weights/best.pt"

tracking:
  reid:
    enabled: true
    weights: "weights/osnet_reid.onnx"
```

Then launch the stream worker through the application entry point.

### 7.3 Config Tuning

Key config parameters for the tracker (`config.yaml` → `tracking`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `track_high_thresh` | `0.5` | Detections above this go to first matching stage |
| `track_low_thresh` | `0.1` | Detections below this are discarded |
| `new_track_thresh` | `0.6` | Minimum confidence to create a new track |
| `appearance_thresh` | `0.25` | Max cosine distance for appearance gating |
| `proximity_thresh` | `0.5` | Max IoU distance for proximity gating |

---

## Phase 8: Evaluate Tracking

### 8.1 Prepare Ground Truth in MOT Format

MOT format (`gt.txt`), one row per object per frame:

```
frame_id, track_id, x, y, w, h, conf, class, visibility
1, 1, 100, 150, 80, 60, 1, 1, 1
1, 2, 300, 400, 90, 70, 1, 1, 1
2, 1, 105, 155, 80, 60, 1, 1, 1
...
```

### 8.2 Prepare Predictions JSON

```json
[
  {"frame_id": 1, "tracks": [{"track_id": 1, "bbox": [100, 150, 180, 210]}]},
  {"frame_id": 2, "tracks": [{"track_id": 1, "bbox": [105, 155, 185, 215]}]}
]
```

### 8.3 Run MOT Evaluation

```bash
python scripts/evaluate_mot.py \
    --gt data/annotations/drone1_gt.txt \
    --pred data/annotations/drone1_predictions.json \
    --output output/metrics_drone1.json
```

### 8.4 Target Metrics

| Metric | Good | Excellent |
|--------|------|-----------|
| MOTA | > 0.60 | > 0.80 |
| IDF1 | > 0.50 | > 0.70 |
| HOTA | > 0.40 | > 0.60 |
| IDs (ID switches) | < 50 | < 20 |

---

## Phase 9: Troubleshooting

### Common Issues

**CUDA out of memory during training:**
```yaml
# configs/train_args.yaml
batch: 4        # reduce from 8
imgsz: 640      # only as last resort, hurts small-object detection
```

**Low mAP on small vehicles (motorcycles, rickshaws):**
- Increase `copy_paste` to `0.2`
- Increase training epochs
- Ensure dataset has sufficient small-object annotations
- Do not reduce `imgsz` below 960

**Tracker producing too many ID switches:**
- Lower `appearance_thresh` in `config.yaml` (try `0.15`)
- Enable Re-ID: set `tracking.reid.enabled: true` and provide weights
- Increase `max_age` to 60

**Tracker losing tracks:**
- Lower `new_track_thresh` to `0.4`
- Increase `track_buffer` to 60
- Verify detections are not too sparse (check `--conf` threshold)

**Slow inference:**
- Use TensorRT engine (Phase 5)
- Disable Re-ID if not needed (`tracking.reid.enabled: false`)
- Reduce `imgsz` to 640 (trade-off with accuracy)

### File Reference

| File | Purpose |
|------|---------|
| `config.yaml` | Runtime pipeline configuration |
| `configs/data.yaml` | YOLO dataset config (classes + paths) |
| `configs/train_args.yaml` | YOLOv26 training hyperparameters |
| `scripts/train.py` | Launch detector training |
| `scripts/validate.py` | Evaluate detector mAP |
| `scripts/export_model.py` | Export to TensorRT |
| `scripts/train_reid.py` | Train OSNet Re-ID model |
| `scripts/prepare_reid_data.py` | Crop vehicle patches from COCO+video tracks |
| `scripts/prepare_reid_data_mot.py` | Crop vehicle patches from VisDrone MOT data |
| `scripts/convert_visdrone_mot_to_yolo.py` | Convert VisDrone MOT to YOLO format |
| `scripts/inference.py` | Single-stream inference + visualization |
| `scripts/evaluate_mot.py` | MOT metrics evaluation |
| `src/drone_traffic/models/yolo26_detector.py` | YOLOv26 detector wrapper |
| `src/drone_traffic/tracking/reid.py` | Re-ID ONNX inference module |
| `src/drone_traffic/tracking/bot_sort.py` | Enhanced BoTSORT tracker |
| `src/drone_traffic/tracking/matching.py` | Cost matrix functions |
| `src/drone_traffic/workers/stream_worker.py` | Main inference pipeline |

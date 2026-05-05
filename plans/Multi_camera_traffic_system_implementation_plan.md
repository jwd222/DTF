# GLM 5.1 Implementation Plan  
**Plug‑and‑Play Aerial Traffic Detection & Tracking Engine**  
*Based on EfficientSAM3 Custom Pipeline with Multi‑Drone Fusion*

This plan is designed to be handed directly to a development team. It is comprehensive, assumes that the agent/team will need to fill in contextual gaps (data, hardware specifics, environment constraints), and structures every component as a configurable module.

---

## 1. Project Overview and Objectives
Build a real‑time, edge‑deployable aerial traffic monitoring system (GLM 5.1) capable of:
- Detecting and classifying vehicles (car, truck, bus, motorcycle, etc.) from single or multiple drones.
- Tracking vehicles across frames with ego‑motion compensation, even in highly congested scenes.
- Fusing multiple drone views (nadir + oblique) into a unified Bird’s‑Eye‑View (BEV) coordinate system for complete intersection coverage.
- Streaming only telemetry metadata (bbox, ID, class, velocity) to a ground station, not video.
- Being fully **plug‑and‑play** with interchangeable backbones, detector heads, and trackers, all controlled by a single configuration file.

---

## 2. System Architecture Overview
The engine is split into two deployment modes selectable via `config.yaml`:

- **Single‑Drone mode**: In‑drone inference + tracking (BoT‑SORT with CMC).
- **Multi‑Drone Fusion mode**: Each drone runs its own detector; tracks and detections are sent to a central ground‑station process that performs homographic projection, track‑to‑track association, and unified BEV tracking.

```
+--------------------+      +----------------------+
| Drone 1 (edge)     |      | Ground Station        |
| ES Encoder+YOLO+   | telemetry |  Fusion Engine       |
| BoT‑SORT+CMC       +------> Homography Projection +
|                    |      | Multi‑Camera Tracking  |
+--------------------+      +----------------------+
| Drone 2 (edge)     +------>                       |
+--------------------+
```

### 2.1 Plug‑and‑Play Component Interfaces
All major building blocks are abstracted behind Python classes with strict interfaces:

- **BackboneInterface** → `extract_features(image) -> feature_map`
- **DetectorInterface** → `detect(feature_map) -> Detections[]`
- **TrackingInterface** → `update(detections, frame_id, camera_motion) -> TrackState`
- **FusionInterface** → `fuse(tracks_from_drones, timestamps) -> GlobalTrackState`

Configuration selects concrete implementations at runtime.

---

## 3. Detailed Component Specifications

### 3.1 Feature Extractor (Backbone)
**Candidates (selectable via config):**
- `EfficientSAM3‑ES‑EV‑L` (EfficientViT‑L Stage 1 distilled encoder)
- `EfficientSAM3‑ES‑RV‑S` (RepViT‑S) for lower compute
- `MobileCLIP‑S1` text encoder (if later Stages 2/3 become available for promptable segmentation; currently not used in detection pipeline)

**Interface requirements:**
- Input: pre‑processed RGB images (configurable resolution: 640, 960, 1280).
- Output: dense feature map (H×W×C), e.g., 80×80×256 for 1280 input.
- Pre‑trained weights loaded from official EfficientSAM3 repository; frozen or fine‑tuned depending on config.

**Parameterisation:**
```yaml
backbone:
  name: "ES-EV-L"
  weights_path: "./weights/es_ev_l.pt"
  freeze: true                # true for initial training of detector head
  input_size: [1280, 1280]    # H, W
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
```

### 3.2 Detection Head (Neck + Bounding Box Head)
**Design:** YOLOv11‑style Feature Pyramid Network (FPN) with decoupled detection head. Attached directly to the backbone feature map.

**Plug‑and‑play sizes:**
- `YOLO-Nano`: minimal latency (depth_mult=0.33, width_mult=0.25)
- `YOLO-Small`: balanced accuracy/speed (depth_mult=0.33, width_mult=0.50)
- `YOLO-Medium`: higher accuracy for ground station fusion (depth_mult=0.67, width_mult=0.75)

**Configurable parameters:**
- Number of classes (e.g., car, truck, bus, motorcycle, bicycle, pedestrian) – defined by dataset.
- Anchor‑free or anchor‑based, as per YOLOv11 specification.
- Detection confidence threshold, NMS IoU threshold.
- Output: `Detections` object (bboxes in pixel coordinates, class IDs, confidence scores).

**Training strategy (agent must fill in):**
- Need significant annotated aerial dataset (nadir, oblique, mixed angles). If none exists, synthetic data generation or transfer learning from COCO/VisDrone with heavy augmentation is required.
- Stage‑wise training: freeze backbone, train head; then optionally unfreeze a few final backbone blocks.

### 3.3 Tracker (Single‑View and Multi‑View)
**Selectable algorithms:**
- **BoT‑SORT** (default): Kalman filter with Camera Motion Compensation and ReID‑like feature bank (optional).
- **ByteTrack**: lightweight, but only for static cameras or after CMC. Not recommended for drones.
- **OC‑SORT**: alternative for high‑motion scenarios; configurable as drop‑in.

**Camera Motion Compensation (CMC) module:**  
Uses Global Motion Compensation (GMC) via ORB/ECC or a simple homography tracker on background features. Must be implemented as a standalone module feeding the tracker:

```yaml
tracker:
  name: "BoTSORT"
  cmc_method: "ORB"       # ORB, ECC, OFF
  track_high_thresh: 0.6
  track_low_thresh: 0.3
  new_track_thresh: 0.7
  max_time_lost: 30       # frames before track death
  feature_extractor: "osnet" # optional, only for re-id, can be null
```

**Parameter tuning required per scene:** agent must calibrate motion thresholds, image scale, and feature bank size based on drone altitude and speed.

### 3.4 Multi‑Drone Fusion Engine (Ground Station)
Works with telemetry streams from two (or more) drones.

**Key sub‑modules:**
1. **Homography Projector** – converts each drone’s pixel coordinates to a global BEV plane.
   - Requires calibration data per drone: `homography_matrix` (3×3), computed offline from known ground control points (GCPs). Agent must place markers and compute using standard tools (OpenCV `findHomography`).
   - Alternatively, if drone telemetry includes accurate GPS+IMU+altitude, compute homography from intrinsic/extrinsic matrices.
2. **Temporal Synchroniser** – buffers incoming detections from each stream, aligns by timestamp using PTP‑synchronised clocks. Configurable max time delta (e.g., 50ms). Out‑of‑sync data is discarded with warning.
3. **Track‑to‑Track Associator** – projected tracks are matched using Mahalanobis distance (position + velocity) in BEV coordinates.
4. **Conflict Resolution Merger** – as described: if nadir track and oblique track overlap in BEV (IoU > threshold), merge into one track ID. If oblique has two tracks where nadir has one, split into two tracks preserving oblique IDs. Rules expressed as a configurable policy.

**Configurable parameters:**
```yaml
fusion:
  merge_iou_threshold: 0.3
  max_time_sync_diff: 0.05        # seconds
  kalman_process_noise: 0.1
  view_priority: "oblique"        # which view to trust for splitting
  homography_source: "file"       # or "gps_imu"
  calibration_paths:
    drone1: "./calib/drone1_homography.npy"
    drone2: "./calib/drone2_homography.npy"
```

**Scope for agent:** define BEV coordinate origin and scale (meters per pixel). Needs accurate field measurements.

---

## 4. Plug‑and‑Play Configuration Schema
All variability is centralised in a single YAML/JSON configuration file. Example structure:

```yaml
# GLM 5.1 Configuration
mode: "multi_drone"    # "single_drone" or "multi_drone"

drone_1:
  input_source: "rtsp://192.168.1.10:8554/stream"
  model:
    backbone: {...}
    detector: {...}
    tracker: {...}
  output_telemetry: "udp://ground.station:5000"
  preprocessing:
    resize: [1280, 1280]
    letterbox: true

drone_2: ... (similar)

ground_station:
  fusion: {...}
  output_tracks: "tcp://localhost:9090"
  recording_path: "/data/telemetry_log/"
```

**Switching components:** The engine factory reads these sections and instantiates the correct classes; e.g., changing `tracker.name` from `BoTSORT` to `OCSORT` requires no code change. The developer must register each component in a registry.

---

## 5. Implementation Phases and Milestones

### Phase 0 – Environment & Prerequisites Setup
1. **Hardware selection:**
   - Single‑board computer for drone: NVIDIA Jetson Orin NX/AGX (16/32GB), with TensorRT support.
   - Ground station: Jetson AGX Orin or x86 workstation with RTX GPU.
   - Drones with RTK‑GPS, PTP‑capable network (IEEE 802.1AS via Wi‑Fi 6 or custom link).
   - Camera: 4K streaming global shutter camera, configurable angle.
2. **Software stack:** Ubuntu 22.04, PyTorch 2.0, TensorRT 8.6, DeepStream (for hw acceleration), MQTT/ZMQ for telemetry transport.
3. **Clocks synchronisation**: All devices run `linuxptp` in master‑slave configuration. Validate with `ptp4l`.

### Phase 1 – Data Acquisition & Annotation (Agent‑dependent)
- Gather raw aerial videos from:
  - Single drone at multiple altitudes (30m–120m) and angles (nadir, 30°, 45°, 60° oblique).
  - Two‑drone set‑ups capturing the same intersection from different views.
- Annotate bounding boxes and class labels. Tool: CVAT or Label‑Studio.
- Create calibration datasets: place 4+ ground control points (GCPs) visible in both nadir and oblique views. Record precise GPS coordinates. Compute homography matrices.
- Validate homography by projecting nadir points to oblique image and vice versa; residual error < 2 pixels.
- Document all parameters (scenario lighting, weather, drone speed) for training robustness.

### Phase 2 – Single‑View Detection and Tracking Engine
2.1 **Build the modular pipeline:**
   - Implement `BackboneRegistry`, `DetectorRegistry`, `TrackerRegistry`.
   - Implement config parser that builds pipeline.

2.2 **Training the detector head:**
   - Script to convert annotations to YOLO format.
   - Use data augmentation: mosaic, mix‑up, perspective transforms (critical for drone angles), motion blur, occlusion simulation.
   - Train YOLO‑Nano and Small heads on frozen EfficientSAM3 backbone. Monitor mAP@0.5:0.95.
   - Fine‑tune with backbone unfrozen for best variant.

2.3 **Tracker integration:**
   - Integrate BoT‑SORT (or other) with CMC module. Test on single‑drone oblique videos.
   - Tune Kalman filter parameters for typical vehicle dynamics (max acceleration, velocity range in BEV).
   - Implement track management (birth, death, re‑identification) and configure thresholds.

2.4 **Edge optimisation:**
   - Export pipeline to ONNX, then build TensorRT engine for Orin.
   - Profile inference time; ensure ≥30 FPS end‑to‑end (detection + tracking) with 1280×1280 inputs.
   - Implement telemetry serialisation (Protobuf or JSON) and streaming via UDP.

### Phase 3 – Multi‑Drone Fusion Engine
3.1 **Set up telemetry receiver on ground station:**
   - Receive and decode streams from both drones, buffering by timestamp.
   - Synchronisation check: log max latency drift.

3.2 **Implement Homography Projection:**
   - Use stored calibration matrices to map each detection’s foot point (bottom centre of bounding box) to BEV coordinates.
   - Define a common BEV grid (e.g., 200m × 200m at 0.1m/pixel).
   - Project tracks from both drones.

3.3 **Fusion policy implementation:**
   - Implement track‑to‑track association with Hungarian algorithm, using position and velocity covariance from respective Kalman filters.
   - Implement conflict resolver with configurable policies.
   - Output a global track list with unique IDs.

3.4 **Global tracking:**
   - Once tracks are fused, maintain a central Kalman filter for each global track, updating from associated drone measurements. This provides smooth, low‑latency estimates even if one drone temporarily loses sight.

### Phase 4 – System Integration and End‑to‑End Validation
- Deploy engine on actual hardware, connect to live drone cameras.
- Perform closed‑loop tests:
  - Ground truth comparison: set up a LiDAR or manual count to validate detection and tracking accuracy.
  - Stress tests: multiple simultaneous vehicles, sudden lighting changes, partial occlusion.
- Parameter sweep: adjust detection thresholds, tracker gates, fusion IoU, etc., using a held‑out validation set.
- Build a dashboard for real‑time visualisation (BEV map with track IDs and velocities).

### Phase 5 – Documentation and Delivery
- Deliver a fully commented code base with:
  - Installation guide (dependencies, Dockerfile).
  - Configuration reference.
  - Calibration SOP.
  - Operations manual (launch procedures, troubleshooting).
- Provide a suite of unit and integration tests.

---

## 6. Scope of Work Filling by the Agent (Explicit Checklist)
The plan cannot be fully executed without the agent filling in these details. This must be addressed before/during each phase:

- [ ] **Physical deployment environment**: GPS coordinates of intersection, BEV origin definition, GCP placement and measurement.
- [ ] **Drone flight parameters**: flight height, speed, camera FOV, gimbal angle, camera intrinsic calibration (chessboard).
- [ ] **Network infrastructure**: PTP grandmaster clock availability, bandwidth for two telemetry streams, fallback if link drops.
- [ ] **Real‑world labelled data**: amount, annotation quality, class distribution. If insufficient, plan for semi‑automatic annotation using foundation models.
- [ ] **Weather/lighting variability**: collect data across different conditions; may require gamma/histogram equalisation pre‑processing.
- [ ] **Legal and safety**: flight permissions, no‑fly zone constraints, data privacy (anonymisation of licence plates/pedestrians if required).
- [ ] **Performance bounds**: exactly what is “real‑time”? 30 FPS? 25 FPS? The pipeline must guarantee maximum latency for safety‑critical applications.
- [ ] **Homography stability**: is the drone gimbal stabilised? If not, dynamic homography must be recalculated per frame using IMU+GPS or feature matching, significantly increasing complexity.
- [ ] **Occlusions maps**: static obstacles (trees, bridges) that invalidate certain areas of the BEV – these should be masked.
- [ ] **Post‑processing requirements**: do we need to export to standard formats (MOTChallenge, YOLO tracking) for evaluation? Plan accordingly.

---

## 7. Error Handling and Foolproof Design Strategies
- **Input validation:** The pipeline starts by checking for valid video stream, config file integrity, and weight file checksums. Fails gracefully with clear log messages.
- **Frame drop handling:** If a frame is missed, the tracker propagates predictions. After N consecutive misses, tracks are deleted. CMC continues to track background motion so that when frames resume, the coordinate system is still consistent.
- **Synchronisation loss:** If PTP drift exceeds threshold, the fusion node raises an alarm and stops merging, falling back to single‑source tracking per drone.
- **False positive suppression:** Implement dynamic confidence threshold based on scene density (e.g., lower threshold in sparse areas, higher in congestion).
- **Failsafe for homography errors:** If projection falls outside expected BEV bounds, detection is flagged and ignored.
- **Automatic recalibration prompt:** If number of unmatched detections in overlapping region exceeds a rolling window threshold, alert the operator to recalibrate.
- **Monitoring dashboard:** Telemetry streams include heartbeats and error codes; ground station monitors health and logs all anomalies.
- **Reproducibility:** All parameters are logged for every run, enabling exact reproduction for debugging.

---

## 8. Testing and Validation Strategy
- **Unit tests:** Each module (CMC, homography projection, Hungarian matching) tested with synthetic scenes.
- **Component‑wise benchmarks:** Speed/accuracy trade‑offs for backbone‑detector pairs logged in a benchmark table.
- **End‑to‑end simulation:** Using AirSim or Unreal Engine to generate synthetic multi‑drone feeds with ground truth; validates tracker and fusion without costly flight tests.
- **Field trials:** Phased approach: first static camera, then one drone, finally two drones. Compare trajectories to RTK‑GPS ground truth (car‑mounted receiver).
- **Acceptance criteria:** 
  - Detection mAP > 0.85 on validation aerial dataset.
  - Tracker MOTA > 0.80 on 30‑frame occlusion sequences.
  - End‑to‑end latency from frame capture to telemetry output < 50ms on edge.
  - Fusion correctly merges >95% of overlapping vehicles.

---

## 9. Deliverables
1. **GLM 5.1 Engine Source Code** (Python/PyTorch, C++ TensorRT runtime, config system).
2. **Pre‑trained models** (backbone + detector head weights) for typical drone altitudes and angles.
3. **Deployment packages** for Jetson Orin (flashable image) and ground station (Docker).
4. **Calibration toolkit** with scripts for GCP‑based homography calculation and PTP setup.
5. **User manual** covering installation, configuration, tuning, and maintenance.
6. **Validation report** with quantitative results on held‑out data.
7. **Agent fill‑in template** – a document that explicitly lists every context‑specific parameter that must be provided by the operations team before deployment (e.g., intersection dimensions, drone IPs, flight plan).

---

By strictly following this plan, the GLM 5.1 system will be a robust, adaptable detection and tracking engine capable of handling the most challenging multi‑drone traffic scenarios, while allowing almost any component to be swapped or re‑parameterised without rewriting core logic.
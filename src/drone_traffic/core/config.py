from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    device: str = "cuda"
    fp16: bool = True
    torch_compile: bool = False
    compile_mode: str = "reduce-overhead"
    num_workers: int = 2


class InputConfig(BaseModel):
    resolution: list[int] = Field(default_factory=lambda: [960, 960])
    target_fps: int = 30
    letterbox: bool = True
    normalize: bool = True
    mean: list[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225])


class StreamConfig(BaseModel):
    name: str
    source: str
    homography: str | None = None


class DetectorConfig(BaseModel):
    type: str = "yolo26_detector"
    weights: str = "yolo26s.pt"
    num_classes: int = 6
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300


class ModelsConfig(BaseModel):
    detector: DetectorConfig = Field(default_factory=DetectorConfig)


class TrainingConfig(BaseModel):
    epochs: int = 300
    imgsz: int = 960
    batch: int = 8
    optimizer: str = "MuSGD"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    copy_paste: float = 0.1
    degrees: float = 5.0
    translate: float = 0.1
    scale: float = 0.5
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.1
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    patience: int = 50
    save_period: int = 10
    workers: int = 8
    project: str = "runs/train"
    name: str = "yolo26_vehicle"


class CMCConfig(BaseModel):
    method: str = "orb"
    max_features: int = 500


class ReIDConfig(BaseModel):
    enabled: bool = False
    weights: str | None = None
    model_type: str = "osnet_x1_0"
    embedding_dim: int = 512
    appearance_thresh: float = 0.25


class TrackingConfig(BaseModel):
    type: str = "bot_sort"
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    cmc: CMCConfig = Field(default_factory=CMCConfig)
    reid: ReIDConfig = Field(default_factory=ReIDConfig)


class AssociationConfig(BaseModel):
    metric: str = "mahalanobis"
    threshold: float = 2.0


class ConflictResolutionConfig(BaseModel):
    policy: str = "merge"


class FusionConfig(BaseModel):
    enabled: bool = False
    max_time_sync_diff: float = 0.05
    homography_method: str = "predefined"
    association: AssociationConfig = Field(default_factory=AssociationConfig)
    conflict_resolution: ConflictResolutionConfig = Field(
        default_factory=ConflictResolutionConfig
    )


class PersistenceConfig(BaseModel):
    db_url: str = "postgresql+asyncpg://drone:drone@localhost:5432/drone_traffic"
    batch_size: int = 100
    flush_interval: float = 0.5


class ZMQConfig(BaseModel):
    telemetry_port: int = 5555
    fusion_port: int = 5556


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    file: str | None = None


class OutputConfig(BaseModel):
    save_annotated: bool = False
    output_dir: str = "output/"
    bev_canvas_size: list[int] = Field(default_factory=lambda: [1000, 1000])


class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    streams: list[StreamConfig] = Field(default_factory=list)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    persistence: PersistenceConfig = Field(default_factory=PersistenceConfig)
    zmq: ZMQConfig = Field(default_factory=ZMQConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    model_config: dict[str, Any] = {"extra": "forbid"}


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r") as f:
        raw = yaml.safe_load(f)
    return AppConfig(**(raw or {}))

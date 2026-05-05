from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def iou(self, other: BBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def to_xywh(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.width, self.height

    def to_cxcyah(self) -> tuple[float, float, float, float]:
        return self.cx, self.cy, self.width / max(self.height, 1e-6), self.height


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    class_label: str = ""
    feature: Any | None = None


@dataclass
class TrackState:
    track_id: int
    bbox: BBox
    confidence: float
    class_id: int
    class_label: str = ""
    velocity: tuple[float, float] = (0.0, 0.0)
    state_vector: list[float] = field(default_factory=list)
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    feature: Any | None = None


@dataclass
class BEVCoord:
    x: float
    y: float


@dataclass
class BEVBBox:
    x: float
    y: float
    w: float
    h: float


class TrackObservation(TypedDict):
    global_track_id: int
    camera_id: int
    timestamp: float
    frame_id: int
    bbox_px: dict[str, float]
    bbox_bev: dict[str, float] | None
    confidence: float
    velocity_bev: dict[str, float]
    source_track_id: int


class TelemetryMessage(TypedDict):
    drone_id: str
    frame_id: int
    timestamp: float
    tracks: list[dict[str, Any]]


class FusionMessage(TypedDict):
    timestamp: float
    global_tracks: list[dict[str, Any]]
    events: list[dict[str, Any]]

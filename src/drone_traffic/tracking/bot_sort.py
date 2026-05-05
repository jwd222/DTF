from __future__ import annotations

import numpy as np

from drone_traffic.core.registry import register_tracker
from drone_traffic.core.types import BBox, Detection, TrackState
from drone_traffic.tracking.cmc import ORBCMC
from drone_traffic.tracking.kalman import KalmanFilter8
from drone_traffic.tracking.matching import hungarian_assignment, iou_distance
from drone_traffic.tracking.tracker_base import TrackingInterface


class _Track:
    _next_id = 1

    def __init__(self, detection: Detection, timestamp: float = 0.0):
        self.track_id = _Track._next_id
        _Track._next_id += 1

        self.class_id = detection.class_id
        self.class_label = detection.class_label
        self.confidence = detection.confidence

        cxcyah = np.array(detection.bbox.to_cxcyah(), dtype=np.float64)
        self.kf = KalmanFilter8()
        self.kf.init_from_bbox(cxcyah)

        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.timestamp = timestamp

    @property
    def state(self) -> np.ndarray:
        return self.kf.get_state()

    @property
    def predicted_bbox(self) -> BBox:
        s = self.state
        cx, cy, a, h = s[0], s[1], s[2], s[3]
        w = a * h
        return BBox(x1=cx - w / 2, y1=cy - h / 2, x2=cx + w / 2, y2=cy + h / 2)

    def predict(self) -> None:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection, timestamp: float = 0.0) -> None:
        cxcyah = np.array(detection.bbox.to_cxcyah(), dtype=np.float64)
        self.kf.update(cxcyah)
        self.hits += 1
        self.time_since_update = 0
        self.confidence = detection.confidence
        self.timestamp = timestamp

    def apply_cmc(self, matrix: np.ndarray) -> None:
        self.kf.apply_affine(matrix)

    def to_track_state(self) -> TrackState:
        s = self.state
        return TrackState(
            track_id=self.track_id,
            bbox=self.predicted_bbox,
            confidence=self.confidence,
            class_id=self.class_id,
            class_label=self.class_label,
            velocity=(s[4], s[5]),
            state_vector=s.tolist(),
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
        )


@register_tracker("bot_sort")
class BoTSORTTracker(TrackingInterface):
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        cmc_method: str = "orb",
        cmc_max_features: int = 500,
        **kwargs,
    ):
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._tracks: list[_Track] = []
        self._frame_count = 0

        self._cmc = ORBCMC(max_features=cmc_max_features)
        self._prev_frame: np.ndarray | None = None

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray | None = None,
        timestamp: float = 0.0,
    ) -> list[TrackState]:
        self._frame_count += 1

        for track in self._tracks:
            track.predict()

        if frame is not None and self._prev_frame is not None:
            cmc_matrix = self._cmc.compute(self._prev_frame, frame)
            for track in self._tracks:
                track.apply_cmc(cmc_matrix)

        if self._tracks and detections:
            predicted = np.array(
                [t.predicted_bbox.to_cxcyah() for t in self._tracks]
            )
            detected = np.array(
                [d.bbox.to_cxcyah() for d in detections]
            )

            cost = iou_distance(predicted, detected)
            matches, unmatched_tracks, unmatched_dets = hungarian_assignment(
                cost, threshold=1.0 - self._iou_threshold
            )

            for t_idx, d_idx in matches:
                self._tracks[t_idx].update(detections[d_idx], timestamp)

            for d_idx in unmatched_dets:
                new_track = _Track(detections[d_idx], timestamp)
                self._tracks.append(new_track)
        else:
            for det in detections:
                new_track = _Track(det, timestamp)
                self._tracks.append(new_track)

        self._tracks = [
            t for t in self._tracks if t.time_since_update <= self._max_age
        ]

        if frame is not None:
            self._prev_frame = frame.copy()

        return self._get_confirmed_tracks()

    def _get_confirmed_tracks(self) -> list[TrackState]:
        results = []
        for track in self._tracks:
            if track.hits >= self._min_hits or self._frame_count <= self._min_hits:
                results.append(track.to_track_state())
        return results

    def get_active_tracks(self) -> list[TrackState]:
        return [t.to_track_state() for t in self._tracks]

    def reset(self) -> None:
        self._tracks.clear()
        self._frame_count = 0
        self._prev_frame = None
        self._cmc.reset()
        _Track._next_id = 1

from __future__ import annotations

import numpy as np

from drone_traffic.core.registry import register_tracker
from drone_traffic.core.types import BBox, Detection, TrackState
from drone_traffic.tracking.cmc import ORBCMC
from drone_traffic.tracking.kalman import KalmanFilter8
from drone_traffic.tracking.matching import (
    combined_cost_matrix,
    cosine_distance,
    hungarian_assignment,
    iou_distance,
)
from drone_traffic.tracking.reid import ReIDExtractor
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

        self.feature: np.ndarray | None = getattr(detection, "feature", None)
        self._features_buffer: list[np.ndarray] = []
        if self.feature is not None:
            self._features_buffer.append(self.feature.copy())

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

        feat = getattr(detection, "feature", None)
        if feat is not None:
            self._features_buffer.append(feat.copy())
            if len(self._features_buffer) > 30:
                self._features_buffer = self._features_buffer[-30:]
            stacked = np.stack(self._features_buffer)
            self.feature = stacked.mean(axis=0)
            norm = np.linalg.norm(self.feature)
            if norm > 0:
                self.feature = self.feature / norm

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
            feature=self.feature,
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
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        reid_weights: str | None = None,
        reid_enabled: bool = False,
        reid_model_type: str = "osnet_x1_0",
        reid_embedding_dim: int = 512,
        reid_appearance_thresh: float = 0.25,
        **kwargs,
    ):
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold
        self._track_high_thresh = track_high_thresh
        self._track_low_thresh = track_low_thresh
        self._new_track_thresh = new_track_thresh
        self._track_buffer = track_buffer
        self._match_thresh = match_thresh
        self._proximity_thresh = proximity_thresh
        self._appearance_thresh = appearance_thresh
        self._tracks: list[_Track] = []
        self._frame_count = 0

        self._cmc = ORBCMC(max_features=cmc_max_features)
        self._prev_frame: np.ndarray | None = None

        self._reid: ReIDExtractor | None = None
        if reid_enabled and reid_weights:
            self._reid = ReIDExtractor(
                weights=reid_weights,
                embedding_dim=reid_embedding_dim,
            )

    def _extract_reid_features(
        self,
        detections: list[Detection],
        frame: np.ndarray | None,
    ) -> None:
        if self._reid is None or not self._reid.is_ready or frame is None:
            return

        features = self._reid.extract_features(detections, frame)
        for det, feat in zip(detections, features):
            det.feature = feat

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray | None = None,
        timestamp: float = 0.0,
    ) -> list[TrackState]:
        self._frame_count += 1

        self._extract_reid_features(detections, frame)

        for track in self._tracks:
            track.predict()

        if frame is not None and self._prev_frame is not None:
            cmc_matrix = self._cmc.compute(self._prev_frame, frame)
            for track in self._tracks:
                track.apply_cmc(cmc_matrix)

        if self._tracks and detections:
            high_dets = [d for d in detections if d.confidence >= self._track_high_thresh]
            low_dets = [d for d in detections if d.confidence < self._track_high_thresh]

            high_indices = [i for i, d in enumerate(detections) if d.confidence >= self._track_high_thresh]
            low_indices = [i for i, d in enumerate(detections) if d.confidence < self._track_high_thresh]

            unmatched_tracks, unmatched_high = self._match_first_stage(
                high_dets, frame
            )

            for d_idx, det in enumerate(high_dets):
                if d_idx in unmatched_high:
                    if det.confidence >= self._new_track_thresh:
                        new_track = _Track(det, timestamp)
                        self._tracks.append(new_track)

            if low_dets and unmatched_tracks:
                self._match_second_stage(
                    low_dets, unmatched_tracks, timestamp
                )
            else:
                for det in low_dets:
                    if det.confidence >= self._new_track_thresh:
                        new_track = _Track(det, timestamp)
                        self._tracks.append(new_track)
        else:
            for det in detections:
                if det.confidence >= self._new_track_thresh:
                    new_track = _Track(det, timestamp)
                    self._tracks.append(new_track)

        self._tracks = [
            t for t in self._tracks if t.time_since_update <= self._max_age
        ]

        if frame is not None:
            self._prev_frame = frame.copy()

        return self._get_confirmed_tracks()

    def _match_first_stage(
        self,
        detections: list[Detection],
        frame: np.ndarray | None,
    ) -> tuple[set[int], set[int]]:
        if not self._tracks or not detections:
            return set(range(len(self._tracks))), set(range(len(detections)))

        predicted = np.array(
            [t.predicted_bbox.to_cxcyah() for t in self._tracks]
        )
        detected = np.array(
            [d.bbox.to_cxcyah() for d in detections]
        )

        iou_cost = iou_distance(predicted, detected)

        has_features = any(t.feature is not None for t in self._tracks) and any(
            d.feature is not None for d in detections
        )

        if has_features:
            track_features = np.array(
                [t.feature if t.feature is not None else np.zeros(1) for t in self._tracks]
            )
            det_features = np.array(
                [d.feature if d.feature is not None else np.zeros(1) for d in detections]
            )

            if track_features.shape[1] == det_features.shape[1] and track_features.shape[1] > 1:
                app_cost = cosine_distance(track_features, det_features)
                cost = combined_cost_matrix(
                    iou_cost, app_cost,
                    appearance_thresh=self._appearance_thresh,
                    proximity_thresh=self._proximity_thresh,
                )
            else:
                cost = iou_cost
        else:
            cost = iou_cost

        matches, unmatched_tracks, unmatched_dets = hungarian_assignment(
            cost, threshold=1.0 - self._iou_threshold
        )

        for t_idx, d_idx in matches:
            self._tracks[t_idx].update(detections[d_idx])

        return set(unmatched_tracks), set(unmatched_dets)

    def _match_second_stage(
        self,
        detections: list[Detection],
        unmatched_track_indices: set[int],
        timestamp: float,
    ) -> None:
        unmatched_tracks = [self._tracks[i] for i in unmatched_track_indices if i < len(self._tracks)]

        if not unmatched_tracks or not detections:
            return

        predicted = np.array(
            [t.predicted_bbox.to_cxcyah() for t in unmatched_tracks]
        )
        detected = np.array(
            [d.bbox.to_cxcyah() for d in detections]
        )

        cost = iou_distance(predicted, detected)

        matches, _, unmatched_dets = hungarian_assignment(
            cost, threshold=1.0 - self._iou_threshold
        )

        for t_idx, d_idx in matches:
            unmatched_tracks[t_idx].update(detections[d_idx], timestamp)

        for d_idx in unmatched_dets:
            if detections[d_idx].confidence >= self._new_track_thresh:
                new_track = _Track(detections[d_idx], timestamp)
                self._tracks.append(new_track)

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

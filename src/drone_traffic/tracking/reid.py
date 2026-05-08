from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from drone_traffic.core.types import Detection

logger = logging.getLogger(__name__)


class ReIDExtractor:
    def __init__(
        self,
        weights: str = "osnet_reid.onnx",
        embedding_dim: int = 512,
        input_size: tuple[int, int] = (256, 128),
        device: str = "cuda",
    ):
        self._weights = weights
        self._embedding_dim = embedding_dim
        self._input_size = input_size
        self._device = device
        self._session = None

        self._load_model(weights)

    def _load_model(self, weights: str) -> None:
        weights_path = Path(weights)
        if not weights_path.exists():
            logger.warning(
                "Re-ID weights not found at %s — Re-ID will be disabled", weights
            )
            return

        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(weights_path), providers=providers)

        input_name = self._session.get_inputs()[0].name
        input_shape = self._session.get_inputs()[0].shape
        logger.info(
            "Re-ID model loaded: input=%s shape=%s", input_name, input_shape
        )

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    def extract_features(
        self,
        detections: list[Detection],
        frame: np.ndarray,
    ) -> list[np.ndarray | None]:
        if not self.is_ready or not detections:
            return [None] * len(detections)

        features: list[np.ndarray | None] = []
        input_name = self._session.get_inputs()[0].name

        for det in detections:
            bbox = det.bbox
            h, w = frame.shape[:2]

            x1 = max(0, int(bbox.x1))
            y1 = max(0, int(bbox.y1))
            x2 = min(w, int(bbox.x2))
            y2 = min(h, int(bbox.y2))

            if x2 <= x1 or y2 <= y1:
                features.append(None)
                continue

            crop = frame[y1:y2, x1:x2]
            resized = cv2.resize(crop, self._input_size)

            blob = resized.astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))
            blob = np.expand_dims(blob, axis=0)

            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
            blob = (blob - mean) / std

            output = self._session.run(None, {input_name: blob})
            embedding = output[0].flatten()

            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            features.append(embedding)

        return features

    def extract_single(
        self,
        detection: Detection,
        frame: np.ndarray,
    ) -> np.ndarray | None:
        results = self.extract_features([detection], frame)
        return results[0] if results else None

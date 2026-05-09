from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from drone_traffic.core.registry import register_detector
from drone_traffic.core.types import BBox, Detection
from drone_traffic.models.detector_base import DetectorInterface

logger = logging.getLogger(__name__)

VEHICLE_CLASSES = {
    0: "car",
    1: "van",
    2: "truck",
    3: "rickshaw",
    4: "bus",
    5: "motorcycle",
}


@register_detector("yolo26_detector")
class YOLOv26Detector(DetectorInterface):
    def __init__(
        self,
        weights: str = "yolo26s.pt",
        num_classes: int = 6,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        device: str = "cuda",
        **kwargs,
    ):
        super().__init__()
        self._weights = weights
        self._num_classes = num_classes
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._max_detections = max_detections
        self._device = device

        self._model = self._load_model(weights, device)

    @staticmethod
    def _load_model(weights: str, device: str):
        from ultralytics import YOLO

        weights_path = Path(weights)
        if not weights_path.exists():
            logger.warning(
                "YOLOv26 weights not found at %s — will download automatically", weights
            )

        model = YOLO(str(weights_path))
        return model

    def forward(
        self,
        features: dict[str, torch.Tensor],
        original_shape: tuple[int, ...],
        scale_ratio: float,
        pad: tuple[int, int, int, int],
        raw_frame: object | None = None,
    ) -> list[Detection]:
        import numpy as np

        if raw_frame is not None:
            input_image = raw_frame
        else:
            tensor = features.get("tensor")
            if tensor is not None:
                if isinstance(tensor, torch.Tensor):
                    arr = tensor.cpu().numpy()
                    if arr.ndim == 4:
                        arr = arr[0]
                    if arr.shape[0] in (1, 3):
                        arr = np.transpose(arr, (1, 2, 0))
                    arr = (arr * 255).astype(np.uint8)
                    input_image = arr
                else:
                    input_image = tensor
            else:
                logger.warning("No input available for YOLOv26 detector")
                return []

        results = self._model.predict(
            input_image,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            max_det=self._max_detections,
            device=self._device,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        return self._convert_detections(result, original_shape, scale_ratio, pad)

    def _convert_detections(
        self,
        result,
        original_shape: tuple[int, ...],
        scale_ratio: float,
        pad: tuple[int, int, int, int],
    ) -> list[Detection]:
        import numpy as np

        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            conf = float(confs[i])
            cls_id = int(cls_ids[i])

            if cls_id >= self._num_classes:
                continue

            detections.append(
                Detection(
                    bbox=BBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    ),
                    confidence=conf,
                    class_id=cls_id,
                    class_label=VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}"),
                )
            )

        return detections

    def predict_raw(self, frame, **kwargs):
        results = self._model.predict(
            frame,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            max_det=self._max_detections,
            device=self._device,
            verbose=False,
            **kwargs,
        )
        return results

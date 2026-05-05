from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from drone_traffic.core.registry import register_detector
from drone_traffic.core.types import BBox, Detection
from drone_traffic.models.detector_base import DetectorInterface
from drone_traffic.models.nms import non_max_suppression


@register_detector("yolo_head")
class YOLODetectionHead(DetectorInterface):
    def __init__(
        self,
        in_channels: list[int] | None = None,
        num_classes: int = 6,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        device: str = "cuda",
    ):
        super().__init__()
        self._in_channels = in_channels or [128, 256, 512]
        self._num_classes = num_classes
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._max_detections = max_detections
        self._device = device

        self._num_anchors = 3
        anchor_reg = 4 + self._num_classes

        self.neck = nn.ModuleList()
        self.heads = nn.ModuleList()

        for i, ch in enumerate(self._in_channels):
            self.neck.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch, 1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.SiLU(inplace=True),
                )
            )
            self.heads.append(
                nn.Conv2d(ch, self._num_anchors * anchor_reg, 1)
            )

        self.to(device)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        original_shape: tuple[int, ...],
        scale_ratio: float,
        pad: tuple[int, int, int, int],
    ) -> list[Detection]:
        feature_keys = sorted(features.keys())
        all_preds = []

        for i, key in enumerate(feature_keys):
            feat = features[key]
            if i >= len(self.neck):
                break
            necked = self.neck[i](feat)
            pred = self.heads[i](necked)
            all_preds.append(pred)

        if not all_preds:
            return []

        combined = self._decode_predictions(all_preds)
        combined = non_max_suppression(
            combined,
            conf_threshold=self._conf_threshold,
            iou_threshold=self._iou_threshold,
            max_detections=self._max_detections,
        )

        return self._to_detections(combined, original_shape, scale_ratio, pad)

    def _decode_predictions(self, preds: list[torch.Tensor]) -> torch.Tensor:
        decoded = []
        for pred in preds:
            B, _, H, W = pred.shape
            pred = pred.view(B, self._num_anchors, 4 + self._num_classes, H, W)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            pred = pred.view(B, -1, 4 + self._num_classes)

            xy = pred[..., :2].sigmoid()
            wh = pred[..., 2:4].exp()
            conf = pred[..., 4:4 + self._num_classes]

            batch_decoded = torch.cat([xy, wh, conf], dim=-1)
            decoded.append(batch_decoded)

        return torch.cat(decoded, dim=1)

    def _to_detections(
        self,
        tensor: torch.Tensor,
        original_shape: tuple[int, ...],
        scale_ratio: float,
        pad: tuple[int, int, int, int],
    ) -> list[Detection]:
        if tensor.dim() == 3:
            tensor = tensor[0]

        detections = []
        top, left, bottom, right = pad

        for row in tensor:
            x, y, w, h = row[:4].tolist()
            class_scores = row[4:].tolist()

            conf = max(class_scores)
            class_id = class_scores.index(conf)

            px1 = (x - w / 2) - left
            py1 = (y - h / 2) - top
            px2 = (x + w / 2) - left
            py2 = (y + h / 2) - top

            px1 = px1 / scale_ratio
            py1 = py1 / scale_ratio
            px2 = px2 / scale_ratio
            py2 = py2 / scale_ratio

            orig_h, orig_w = original_shape[:2]
            px1 = max(0, min(px1, orig_w))
            py1 = max(0, min(py1, orig_h))
            px2 = max(0, min(px2, orig_w))
            py2 = max(0, min(py2, orig_h))

            detections.append(
                Detection(
                    bbox=BBox(x1=px1, y1=py1, x2=px2, y2=py2),
                    confidence=conf,
                    class_id=class_id,
                )
            )

        return detections

from __future__ import annotations

import logging
import time
from typing import Any

import msgpack
import numpy as np
import torch
import zmq

from drone_traffic.core.config import AppConfig
from drone_traffic.core.registry import build_component
from drone_traffic.core.types import Detection, TelemetryMessage
from drone_traffic.ingestion.reader import VideoReader
from drone_traffic.models import efficient_sam3, yolo_head  # noqa: F401 – trigger registration
from drone_traffic.tracking import bot_sort, cmc  # noqa: F401 – trigger registration

logger = logging.getLogger(__name__)


def run_stream_worker(
    config_dict: dict[str, Any],
    stream_name: str,
    stream_source: str,
    zmq_port: int = 5555,
) -> None:
    import yaml
    from pydantic import TypeAdapter

    config = AppConfig(**config_dict)

    device = torch.device(config.system.device if torch.cuda.is_available() else "cpu")

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.connect(f"tcp://localhost:{zmq_port}")

    backbone = build_component(
        "backbone",
        config.models.backbone.type,
        weights=config.models.backbone.weights,
        frozen=config.models.backbone.frozen,
        device=str(device),
    )

    detector = build_component(
        "detector",
        config.models.detector.type,
        in_channels=backbone.output_channels,
        num_classes=config.models.detector.num_classes,
        conf_threshold=config.models.detector.conf_threshold,
        iou_threshold=config.models.detector.iou_threshold,
        max_detections=config.models.detector.max_detections,
        device=str(device),
    )

    tracker = build_component(
        "tracker",
        config.tracking.type,
        max_age=config.tracking.max_age,
        min_hits=config.tracking.min_hits,
        iou_threshold=config.tracking.iou_threshold,
        cmc_method=config.tracking.cmc.method,
        cmc_max_features=config.tracking.cmc.max_features,
    )

    if config.system.torch_compile:
        backbone = torch.compile(backbone, mode=config.system.compile_mode)
        detector = torch.compile(detector, mode=config.system.compile_mode)

    logger.info("Stream worker '%s' initialized", stream_name)

    with VideoReader(
        source=stream_source,
        target_fps=config.input.target_fps,
        resolution=tuple(config.input.resolution),
        letterbox=config.input.letterbox,
        normalize=config.input.normalize,
        mean=config.input.mean,
        std=config.input.std,
    ) as reader:
        raw_cap = None
        try:
            import cv2
            raw_cap = cv2.VideoCapture(stream_source)
        except Exception:
            pass

        for packet in reader.read_frames():
            frame_start = time.perf_counter()

            tensor = torch.from_numpy(packet.image).unsqueeze(0).to(device)
            if config.system.fp16:
                tensor = tensor.half()
                if hasattr(backbone, "half"):
                    backbone = backbone.half()
                if hasattr(detector, "half"):
                    detector = detector.half()

            with torch.no_grad():
                features = backbone(tensor)
                detections = detector(
                    features,
                    packet.original_shape,
                    packet.scale_ratio,
                    packet.pad,
                )

            raw_frame = None
            if raw_cap is not None:
                ret, f = raw_cap.read()
                if ret:
                    raw_frame = f

            tracks = tracker.update(detections, raw_frame, packet.timestamp)

            message: TelemetryMessage = {
                "drone_id": stream_name,
                "frame_id": packet.frame_id,
                "timestamp": packet.timestamp,
                "tracks": [
                    {
                        "id": t.track_id,
                        "bbox": {"x1": t.bbox.x1, "y1": t.bbox.y1, "x2": t.bbox.x2, "y2": t.bbox.y2},
                        "class_id": t.class_id,
                        "confidence": t.confidence,
                        "velocity": {"vx": t.velocity[0], "vy": t.velocity[1]},
                    }
                    for t in tracks
                ],
            }

            packed = msgpack.packb(message, use_bin_type=True)
            pub.send(stream_name.encode(), zmq.SNDMORE)
            pub.send(packed)

            elapsed = time.perf_counter() - frame_start
            if packet.frame_id % 100 == 0:
                logger.info(
                    "Stream %s frame %d: %d detections, %d tracks, %.1f ms",
                    stream_name,
                    packet.frame_id,
                    len(detections),
                    len(tracks),
                    elapsed * 1000,
                )

    pub.close()
    ctx.term()
    logger.info("Stream worker '%s' finished", stream_name)

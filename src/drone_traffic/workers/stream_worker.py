from __future__ import annotations

import logging
import time
from typing import Any

import msgpack
import numpy as np
import zmq

from drone_traffic.core.config import AppConfig
from drone_traffic.core.registry import build_component
from drone_traffic.core.types import Detection, TelemetryMessage
from drone_traffic.ingestion.reader import VideoReader
from drone_traffic.models import yolo26_detector  # noqa: F401 – trigger registration
from drone_traffic.tracking import bot_sort, cmc  # noqa: F401 – trigger registration

logger = logging.getLogger(__name__)


def run_stream_worker(
    config_dict: dict[str, Any],
    stream_name: str,
    stream_source: str,
    zmq_port: int = 5555,
) -> None:
    config = AppConfig(**config_dict)

    device = config.system.device
    if device == "cuda":
        import torch as _torch
        if not _torch.cuda.is_available():
            device = "cpu"
            logger.warning("CUDA not available, falling back to CPU")

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.connect(f"tcp://localhost:{zmq_port}")

    detector = build_component(
        "detector",
        config.models.detector.type,
        weights=config.models.detector.weights,
        num_classes=config.models.detector.num_classes,
        conf_threshold=config.models.detector.conf_threshold,
        iou_threshold=config.models.detector.iou_threshold,
        max_detections=config.models.detector.max_detections,
        device=device,
    )

    reid_kwargs = {}
    if config.tracking.reid.enabled:
        reid_kwargs = {
            "reid_enabled": True,
            "reid_weights": config.tracking.reid.weights,
            "reid_model_type": config.tracking.reid.model_type,
            "reid_embedding_dim": config.tracking.reid.embedding_dim,
            "reid_appearance_thresh": config.tracking.reid.appearance_thresh,
        }

    tracker = build_component(
        "tracker",
        config.tracking.type,
        max_age=config.tracking.max_age,
        min_hits=config.tracking.min_hits,
        iou_threshold=config.tracking.iou_threshold,
        cmc_method=config.tracking.cmc.method,
        cmc_max_features=config.tracking.cmc.max_features,
        track_high_thresh=config.tracking.track_high_thresh,
        track_low_thresh=config.tracking.track_low_thresh,
        new_track_thresh=config.tracking.new_track_thresh,
        track_buffer=config.tracking.track_buffer,
        match_thresh=config.tracking.match_thresh,
        proximity_thresh=config.tracking.proximity_thresh,
        appearance_thresh=config.tracking.appearance_thresh,
        **reid_kwargs,
    )

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

            raw_frame = None
            if raw_cap is not None:
                ret, f = raw_cap.read()
                if ret:
                    raw_frame = f

            detections = detector(
                {},
                packet.original_shape,
                packet.scale_ratio,
                packet.pad,
                raw_frame=raw_frame if raw_frame is not None else packet.image,
            )

            tracker_frame = raw_frame
            if tracker_frame is None:
                frame_h, frame_w = packet.original_shape[:2]
                tracker_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

            tracks = tracker.update(detections, tracker_frame, packet.timestamp)

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

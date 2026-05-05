from __future__ import annotations

import logging
from typing import Any

import msgpack
import zmq

from drone_traffic.core.config import AppConfig
from drone_traffic.core.registry import build_component
from drone_traffic.core.types import TelemetryMessage
from drone_traffic.fusion import homography_fusion  # noqa: F401 – trigger registration

logger = logging.getLogger(__name__)


def run_fusion_worker(
    config_dict: dict[str, Any],
    sub_port: int = 5555,
    pub_port: int = 5556,
) -> None:
    config = AppConfig(**config_dict)

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.bind(f"tcp://*:{sub_port}")
    sub.setsockopt(zmq.SUBSCRIBE, b"")

    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{pub_port}")

    homographies: dict[str, Any] = {}
    for stream_cfg in config.streams:
        if stream_cfg.homography:
            import numpy as np
            from pathlib import Path
            p = Path(stream_cfg.homography)
            if p.exists():
                homographies[stream_cfg.name] = np.load(str(p))

    fusion = build_component(
        "fusion",
        "homography",
        max_time_sync_diff=config.fusion.max_time_sync_diff,
        homographies=homographies if homographies else None,
        association_threshold=config.fusion.association.threshold,
        conflict_policy=config.fusion.conflict_resolution.policy,
    )

    logger.info("Fusion worker started on SUB=%d, PUB=%d", sub_port, pub_port)

    pending: dict[str, TelemetryMessage] = {}

    try:
        while True:
            try:
                topic = sub.recv_string(zmq.NOBLOCK)
                payload = sub.recv()
                msg: TelemetryMessage = msgpack.unpackb(payload, raw=False)
                pending[msg["drone_id"]] = msg

                if len(pending) >= len(config.streams):
                    result = fusion.process(pending)
                    packed = msgpack.packb(result, use_bin_type=True)
                    pub.send(b"fusion", zmq.SNDMORE)
                    pub.send(packed)
                    pending.clear()

            except zmq.Again:
                import time
                time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        sub.close()
        pub.close()
        ctx.term()
        logger.info("Fusion worker stopped")

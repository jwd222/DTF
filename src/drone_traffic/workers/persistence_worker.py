from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

import msgpack
import zmq
import zmq.asyncio

from drone_traffic.core.config import AppConfig
from drone_traffic.persistence.database import init_db, get_session_factory
from drone_traffic.persistence.models import TrackObservation

logger = logging.getLogger(__name__)


async def run_persistence_worker(
    config_dict: dict[str, Any],
    sub_url: str = "tcp://localhost:5556",
) -> None:
    config = AppConfig(**config_dict)

    init_db(config.persistence.db_url)

    ctx = zmq.asyncio.Context()
    sub = ctx.socket(zmq.SUB)
    sub.connect(sub_url)
    sub.setsockopt(zmq.SUBSCRIBE, b"")

    batch: list[dict] = []
    last_flush = time.monotonic()
    flush_interval = config.persistence.flush_interval
    batch_size = config.persistence.batch_size

    logger.info("Persistence worker connected to %s", sub_url)

    try:
        while True:
            try:
                topic = await asyncio.wait_for(sub.recv(), timeout=0.1)
                payload = await sub.recv()
                data = msgpack.unpackb(payload, raw=False)

                observations = _extract_observations(data)
                batch.extend(observations)

                if len(batch) >= batch_size or (time.monotonic() - last_flush) > flush_interval:
                    await _flush_batch(batch)
                    batch.clear()
                    last_flush = time.monotonic()

            except asyncio.TimeoutError:
                if batch and (time.monotonic() - last_flush) > flush_interval:
                    await _flush_batch(batch)
                    batch.clear()
                    last_flush = time.monotonic()

    except KeyboardInterrupt:
        if batch:
            await _flush_batch(batch)
    finally:
        sub.close()
        ctx.term()
        logger.info("Persistence worker stopped")


def _extract_observations(data: dict) -> list[dict]:
    observations = []
    for gt in data.get("global_tracks", []):
        obs = {
            "global_track_id": gt.get("global_id"),
            "timestamp": datetime.now(timezone.utc),
            "confidence": gt.get("confidence"),
        }
        sources = gt.get("sources", {})
        for source_id, track_data in sources.items():
            obs["camera_id"] = None
            obs["bbox_px"] = track_data.get("bbox")
            obs["velocity_bev"] = track_data.get("velocity")
            obs["source_track_id"] = track_data.get("id")
        observations.append(obs)
    return observations


async def _flush_batch(batch: list[dict]) -> None:
    if not batch:
        return

    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            objects = [TrackObservation(**obs) for obs in batch]
            session.add_all(objects)
            await session.commit()
        logger.info("Flushed %d observations", len(batch))
    except Exception as e:
        logger.error("Failed to flush batch: %s", e)

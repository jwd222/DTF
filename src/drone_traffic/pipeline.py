from __future__ import annotations

import logging
import multiprocessing as mp
import time
from typing import Any

from drone_traffic.core.config import AppConfig
from drone_traffic.workers.fusion_worker import run_fusion_worker
from drone_traffic.workers.persistence_worker import run_persistence_worker
from drone_traffic.workers.stream_worker import run_stream_worker

logger = logging.getLogger(__name__)


class PipelineManager:
    def __init__(self, config: AppConfig):
        self._config = config
        self._processes: list[mp.Process] = []
        self._setup_logging()

    def _setup_logging(self) -> None:
        level = getattr(logging, self._config.logging.level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(process)d] %(name)s %(levelname)s: %(message)s",
        )

    def run(self) -> None:
        config_dict = self._config.model_dump()

        if self._config.fusion.enabled and len(self._config.streams) >= 2:
            fusion_proc = mp.Process(
                target=run_fusion_worker,
                args=(config_dict, self._config.zmq.telemetry_port, self._config.zmq.fusion_port),
                name="fusion-worker",
                daemon=True,
            )
            self._processes.append(fusion_proc)
        else:
            logger.info("Fusion disabled or < 2 streams — running single-stream mode")

        for stream_cfg in self._config.streams:
            proc = mp.Process(
                target=run_stream_worker,
                args=(
                    config_dict,
                    stream_cfg.name,
                    stream_cfg.source,
                    self._config.zmq.telemetry_port,
                ),
                name=f"stream-{stream_cfg.name}",
                daemon=True,
            )
            self._processes.append(proc)

        persistence_proc = mp.Process(
            target=_run_sync_persistence,
            args=(config_dict, f"tcp://localhost:{self._config.zmq.fusion_port}"),
            name="persistence-worker",
            daemon=True,
        )
        self._processes.append(persistence_proc)

        for proc in self._processes:
            proc.start()
            logger.info("Started process: %s (PID %d)", proc.name, proc.pid)

        try:
            while True:
                alive = [p for p in self._processes if p.is_alive()]
                if not alive:
                    logger.info("All worker processes have exited")
                    break
                for proc in alive:
                    if proc.exitcode is not None and proc.exitcode != 0:
                        logger.error(
                            "Process %s exited with code %d", proc.name, proc.exitcode
                        )
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Shutting down pipeline...")
        finally:
            for proc in self._processes:
                if proc.is_alive():
                    proc.terminate()
            for proc in self._processes:
                proc.join(timeout=5.0)
            logger.info("Pipeline stopped")


def _run_sync_persistence(config_dict: dict[str, Any], sub_url: str) -> None:
    import asyncio
    from drone_traffic.workers.persistence_worker import run_persistence_worker
    asyncio.run(run_persistence_worker(config_dict, sub_url))

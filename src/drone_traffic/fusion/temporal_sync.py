from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from drone_traffic.core.types import TelemetryMessage


class TemporalSyncBuffer:
    def __init__(
        self,
        max_time_diff: float = 0.05,
        sources_expected: int = 2,
    ):
        self._max_time_diff = max_time_diff
        self._sources_expected = sources_expected
        self._buffer: dict[str, list[TelemetryMessage]] = defaultdict(list)
        self._latest: dict[str, TelemetryMessage] = {}

    def add(self, source_id: str, message: TelemetryMessage) -> None:
        self._buffer[source_id].append(message)
        self._latest[source_id] = message

    def try_flush(self) -> dict[str, TelemetryMessage] | None:
        if len(self._latest) < self._sources_expected:
            return None

        timestamps = [msg["timestamp"] for msg in self._latest.values()]
        time_range = max(timestamps) - min(timestamps)

        if time_range <= self._max_time_diff:
            result = dict(self._latest)
            self._latest.clear()
            return result

        oldest_source = min(self._latest, key=lambda k: self._latest[k]["timestamp"])
        self._latest.pop(oldest_source, None)
        return None

    def reset(self) -> None:
        self._buffer.clear()
        self._latest.clear()

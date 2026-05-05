from __future__ import annotations

from collections import defaultdict
from typing import Any

from drone_traffic.core.types import TelemetryMessage
from drone_traffic.core.registry import register_fusion
from drone_traffic.fusion.conflict_resolver import ConflictResolver
from drone_traffic.fusion.engine_base import FusionInterface
from drone_traffic.fusion.temporal_sync import TemporalSyncBuffer


@register_fusion("homography")
class HomographyFusionEngine(FusionInterface):
    def __init__(
        self,
        max_time_sync_diff: float = 0.05,
        homographies: dict[str, Any] | None = None,
        association_threshold: float = 2.0,
        conflict_policy: str = "merge",
        **kwargs,
    ):
        self._sync_buffer = TemporalSyncBuffer(
            max_time_diff=max_time_sync_diff,
            sources_expected=2,
        )
        self._conflict_resolver = ConflictResolver(policy=conflict_policy)
        self._association_threshold = association_threshold
        self._homographies = homographies
        self._global_track_counter = 0

    def process(
        self, messages: dict[str, TelemetryMessage]
    ) -> dict[str, Any]:
        for source_id, msg in messages.items():
            self._sync_buffer.add(source_id, msg)

        synced = self._sync_buffer.try_flush()
        if synced is None:
            return {"global_tracks": [], "events": []}

        return self._conflict_resolver.resolve(synced, homographies=self._homographies)

    def reset(self) -> None:
        self._sync_buffer.reset()
        self._conflict_resolver.reset()
        self._global_track_counter = 0

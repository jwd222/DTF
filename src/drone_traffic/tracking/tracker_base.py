from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from drone_traffic.core.types import Detection, TrackState


class TrackingInterface(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray | None = None,
        timestamp: float = 0.0,
    ) -> list[TrackState]:
        raise NotImplementedError

    @abstractmethod
    def get_active_tracks(self) -> list[TrackState]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

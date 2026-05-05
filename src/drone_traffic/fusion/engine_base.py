from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from drone_traffic.core.types import TelemetryMessage


class FusionInterface(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def process(
        self, messages: dict[str, TelemetryMessage]
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from drone_traffic.core.types import Detection


class DetectorInterface(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        features: dict[str, torch.Tensor],
        original_shape: tuple[int, ...],
        scale_ratio: float,
        pad: tuple[int, int, int, int],
    ) -> list[Detection]:
        raise NotImplementedError

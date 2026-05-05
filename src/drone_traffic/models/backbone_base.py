from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn


class BackboneInterface(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_channels(self) -> list[int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_strides(self) -> list[int]:
        raise NotImplementedError


class DummyBackbone(BackboneInterface):
    def __init__(
        self,
        in_channels: int = 3,
        channels: list[int] | None = None,
        strides: list[int] | None = None,
    ):
        super().__init__()
        self._channels = channels or [64, 128, 256]
        self._strides = strides or [4, 8, 16]
        self.layers = nn.ModuleList()
        for ch in self._channels:
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, ch, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = ch

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = {}
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            features[f"c{i + 1}"] = out
        return features

    @property
    def output_channels(self) -> list[int]:
        return self._channels

    @property
    def output_strides(self) -> list[int]:
        return self._strides

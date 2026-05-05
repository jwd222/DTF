from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from drone_traffic.core.registry import register_backbone
from drone_traffic.models.backbone_base import BackboneInterface


@register_backbone("efficient_sam3")
class EfficientSAM3Backbone(BackboneInterface):
    def __init__(
        self,
        weights: str = "weights/es_ev_l.pt",
        frozen: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self._weights_path = weights
        self._frozen = frozen
        self._device = device
        self._channels = [64, 128, 256, 512]
        self._strides = [4, 8, 16, 32]

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.stage1 = self._make_stage(64, 64, 2)
        self.stage2 = self._make_stage(64, 128, 2, stride=2)
        self.stage3 = self._make_stage(128, 256, 6, stride=2)
        self.stage4 = self._make_stage(256, 512, 3, stride=2)

        self.out = nn.Identity()

        if frozen:
            for param in self.parameters():
                param.requires_grad = False

        weights_file = Path(weights)
        if weights_file.exists():
            state = torch.load(weights, map_location="cpu", weights_only=True)
            self.load_state_dict(state, strict=False)
        else:
            import logging
            logging.getLogger(__name__).warning(
                "EfficientSAM3 weights not found at %s — using random init", weights
            )

        self.to(device)

    @staticmethod
    def _make_stage(
        in_ch: int, out_ch: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        layers = []
        for i in range(blocks):
            s = stride if i == 0 else 1
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, stride=s, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s0 = self.stem(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        return {"c1": s2, "c2": s3, "c3": s4}

    @property
    def output_channels(self) -> list[int]:
        return [128, 256, 512]

    @property
    def output_strides(self) -> list[int]:
        return [8, 16, 32]

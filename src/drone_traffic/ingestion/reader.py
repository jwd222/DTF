from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generator

import cv2
import numpy as np


@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    image: np.ndarray
    original_shape: tuple[int, int, int]
    scale_ratio: float
    pad: tuple[int, int, int, int]


class VideoReader:
    def __init__(
        self,
        source: str,
        target_fps: int = 30,
        resolution: tuple[int, int] = (640, 640),
        letterbox: bool = True,
        normalize: bool = True,
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ):
        self.source = source
        self.target_fps = target_fps
        self.resolution = resolution
        self.letterbox = letterbox
        self.normalize = normalize
        self.mean = np.array(mean or [0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array(std or [0.229, 0.224, 0.225], dtype=np.float32)

        self._cap: cv2.VideoCapture | None = None
        self._frame_id = 0

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {self.source}")

        self._source_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_skip = max(1, int(self._source_fps / self.target_fps))
        self._frame_id = 0

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def source_fps(self) -> float:
        return getattr(self, "_source_fps", 30.0)

    @property
    def total_frames(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int, int, int]]:
        original_shape = frame.shape
        target_w, target_h = self.resolution

        if self.letterbox:
            h, w = frame.shape[:2]
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            pad_w = target_w - new_w
            pad_h = target_h - new_h
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(114, 114, 114),
            )
            pad = (top, left, bottom, right)
            ratio = scale
        else:
            padded = cv2.resize(frame, (target_w, target_h))
            h, w = frame.shape[:2]
            ratio = min(target_w / w, target_h / h)
            pad = (0, 0, 0, 0)

        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = blob.astype(np.float32) / 255.0

        if self.normalize:
            blob = (blob - self.mean) / self.std

        blob = blob.transpose(2, 0, 1)
        return blob, ratio, pad

    def read_frames(self) -> Generator[FramePacket, None, None]:
        if self._cap is None:
            raise RuntimeError("VideoReader not opened. Call open() first.")

        start_time = time.time()

        while True:
            for _ in range(self._frame_skip - 1):
                ret = self._cap.grab()
                if not ret:
                    return

            ret, frame = self._cap.read()
            if not ret:
                return

            original_shape = frame.shape
            processed, ratio, pad = self._preprocess(frame)
            elapsed = time.time() - start_time

            yield FramePacket(
                frame_id=self._frame_id,
                timestamp=elapsed,
                image=processed,
                original_shape=original_shape,
                scale_ratio=ratio,
                pad=pad,
            )
            self._frame_id += 1

    def read_all(self) -> list[FramePacket]:
        return list(self.read_frames())

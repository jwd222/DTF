from __future__ import annotations

from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_video(tmp_path: Path) -> str:
    video_path = str(tmp_path / "test_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 640))

    for i in range(90):
        frame = np.full((640, 640, 3), (40, 40, 40), dtype=np.uint8)
        x = (i * 5) % 580
        y = (i * 3) % 580
        cv2.rectangle(frame, (x, y), (x + 60, y + 30), (0, 255, 0), -1)
        cv2.rectangle(frame, (x + 100, y + 50), (x + 160, y + 80), (0, 0, 255), -1)
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def sample_frame() -> np.ndarray:
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_homography() -> np.ndarray:
    return np.array(
        [
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def sample_detections() -> list:
    from drone_traffic.core.types import BBox, Detection

    return [
        Detection(bbox=BBox(x1=100, y1=100, x2=200, y2=200), confidence=0.9, class_id=0),
        Detection(bbox=BBox(x1=300, y1=300, x2=400, y2=400), confidence=0.8, class_id=1),
        Detection(bbox=BBox(x1=500, y1=100, x2=600, y2=200), confidence=0.7, class_id=0),
    ]

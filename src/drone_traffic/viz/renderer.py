from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
    (255, 128, 0),
]


class BEVRenderer:
    def __init__(
        self,
        canvas_size: tuple[int, int] = (1000, 1000),
        scale: float = 10.0,
        bg_color: tuple[int, int, int] = (40, 40, 40),
    ):
        self._canvas_size = canvas_size
        self._scale = scale
        self._bg_color = bg_color
        self._canvas = np.full(
            (canvas_size[1], canvas_size[0], 3), bg_color, dtype=np.uint8
        )
        self._trajectories: dict[int, list[tuple[float, float]]] = {}

    def clear(self) -> None:
        self._canvas[:] = self._bg_color

    def draw_grid(self, spacing: float = 10.0, color: tuple[int, int, int] = (60, 60, 60)) -> None:
        h, w = self._canvas.shape[:2]
        pixel_spacing = int(spacing * self._scale)
        for x in range(0, w, pixel_spacing):
            cv2.line(self._canvas, (x, 0), (x, h), color, 1)
        for y in range(0, h, pixel_spacing):
            cv2.line(self._canvas, (0, y), (w, y), color, 1)

    def draw_track(
        self,
        global_id: int,
        bev_position: tuple[float, float],
        confidence: float = 1.0,
        velocity: tuple[float, float] | None = None,
        class_id: int = 0,
    ) -> None:
        color = COLORS[global_id % len(COLORS)]
        px = int(bev_position[0] * self._scale)
        py = int(bev_position[1] * self._scale)

        radius = max(4, int(8 * confidence))
        cv2.circle(self._canvas, (px, py), radius, color, -1)
        cv2.circle(self._canvas, (px, py), radius, (255, 255, 255), 1)

        label = f"ID:{global_id}"
        cv2.putText(
            self._canvas, label, (px + radius + 2, py - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
        )

        if velocity is not None:
            vx, vy = velocity
            end_x = int(px + vx * self._scale)
            end_y = int(py + vy * self._scale)
            cv2.arrowedLine(self._canvas, (px, py), (end_x, end_y), color, 2)

        if global_id not in self._trajectories:
            self._trajectories[global_id] = []
        self._trajectories[global_id].append(bev_position)

    def draw_trajectory(self, global_id: int, max_points: int = 100) -> None:
        pts = self._trajectories.get(global_id, [])
        if len(pts) < 2:
            return
        color = COLORS[global_id % len(COLORS)]
        recent = pts[-max_points:]
        pixel_pts = [(int(x * self._scale), int(y * self._scale)) for x, y in recent]
        for i in range(1, len(pixel_pts)):
            alpha = i / len(pixel_pts)
            thickness = max(1, int(alpha * 2))
            cv2.line(self._canvas, pixel_pts[i - 1], pixel_pts[i], color, thickness)

    def draw_frame_overlay(
        self,
        frame: np.ndarray,
        tracks: list[dict[str, Any]],
    ) -> np.ndarray:
        annotated = frame.copy()
        for t in tracks:
            color = COLORS[t.get("global_id", 0) % len(COLORS)]
            bbox = t.get("bbox", {})
            x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
            x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f'{t.get("global_id", "?")} {t.get("confidence", 0):.2f}'
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )
        return annotated

    def get_canvas(self) -> np.ndarray:
        return self._canvas.copy()

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), self._canvas)

    def reset_trajectories(self) -> None:
        self._trajectories.clear()

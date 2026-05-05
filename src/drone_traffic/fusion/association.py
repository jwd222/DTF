from __future__ import annotations

import numpy as np
from typing import Any

from drone_traffic.core.types import TelemetryMessage
from drone_traffic.tracking.matching import hungarian_assignment


def mahalanobis_distance(
    points_a: np.ndarray, points_b: np.ndarray, cov: np.ndarray | None = None
) -> np.ndarray:
    diff = points_a[:, None, :] - points_b[None, :, :]
    if cov is None:
        cov = np.eye(points_a.shape[-1])
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.eye(points_a.shape[-1])

    dist = np.sqrt(np.einsum("...i,ij,...j->...", diff, inv_cov, diff))
    return dist


def associate_tracks(
    tracks_a: list[dict[str, Any]],
    tracks_b: list[dict[str, Any]],
    threshold: float = 2.0,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks_a or not tracks_b:
        return [], list(range(len(tracks_a))), list(range(len(tracks_b)))

    pts_a = np.array([t.get("bev_position", [0, 0])[:2] for t in tracks_a])
    pts_b = np.array([t.get("bev_position", [0, 0])[:2] for t in tracks_b])

    cost = mahalanobis_distance(pts_a, pts_b)
    matches, unmatched_a, unmatched_b = hungarian_assignment(cost, threshold)
    return matches, unmatched_a, unmatched_b

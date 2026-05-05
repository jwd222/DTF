from __future__ import annotations

import numpy as np


def iou_distance(tracks: np.ndarray, detections: np.ndarray) -> np.ndarray:
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)))

    t_x1 = tracks[:, 0] - tracks[:, 2] / 2
    t_y1 = tracks[:, 1] - tracks[:, 3] / 2
    t_x2 = tracks[:, 0] + tracks[:, 2] / 2
    t_y2 = tracks[:, 1] + tracks[:, 3] / 2

    d_x1 = detections[:, 0] - detections[:, 2] / 2
    d_y1 = detections[:, 1] - detections[:, 3] / 2
    d_x2 = detections[:, 0] + detections[:, 2] / 2
    d_y2 = detections[:, 1] + detections[:, 3] / 2

    ix1 = np.maximum(t_x1[:, None], d_x1[None, :])
    iy1 = np.maximum(t_y1[:, None], d_y1[None, :])
    ix2 = np.minimum(t_x2[:, None], d_x2[None, :])
    iy2 = np.minimum(t_y2[:, None], d_y2[None, :])

    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)

    t_area = (t_x2 - t_x1) * (t_y2 - t_y1)
    d_area = (d_x2 - d_x1) * (d_y2 - d_y1)
    union = t_area[:, None] + d_area[None, :] - inter

    iou = np.where(union > 0, inter / union, 0.0)
    return 1.0 - iou


def hungarian_assignment(cost_matrix: np.ndarray, threshold: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    try:
        from scipy.optimize import linear_sum_assignment

        row_idx, col_idx = linear_sum_assignment(cost_matrix)
    except ImportError:
        return _greedy_assignment(cost_matrix, threshold)

    matches = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] <= threshold:
            matches.append((r, c))
            unmatched_rows.discard(r)
            unmatched_cols.discard(c)

    return matches, list(unmatched_rows), list(unmatched_cols)


def _greedy_assignment(
    cost_matrix: np.ndarray, threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    matches = []
    used_rows = set()
    used_cols = set()

    flat_idx = np.argsort(cost_matrix.ravel())
    for idx in flat_idx:
        r, c = divmod(int(idx), cost_matrix.shape[1])
        if r in used_rows or c in used_cols:
            continue
        if cost_matrix[r, c] > threshold:
            break
        matches.append((r, c))
        used_rows.add(r)
        used_cols.add(c)

    unmatched_rows = [r for r in range(cost_matrix.shape[0]) if r not in used_rows]
    unmatched_cols = [c for c in range(cost_matrix.shape[1]) if c not in used_cols]
    return matches, unmatched_rows, unmatched_cols

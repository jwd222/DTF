from __future__ import annotations

import torch


def non_max_suppression(
    predictions: torch.Tensor,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> torch.Tensor:
    if predictions.dim() == 3:
        results = []
        for i in range(predictions.shape[0]):
            results.append(
                _nms_single(
                    predictions[i], conf_threshold, iou_threshold, max_detections
                )
            )
        max_len = max(r.shape[0] for r in results)
        padded = []
        for r in results:
            if r.shape[0] < max_len:
                pad = torch.zeros(
                    max_len - r.shape[0], r.shape[1], device=r.device, dtype=r.dtype
                )
                padded.append(torch.cat([r, pad]))
            else:
                padded.append(r)
        return torch.stack(padded)

    return _nms_single(predictions, conf_threshold, iou_threshold, max_detections)


def _nms_single(
    pred: torch.Tensor,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> torch.Tensor:
    num_classes = pred.shape[1] - 4
    if num_classes <= 0:
        return torch.zeros(0, pred.shape[1], device=pred.device)

    scores = pred[:, 4]
    mask = scores > conf_threshold
    if mask.sum() == 0:
        return torch.zeros(0, pred.shape[1], device=pred.device)

    filtered = pred[mask]
    scores = scores[mask]

    order = scores.argsort(descending=True)[:max_detections]
    filtered = filtered[order]
    scores = scores[order]

    x1 = filtered[:, 0] - filtered[:, 2] / 2
    y1 = filtered[:, 1] - filtered[:, 3] / 2
    x2 = filtered[:, 0] + filtered[:, 2] / 2
    y2 = filtered[:, 1] + filtered[:, 3] / 2

    keep = []
    areas = (x2 - x1) * (y2 - y1)

    for i in range(len(filtered)):
        should_keep = True
        for j in keep:
            xx1 = max(x1[i].item(), x1[j].item())
            yy1 = max(y1[i].item(), y1[j].item())
            xx2 = min(x2[i].item(), x2[j].item())
            yy2 = min(y2[i].item(), y2[j].item())

            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            union = areas[i].item() + areas[j].item() - inter
            if union > 0 and inter / union > iou_threshold:
                should_keep = False
                break

        if should_keep:
            keep.append(i)

    if not keep:
        return torch.zeros(0, pred.shape[1], device=pred.device)

    result = filtered[keep]
    return result

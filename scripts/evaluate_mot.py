"""MOT evaluation using py-motmetrics for tracking benchmarking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_mot_ground_truth(gt_path: str) -> dict[int, list[dict]]:
    frames: dict[int, list[dict]] = {}

    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue

            frame_id = int(parts[0])
            track_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])

            if frame_id not in frames:
                frames[frame_id] = []

            frames[frame_id].append({
                "track_id": track_id,
                "bbox": [x, y, x + w, y + h],
            })

    return frames


def load_predictions(pred_path: str) -> dict[int, list[dict]]:
    frames: dict[int, list[dict]] = {}

    with open(pred_path, "r") as f:
        data = json.load(f)

    for frame_data in data:
        frame_id = frame_data["frame_id"]
        frames[frame_id] = frame_data.get("tracks", [])

    return frames


def compute_mot_metrics(gt_frames: dict, pred_frames: dict) -> dict:
    import motmetrics as mm

    mh = mm.metrics.create()
    accumulator = mm.MOTAccumulator()

    all_frame_ids = sorted(set(list(gt_frames.keys()) + list(pred_frames.keys())))

    for frame_id in all_frame_ids:
        gt_objects = gt_frames.get(frame_id, [])
        pred_objects = pred_frames.get(frame_id, [])

        gt_ids = [o["track_id"] for o in gt_objects]
        pred_ids = [o["track_id"] for o in pred_objects]

        if gt_objects and pred_objects:
            gt_bboxes = np.array([o["bbox"] for o in gt_objects])
            pred_bboxes = np.array([o["bbox"] for o in pred_objects])

            distances = mm.distances.iou_matrix(gt_bboxes, pred_bboxes, max_iou=0.5)
        else:
            distances = np.empty((len(gt_ids), len(pred_ids)))

        accumulator.update(gt_ids, pred_ids, distances)

    summary = mh.compute(accumulator, metrics=mm.metrics.motchallenge_metrics, name="eval")

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    print(strsummary)

    results = {}
    if hasattr(summary, "columns"):
        for col in summary.columns:
            val = summary[col].iloc[0] if len(summary) > 0 else None
            results[col] = float(val) if val is not None and not isinstance(val, str) else val

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MOT metrics")
    parser.add_argument("--gt", type=str, required=True, help="Ground truth MOT file (MOT format)")
    parser.add_argument("--pred", type=str, required=True, help="Predictions JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output JSON for metrics")
    args = parser.parse_args()

    try:
        import motmetrics as mm
    except ImportError:
        print("motmetrics is required. Install with: pip install motmetrics")
        return

    gt_frames = load_mot_ground_truth(args.gt)
    pred_frames = load_predictions(args.pred)

    print(f"Ground truth: {len(gt_frames)} frames, total {sum(len(v) for v in gt_frames.values())} annotations")
    print(f"Predictions: {len(pred_frames)} frames, total {sum(len(v) for v in pred_frames.values())} detections")

    metrics = compute_mot_metrics(gt_frames, pred_frames)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()

"""Validate a trained YOLOv26 model on the vehicle detection dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Validate YOLOv26 vehicle detector")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Dataset config")
    parser.add_argument("--imgsz", type=int, default=960, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
    )

    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
        class_names = model.model.names if hasattr(model, "model") else {}
        for i, ap in enumerate(metrics.box.maps):
            name = class_names.get(i, f"class_{i}")
            print(f"  {name}: AP@50-95 = {ap:.4f}")


if __name__ == "__main__":
    main()

"""Export YOLOv26 model to TensorRT engine."""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv26 to TensorRT engine")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt weights")
    parser.add_argument("--imgsz", type=int, default=960, help="Image size for export")
    parser.add_argument("--half", action="store_true", default=True, help="FP16 export")
    parser.add_argument("--int8", action="store_true", default=True, help="INT8 quantization")
    parser.add_argument("--no-int8", action="store_true", help="Disable INT8")
    parser.add_argument(
        "--calib-data",
        type=str,
        default=None,
        help="Path to calibration data for INT8",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))

    use_int8 = args.int8 and not args.no_int8

    export_kwargs = {
        "format": "engine",
        "imgsz": args.imgsz,
        "half": args.half,
        "int8": use_int8,
        "dynamic": False,
        "batch": 1,
    }

    if use_int8 and args.calib_data:
        export_kwargs["data"] = args.calib_data

    exported_path = model.export(**export_kwargs)
    print(f"Model exported to: {exported_path}")


if __name__ == "__main__":
    main()

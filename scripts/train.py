"""Train YOLOv26 on the vehicle detection dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv26 vehicle detector")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_args.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with open(config_path, "r") as f:
        train_cfg = yaml.safe_load(f)

    model_path = train_cfg.pop("model", "yolo26s.pt")

    if args.resume:
        model = YOLO(args.resume)
    else:
        model = YOLO(model_path)

    results = model.train(**train_cfg)
    print(f"Training complete. Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()

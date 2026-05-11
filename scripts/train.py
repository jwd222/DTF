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
        results = model.train(**train_cfg)
        print(f"Training complete. Results saved to: {results.save_dir}")
        return

    phase1_epochs = 20
    total_epochs = train_cfg.get("epochs", 300)
    phase2_epochs = max(1, total_epochs - phase1_epochs)

    base_name = train_cfg.get("name", "yolo26_vehicle")

    phase1_cfg = {
        **train_cfg,
        "freeze": 10,
        "epochs": phase1_epochs,
        "lr0": 0.001,
        "name": base_name + "_phase1",
    }

    model = YOLO(model_path)
    print(f"Phase 1: frozen backbone, {phase1_epochs} epochs, lr={phase1_cfg['lr0']}")
    phase1_results = model.train(**phase1_cfg)

    phase1_weights = Path(phase1_results.save_dir) / "weights" / "last.pt"

    phase2_cfg = {
        **train_cfg,
        "epochs": phase2_epochs,
        "name": base_name + "_phase2",
    }

    model = YOLO(str(phase1_weights))
    print(f"Phase 2: full fine-tune, {phase2_epochs} epochs, lr={phase2_cfg.get('lr0', 0.0002)}")
    results = model.train(**phase2_cfg)
    print(f"Training complete. Results saved to: {results.save_dir}")


if __name__ == "__main__":
    main()

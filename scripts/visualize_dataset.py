"""Visualize YOLO-format dataset by overlaying labels on sample images.

Randomly picks images from train/val splits, draws bounding boxes with class
names, and saves each annotated image individually to disk.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml


CLASS_COLORS = [
    (0, 255, 0),
    (255, 165, 0),
    (0, 165, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
]


def draw_labels(
    img: np.ndarray,
    labels: list[tuple[int, float, float, float, float]],
    names: list[str],
) -> np.ndarray:
    h, w = img.shape[:2]
    vis = img.copy()
    for cls_id, xc, yc, bw, bh in labels:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = names[cls_id] if cls_id < len(names) else str(cls_id)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(
            vis, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
    return vis


def load_labels(lbl_path: Path) -> list[tuple[int, float, float, float, float]]:
    labels = []
    if not lbl_path.exists():
        return labels
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                coords = tuple(float(x) for x in parts[1:5])
                labels.append((cls_id, *coords))
    return labels


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO dataset samples with labels")
    parser.add_argument("--data", type=str, default="configs/data.yaml", help="Data config YAML")
    parser.add_argument("--n-samples", type=int, default=16, help="Total samples to visualize")
    parser.add_argument("--output", type=str, default="visualizations", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    with open(args.data, "r") as f:
        data_cfg = yaml.safe_load(f)

    base = Path(data_cfg["path"])
    names = list(data_cfg.get("names", {}).values())

    splits = {}
    for split_key in ("train", "val"):
        split_img_dir = base / data_cfg[split_key]
        if not split_img_dir.exists():
            print(f"  {split_key} dir not found: {split_img_dir}, skipping")
            continue
        images = sorted(
            p for p in split_img_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )
        splits[split_key] = images
        print(f"  {split_key}: {len(images)} images")

    if not splits:
        raise FileNotFoundError("No image directories found")

    if args.seed is not None:
        random.seed(args.seed)

    out_dir = Path(args.output)

    for split_key, image_paths in splits.items():
        n = min(args.n_samples, len(image_paths))
        sampled = random.sample(image_paths, n)
        split_out = out_dir / split_key
        split_out.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(sampled):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  WARNING: Could not read {img_path}")
                continue
            lbl_path = base / "labels" / split_key / f"{img_path.stem}.txt"
            labels = load_labels(lbl_path)
            vis = draw_labels(img, labels, names)

            seq_name = img_path.stem.split("_")[0] if "_" in img_path.stem else ""
            info = f"{seq_name} | {len(labels)} objs | {split_key}"
            cv2.putText(vis, info, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            out_path = split_out / f"{i:03d}_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), vis)

        print(f"  Saved {n} images to {split_out}/")

    print("Done.")


if __name__ == "__main__":
    main()

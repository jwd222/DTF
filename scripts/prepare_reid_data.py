"""Prepare Re-ID training data by cropping vehicle patches from annotated tracks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Prepare Re-ID training crops")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--annotations", type=str, required=True, help="Annotations JSON/COCO path")
    parser.add_argument("--output-dir", type=str, default="data/reid_crops", help="Output directory")
    parser.add_argument("--min-track-len", type=int, default=5, help="Minimum track length")
    parser.add_argument("--crop-size", type=int, default=256, help="Crop size for output patches")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    gallery_dir = output_dir / "gallery"
    query_dir = output_dir / "query"
    train_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = Path(args.annotations)
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    tracks: dict[int, list[dict]] = {}
    for ann in annotations.get("annotations", []):
        tid = ann.get("track_id", ann.get("id", 0))
        if tid not in tracks:
            tracks[tid] = []
        tracks[tid].append(ann)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.video}")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for tid, anns in tracks.items():
            if len(anns) < args.min_track_len:
                continue

            for ann in anns:
                if ann.get("frame_id", ann.get("image_id", -1)) != frame_idx:
                    continue

                bbox = ann.get("bbox", [])
                if len(bbox) != 4:
                    continue

                x, y, w, h = [int(v) for v in bbox]
                x2 = min(x + w, frame.shape[1])
                y2 = min(y + h, frame.shape[0])
                x = max(0, x)
                y = max(0, y)

                crop = frame[y:y2, x:x2]
                if crop.size == 0:
                    continue

                aspect = w / max(h, 1)
                if aspect >= 1:
                    new_w = args.crop_size
                    new_h = max(1, int(args.crop_size / aspect))
                else:
                    new_h = args.crop_size
                    new_w = max(1, int(args.crop_size * aspect))

                resized = cv2.resize(crop, (new_w, new_h))

                padded = np.full((args.crop_size, args.crop_size, 3), 114, dtype=np.uint8)
                pad_x = (args.crop_size - new_w) // 2
                pad_y = (args.crop_size - new_h) // 2
                padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

                identity_dir = train_dir / f"{tid:04d}"
                identity_dir.mkdir(parents=True, exist_ok=True)
                crop_path = identity_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(crop_path), padded)
                saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"Extracted {saved_count} crops for {len([t for t in tracks if len(tracks[t]) >= args.min_track_len])} tracks")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()

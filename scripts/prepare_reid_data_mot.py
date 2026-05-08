"""Prepare Re-ID training data from VisDrone MOT annotations and image sequences.

Reads MOT-format annotation files, groups detections by target_id (vehicle identity),
crops and resizes each detection from the source images, and saves them organized
by identity for torchreid-compatible training.

VisDrone MOT annotation format (per line):
    <frame_id>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,
    <score>,<object_category>,<truncation>,<occlusion>

Output structure (torchreid ImageDataManager compatible):
    data/reid_crops/
    ├── train/
    │   ├── 000000/
    │   │   ├── seq1_f000045.jpg
    │   │   └── ...
    │   └── 000001/
    ├── gallery/
    │   └── ...
    └── query/
        └── ...
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._visdrone_constants import VISEDRONE_TO_YOLO, VISDRONE_IGNORE


CROP_H = 256
CROP_W = 128

MIN_FRAMES_FOR_QUERY_SPLIT = 10


def _crop_and_pad(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    target_h: int = CROP_H,
    target_w: int = CROP_W,
    pad_value: int = 114,
) -> np.ndarray | None:
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(x + w, frame.shape[1])
    y2 = min(y + h, frame.shape[0])

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    padded = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    resized = cv2.resize(crop, (target_w, target_h))
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return padded


def _parse_mot_annotations(
    ann_file: Path,
) -> dict[int, dict[int, tuple[int, int, int, int]]]:
    """Parse MOT annotation file into {frame_id: {target_id: (x, y, w, h)}}."""
    frame_dets: dict[int, dict[int, tuple[int, int, int, int]]] = {}
    with open(ann_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 10:
                continue
            frame_id = int(parts[0])
            target_id = int(parts[1])
            bbox_left = int(parts[2])
            bbox_top = int(parts[3])
            bbox_width = int(parts[4])
            bbox_height = int(parts[5])
            obj_category = int(parts[7])
            occlusion = int(parts[9])

            if obj_category in VISDRONE_IGNORE:
                continue
            if obj_category not in VISEDRONE_TO_YOLO:
                continue
            if bbox_width < 2 or bbox_height < 2:
                continue
            if occlusion >= 2:
                continue

            if frame_id not in frame_dets:
                frame_dets[frame_id] = {}
            frame_dets[frame_id][target_id] = (bbox_left, bbox_top, bbox_width, bbox_height)
    return frame_dets


def _build_track_index(
    frame_dets: dict[int, dict[int, tuple[int, int, int, int]]],
) -> dict[int, list[tuple[int, int, int, int, int]]]:
    """Build track index: {target_id: [(frame_id, x, y, w, h), ...]}."""
    tracks: dict[int, list[tuple[int, int, int, int, int]]] = {}
    for frame_id, targets in frame_dets.items():
        for tid, bbox in targets.items():
            if tid not in tracks:
                tracks[tid] = []
            tracks[tid].append((frame_id, *bbox))
    for tid in tracks:
        tracks[tid].sort(key=lambda d: d[0])
    return tracks


def _split_train_query(
    track_frames: list[tuple[int, int, int, int, int]],
    query_ratio: float = 0.2,
) -> tuple[list[tuple[int, int, int, int, int]], list[tuple[int, int, int, int, int]]]:
    """Split a track's frames into train and query sets (temporal split).

    Short tracks (fewer than MIN_FRAMES_FOR_QUERY_SPLIT) keep all frames in train.
    """
    if len(track_frames) < MIN_FRAMES_FOR_QUERY_SPLIT:
        return track_frames, []
    n_query = max(1, int(len(track_frames) * query_ratio))
    return track_frames[:-n_query], track_frames[-n_query:]


def _process_sequence(
    seq_name: str,
    sequences_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    tid_offset: int,
    min_track_len: int,
    query_ratio: float,
    gallery_ratio: float,
    max_crops_per_track: int,
) -> dict:
    seq_img_dir = sequences_dir / seq_name
    ann_file = annotations_dir / f"{seq_name}.txt"

    image_files = sorted(
        p for p in seq_img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )
    if not image_files:
        return {"identities": 0, "train": 0, "gallery": 0, "query": 0, "skipped": 0}

    img_index: dict[int, Path] = {int(p.stem): p for p in image_files}

    frame_dets = _parse_mot_annotations(ann_file)
    tracks = _build_track_index(frame_dets)

    frame_crops: dict[int, list[tuple[tuple[int, int, int, int], Path, str]]] = defaultdict(list)

    global_tid = tid_offset
    identities = 0
    skipped_short = 0

    for tid, detections in tracks.items():
        if len(detections) < min_track_len:
            skipped_short += 1
            continue

        sampled = detections
        if len(sampled) > max_crops_per_track:
            step = len(sampled) / max_crops_per_track
            sampled = [detections[int(i * step)] for i in range(max_crops_per_track)]

        train_frames, query_frames = _split_train_query(sampled, query_ratio)

        gallery_frames: list[tuple[int, int, int, int, int]] = []
        if len(sampled) >= MIN_FRAMES_FOR_QUERY_SPLIT and gallery_ratio > 0:
            n_gallery = max(1, int(len(train_frames) * gallery_ratio))
            gallery_frames = train_frames[-n_gallery:]
            train_frames = train_frames[:-n_gallery]

        identity_name = f"{global_tid:06d}"
        for split_name, frames in [
            ("train", train_frames),
            ("gallery", gallery_frames),
            ("query", query_frames),
        ]:
            split_dir = output_dir / split_name / identity_name
            split_dir.mkdir(parents=True, exist_ok=True)
            for frame_id, bx, by, bw, bh in frames:
                out_path = split_dir / f"{seq_name}_f{frame_id:06d}.jpg"
                frame_crops[frame_id].append(((bx, by, bw, bh), out_path, split_name))

        global_tid += 1
        identities += 1

    counts = {"train": 0, "gallery": 0, "query": 0}
    for frame_id, crops in frame_crops.items():
        img_path = img_index.get(frame_id)
        if img_path is None:
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        for (bx, by, bw, bh), out_path, split_name in crops:
            crop = _crop_and_pad(frame, bx, by, bw, bh)
            if crop is None:
                continue
            cv2.imwrite(str(out_path), crop)
            counts[split_name] += 1

    print(f"  {seq_name}: {identities} IDs, {sum(counts.values())} crops")
    return {
        "identities": identities,
        "train": counts["train"],
        "gallery": counts["gallery"],
        "query": counts["query"],
        "skipped": skipped_short,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare Re-ID crops from VisDrone MOT data")
    parser.add_argument("--input-dir", type=str, required=True, help="VisDrone MOT base directory (contains sequences/ and annotations/)")
    parser.add_argument("--output-dir", type=str, default="data/reid_crops", help="Output directory for Re-ID crops")
    parser.add_argument("--min-track-len", type=int, default=5, help="Minimum detections per track to include")
    parser.add_argument("--query-ratio", type=float, default=0.2, help="Fraction of each track's frames for query set (tracks shorter than 10 frames go entirely to train/gallery)")
    parser.add_argument("--gallery-ratio", type=float, default=0.3, help="Fraction of each track's frames for gallery set (only for tracks >= 10 frames)")
    parser.add_argument("--max-crops-per-track", type=int, default=30, help="Max crops per track per sequence (sample if exceeded)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for processing sequences")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    sequences_dir = input_dir / "sequences"
    annotations_dir = input_dir / "annotations"
    output_dir = Path(args.output_dir)

    if not sequences_dir.exists():
        raise FileNotFoundError(f"Sequences directory not found: {sequences_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    train_dir = output_dir / "train"
    gallery_dir = output_dir / "gallery"
    query_dir = output_dir / "query"
    train_dir.mkdir(parents=True, exist_ok=True)
    gallery_dir.mkdir(exist_ok=True)
    query_dir.mkdir(exist_ok=True)

    sequence_names = sorted(
        p.name for p in sequences_dir.iterdir() if p.is_dir()
    )

    if not sequence_names:
        raise FileNotFoundError(f"No sequence folders found in {sequences_dir}")

    sequence_track_counts = []
    for seq_name in sequence_names:
        ann_file = annotations_dir / f"{seq_name}.txt"
        if not ann_file.exists():
            sequence_track_counts.append(0)
            continue
        frame_dets = _parse_mot_annotations(ann_file)
        tracks = _build_track_index(frame_dets)
        count = sum(1 for d in tracks.values() if len(d) >= args.min_track_len)
        sequence_track_counts.append(count)

    tid_offsets = []
    offset = 0
    for count in sequence_track_counts:
        tid_offsets.append(offset)
        offset += count

    def _dispatch():
        for i, seq_name in enumerate(sequence_names):
            ann_file = annotations_dir / f"{seq_name}.txt"
            if not ann_file.exists():
                continue
            yield i, seq_name

    print(f"Processing {sum(1 for _, _ in _dispatch())} sequences with {args.workers} worker(s) ...")

    all_stats = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for i, seq_name in _dispatch():
                futures[executor.submit(
                    _process_sequence,
                    seq_name,
                    sequences_dir,
                    annotations_dir,
                    output_dir,
                    tid_offsets[i],
                    args.min_track_len,
                    args.query_ratio,
                    args.gallery_ratio,
                    args.max_crops_per_track,
                )] = seq_name
            for future in as_completed(futures):
                all_stats.append(future.result())
    else:
        for i, seq_name in _dispatch():
            stats = _process_sequence(
                seq_name,
                sequences_dir,
                annotations_dir,
                output_dir,
                tid_offsets[i],
                args.min_track_len,
                args.query_ratio,
                args.gallery_ratio,
                args.max_crops_per_track,
            )
            all_stats.append(stats)

    total_identities = sum(s["identities"] for s in all_stats)
    total_crops_train = sum(s["train"] for s in all_stats)
    total_crops_gallery = sum(s["gallery"] for s in all_stats)
    total_crops_query = sum(s["query"] for s in all_stats)
    skipped_short = sum(s["skipped"] for s in all_stats)

    print()
    print(f"Re-ID data preparation complete. Output: {output_dir}")
    print(f"  Identities:    {total_identities}")
    print(f"  Train crops:   {total_crops_train}")
    print(f"  Gallery crops: {total_crops_gallery}")
    print(f"  Query crops:   {total_crops_query}")
    print(f"  Skipped (short tracks): {skipped_short}")
    print()
    print(f"Train:   {train_dir}")
    print(f"Gallery: {gallery_dir}")
    print(f"Query:   {query_dir}")


if __name__ == "__main__":
    main()

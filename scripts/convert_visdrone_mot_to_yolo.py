"""Convert VisDrone MOT annotations to YOLO format for vehicle detection training.

VisDrone MOT annotation format (per line):
    <frame_id>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

    Raw MOT object_category matches the official VisDrone.yaml (0-indexed):
    0: pedestrian, 1: people, 2: bicycle, 3: car, 4: van, 5: truck,
    6: tricycle, 7: awning-tricycle, 8: bus, 9: motor

YOLO label format (per line):
    <class_id> <x_center> <y_center> <width> <height>  (all normalized 0-1)
"""
from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._visdrone_constants import VISEDRONE_TO_YOLO, YOLO_NAMES, VISDRONE_IGNORE


def _get_image_size(path: Path) -> tuple[int, int]:
    """Read image dimensions from JPEG/PNG/BMP header without full decode."""
    with open(path, "rb") as f:
        head = f.read(32)

    if head[:8] == b"\x89PNG\r\n\x1a\n":
        w = struct.unpack(">I", head[16:20])[0]
        h = struct.unpack(">I", head[20:24])[0]
        return w, h

    if head[:2] == b"\xff\xd8":
        with open(path, "rb") as f:
            f.read(2)
            while True:
                marker = f.read(2)
                if len(marker) < 2:
                    break
                if marker[0] != 0xFF:
                    break
                if marker[1] in (0xC0, 0xC1, 0xC2):
                    f.read(3)
                    h = struct.unpack(">H", f.read(2))[0]
                    w = struct.unpack(">H", f.read(2))[0]
                    return w, h
                length_bytes = f.read(2)
                if len(length_bytes) < 2:
                    break
                length = struct.unpack(">H", length_bytes)[0]
                if length < 2:
                    break
                f.read(length - 2)

    if head[:2] == b"BM":
        w = struct.unpack("<I", head[18:22])[0]
        h = abs(struct.unpack("<i", head[22:26])[0])
        return w, h

    raise ValueError(f"Unsupported image format: {path}")


def main():
    parser = argparse.ArgumentParser(description="Convert VisDrone MOT to YOLO format")
    parser.add_argument("--input-dir", type=str, required=True, help="VisDrone MOT base directory (contains sequences/ and annotations/)")
    parser.add_argument("--output-dir", type=str, default="data/vehicle_dataset", help="Output YOLO dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of sequences to use for validation")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of creating symlinks")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    sequences_dir = input_dir / "sequences"
    annotations_dir = input_dir / "annotations"
    output_dir = Path(args.output_dir)

    if not sequences_dir.exists():
        raise FileNotFoundError(f"Sequences directory not found: {sequences_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    sequence_names = sorted(
        p.name for p in sequences_dir.iterdir() if p.is_dir()
    )

    if not sequence_names:
        raise FileNotFoundError(f"No sequence folders found in {sequences_dir}")

    n_val = max(1, int(len(sequence_names) * args.val_ratio))
    val_sequences = set(sequence_names[:n_val])
    train_sequences = set(sequence_names[n_val:])

    print(f"Total sequences: {len(sequence_names)}")
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Val sequences:   {len(val_sequences)}")

    for split, seq_set in [("train", train_sequences), ("val", val_sequences)]:
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = {"train": {"images": 0, "labels": 0}, "val": {"images": 0, "labels": 0}}
    skipped_no_ann = 0
    skipped_ignored = 0
    skipped_small = 0

    for seq_name in sequence_names:
        split = "val" if seq_name in val_sequences else "train"
        seq_img_dir = sequences_dir / seq_name
        ann_file = annotations_dir / f"{seq_name}.txt"

        if not ann_file.exists():
            print(f"  WARNING: No annotation file for {seq_name}, skipping")
            continue

        print(f"  Processing {seq_name} -> {split}")

        image_files = sorted(
            p for p in seq_img_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )

        if not image_files:
            continue

        img_size_cache: dict[str, tuple[int, int]] = {}
        first_size = _get_image_size(image_files[0])
        for img_path in image_files:
            img_size_cache[img_path.stem] = first_size

        frame_anns: dict[int, list[tuple[int, int, int, int, int]]] = {}
        with open(ann_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 10:
                    continue
                frame_id = int(parts[0])
                bbox_left = int(parts[2])
                bbox_top = int(parts[3])
                bbox_width = int(parts[4])
                bbox_height = int(parts[5])
                obj_category = int(parts[7])

                if obj_category in VISDRONE_IGNORE:
                    skipped_ignored += 1
                    continue

                yolo_class = VISEDRONE_TO_YOLO.get(obj_category)
                if yolo_class is None:
                    skipped_ignored += 1
                    continue

                if bbox_width < 2 or bbox_height < 2:
                    skipped_small += 1
                    continue

                if frame_id not in frame_anns:
                    frame_anns[frame_id] = []
                frame_anns[frame_id].append((yolo_class, bbox_left, bbox_top, bbox_width, bbox_height))

        img_dir_out = output_dir / "images" / split
        lbl_dir_out = output_dir / "labels" / split

        for img_path in image_files:
            frame_id = int(img_path.stem)
            dest_name = f"{seq_name}_{img_path.name}"

            if args.copy_images:
                shutil.copy2(img_path, img_dir_out / dest_name)
            else:
                try:
                    (img_dir_out / dest_name).symlink_to(img_path.resolve())
                except OSError:
                    shutil.copy2(img_path, img_dir_out / dest_name)

            anns = frame_anns.get(frame_id, [])
            lbl_path = lbl_dir_out / f"{seq_name}_{img_path.stem}.txt"

            with open(lbl_path, "w") as f:
                img_w, img_h = img_size_cache[img_path.stem]
                for yolo_class, bx, by, bw, bh in anns:
                    x_center = (bx + bw / 2) / img_w
                    y_center = (by + bh / 2) / img_h
                    norm_w = bw / img_w
                    norm_h = bh / img_h
                    f.write(f"{yolo_class} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

            stats[split]["images"] += 1
            stats[split]["labels"] += len(anns)

            if frame_id not in frame_anns:
                skipped_no_ann += 1

    print()
    print(f"Conversion complete. Output: {output_dir}")
    print(f"  Train: {stats['train']['images']} images, {stats['train']['labels']} labels")
    print(f"  Val:   {stats['val']['images']} images, {stats['val']['labels']} labels")
    print(f"  Skipped (no annotations): {skipped_no_ann}")
    print(f"  Skipped (ignored categories): {skipped_ignored}")
    print(f"  Skipped (too small): {skipped_small}")
    print()
    print("Classes mapped (VisDrone 0-indexed -> YOLO):")
    print("  3: car               -> 0: compact_car")
    print("  4: van               -> 2: van")
    print("  5: truck             -> 3: truck")
    print("  6: tricycle          -> 5: rickshaw")
    print("  7: awning-tricycle   -> 5: rickshaw")
    print("  8: bus               -> 4: bus")
    print("  9: motor             -> 6: motorcycle")
    print()
    print("Ignored VisDrone classes: 0: pedestrian, 1: people, 2: bicycle")
    print("NOTE: YOLO class 1 (suv) has no VisDrone equivalent.")


if __name__ == "__main__":
    main()

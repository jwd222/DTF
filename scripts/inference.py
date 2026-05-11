"""Single-stream inference with YOLOv26 + BoTSORT tracking and visualization.

Supports three source types:
  - video:  a single video file (mp4, avi, etc.)
  - images: a directory of images
  - mot:    a VisDrone MOT dataset directory (contains sequences/) — runs
            inference on each sequence and saves per-sequence output
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
    (255, 128, 0),
    (0, 128, 255),
    (128, 255, 0),
]

VEHICLE_CLASSES = {
    0: "car",
    1: "van",
    2: "truck",
    3: "rickshaw",
    4: "bus",
    5: "motorcycle",
}


def _draw_detections(annotated, results):
    if not results or len(results) == 0:
        return
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)

    track_ids = None
    if boxes.id is not None:
        track_ids = boxes.id.cpu().numpy().astype(int)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = map(int, xyxy[i])
        conf = confs[i]
        cls_id = cls_ids[i]
        label_name = VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}")

        color = COLORS[cls_id % len(COLORS)]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        tid_str = ""
        if track_ids is not None:
            tid = track_ids[i]
            tid_str = f" #{tid}"

        text = f"{label_name}{tid_str} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(
            annotated,
            text,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _run_on_images(image_paths, model, args, writer=None, save_dir=None, fps_target=30.0):
    frame_count = 0
    total_time = 0.0
    local_writer = writer

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        start = time.perf_counter()

        if args.track:
            results = model.track(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                tracker=args.tracker,
                persist=True,
                verbose=False,
            )
        else:
            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )

        elapsed = time.perf_counter() - start
        total_time += elapsed
        frame_count += 1

        annotated = frame.copy()
        _draw_detections(annotated, results)

        fps_text = f"FPS: {1.0 / max(elapsed, 1e-6):.1f} | Frame: {frame_count}/{len(image_paths)}"
        cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if local_writer:
            local_writer.write(annotated)

        if save_dir:
            cv2.imwrite(str(save_dir / img_path.name), annotated)

        if args.show:
            cv2.imshow("YOLOv26 Detection", annotated)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

    if args.show:
        cv2.destroyAllWindows()

    avg_fps = frame_count / max(total_time, 1e-6)
    return frame_count, total_time, avg_fps


def _run_on_video(source, model, args):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    save_dir = None
    if args.save_video:
        Path(args.save_video).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (w, h))
    if args.save_images:
        save_dir = Path(args.save_images)
        save_dir.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    total_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()

        if args.track:
            results = model.track(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                tracker=args.tracker,
                persist=True,
                verbose=False,
            )
        else:
            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )

        elapsed = time.perf_counter() - start
        total_time += elapsed
        frame_count += 1

        annotated = frame.copy()
        _draw_detections(annotated, results)

        fps_text = f"FPS: {1.0 / max(elapsed, 1e-6):.1f} | Frame: {frame_count}"
        cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if writer:
            writer.write(annotated)

        if save_dir:
            cv2.imwrite(str(save_dir / f"{frame_count:06d}.jpg"), annotated)

        if args.show:
            cv2.imshow("YOLOv26 Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    avg_fps = frame_count / max(total_time, 1e-6)
    print(f"\nProcessed {frame_count} frames in {total_time:.2f}s (avg {avg_fps:.1f} FPS)")
    if args.save_video:
        print(f"Output video saved to: {args.save_video}")
    if args.save_images:
        print(f"Annotated images saved to: {save_dir}")


def _collect_image_files(directory: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def _run_on_mot(source_dir, model, args):
    source_path = Path(source_dir)
    sequences_dir = source_path / "sequences"

    if sequences_dir.exists():
        seq_dirs = sorted(
            p for p in sequences_dir.iterdir()
            if p.is_dir()
        )
    else:
        seq_dirs = [source_path]

    if not seq_dirs:
        raise FileNotFoundError(f"No sequence directories found in: {source_dir}")

    print(f"Found {len(seq_dirs)} sequence(s)")

    output_base = Path(args.save_images) if args.save_images else Path(args.save_video).parent if args.save_video else Path("output/inference_mot")
    output_base.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_time = 0.0

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        print(f"\n  Processing sequence: {seq_name}")

        image_paths = _collect_image_files(seq_dir)
        if not image_paths:
            print(f"    No images found in {seq_dir}, skipping")
            continue

        print(f"    {len(image_paths)} images")

        writer = None
        save_dir = None

        if args.save_video:
            seq_video_path = output_base / f"{seq_name}.mp4"
            first = cv2.imread(str(image_paths[0]))
            if first is not None:
                h, w = first.shape[:2]
                fps = args.mot_fps
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(seq_video_path), fourcc, fps, (w, h))

        if args.save_images:
            save_dir = output_base / seq_name
            save_dir.mkdir(parents=True, exist_ok=True)

        fc, tt, avg_fps = _run_on_images(image_paths, model, args, writer=writer, save_dir=save_dir)
        total_frames += fc
        total_time += tt

        if writer:
            writer.release()
            print(f"    Video saved: {seq_video_path}")
        if save_dir:
            print(f"    Images saved: {save_dir}")
        print(f"    {fc} frames, avg {avg_fps:.1f} FPS")

    print(f"\nAll sequences done: {total_frames} frames in {total_time:.2f}s (avg {total_frames / max(total_time, 1e-6):.1f} FPS)")
    print(f"Output directory: {output_base}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 inference with tracking visualization")
    parser.add_argument("--source", type=str, required=True, help="Video file, image directory, or MOT dataset directory")
    parser.add_argument("--source-type", type=str, choices=["auto", "video", "images", "mot"], default="auto",
                        help="Source type. 'auto' detects from path. Use 'mot' for VisDrone MOT datasets.")
    parser.add_argument("--weights", type=str, default="yolo26s.pt", help="YOLOv26 weights")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--save-video", type=str, default=None, help="Save annotated output as video (.mp4)")
    parser.add_argument("--save-images", type=str, default=None, help="Save annotated output as images to directory")
    parser.add_argument("--show", action="store_true", help="Display frames in window (requires GUI)")
    parser.add_argument("--device", type=str, default="0", help="Device (0, cpu, etc.)")
    parser.add_argument("--tracker", type=str, default="botsort.yaml", help="Tracker config")
    parser.add_argument("--track", action="store_true", help="Enable tracking")
    parser.add_argument("--mot-fps", type=float, default=30.0, help="FPS for MOT sequence output videos")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)

    source_type = args.source_type
    if source_type == "auto":
        source_path = Path(args.source)
        if source_path.is_dir():
            if (source_path / "sequences").exists():
                source_type = "mot"
            else:
                source_type = "images"
        else:
            source_type = "video"

    if source_type == "mot":
        _run_on_mot(args.source, model, args)
    elif source_type == "images":
        source_path = Path(args.source)
        image_paths = _collect_image_files(source_path)
        if not image_paths:
            raise FileNotFoundError(f"No images found in: {source_path}")
        print(f"Found {len(image_paths)} images in {source_path}")

        writer = None
        save_dir = None
        if args.save_video:
            Path(args.save_video).parent.mkdir(parents=True, exist_ok=True)
            first = cv2.imread(str(image_paths[0]))
            if first is not None:
                h, w = first.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save_video, fourcc, 30.0, (w, h))
        if args.save_images:
            save_dir = Path(args.save_images)
            save_dir.mkdir(parents=True, exist_ok=True)

        fc, tt, avg_fps = _run_on_images(image_paths, model, args, writer=writer, save_dir=save_dir)
        print(f"\nProcessed {fc}/{len(image_paths)} images in {tt:.2f}s (avg {avg_fps:.1f} FPS)")
        if writer:
            writer.release()
            print(f"Output video saved to: {args.save_video}")
        if save_dir:
            print(f"Annotated images saved to: {save_dir}")
    else:
        _run_on_video(args.source, model, args)


if __name__ == "__main__":
    main()

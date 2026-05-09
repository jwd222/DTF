"""Single-stream inference with YOLOv26 + BoTSORT tracking and visualization."""
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


def _run_on_images(image_paths, model, args):
    frame_count = 0
    total_time = 0.0
    writer = None
    save_dir = None

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        first = cv2.imread(str(image_paths[0]))
        if first is None:
            raise IOError(f"Cannot read image: {image_paths[0]}")
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, 30.0, (w, h))

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

        if writer:
            writer.write(annotated)

        if save_dir:
            cv2.imwrite(str(save_dir / img_path.name), annotated)

        if args.show:
            cv2.imshow("YOLOv26 Detection", annotated)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    avg_fps = frame_count / max(total_time, 1e-6)
    print(f"\nProcessed {frame_count}/{len(image_paths)} images in {total_time:.2f}s (avg {avg_fps:.1f} FPS)")
    if args.output:
        print(f"Output video saved to: {args.output}")
    if save_dir:
        print(f"Annotated images saved to: {save_dir}")


def _run_on_video(source, model, args):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    frame_count = 0
    total_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()

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

        elapsed = time.perf_counter() - start
        total_time += elapsed
        frame_count += 1

        annotated = frame.copy()
        _draw_detections(annotated, results)

        fps_text = f"FPS: {1.0 / max(elapsed, 1e-6):.1f} | Frame: {frame_count}"
        cv2.putText(annotated, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if writer:
            writer.write(annotated)

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
    if args.output:
        print(f"Output saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 inference with tracking visualization")
    parser.add_argument("--source", type=str, required=True, help="Video file, camera index, or image directory")
    parser.add_argument("--weights", type=str, default="yolo26s.pt", help="YOLOv26 weights")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--output", type=str, default=None, help="Output video path or directory (for image sources)")
    parser.add_argument("--show", action="store_true", help="Display frames in window (requires GUI)")
    parser.add_argument("--save-dir", type=str, default=None, help="Save annotated images to this directory")
    parser.add_argument("--device", type=str, default="0", help="Device (0, cpu, etc.)")
    parser.add_argument("--tracker", type=str, default="botsort.yaml", help="Tracker config")
    parser.add_argument("--track", action="store_true", help="Enable tracking (only for video)")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)

    source_path = Path(args.source)
    if source_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_paths = sorted(
            p for p in source_path.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in: {source_path}")
        print(f"Found {len(image_paths)} images in {source_path}")
        _run_on_images(image_paths, model, args)
    else:
        _run_on_video(args.source, model, args)


if __name__ == "__main__":
    main()

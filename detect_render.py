"""Render a YOLO-only and a simple IoU-tracked video for experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def draw(frame, detections, color, show_ids):
    for x1, y1, x2, y2, score, track_id in detections:
        box = tuple(map(int, (x1, y1, x2, y2)))
        cv2.rectangle(frame, box[:2], box[2:], color, max(2, frame.shape[1] // 1000))
        text = f"ID {track_id} {score:.2f}" if show_ids else f"{score:.2f}"
        cv2.putText(
            frame,
            text,
            (box[0], max(25, box[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )


def iou(first, second):
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_first = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    area_second = max(1, bx2 - bx1) * max(1, by2 - by1)
    return intersection / (area_first + area_second - intersection)


def run(source, output, model, step, conf, imgsz, device, nms_iou):
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    raw_writer = cv2.VideoWriter(
        str(output_path / "yolo_only.mp4"), codec, fps, (width, height)
    )
    tracked_writer = cv2.VideoWriter(
        str(output_path / "tracked.mp4"), codec, fps, (width, height)
    )
    network = YOLO(model)
    active = {}
    next_id = 1
    frame_number = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        current = []
        if frame_number % step == 0:
            result = network(
                frame,
                classes=[0],
                conf=conf,
                iou=nms_iou,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )[0]
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                current.append((x1, y1, x2, y2, float(box.conf[0]), None))
            used = set()
            updated = {}
            for index, detection in enumerate(current):
                box = detection[:4]
                best_id, best_iou = None, 0.0
                for track_id, (old_box, last_frame) in active.items():
                    if track_id in used or frame_number - last_frame > step * 4:
                        continue
                    value = iou(box, old_box)
                    if value > best_iou:
                        best_iou, best_id = value, track_id
                if best_id is None:
                    best_id, next_id = next_id, next_id + 1
                used.add(best_id)
                updated[best_id] = (box, frame_number)
                current[index] = (*detection[:5], best_id)
            active = {
                **{
                    key: value
                    for key, value in active.items()
                    if frame_number - value[1] <= step * 4
                },
                **updated,
            }
        raw_frame, tracked_frame = frame.copy(), frame.copy()
        draw(raw_frame, current, (0, 255, 255), False)
        draw(tracked_frame, current, (0, 220, 80), True)
        raw_writer.write(raw_frame)
        tracked_writer.write(tracked_frame)
        frame_number += 1
        if frame_number % 20 == 0:
            print(f"Processed: {frame_number}/{total_frames}", flush=True)

    capture.release()
    raw_writer.release()
    tracked_writer.release()
    print(f"Processed {total_frames} frames -> {output_path}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="videos_proms/camera_new_4 - Trim.mp4")
    parser.add_argument("--output", default="videos_proms_results")
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nms-iou", type=float, default=0.4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.source,
        args.output,
        args.model,
        args.step,
        args.conf,
        args.imgsz,
        args.device,
        args.nms_iou,
    )

"""Render tracked bounding boxes on source videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def draw_label(frame, box, global_id, color=(0, 220, 80)):
    x1, y1, x2, y2 = map(int, box)
    thickness = max(2, frame.shape[1] // 1000)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = f"ID {global_id}"
    (text_width, text_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    )
    top = max(0, y1 - text_height - 12)
    cv2.rectangle(frame, (x1, top), (x1 + text_width + 10, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 5, max(text_height + 2, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def render(videos: str, report: str, output: str, hold: int = 0) -> None:
    if hold < 0:
        raise ValueError("hold must be >= 0")
    data = json.loads(Path(report).read_text(encoding="utf-8"))
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(videos).glob("*.mp4"))

    for camera, source in enumerate(files, 1):
        capture = cv2.VideoCapture(str(source))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        destination = output_path / f"{source.stem}_tracked.mp4"
        writer = cv2.VideoWriter(str(destination), codec, fps, (width, height))

        boxes = sorted(
            (
                item["frame"],
                item["box"],
                track["global_id"],
            )
            for track in data["tracks"]
            for item in track.get("boxes", [])
            if item["camera"] == camera
        )
        box_index = 0
        active = []
        frame_number = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            while box_index < len(boxes) and boxes[box_index][0] <= frame_number:
                active.append(boxes[box_index])
                box_index += 1
            kept = []
            for seen_frame, box, global_id in active:
                if frame_number - seen_frame > hold:
                    continue
                kept.append((seen_frame, box, global_id))
                draw_label(frame, box, global_id)
            active = kept
            writer.write(frame)
            frame_number += 1

        capture.release()
        writer.release()
        print(destination, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", default="videos")
    parser.add_argument("--report", default="outputs/report.json")
    parser.add_argument("--output", default="rendered")
    parser.add_argument("--hold", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render(args.videos, args.report, args.output, args.hold)

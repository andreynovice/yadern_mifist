"""Detect and track people in synchronized camera videos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


class ReID:
    """OSNet feature extractor for local and cross-camera association."""

    def __init__(self, enabled: bool, device: str = "cpu") -> None:
        self.extractor = None
        self.model_name = None
        if not enabled:
            return
        try:
            from torchreid.reid.utils import FeatureExtractor

            self.model_name = "osnet_x1_0"
            self.extractor = FeatureExtractor(
                model_name=self.model_name,
                device=device,
            )
            print(f"OSNet Re-ID enabled ({device})", flush=True)
        except Exception as exc:
            raise RuntimeError(
                "OSNet could not be initialized. Install requirements.txt."
            ) from exc

    def vector(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        if self.extractor is None:
            return None
        x1, y1, x2, y2 = box
        crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        if crop.size == 0:
            return None
        value = self.extractor([crop])[0].detach().cpu().numpy().astype(np.float32)
        return value / (np.linalg.norm(value) + 1e-8)


def appearance(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Return an HSV histogram for the no-ReID baseline."""
    x1, y1, x2, y2 = box
    crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
    if crop.size == 0:
        return np.zeros(48, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [8, 6], [0, 180, 0, 256])
    return cv2.normalize(histogram, histogram).flatten()


def similarity(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        cv2.compareHist(
            first.astype(np.float32),
            second.astype(np.float32),
            cv2.HISTCMP_CORREL,
        )
    )


class Tracker:
    """One-camera Deep SORT tracker with externally supplied embeddings."""

    def __init__(self, max_age: int = 50, n_init: int = 3) -> None:
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=0.35,
            nn_budget=100,
            embedder=None,
        )

    def update(self, detections: list[dict]) -> list[dict]:
        raw_detections = [
            (
                [
                    item["box"][0],
                    item["box"][1],
                    item["box"][2] - item["box"][0],
                    item["box"][3] - item["box"][1],
                ],
                item["score"],
                "person",
            )
            for item in detections
        ]
        embeddings = [
            item["reid"] if item["reid"] is not None else item["appearance"]
            for item in detections
        ]
        tracks = self.tracker.update_tracks(raw_detections, embeds=embeddings)
        confirmed = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            box = tuple(map(int, track.to_ltrb(orig=True)))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            feature = np.asarray(track.get_feature(), dtype=np.float32)
            feature /= np.linalg.norm(feature) + 1e-8
            confirmed.append(
                {
                    "box": box,
                    "local_id": str(track.track_id),
                    "score": track.det_conf or 0.0,
                    "reid": feature,
                }
            )
        return confirmed


def make_detector(
    model: str | None,
    conf: float,
    imgsz: int,
) -> tuple[Callable, str]:
    if model:
        from ultralytics import YOLO

        network = YOLO(model)

        def detect(frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
            result = []
            for box in network(
                frame,
                verbose=False,
                classes=[0],
                conf=conf,
                imgsz=imgsz,
            )[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                result.append((x1, y1, x2, y2, float(box.conf[0])))
            return result

        return detect, "YOLO person detector"

    subtractors = {}

    def detect_fallback(
        frame: np.ndarray,
        camera: str,
    ) -> list[tuple[int, int, int, int, float]]:
        subtractor = subtractors.setdefault(
            camera,
            cv2.createBackgroundSubtractorMOG2(
                history=300,
                varThreshold=32,
                detectShadows=True,
            ),
        )
        mask = subtractor.apply(frame)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )
        mask = cv2.dilate(
            mask,
            np.ones((5, 5), dtype=np.uint8),
            iterations=2,
        )
        detections = []
        contours = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )[0]
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            large_enough = width * height > frame.shape[0] * frame.shape[1] * 0.002
            person_shape = height > 1.25 * width and height > 45
            if large_enough and person_shape:
                detections.append((x, y, x + width, y + height, 0.5))
        return detections

    return detect_fallback, "OpenCV MOG2 fallback"


def process_camera(
    path: Path,
    camera: int,
    detector: Callable,
    model: str | None,
    step: int,
    conf: float,
    imgsz: int,
    max_frames: int,
    reid: ReID,
) -> tuple[list[dict], float, int]:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker = Tracker(max_age=max(30, int(fps * 2)), n_init=3)
    segments = {}
    frame_number = 0

    while max_frames == 0 or frame_number < max_frames:
        ok, image = capture.read()
        if not ok:
            break
        if frame_number % step == 0:
            raw = detector(image) if model else detector(image, str(camera))
            detections = []
            for x1, y1, x2, y2, score in raw:
                box = (x1, y1, x2, y2)
                detections.append(
                    {
                        "box": box,
                        "score": score,
                        "appearance": appearance(image, box),
                        "reid": reid.vector(image, box),
                    }
                )
            for item in tracker.update(detections):
                key = item["local_id"]
                segment = segments.setdefault(
                    key,
                    {
                        "camera": camera,
                        "local_id": key,
                        "start_time": frame_number / fps,
                        "end_time": frame_number / fps,
                        "frames": 0,
                        "reids": [],
                        "boxes": [],
                    },
                )
                segment["end_time"] = frame_number / fps
                segment["frames"] += 1
                segment["reids"].append(item["reid"])
                if len(segment["boxes"]) < 300:
                    segment["boxes"].append(
                        {
                            "camera": camera,
                            "frame": frame_number,
                            "box": list(item["box"]),
                        }
                    )
        frame_number += 1

    capture.release()
    valid_segments = []
    for segment in segments.values():
        segment["appearance"] = np.zeros(48, dtype=np.float32)
        segment["reid"] = np.mean(segment.pop("reids"), axis=0)
        segment["reid"] /= np.linalg.norm(segment["reid"]) + 1e-8
        if segment["frames"] >= 3:
            valid_segments.append(segment)
    return valid_segments, fps, total_frames


def merge_segments(segments: list[dict], gap: float) -> list[dict]:
    parents = list(range(len(segments)))
    members = {index: [index] for index in range(len(segments))}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def compatible(first_root: int, second_root: int) -> bool:
        for first in members[first_root]:
            for second in members[second_root]:
                left, right = segments[first], segments[second]
                overlap = min(left["end_time"], right["end_time"]) >= max(
                    left["start_time"], right["start_time"]
                )
                if left["camera"] == right["camera"] and overlap:
                    return False
        return True

    order = sorted(
        range(len(segments)), key=lambda index: segments[index]["start_time"]
    )
    for position, first_index in enumerate(order):
        first = segments[first_index]
        for second_index in order[position + 1 :]:
            second = segments[second_index]
            if second["start_time"] - first["end_time"] > gap:
                break
            if first["camera"] == second["camera"]:
                continue
            overlap = min(first["end_time"], second["end_time"]) >= max(
                first["start_time"], second["start_time"]
            )
            near_in_time = (
                abs(first["start_time"] - second["end_time"]) <= gap
                or abs(second["start_time"] - first["end_time"]) <= gap
            )
            if not (overlap or near_in_time):
                continue
            if similarity(first["reid"], second["reid"]) < 0.58:
                continue
            first_root, second_root = find(first_index), find(second_index)
            if first_root != second_root and compatible(first_root, second_root):
                parents[second_root] = first_root
                members[first_root].extend(members[second_root])
                del members[second_root]

    groups = {}
    for index, segment in enumerate(segments):
        groups.setdefault(find(index), []).append(segment)
    tracks = []
    for global_id, items in enumerate(groups.values(), 1):
        boxes = [box for item in items for box in item["boxes"]]
        tracks.append(
            {
                "global_id": global_id,
                "start_time": round(min(item["start_time"] for item in items), 2),
                "end_time": round(max(item["end_time"] for item in items), 2),
                "cameras": sorted({item["camera"] for item in items}),
                "frames": sum(item["frames"] for item in items),
                "boxes": boxes[:500],
            }
        )
    return tracks


def analyze(
    videos: str,
    output: str,
    model: str | None = None,
    step: int = 5,
    conf: float = 0.15,
    imgsz: int = 1280,
    gap: float = 8,
    max_frames: int = 0,
    reid_enabled: bool = True,
    device: str = "cpu",
) -> None:
    files = sorted(Path(videos).glob("*.mp4"))
    if not files:
        raise FileNotFoundError(f"No MP4 files found in {videos}")
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    detector, detector_name = make_detector(model, conf, imgsz)
    reid_engine = ReID(bool(model) and reid_enabled, device)
    segments, camera_metadata = [], []

    for camera, path in enumerate(files, 1):
        print(f"camera {camera}/{len(files)}: {path.name}", flush=True)
        local, fps, total = process_camera(
            path, camera, detector, model, step, conf, imgsz, max_frames, reid_engine
        )
        segments.extend(local)
        camera_metadata.append(
            {
                "camera": camera,
                "file": path.name,
                "fps": fps,
                "frames": total,
                "local_segments": len(local),
            }
        )

    tracks = merge_segments(segments, gap)
    report = {
        "detector": detector_name,
        "reid": reid_engine.model_name,
        "reid_device": device if reid_engine.extractor else None,
        "sample_step": step,
        "association_gap_sec": gap,
        "cameras": camera_metadata,
        "total_tracks": len(tracks),
        "tracks": tracks,
    }
    (output_path / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (output_path / "tracks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["track_id", "start_sec", "end_sec", "cameras", "sampled_frames"]
        )
        for track in tracks:
            writer.writerow(
                [
                    track["global_id"],
                    track["start_time"],
                    track["end_time"],
                    ",".join(map(str, track["cameras"])),
                    track["frames"],
                ]
            )
    print(
        json.dumps(
            {
                "detector": detector_name,
                "local_segments": len(segments),
                "total_tracks": len(tracks),
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", default="videos")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--model")
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--gap", type=float, default=8)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-reid", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(
        args.videos,
        args.output,
        args.model,
        args.step,
        args.conf,
        args.imgsz,
        args.gap,
        args.max_frames,
        not args.no_reid,
        args.device,
    )

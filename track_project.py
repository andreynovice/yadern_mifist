"""Unified OSNet tracking, graph association, and floor projection."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from analyze import ReID, appearance

DEFAULT_GRAPH = {
    "camera_edges": {
        "2": ["1"],
        "1": ["3"],
        "3": ["5"],
        "5": ["3", "4"],
        "4": [],
    },
    "visibility_states": {
        "S2": ["2"],
        "S21": ["2", "1"],
        "S13": ["1", "3"],
        "S35": ["3", "5"],
        "S354": ["3", "5", "4"],
        "S3": ["3"],
        "S4": ["4"],
    },
    "state_transitions": {
        "S2": ["S21"],
        "S21": ["S13"],
        "S13": ["S21", "S35"],
        "S35": ["S354"],
        "S354": ["S3", "S4"],
        "S3": [],
        "S4": [],
    },
}

DEFAULT_CALIBRATION = {
    "coordinate_system": {
        "units": "metres",
        "origin": "first table corner",
        "x_axis": "table width",
        "y_axis": "table depth",
        "z_axis": "up from floor",
    },
    "common": {
        "focal_length": 1400.0,
        "table_width": 2.74,
        "table_depth": 1.525,
        "table_height": 0.76,
        "table_points": [
            [689.0, 618.0],
            [1603.0, 499.0],
            [1888.0, 833.0],
            [751.0, 1081.0],
        ],
    },
    "cameras": {},
}


@dataclass
class CameraCalibration:
    mode: str
    intrinsics: np.ndarray | None = None
    rotation: np.ndarray | None = None
    translation: np.ndarray | None = None
    focal_length: float | None = None
    reprojection_error_px: float | None = None
    camera_world: list[float] | None = None


class LocalTracker:
    """Deep SORT with OSNet or HSV fallback features supplied by the caller."""

    def __init__(
        self,
        max_age: int,
        n_init: int,
        iou_distance: float,
        cosine_distance: float,
        nn_budget: int,
    ) -> None:
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=iou_distance,
            max_cosine_distance=cosine_distance,
            nn_budget=nn_budget,
            embedder=None,
        )

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        output = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            box = tuple(map(int, track.to_ltrb(orig=True)))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            feature = np.asarray(track.get_feature(), dtype=np.float32)
            feature /= np.linalg.norm(feature) + 1e-8
            output.append(
                {
                    "local_id": str(track.track_id),
                    "box": box,
                    "score": float(track.det_conf or 0.0),
                    "feature": feature,
                }
            )
        return suppress_duplicate_boxes(output)


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    first = first / (np.linalg.norm(first) + 1e-8)
    second = second / (np.linalg.norm(second) + 1e-8)
    return float(np.dot(first, second))


def box_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second
    intersection_x1 = max(first_x1, second_x1)
    intersection_y1 = max(first_y1, second_y1)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)
    intersection = max(0, intersection_x2 - intersection_x1) * max(
        0, intersection_y2 - intersection_y1
    )
    first_area = max(1, first_x2 - first_x1) * max(1, first_y2 - first_y1)
    second_area = max(1, second_x2 - second_x1) * max(
        1, second_y2 - second_y1
    )
    return intersection / (first_area + second_area - intersection)


def overlap_fraction(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second
    intersection = max(0, min(first_x2, second_x2) - max(first_x1, second_x1)) * max(
        0, min(first_y2, second_y2) - max(first_y1, second_y1)
    )
    first_area = max(1, first_x2 - first_x1) * max(1, first_y2 - first_y1)
    second_area = max(1, second_x2 - second_x1) * max(
        1, second_y2 - second_y1
    )
    return intersection / max(1, min(first_area, second_area))


def likely_duplicate(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_box = first["box"]
    second_box = second["box"]
    first_score = float(first.get("score", 0.0))
    second_score = float(second.get("score", 0.0))
    if box_iou(first_box, second_box) >= 0.50:
        return True
    if min(first_score, second_score) >= 0.55:
        return False
    if overlap_fraction(first_box, second_box) < 0.18:
        return False
    first_width = max(1, first_box[2] - first_box[0])
    second_width = max(1, second_box[2] - second_box[0])
    first_height = max(1, first_box[3] - first_box[1])
    second_height = max(1, second_box[3] - second_box[1])
    first_center = ((first_box[0] + first_box[2]) / 2, (first_box[1] + first_box[3]) / 2)
    second_center = (
        (second_box[0] + second_box[2]) / 2,
        (second_box[1] + second_box[3]) / 2,
    )
    return (
        abs(first_center[0] - second_center[0])
        <= 0.85 * max(first_width, second_width)
        and abs(first_center[1] - second_center[1])
        <= 0.65 * max(first_height, second_height)
    )


def suppress_duplicate_boxes(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for item in sorted(items, key=lambda value: value.get("score", 0.0), reverse=True):
        if any(likely_duplicate(item, other) for other in selected):
            continue
        selected.append(item)
    return selected


def camera_matrix(
    width: int,
    height: int,
    focal_length: float,
    cx: float | None,
    cy: float | None,
) -> np.ndarray:
    return np.array(
        [
            [focal_length, 0.0, width / 2.0 if cx is None else cx],
            [0.0, focal_length, height / 2.0 if cy is None else cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_focal_length(points: np.ndarray, width: int, height: int) -> float | None:
    """Estimate focal length from perpendicular table edges if possible."""
    homogeneous = np.column_stack((points, np.ones(4, dtype=np.float64)))

    def line(first: int, second: int) -> np.ndarray:
        value = np.cross(homogeneous[first], homogeneous[second])
        return value / (np.linalg.norm(value[:2]) + 1e-12)

    def vanishing(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
        value = np.cross(first, second)
        if abs(value[2]) < 1e-9:
            return None
        return value[:2] / value[2]

    width_vanishing = vanishing(line(0, 1), line(3, 2))
    depth_vanishing = vanishing(line(1, 2), line(0, 3))
    if width_vanishing is None or depth_vanishing is None:
        return None
    principal = np.array([width / 2.0, height / 2.0])
    focal_squared = -float(
        np.dot(width_vanishing - principal, depth_vanishing - principal)
    )
    if not np.isfinite(focal_squared) or focal_squared <= 0:
        return None
    return float(np.sqrt(focal_squared))


def make_calibration(
    settings: dict[str, Any] | None,
    width: int,
    height: int,
    focal_length: float,
) -> CameraCalibration:
    if not settings or not settings.get("table_points"):
        return CameraCalibration(mode="image_normalized")
    points = np.asarray(settings["table_points"], dtype=np.float64)
    if points.shape != (4, 2):
        raise ValueError("table_points must contain four [x, y] points")
    table_width = float(settings.get("table_width", 2.74))
    table_depth = float(settings.get("table_depth", 1.525))
    table_height = float(settings.get("table_height", 0.76))
    focal = settings.get("focal_length")
    if focal is None:
        focal = estimate_focal_length(points, width, height) or focal_length
    focal = float(focal)
    intrinsics = camera_matrix(
        width,
        height,
        focal,
        settings.get("cx"),
        settings.get("cy"),
    )
    distortion = np.asarray(
        settings.get("distortion", [0.0, 0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    world_points = np.array(
        [
            [0.0, table_depth, table_height],
            [table_width, table_depth, table_height],
            [table_width, 0.0, table_height],
            [0.0, 0.0, table_height],
        ],
        dtype=np.float64,
    )
    ok, rotation_vector, translation = cv2.solvePnP(
        world_points,
        points,
        intrinsics,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("Could not estimate camera pose from table points")
    projected, _ = cv2.projectPoints(
        world_points,
        rotation_vector,
        translation,
        intrinsics,
        distortion,
    )
    error = np.linalg.norm(projected.reshape(-1, 2) - points, axis=1)
    rotation, _ = cv2.Rodrigues(rotation_vector)
    camera_world = -rotation.T @ translation.reshape(3)
    if camera_world[2] <= table_height:
        raise ValueError(
            "Calibration places the camera at or below the table. "
            "Check table point order, focal length, and image coordinates."
        )
    return CameraCalibration(
        mode="metric",
        intrinsics=intrinsics,
        rotation=rotation,
        translation=translation,
        focal_length=focal,
        reprojection_error_px=float(np.mean(error)),
        camera_world=[float(value) for value in camera_world],
    )


def project_ray(
    pixel: tuple[float, float],
    plane_z: float,
    calibration: CameraCalibration,
) -> list[float] | None:
    """Intersect a pixel ray with a horizontal plane in metres."""
    if calibration.mode != "metric":
        return None
    homogeneous = np.array([pixel[0], pixel[1], 1.0], dtype=np.float64)
    ray_camera = np.linalg.inv(calibration.intrinsics) @ homogeneous
    ray_world = calibration.rotation.T @ ray_camera
    camera_world = -calibration.rotation.T @ calibration.translation.reshape(3)
    if abs(ray_world[2]) < 1e-9:
        return None
    scale = (plane_z - camera_world[2]) / ray_world[2]
    if scale <= 0:
        return None
    point = camera_world + scale * ray_world
    return [
        round(float(point[0]), 4),
        round(float(point[1]), 4),
        round(float(point[2]), 4),
    ]


def normalized_floor(
    pixel: tuple[float, float],
    width: int,
    height: int,
    map_width: float,
    map_height: float,
) -> list[float]:
    return [
        round(float(pixel[0] / max(width, 1) * map_width), 4),
        round(float((1.0 - pixel[1] / max(height, 1)) * map_height), 4),
        0.0,
    ]


def load_or_create(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        path.write_text(json.dumps(default, indent=2), encoding="utf-8")
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def camera_calibration_settings(
    calibration_data: dict[str, Any], camera_id: str
) -> dict[str, Any] | None:
    """Combine shared calibration values with optional per-camera overrides."""
    common = calibration_data.get("common", {})
    per_camera = calibration_data.get("cameras", {}).get(camera_id, {})
    if not common and not per_camera:
        return None
    settings = dict(common)
    settings.update(per_camera)
    return settings


def normalize_graph(graph: dict[str, Any]) -> dict[str, Any]:
    """Return a visibility-state graph, accepting the legacy camera graph too."""
    if "visibility_states" in graph and "state_transitions" in graph:
        states = {
            name: [str(camera) for camera in cameras]
            for name, cameras in graph["visibility_states"].items()
        }
        transitions = {
            name: [str(target) for target in targets]
            for name, targets in graph["state_transitions"].items()
        }
        camera_edges = graph.get("camera_edges", {})
        return {
            "camera_edges": {
                str(camera): [str(target) for target in targets]
                for camera, targets in camera_edges.items()
            },
            "visibility_states": states,
            "state_transitions": transitions,
        }

    # Keep old {"2": ["1"], ...} files readable while using the new
    # visibility-state semantics for all newly created runs.
    camera_edges = {
        str(camera): [str(target) for target in targets]
        for camera, targets in graph.items()
    }
    states = {f"S{camera}": [str(camera)] for camera in camera_edges}
    transitions = {
        f"S{camera}": [f"S{target}" for target in targets]
        for camera, targets in camera_edges.items()
    }
    return {
        "camera_edges": camera_edges,
        "visibility_states": states,
        "state_transitions": transitions,
    }


def graph_camera_edges(graph: dict[str, Any]) -> dict[str, list[str]]:
    """Return the directed camera-level edges from a normalized graph."""
    if "camera_edges" in graph:
        return graph["camera_edges"]
    return normalize_graph(graph)["camera_edges"]


def visibility_state_for_cameras(graph: dict[str, Any], cameras: set[str]) -> str:
    """Name the exact visibility state, or describe an unlisted subset."""
    states = graph.get("visibility_states", {})
    for name, members in states.items():
        if set(members) == cameras:
            return name
    return "custom[" + ",".join(sorted(cameras)) + "]"


def resolve_camera_ids(files: list[Path], explicit: str | None) -> list[str]:
    """Resolve stable camera IDs from names or an explicit CLI list."""
    if explicit:
        camera_ids = [value.strip() for value in explicit.split(",") if value.strip()]
        if len(camera_ids) != len(files):
            raise ValueError("--camera-ids must contain one ID per input MP4")
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("--camera-ids must contain unique camera IDs")
        return camera_ids
    parsed = []
    for path in files:
        match = re.search(r"camera(?:_new)?_(\d+)", path.stem.lower())
        parsed.append(match.group(1) if match else None)
    if all(value is not None for value in parsed):
        if len(set(parsed)) != len(files):
            raise ValueError(
                "Input filenames contain duplicate camera IDs. Use --camera-ids."
            )
        return [value for value in parsed if value is not None]
    return [str(index) for index in range(1, len(files) + 1)]


def parse_camera_offsets(value: str | None, camera_ids: list[str]) -> dict[str, float]:
    """Parse comma-separated camera=seconds synchronization offsets."""
    offsets = {camera: 0.0 for camera in camera_ids}
    if not value:
        return offsets
    for entry in value.split(","):
        parts = entry.split("=", maxsplit=1)
        if len(parts) != 2:
            raise ValueError("--camera-offsets must look like 1=0,2=1.25")
        camera, raw_offset = parts[0].strip(), parts[1].strip()
        if camera not in offsets:
            raise ValueError(f"Unknown camera ID in --camera-offsets: {camera}")
        try:
            offsets[camera] = float(raw_offset)
        except ValueError as exc:
            raise ValueError("Camera offsets must be numbers in seconds") from exc
    return offsets


def parse_devices(value: str | None, fallback: str) -> list[str]:
    """Parse one or more inference devices for independent camera workers."""
    if not value:
        return [fallback]
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("--devices must contain at least one device")
    return devices


def detect(
    model: Any, frame: np.ndarray, args: argparse.Namespace
) -> list[dict[str, Any]]:
    result = model(
        frame,
        classes=[0],
        conf=args.conf,
        iou=args.det_iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
    )[0]
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({"box": (x1, y1, x2, y2), "score": float(box.conf[0])})
    return suppress_duplicate_boxes(
        [
            item
            for item in detections
            if item["score"] >= args.min_person_conf
            and (item["box"][3] - item["box"][1])
            / max(item["box"][2] - item["box"][0], 1)
            >= args.min_person_aspect
        ]
    )


def process_camera(
    path: Path,
    camera: str,
    model: Any,
    reid: ReID,
    args: argparse.Namespace,
    time_offset: float = 0.0,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_interval = max(1, round(fps * 10))
    updates_per_second = fps / args.step
    max_age = args.track_max_age
    if args.track_max_age_sec is not None:
        max_age = max(1, round(args.track_max_age_sec * updates_per_second))
    tracker = LocalTracker(
        max_age=max_age,
        n_init=args.n_init,
        iou_distance=args.track_iou,
        cosine_distance=args.track_cosine,
        nn_budget=args.nn_budget,
    )
    segments: dict[str, dict[str, Any]] = {}
    embeddings = []
    frame_number = 0
    while args.max_frames == 0 or frame_number < args.max_frames:
        ok, image = capture.read()
        if not ok:
            break
        if frame_number % args.step == 0:
            detections = detect(model, image, args)
            for item in detections:
                item["appearance"] = appearance(image, item["box"])
                item["reid"] = reid.vector(image, item["box"])
            for tracked in tracker.update(detections):
                local_id = tracked["local_id"]
                feature = tracked["feature"]
                box = tracked["box"]
                observation = {
                    "camera": camera,
                    "frame": frame_number,
                    "time_sec": round(frame_number / fps, 4),
                    "sync_time_sec": round(frame_number / fps + time_offset, 4),
                    "local_id": local_id,
                    "box": list(box),
                    "score": round(tracked["score"], 5),
                }
                segment = segments.setdefault(
                    local_id,
                    {
                        "camera": camera,
                        "local_id": local_id,
                        "start_time": frame_number / fps,
                        "end_time": frame_number / fps,
                        "sync_start_time": frame_number / fps + time_offset,
                        "sync_end_time": frame_number / fps + time_offset,
                        "features": [],
                        "observations": [],
                    },
                )
                segment["end_time"] = frame_number / fps
                segment["sync_end_time"] = frame_number / fps + time_offset
                segment["features"].append(feature)
                segment["observations"].append(observation)
                embeddings.append(
                    {
                        "camera": camera,
                        "local_id": local_id,
                        "frame": frame_number,
                        "feature_type": "osnet" if feature.size > 100 else "hsv",
                        "embedding": [round(float(value), 8) for value in feature],
                    }
                )
        if frame_number and frame_number % progress_interval == 0:
            progress = (
                "unknown"
                if total_frames <= 0
                else f"{100.0 * frame_number / total_frames:.1f}%"
            )
            print(
                f"camera {camera}: frame {frame_number}/{total_frames} ({progress})",
                flush=True,
            )
        frame_number += 1
    capture.release()
    valid_segments = []
    for segment in segments.values():
        if len(segment["observations"]) < args.min_segment_frames:
            continue
        features = np.asarray(segment.pop("features"), dtype=np.float32)
        feature = np.mean(features, axis=0)
        feature /= np.linalg.norm(feature) + 1e-8
        segment["feature"] = feature
        valid_segments.append(segment)
    return {
        "camera": camera,
        "file": path.name,
        "path": str(path),
        "fps": fps,
        "time_offset_sec": time_offset,
        "width": width,
        "height": height,
        "segments": valid_segments,
        "embeddings": embeddings,
    }


def process_camera_worker(
    device: str,
    jobs: list[tuple[str, str, float]],
    model_path: str,
    args_values: dict[str, Any],
) -> dict[str, Any]:
    """Process a group of cameras in a separate process and GPU context."""
    from ultralytics import YOLO

    worker_args = argparse.Namespace(**args_values)
    worker_args.device = device
    model = YOLO(model_path)
    reid = ReID(not worker_args.no_reid, device)
    cameras = []
    for position, (camera_id, raw_path, time_offset) in enumerate(jobs, 1):
        path = Path(raw_path)
        print(
            f"[{device}] processing camera {camera_id} "
            f"({position}/{len(jobs)}): {path.name}",
            flush=True,
        )
        cameras.append(
            process_camera(
                path,
                camera_id,
                model,
                reid,
                worker_args,
                time_offset=time_offset,
            )
        )
    return {"cameras": cameras, "reid_model": reid.model_name}


def allowed_transition(
    graph: dict[str, list[str]], source: str, target: str, max_hops: int
) -> bool:
    """Allow a directed camera route when intermediate cameras were missed."""
    graph = graph_camera_edges(graph)
    frontier = {source}
    visited = {source}
    for _ in range(max_hops):
        frontier = {
            neighbor
            for node in frontier
            for neighbor in graph.get(node, [])
            if neighbor not in visited
        }
        if target in frontier:
            return True
        visited.update(frontier)
    return False


def state_transition_allowed(
    graph: dict[str, Any],
    source_states: set[str],
    target_states: set[str],
    max_hops: int,
) -> bool:
    """Check a path between visibility states in the state graph."""
    transitions = graph.get("state_transitions", {})
    frontier = set(source_states)
    if frontier & target_states:
        return True
    visited = set(frontier)
    for _ in range(max_hops):
        frontier = {
            target
            for state in frontier
            for target in transitions.get(state, [])
            if target not in visited
        }
        if frontier & target_states:
            return True
        visited.update(frontier)
    return False


def state_candidates_for_active_cameras(
    graph: dict[str, Any], active_cameras: set[str]
) -> tuple[set[str], list[tuple[str, str]]]:
    """Return exact states and legal state-boundary overlaps for a snapshot."""
    states = graph.get("visibility_states", {})
    exact = {
        name
        for name, members in states.items()
        if active_cameras.issubset(set(members))
    }
    if exact:
        return exact, []

    overlaps = []
    for source, targets in graph.get("state_transitions", {}).items():
        source_cameras = set(states.get(source, []))
        for target in targets:
            target_cameras = set(states.get(target, []))
            shared = source_cameras & target_cameras
            union = source_cameras | target_cameras
            if (
                shared
                and shared.issubset(active_cameras)
                and active_cameras.issubset(union)
            ):
                overlaps.append((source, target))
    return set(), overlaps


def state_path_compatible(
    graph: dict[str, Any],
    items: list[dict[str, Any]],
    max_hops: int,
    transition_overlap: float = 2.0,
) -> bool:
    """Validate simultaneous visibility and its order for one global track."""
    if not items:
        return True
    events = sorted(
        {float(item.get("sync_start_time", item["start_time"])) for item in items}
        | {float(item.get("sync_end_time", item["end_time"])) for item in items}
    )
    if len(events) == 1:
        intervals = [(events[0], events[0])]
    else:
        intervals = [
            (left, right) for left, right in zip(events, events[1:]) if right > left
        ]
    reachable: set[str] | None = None
    # One camera route can require one extra state edge for an overlap state,
    # e.g. S2 -> S21 -> S13 -> S35 -> S354 -> S4.
    state_hops = max_hops + 1
    for left, right in intervals:
        sample = left if left == right else (left + right) / 2.0
        active = {
            str(item["camera"])
            for item in items
            if item.get("sync_start_time", item["start_time"])
            <= sample
            <= item.get("sync_end_time", item["end_time"])
        }
        if not active:
            continue
        exact, overlaps = state_candidates_for_active_cameras(graph, active)
        if not exact and not overlaps:
            return False
        if overlaps and right - left > transition_overlap:
            return False
        if reachable is None:
            if exact:
                reachable = exact
            else:
                reachable = {state for transition in overlaps for state in transition}
            continue
        if exact:
            next_reachable = {
                target
                for target in exact
                if state_transition_allowed(graph, reachable, {target}, state_hops)
            }
        else:
            next_reachable = set()
            for source, target in overlaps:
                if state_transition_allowed(graph, reachable, {source}, state_hops):
                    next_reachable.update((source, target))
        if not next_reachable:
            return False
        reachable = next_reachable
    return reachable is not None


def interval_distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    """Return the time gap between two segment intervals, ignoring overlap."""
    first_start = first.get("sync_start_time", first["start_time"])
    first_end = first.get("sync_end_time", first["end_time"])
    second_start = second.get("sync_start_time", second["start_time"])
    second_end = second.get("sync_end_time", second["end_time"])
    return max(0.0, first_start - second_end, second_start - first_end)


def interval_overlap(first: dict[str, Any], second: dict[str, Any]) -> float:
    """Return the duration for which two segments are visible simultaneously."""
    first_start = first.get("sync_start_time", first["start_time"])
    first_end = first.get("sync_end_time", first["end_time"])
    second_start = second.get("sync_start_time", second["start_time"])
    second_end = second.get("sync_end_time", second["end_time"])
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def merge_segments(
    cameras: list[dict[str, Any]],
    graph: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    segments = [segment for camera in cameras for segment in camera["segments"]]
    parents = list(range(len(segments)))
    members = {index: [index] for index in range(len(segments))}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def compatible(first_root: int, second_root: int) -> bool:
        combined = members[first_root] + members[second_root]
        for position, first_index in enumerate(combined):
            first = segments[first_index]
            for second_index in combined[position + 1 :]:
                second = segments[second_index]
                if first["camera"] != second["camera"]:
                    continue
                first_start = first.get("sync_start_time", first["start_time"])
                first_end = first.get("sync_end_time", first["end_time"])
                second_start = second.get("sync_start_time", second["start_time"])
                second_end = second.get("sync_end_time", second["end_time"])
                if min(first_end, second_end) >= max(first_start, second_start):
                    return False
        return state_path_compatible(
            graph,
            [segments[index] for index in combined],
            args.max_graph_hops,
            args.transition_overlap,
        )

    local_candidates = []
    cross_camera_candidates = []
    for first_index, original_first in enumerate(segments):
        for second_index in range(first_index + 1, len(segments)):
            first = original_first
            second = segments[second_index]
            candidate_first_index = first_index
            candidate_second_index = second_index
            first_start = first.get("sync_start_time", first["start_time"])
            second_start = second.get("sync_start_time", second["start_time"])
            if first["camera"] == second["camera"]:
                # A local camera track can only continue forward in time.
                if first_start > second_start:
                    first, second = second, first
                    candidate_first_index, candidate_second_index = (
                        second_index,
                        first_index,
                    )
                    first_start, second_start = second_start, first_start
                if interval_distance(first, second) > args.global_gap:
                    continue
                threshold = args.local_reid_threshold
                candidates = local_candidates
            else:
                # Different cameras may observe the person at the same time.
                # The state graph, rather than an end-to-start rule, decides
                # whether this pair can belong to one person.
                if interval_distance(first, second) > args.global_gap:
                    continue
                if not state_path_compatible(
                    graph,
                    [first, second],
                    args.max_graph_hops,
                    args.transition_overlap,
                ):
                    continue
                threshold = args.reid_threshold
                candidates = cross_camera_candidates
            score = cosine(first["feature"], second["feature"])
            if score < threshold:
                continue
            time_gap = interval_distance(first, second)
            overlap = interval_overlap(first, second)
            combined_score = score - args.time_penalty * time_gap / max(
                args.global_gap, 1e-6
            )
            candidates.append(
                {
                    "first_index": candidate_first_index,
                    "second_index": candidate_second_index,
                    "score": score,
                    "combined_score": combined_score,
                    "time_gap": time_gap,
                    "overlap": overlap,
                }
            )

    def merge_candidate(candidate: dict[str, Any]) -> bool:
        first_root = find(candidate["first_index"])
        second_root = find(candidate["second_index"])
        if first_root == second_root or not compatible(first_root, second_root):
            return False
        parents[second_root] = first_root
        members[first_root].extend(members[second_root])
        del members[second_root]
        return True

    # Close same-camera fragments first. Sorting by temporal distance prevents
    # a high-similarity, long-range false match from swallowing a nearby track.
    for candidate in sorted(
        local_candidates,
        key=lambda item: (item["time_gap"], -item["score"]),
    ):
        merge_candidate(candidate)

    # Cross-camera associations are one-to-one for each camera pair. This
    # prevents one local segment from absorbing several visually similar
    # segments before the correct overlapping observation is considered.
    from scipy.optimize import linear_sum_assignment

    camera_pair_rank: dict[tuple[str, str], int] = {}
    for source, targets in graph_camera_edges(graph).items():
        for target in targets:
            pair = tuple(sorted((str(source), str(target))))
            if pair not in camera_pair_rank:
                camera_pair_rank[pair] = len(camera_pair_rank)

    candidates_by_camera_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in cross_camera_candidates:
        first = segments[candidate["first_index"]]
        second = segments[candidate["second_index"]]
        camera_pair = tuple(sorted((first["camera"], second["camera"])))
        candidates_by_camera_pair.setdefault(camera_pair, []).append(candidate)

    ordered_camera_pairs = sorted(
        candidates_by_camera_pair,
        key=lambda pair: (camera_pair_rank.get(pair, len(camera_pair_rank)), pair),
    )
    for camera_pair in ordered_camera_pairs:
        pair_candidates = candidates_by_camera_pair[camera_pair]
        left_camera, right_camera = camera_pair
        left_segments = sorted(
            {
                (
                    candidate["first_index"]
                    if segments[candidate["first_index"]]["camera"] == left_camera
                    else candidate["second_index"]
                )
                for candidate in pair_candidates
            }
        )
        right_segments = sorted(
            {
                (
                    candidate["second_index"]
                    if segments[candidate["second_index"]]["camera"] == right_camera
                    else candidate["first_index"]
                )
                for candidate in pair_candidates
            }
        )
        candidate_lookup = {}
        for candidate in pair_candidates:
            first_index = candidate["first_index"]
            second_index = candidate["second_index"]
            if segments[first_index]["camera"] != left_camera:
                first_index, second_index = second_index, first_index
            candidate_lookup[(first_index, second_index)] = candidate

        unavailable = set()
        used_left = set()
        used_right = set()
        while True:
            available_left = [
                index for index in left_segments if index not in used_left
            ]
            available_right = [
                index for index in right_segments if index not in used_right
            ]
            if not available_left or not available_right:
                break

            left_positions = {
                segment_index: position
                for position, segment_index in enumerate(available_left)
            }
            right_positions = {
                segment_index: position
                for position, segment_index in enumerate(available_right)
            }
            costs = np.zeros(
                (len(available_left), len(available_right) + len(available_left)),
                dtype=np.float64,
            )
            costs[:, : len(available_right)] = 1e6
            for (left_index, right_index), candidate in candidate_lookup.items():
                if (left_index, right_index) in unavailable:
                    continue
                if (
                    left_index not in left_positions
                    or right_index not in right_positions
                ):
                    continue
                costs[left_positions[left_index], right_positions[right_index]] = (
                    -candidate["combined_score"]
                )

            row_indices, column_indices = linear_sum_assignment(costs)
            assignments = []
            for row, column in zip(row_indices, column_indices):
                if column >= len(available_right):
                    continue
                pair = (available_left[row], available_right[column])
                candidate = candidate_lookup.get(pair)
                if candidate is not None and pair not in unavailable:
                    assignments.append(candidate)
            if not assignments:
                break

            assignments.sort(key=lambda item: item["combined_score"], reverse=True)
            rejected = False
            for candidate in assignments:
                first_index = candidate["first_index"]
                second_index = candidate["second_index"]
                if segments[first_index]["camera"] != left_camera:
                    first_index, second_index = second_index, first_index
                if merge_candidate(candidate):
                    used_left.add(first_index)
                    used_right.add(second_index)
                else:
                    unavailable.add((first_index, second_index))
                    rejected = True
            if not rejected:
                break

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, segment in enumerate(segments):
        groups.setdefault(find(index), []).append(segment)
    tracks = []
    for global_id, items in enumerate(groups.values(), 1):
        observations = [
            observation for item in items for observation in item["observations"]
        ]
        observations.sort(
            key=lambda item: (
                item.get("sync_time_sec", item["time_sec"]),
                item["camera"],
            )
        )
        last_seen = max(item.get("sync_end_time", item["end_time"]) for item in items)
        tracks.append(
            {
                "global_id": global_id,
                "start_time": round(
                    min(
                        item.get("sync_start_time", item["start_time"])
                        for item in items
                    ),
                    4,
                ),
                "last_seen": round(last_seen, 4),
                "memory_until": round(last_seen + args.global_gap, 4),
                "cameras": sorted({item["camera"] for item in items}),
                "segments": [
                    {
                        "camera": item["camera"],
                        "local_id": item["local_id"],
                        "start_time": round(item["start_time"], 4),
                        "end_time": round(item["end_time"], 4),
                        "sync_start_time": round(
                            item.get("sync_start_time", item["start_time"]), 4
                        ),
                        "sync_end_time": round(
                            item.get("sync_end_time", item["end_time"]), 4
                        ),
                        "frames": len(item["observations"]),
                    }
                    for item in sorted(
                        items,
                        key=lambda value: value.get(
                            "sync_start_time", value["start_time"]
                        ),
                    )
                ],
                "observations": observations,
            }
        )
    return tracks


def assign_projections(
    tracks: list[dict[str, Any]],
    cameras: list[dict[str, Any]],
    calibrations: dict[str, CameraCalibration],
    args: argparse.Namespace,
) -> None:
    dimensions = {
        camera["camera"]: (camera["width"], camera["height"]) for camera in cameras
    }
    for track in tracks:
        for observation in track["observations"]:
            camera = observation["camera"]
            width, height = dimensions[camera]
            x1, y1, x2, y2 = observation["box"]
            head = ((x1 + x2) / 2.0, float(y1))
            feet = ((x1 + x2) / 2.0, float(y2))
            calibration = calibrations[camera]
            floor_map = None
            head_world = project_ray(head, args.person_height, calibration)
            feet_world = project_ray(feet, 0.0, calibration)
            if calibration.mode == "metric":
                if args.floor_point == "feet":
                    floor = feet_world
                elif head_world is None:
                    floor = None
                else:
                    floor = [head_world[0], head_world[1], 0.0]
            else:
                floor_pixel = feet if args.floor_point == "feet" else head
                floor = None
                floor_map = normalized_floor(
                    floor_pixel,
                    width,
                    height,
                    1.0,
                    1.0,
                )
            observation["head_pixel"] = [round(head[0], 2), round(head[1], 2)]
            observation["feet_pixel"] = [round(feet[0], 2), round(feet[1], 2)]
            observation["head_world"] = head_world
            observation["feet_world"] = feet_world
            observation["floor_world"] = floor
            observation["floor_map"] = (
                floor_map if calibration.mode != "metric" else None
            )
            observation["projection_mode"] = calibration.mode


def draw_box(
    frame: np.ndarray,
    box: list[int],
    global_id: int,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        frame,
        f"ID {global_id}",
        (x1, max(30, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )


def render_videos(
    cameras: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    output: Path,
    args: argparse.Namespace,
) -> None:
    colors = {
        track["global_id"]: (
            int((37 * track["global_id"]) % 220 + 20),
            int((97 * track["global_id"]) % 220 + 20),
            int((173 * track["global_id"]) % 220 + 20),
        )
        for track in tracks
    }
    observations_by_camera: dict[str, dict[int, list[tuple]]] = {}
    projections_by_camera: dict[str, dict[int, list[tuple]]] = {}
    for track in tracks:
        for observation in track["observations"]:
            camera = observation["camera"]
            frame = observation["frame"]
            value = (observation["box"], track["global_id"])
            observations_by_camera.setdefault(camera, {}).setdefault(frame, []).append(
                value
            )
            point = observation.get("floor_world")
            mode = observation.get("projection_mode")
            if mode != "metric":
                point = observation.get("floor_map")
            projections_by_camera.setdefault(camera, {}).setdefault(frame, []).append(
                (point, mode, track["global_id"])
            )
    tracked_dir = output / "tracked"
    projection_dir = output / "projections"
    tracked_dir.mkdir(parents=True, exist_ok=True)
    projection_dir.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        camera_id = camera["camera"]
        source = Path(camera["path"])
        capture = cv2.VideoCapture(str(source))
        tracked_path = tracked_dir / f"{source.stem}_tracked.mp4"
        projection_path = projection_dir / f"{source.stem}_projection.mp4"
        tracked_writer = cv2.VideoWriter(
            str(tracked_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            camera["fps"] / args.step,
            (camera["width"], camera["height"]),
        )
        projection_writer = cv2.VideoWriter(
            str(projection_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            camera["fps"] / args.step,
            (args.map_width_px, args.map_height_px),
        )
        frame_number = 0
        while args.max_frames == 0 or frame_number < args.max_frames:
            ok, image = capture.read()
            if not ok:
                break
            if frame_number % args.step != 0:
                frame_number += 1
                continue
            tracked_image = image.copy()
            for box, global_id in observations_by_camera.get(camera_id, {}).get(
                frame_number, []
            ):
                draw_box(tracked_image, box, global_id, colors[global_id])
            projection = np.full(
                (args.map_height_px, args.map_width_px, 3),
                255,
                dtype=np.uint8,
            )
            for point, mode, global_id in projections_by_camera.get(camera_id, {}).get(
                frame_number, []
            ):
                if point is None:
                    continue
                if mode == "metric":
                    x = (point[0] - args.map_origin_x) / max(args.map_width, 1e-6)
                    y = (point[1] - args.map_origin_y) / max(args.map_height, 1e-6)
                else:
                    x, y = point[0], point[1]
                px = int(
                    args.map_margin + x * (args.map_width_px - 2 * args.map_margin)
                )
                py = int(
                    args.map_height_px
                    - args.map_margin
                    - y * (args.map_height_px - 2 * args.map_margin)
                )
                color = colors[global_id]
                cv2.circle(projection, (px, py), 9, color, -1)
                cv2.putText(
                    projection,
                    f"ID {global_id}",
                    (px + 12, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (20, 20, 20),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                projection,
                f"Camera {camera_id} | {args.floor_point} | "
                f"{camera.get('projection_mode', 'none')}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
            tracked_writer.write(tracked_image)
            projection_writer.write(projection)
            frame_number += 1
        capture.release()
        tracked_writer.release()
        projection_writer.release()
        print(f"camera {camera_id}: {tracked_path} and {projection_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", default="videos", help="Папка с MP4")
    parser.add_argument("--output", default="outputs_unified", help="Папка результата")
    parser.add_argument("--model", default="yolo11m.pt", help="Файл весов YOLO")
    parser.add_argument("--device", default="cpu", help="cpu или cuda")
    parser.add_argument(
        "--devices",
        default=None,
        help="Параллельные устройства для камер, например cuda:0,cuda:1",
    )
    parser.add_argument(
        "--camera-ids", default=None, help="ID камер через запятую, в порядке MP4"
    )
    parser.add_argument(
        "--camera-offsets",
        default=None,
        help="Синхронизация камер: 1=0,2=1.25 (секунды)",
    )
    parser.add_argument(
        "--step", type=int, default=5, help="Обрабатывать каждый N-й кадр"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Лимит кадров; 0 = весь файл"
    )
    parser.add_argument(
        "--conf", type=float, default=0.15, help="Порог confidence YOLO"
    )
    parser.add_argument("--det-iou", type=float, default=0.5, help="IoU NMS YOLO")
    parser.add_argument(
        "--min-person-conf",
        type=float,
        default=0.45,
        help="Минимальная confidence после YOLO для person-бокса",
    )
    parser.add_argument(
        "--min-person-aspect",
        type=float,
        default=0.45,
        help="Минимальное отношение высоты person-бокса к ширине",
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="Размер входа YOLO")
    parser.add_argument(
        "--max-det", type=int, default=100, help="Максимум детекций на кадр"
    )
    parser.add_argument(
        "--track-iou", type=float, default=0.7, help="Порог IoU Deep SORT"
    )
    parser.add_argument(
        "--track-cosine",
        type=float,
        default=0.35,
        help="Максимальная cosine distance Deep SORT",
    )
    parser.add_argument(
        "--nn-budget", type=int, default=100, help="Размер галереи признаков Deep SORT"
    )
    parser.add_argument(
        "--track-max-age", type=int, default=30, help="Запас обновлений трекера"
    )
    parser.add_argument(
        "--track-max-age-sec", type=float, default=2.0, help="Запас трекера в секундах"
    )
    parser.add_argument(
        "--n-init", type=int, default=3, help="Обновлений до подтверждения трека"
    )
    parser.add_argument(
        "--min-segment-frames", type=int, default=3, help="Минимум наблюдений сегмента"
    )
    parser.add_argument(
        "--global-gap",
        type=float,
        default=60.0,
        help="Максимальный разрыв глобального трека, сек",
    )
    parser.add_argument(
        "--transition-overlap",
        type=float,
        default=2.0,
        help="Допустимое перекрытие сегментов, сек",
    )
    # 2 -> 1 -> 3 -> 5 -> 4 is the longest route in the default graph.
    parser.add_argument(
        "--max-graph-hops", type=int, default=4, help="Максимум рёбер пути в графе"
    )
    parser.add_argument(
        "--reid-threshold",
        type=float,
        default=0.58,
        help="Минимальная cosine similarity между камерами",
    )
    parser.add_argument(
        "--local-reid-threshold",
        type=float,
        default=0.68,
        help="Минимальная similarity для сегментов одной камеры",
    )
    parser.add_argument(
        "--time-penalty",
        type=float,
        default=0.10,
        help="Штраф cosine за временной разрыв",
    )
    parser.add_argument(
        "--graph", default="camera_graph.json", help="JSON ориентированного графа камер"
    )
    parser.add_argument(
        "--calibration",
        default="camera_calibrations.json",
        help="JSON калибровок камер",
    )
    parser.add_argument(
        "--person-height",
        type=float,
        default=1.78,
        help="Рост человека для head-проекции, м",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=1400.0,
        help="Запасное фокусное расстояние, px",
    )
    parser.add_argument(
        "--map-width", type=float, default=10.0, help="Ширина метрической карты, м"
    )
    parser.add_argument(
        "--map-height", type=float, default=10.0, help="Высота метрической карты, м"
    )
    parser.add_argument(
        "--map-origin-x",
        type=float,
        default=-5.0,
        help="Мировой X левого края белой карты, м",
    )
    parser.add_argument(
        "--map-origin-y",
        type=float,
        default=-5.0,
        help="Мировой Y нижнего края белой карты, м",
    )
    parser.add_argument(
        "--map-width-px", type=int, default=1000, help="Ширина видео проекции, px"
    )
    parser.add_argument(
        "--map-height-px", type=int, default=700, help="Высота видео проекции, px"
    )
    parser.add_argument(
        "--map-margin", type=int, default=60, help="Отступ точки от края карты, px"
    )
    parser.add_argument(
        "--floor-point",
        choices=["head", "feet"],
        default="head",
        help="Точка пола: head или feet",
    )
    parser.add_argument(
        "--no-reid",
        action="store_true",
        help="Отключить OSNet и использовать HSV baseline",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.step < 1 or args.n_init < 1 or args.min_segment_frames < 1:
        raise ValueError("step, n-init and min-segment-frames must be positive")
    if not 0.0 <= args.min_person_conf <= 1.0:
        raise ValueError("min-person-conf must be between 0 and 1")
    if args.min_person_aspect <= 0:
        raise ValueError("min-person-aspect must be positive")
    if args.global_gap <= 0 or args.person_height <= 0:
        raise ValueError("global-gap and person-height must be positive")
    if args.max_graph_hops < 1:
        raise ValueError("max-graph-hops must be positive")
    files = sorted(Path(args.videos).glob("*.mp4"))
    if not files:
        raise FileNotFoundError(f"No MP4 files found in {args.videos}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    graph_path = Path(args.graph)
    graph = normalize_graph(load_or_create(graph_path, DEFAULT_GRAPH))
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    calibration_path = Path(args.calibration)
    calibration_data = load_or_create(calibration_path, DEFAULT_CALIBRATION)
    camera_ids = resolve_camera_ids(files, args.camera_ids)
    camera_offsets = parse_camera_offsets(args.camera_offsets, camera_ids)
    devices = parse_devices(args.devices, args.device)
    normalized_devices = [
        "cuda:0" if device == "cuda" else device for device in devices
    ]
    if len(set(normalized_devices)) != len(normalized_devices):
        raise ValueError("--devices must not contain duplicate devices")
    devices = normalized_devices

    if len(devices) > 1 and any(device.startswith("cuda:") for device in devices):
        import torch

        visible_gpus = torch.cuda.device_count()
        requested_gpus = [
            int(device.split(":", maxsplit=1)[1])
            for device in devices
            if device.startswith("cuda:")
        ]
        if not torch.cuda.is_available() or max(requested_gpus) >= visible_gpus:
            raise ValueError(
                f"Requested devices {devices}, but only {visible_gpus} CUDA GPU(s) "
                "are visible to this process"
            )

    if len(devices) > 1:
        from concurrent.futures import ProcessPoolExecutor

        worker_count = min(len(devices), len(files))
        devices = devices[:worker_count]
        jobs_by_worker = [[] for _ in devices]
        for index, (camera_id, path) in enumerate(zip(camera_ids, files)):
            jobs_by_worker[index % worker_count].append(
                (camera_id, str(path), camera_offsets[camera_id])
            )
        worker_args = vars(args).copy()
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as executor:
            futures = [
                executor.submit(
                    process_camera_worker,
                    device,
                    jobs,
                    str(args.model),
                    worker_args,
                )
                for device, jobs in zip(devices, jobs_by_worker)
            ]
            worker_results = [future.result() for future in futures]

        cameras_by_id = {
            camera["camera"]: camera
            for result in worker_results
            for camera in result["cameras"]
        }
        cameras = [cameras_by_id[camera_id] for camera_id in camera_ids]
        reid_model = next(
            (
                result["reid_model"]
                for result in worker_results
                if result["reid_model"] is not None
            ),
            None,
        )
    else:
        from ultralytics import YOLO

        args.device = devices[0]
        model = YOLO(args.model)
        reid = ReID(not args.no_reid, args.device)
        cameras = []
        for number, (camera_id, path) in enumerate(zip(camera_ids, files), 1):
            print(
                f"processing camera {camera_id} ({number}/{len(files)}): {path.name}",
                flush=True,
            )
            cameras.append(
                process_camera(
                    path,
                    camera_id,
                    model,
                    reid,
                    args,
                    time_offset=camera_offsets[camera_id],
                )
            )
        reid_model = reid.model_name

    tracks = merge_segments(cameras, graph, args)
    calibrations = {}
    for camera in cameras:
        settings = camera_calibration_settings(calibration_data, camera["camera"])
        calibrations[camera["camera"]] = make_calibration(
            settings,
            camera["width"],
            camera["height"],
            args.focal_length,
        )
        camera["projection_mode"] = calibrations[camera["camera"]].mode
    assign_projections(tracks, cameras, calibrations, args)
    embedding_records = [
        record for camera in cameras for record in camera["embeddings"]
    ]
    global_ids = {
        (observation["camera"], observation["local_id"], observation["frame"]): track[
            "global_id"
        ]
        for track in tracks
        for observation in track["observations"]
    }
    for record in embedding_records:
        record["global_id"] = global_ids.get(
            (record["camera"], record["local_id"], record["frame"])
        )
    report = {
        "detector": "YOLO person",
        "reid": None if args.no_reid else reid_model,
        "reid_device": None if args.no_reid else devices,
        "inference_devices": devices,
        "camera_graph": graph,
        "global_gap_sec": args.global_gap,
        "parameters": vars(args),
        "cameras": [
            {
                key: value
                for key, value in camera.items()
                if key not in {"segments", "embeddings"}
            }
            for camera in cameras
        ],
        "calibrations": {
            camera: {
                "mode": calibration.mode,
                "focal_length": calibration.focal_length,
                "reprojection_error_px": calibration.reprojection_error_px,
                "camera_world": calibration.camera_world,
            }
            for camera, calibration in calibrations.items()
        },
        "total_tracks": len(tracks),
        "tracks": tracks,
    }
    (output / "tracks_projection.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output / "embeddings.json").write_text(
        json.dumps(
            {
                "model": None if args.no_reid else reid_model,
                "device": None if args.no_reid else devices,
                "feature_type": "hsv" if args.no_reid else "osnet",
                "normalization": "L2",
                "records": embedding_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    render_videos(cameras, tracks, output, args)
    print(
        json.dumps({"tracks": len(tracks), "output": str(output)}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main(parse_args())

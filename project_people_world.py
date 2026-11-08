"""Detect people and project their head points into table-based world coordinates.

The camera pose is estimated from four image points of a ping-pong table.
World coordinates are metres: the table starts at (0, 0, table_height), its
long side is the X axis, and its short side is the Y axis. The output video is
a top-down white map with projected head points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_TABLE_IMAGE_POINTS = np.array(
    [
        [689.0, 618.0],
        [1603.0, 499.0],
        [1888.0, 833.0],
        [751.0, 1081.0],
    ],
    dtype=np.float32,
)


def camera_matrix(width: int, height: int, focal_length: float) -> np.ndarray:
    """Build an approximate intrinsic matrix from image size and focal length."""
    return np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_camera_pose(
    image_points: np.ndarray,
    width: int,
    height: int,
    focal_length: float,
    table_width: float,
    table_depth: float,
    table_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate camera pose from the known table corners."""
    world_points = np.array(
        [
            [0.0, 0.0, table_height],
            [table_width, 0.0, table_height],
            [table_width, table_depth, table_height],
            [0.0, table_depth, table_height],
        ],
        dtype=np.float64,
    )
    intrinsics = camera_matrix(width, height, focal_length)
    distortion = np.zeros((4, 1), dtype=np.float64)
    ok, rotation_vector, translation = cv2.solvePnP(
        world_points,
        image_points.astype(np.float64),
        intrinsics,
        distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok:
        raise RuntimeError("Could not estimate camera pose from table points")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    return intrinsics, rotation, translation


def image_point_to_world_plane(
    point: tuple[float, float],
    plane_z: float,
    intrinsics: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[float, float, float] | None:
    """Intersect a camera ray with the horizontal world plane z=plane_z."""
    pixel = np.array([point[0], point[1], 1.0], dtype=np.float64)
    ray_camera = np.linalg.inv(intrinsics) @ pixel
    ray_world = rotation.T @ ray_camera
    camera_world = -rotation.T @ translation.reshape(3)
    if abs(ray_world[2]) < 1e-9:
        return None
    scale = (plane_z - camera_world[2]) / ray_world[2]
    if scale <= 0:
        return None
    world = camera_world + scale * ray_world
    return float(world[0]), float(world[1]), float(world[2])


def detect_people(
    model: YOLO, frame: np.ndarray, confidence: float, image_size: int
) -> list[dict]:
    """Return person detections in JSON-friendly form."""
    detections = []
    result = model(
        frame,
        classes=[0],
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )[0]
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float(box.conf[0]),
                "head_pixel": [round((x1 + x2) / 2.0, 2), float(y1)],
            }
        )
    return detections


def map_to_pixel(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    margin: int,
) -> tuple[int, int]:
    min_x, max_x, min_y, max_y = bounds
    scale = min(
        (width - 2 * margin) / max(max_x - min_x, 1e-6),
        (height - 2 * margin) / max(max_y - min_y, 1e-6),
    )
    return int(margin + (x - min_x) * scale), int(height - margin - (y - min_y) * scale)


def render_world_video(
    frames: list[dict],
    output: Path,
    fps: float,
    width: int,
    height: int,
    bounds: tuple[float, float, float, float],
) -> None:
    """Render top-down projected head points on a white background."""
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame_data in frames:
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        for index, person in enumerate(frame_data["people"], 1):
            world = person.get("head_world")
            if world is None:
                continue
            px, py = map_to_pixel(world[0], world[1], bounds, width, height, 60)
            cv2.circle(canvas, (px, py), 9, (30, 50, 220), -1)
            cv2.putText(
                canvas,
                f"person {index}",
                (px + 12, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            canvas,
            "World coordinates, metres",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        writer.write(canvas)
    writer.release()


def project_video(args: argparse.Namespace) -> None:
    if args.step < 1:
        raise ValueError("--step must be at least 1")
    if args.person_height <= 0:
        raise ValueError("--person-height must be positive")
    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = YOLO(args.model)
    image_points = np.array(args.table_points, dtype=np.float32).reshape(4, 2)
    intrinsics, rotation, translation = estimate_camera_pose(
        image_points,
        source_width,
        source_height,
        args.focal_length,
        args.table_width,
        args.table_depth,
        args.table_height,
    )
    frames = []
    frame_number = 0
    while args.max_frames == 0 or frame_number < args.max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_number % args.step == 0:
            people = detect_people(model, frame, args.confidence, args.image_size)
            for person in people:
                head_pixel = tuple(person["head_pixel"])
                head_world = image_point_to_world_plane(
                    head_pixel,
                    args.person_height,
                    intrinsics,
                    rotation,
                    translation,
                )
                floor_world = (
                    None if head_world is None else [head_world[0], head_world[1], 0.0]
                )
                person["head_world"] = (
                    None
                    if head_world is None
                    else list(map(lambda value: round(value, 4), head_world))
                )
                person["floor_world"] = (
                    None
                    if floor_world is None
                    else [round(value, 4) for value in floor_world]
                )
            frames.append(
                {
                    "frame": frame_number,
                    "time_sec": round(frame_number / fps, 4),
                    "people": people,
                }
            )
        frame_number += 1
    capture.release()
    all_points = [
        person["floor_world"]
        for frame in frames
        for person in frame["people"]
        if person["floor_world"] is not None
    ]
    if all_points:
        xs, ys = zip(*[(point[0], point[1]) for point in all_points])
        padding = args.map_padding
        bounds = (
            min(xs) - padding,
            max(xs) + padding,
            min(ys) - padding,
            max(ys) + padding,
        )
    else:
        bounds = (0.0, args.map_width, 0.0, args.map_height)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "coordinate_system": {
            "units": "metres",
            "origin": "table corner corresponding to the first table point",
            "x_axis": "along table width",
            "y_axis": "along table depth",
            "z_axis": "up from the floor",
            "floor_z": 0.0,
        },
        "source": str(args.source),
        "model": args.model,
        "fps": fps,
        "source_size": [source_width, source_height],
        "step": args.step,
        "person_height_m": args.person_height,
        "table": {
            "width_m": args.table_width,
            "depth_m": args.table_depth,
            "height_m": args.table_height,
            "image_points_xy": image_points.tolist(),
        },
        "camera_focal_length_px": args.focal_length,
        "camera_matrix": intrinsics.tolist(),
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0],
        "pose_method": "cv2.solvePnP(SOLVEPNP_IPPE)",
        "world_bounds_xy": list(bounds),
        "frames": frames,
    }
    (output_dir / "world_tracks.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    render_world_video(
        frames,
        output_dir / "world_projection.mp4",
        fps / args.step,
        args.map_width_px,
        args.map_height_px,
        bounds,
    )
    print(f"JSON: {output_dir / 'world_tracks.json'}")
    print(f"Video: {output_dir / 'world_projection.mp4'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="videos_proms/camera_4_lots.mp4")
    parser.add_argument("--output", default="world_projection_output")
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum source frames; 0 means the whole video",
    )
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--person-height", type=float, default=1.78)
    parser.add_argument("--table-width", type=float, default=2.74)
    parser.add_argument("--table-depth", type=float, default=1.525)
    parser.add_argument("--table-height", type=float, default=0.76)
    parser.add_argument(
        "--focal-length",
        type=float,
        default=1800.0,
        help="Approximate camera focal length in pixels; calibrate for accurate coordinates",
    )
    parser.add_argument("--map-width-px", type=int, default=1000)
    parser.add_argument("--map-height-px", type=int, default=700)
    parser.add_argument("--map-padding", type=float, default=1.0)
    parser.add_argument("--map-width", type=float, default=10.0)
    parser.add_argument("--map-height", type=float, default=10.0)
    parser.add_argument(
        "--table-points",
        type=float,
        nargs=8,
        default=DEFAULT_TABLE_IMAGE_POINTS.flatten().tolist(),
        metavar=("X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    project_video(parse_args())

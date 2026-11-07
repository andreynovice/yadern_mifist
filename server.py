"""Local HTTP API and web host for the people-tracking demonstration."""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
RUNS_ROOT = ROOT / "runs"
RUNS_ROOT.mkdir(exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobCreate(BaseModel):
    videos: str = Field(..., description="Input video directory")
    model: str | None = "yolo11m.pt"
    step: int = Field(5, ge=1)
    conf: float = Field(0.15, ge=0.0, le=1.0)
    imgsz: int = Field(1280, ge=32)
    gap: float = Field(8.0, ge=0.0)
    max_frames: int = Field(0, ge=0)
    device: str = "cpu"
    reid: bool = True
    render: bool = True


class ImportJob(BaseModel):
    result_dir: str
    videos: str | None = None
    world_dir: str | None = None
    name: str | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "name": data.get("name") or f"Tracking job {job_id}",
            "status": "queued",
            "stage": "queued",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            **data,
        }
        with self._lock:
            self._jobs[job_id] = job
        return dict(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else dict(job)

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(job) for job in self._jobs.values()]

    def update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(values, updated_at=utc_now())

    def submit(self, task: Any) -> None:
        self._executor.submit(task)


def model_data(model: BaseModel) -> dict[str, Any]:
    """Support both Pydantic v1 and v2 in local environments."""
    dump = getattr(model, "model_dump", None)
    return dump() if dump is not None else model.dict()


store = JobStore()
app = FastAPI(title="People Tracking Demo API", version="1.0.0")


def require_job(job_id: str) -> dict[str, Any]:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def directory(value: str | None, field: str, required: bool = True) -> Path | None:
    if not value:
        if required:
            raise HTTPException(status_code=400, detail=f"{field} is required")
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {path}")
    return path


def result_file(job: dict[str, Any], filename: str) -> Path:
    path = Path(job["result_dir"]) / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{filename} is not available")
    return path


def run_analysis(job_id: str, request: JobCreate, output_dir: Path) -> None:
    try:
        store.update(job_id, status="running", stage="analysis")
        from analyze import analyze

        analyze(
            videos=request.videos,
            output=str(output_dir),
            model=request.model,
            step=request.step,
            conf=request.conf,
            imgsz=request.imgsz,
            gap=request.gap,
            max_frames=request.max_frames,
            reid_enabled=request.reid,
            device=request.device,
        )
        if request.render:
            store.update(job_id, stage="rendering")
            from render import render

            rendered_dir = output_dir / "rendered"
            render(request.videos, str(output_dir / "report.json"), str(rendered_dir))
            store.update(job_id, rendered_dir=str(rendered_dir))
        store.update(job_id, status="completed", stage="completed")
    except Exception as exc:
        store.update(job_id, status="failed", stage="error", error=str(exc))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": store.all()}


@app.post("/api/jobs")
def create_job(request: JobCreate) -> dict[str, Any]:
    videos = directory(request.videos, "videos")
    output_dir = RUNS_ROOT / uuid.uuid4().hex[:12]
    output_dir.mkdir(parents=True, exist_ok=True)
    job = store.create({
        "kind": "analysis",
        "videos": str(videos),
        "result_dir": str(output_dir),
        "params": model_data(request),
    })
    store.submit(lambda: run_analysis(job["id"], request, output_dir))
    return job


@app.post("/api/jobs/import")
def import_job(request: ImportJob) -> dict[str, Any]:
    result_dir = directory(request.result_dir, "result_dir")
    assert result_dir is not None
    if not (result_dir / "report.json").is_file():
        raise HTTPException(status_code=400, detail="result_dir must contain report.json")
    videos = directory(request.videos, "videos", required=False)
    world_dir = directory(request.world_dir, "world_dir", required=False)
    return store.create({
        "kind": "import",
        "name": request.name or result_dir.name,
        "videos": str(videos) if videos else None,
        "result_dir": str(result_dir),
        "world_dir": str(world_dir) if world_dir else None,
        "status": "completed",
        "stage": "completed",
    })


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return require_job(job_id)


@app.get("/api/jobs/{job_id}/report")
def get_report(job_id: str) -> Any:
    path = result_file(require_job(job_id), "report.json")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/tracks")
def get_tracks(job_id: str) -> dict[str, Any]:
    report = get_report(job_id)
    return {"total": report.get("total_tracks", 0), "tracks": report.get("tracks", [])}


@app.get("/api/jobs/{job_id}/tracks/{track_id}")
def get_track(job_id: str, track_id: int) -> Any:
    report = get_report(job_id)
    for track in report.get("tracks", []):
        if int(track.get("global_id", -1)) == track_id:
            return track
    raise HTTPException(status_code=404, detail="Track not found")


@app.get("/api/jobs/{job_id}/embeddings")
def get_embeddings(job_id: str) -> Any:
    path = result_file(require_job(job_id), "embeddings.json")
    return json.loads(path.read_text(encoding="utf-8"))


def camera_files(job: dict[str, Any]) -> list[Path]:
    if not job.get("videos"):
        return []
    directory_path = Path(job["videos"])
    if not directory_path.is_dir():
        return []
    suffixes = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    return sorted(path for path in directory_path.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


@app.get("/api/jobs/{job_id}/cameras")
def get_cameras(job_id: str) -> dict[str, Any]:
    job = require_job(job_id)
    report = get_report(job_id)
    rendered_dir = Path(job.get("rendered_dir", Path(job["result_dir"]) / "rendered"))
    rendered = sorted(rendered_dir.glob("*.mp4")) if rendered_dir.is_dir() else []
    sources = camera_files(job)
    cameras = []
    for metadata in report.get("cameras", []):
        camera_id = int(metadata["camera"])
        cameras.append({
            **metadata,
            "source_available": camera_id <= len(sources),
            "rendered_available": camera_id <= len(rendered),
        })
    return {"cameras": cameras}


def video_response(job: dict[str, Any], camera_id: int) -> FileResponse:
    rendered_dir = Path(job.get("rendered_dir", Path(job["result_dir"]) / "rendered"))
    rendered = sorted(rendered_dir.glob("*.mp4")) if rendered_dir.is_dir() else []
    sources = camera_files(job)
    # Use source video first: the browser draws report boxes on a canvas.
    # Serving a rendered MP4 here would draw a second, stale set of boxes.
    path = sources[camera_id - 1] if 0 < camera_id <= len(sources) else None
    if path is None and 0 < camera_id <= len(rendered):
        path = rendered[camera_id - 1]
    if path is None:
        raise HTTPException(status_code=404, detail="Camera video not found")
    media_types = {
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
    }
    return FileResponse(path, media_type=media_types.get(path.suffix.lower(), "application/octet-stream"))


@app.get("/api/jobs/{job_id}/videos/{camera_id}")
def get_video(job_id: str, camera_id: int) -> FileResponse:
    return video_response(require_job(job_id), camera_id)


def world_path(job: dict[str, Any]) -> Path:
    candidates = []
    if job.get("world_dir"):
        candidates.append(Path(job["world_dir"]))
    candidates.append(Path(job["result_dir"]))
    for directory_path in candidates:
        path = directory_path / "world_tracks.json"
        if path.is_file():
            return path
    raise HTTPException(status_code=404, detail="World projection is not available")


@app.get("/api/jobs/{job_id}/world")
def get_world(job_id: str) -> Any:
    path = world_path(require_job(job_id))
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/world/video")
def get_world_video(job_id: str) -> FileResponse:
    path = world_path(require_job(job_id)).with_name("world_projection.mp4")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="World projection video is not available")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/export/{filename}")
def export_file(job_id: str, filename: str) -> FileResponse:
    job = require_job(job_id)
    if filename == "world_tracks.json":
        path = world_path(job)
    elif filename in {"report.json", "tracks.csv", "embeddings.json"}:
        path = result_file(job, filename)
    else:
        raise HTTPException(status_code=404, detail="Export is not allowed")
    return FileResponse(path, filename=filename)


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")

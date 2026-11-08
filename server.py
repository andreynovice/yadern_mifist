"""HTTP API and web host for the unified people-tracking pipeline."""
from __future__ import annotations

import json
import logging
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
logger = logging.getLogger("people_tracking")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
_runs_lock = threading.Lock()


def next_experiment_number() -> int:
    numbers = [int(path.name) for path in RUNS_ROOT.iterdir() if path.is_dir() and path.name.isdigit()]
    numbers.extend(
        int(job["experiment_number"])
        for job in store.all()
        if str(job.get("experiment_number", "")).isdigit()
    )
    return max(numbers, default=0) + 1


def allocate_experiment() -> tuple[int, Path]:
    with _runs_lock:
        number = next_experiment_number()
        path = RUNS_ROOT / str(number)
        path.mkdir(parents=True, exist_ok=False)
        return number, path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobCreate(BaseModel):
    videos: str
    model: str = "yolo11m.pt"
    step: int = Field(20, ge=1)
    conf: float = Field(0.15, ge=0.0, le=1.0)
    imgsz: int = Field(1280, ge=32)
    gap: float = Field(60.0, ge=0.0)
    max_frames: int = Field(0, ge=0)
    start_sec: float = Field(60.0, ge=0.0)
    end_sec: float | None = Field(90.0, ge=0.0)
    device: str = "cpu"
    reid: bool = True


class ImportJob(BaseModel):
    result_dir: str
    name: str | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        job = {"id": job_id, "name": data.get("name") or f"Эксперимент {data.get('experiment_number', job_id)}",
               "status": "queued", "stage": "queued", "created_at": utc_now(),
               "updated_at": utc_now(), **data}
        with self._lock:
            self._jobs[job_id] = job
        return dict(job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return None if job_id not in self._jobs else dict(self._jobs[job_id])

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(job) for job in self._jobs.values()]

    def update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(values, updated_at=utc_now())

    def submit(self, task: Any) -> None:
        self._executor.submit(task)


store = JobStore()
app = FastAPI(title="People Tracking Demo API", version="2.0.0")


def model_data(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    return dump() if dump is not None else model.dict()


def require_job(job_id: str) -> dict[str, Any]:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def directory(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {path}")
    return path


def result_directory(value: str) -> Path:
    """Accept a result root or its tracked/ subdirectory."""
    path = directory(value)
    if path.name.lower() == "tracked" and (path.parent / "tracks_projection.json").is_file():
        return path.parent
    if not (path / "tracks_projection.json").is_file() and (path.parent / "tracks_projection.json").is_file():
        return path.parent
    return path


def tracked_video_path(result_dir: Path, camera: dict[str, Any], index: int) -> Path | None:
    tracked_dir = result_dir / "tracked"
    if not tracked_dir.is_dir():
        return None
    filename = Path(str(camera.get("file", "")))
    candidates = [
        tracked_dir / filename.name,
        tracked_dir / f"{filename.stem}_tracked.mp4",
        tracked_dir / f"{filename.stem}_tracked_tracked.mp4",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
def read_unified(job: dict[str, Any]) -> dict[str, Any]:
    path = Path(job["result_dir"]) / "tracks_projection.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Unified result is not available")
    return json.loads(path.read_text(encoding="utf-8"))


def run_analysis(job_id: str, request: JobCreate, output_dir: Path) -> None:
    try:
        logger.info("job=%s started analysis output=%s videos=%s", job_id, output_dir, request.videos)
        store.update(job_id, status="running", stage="analysis")
        import track_project
        argv = ["--videos", request.videos, "--output", str(output_dir), "--model", request.model,
                "--device", request.device, "--step", str(request.step), "--conf", str(request.conf),
                "--imgsz", str(request.imgsz), "--global-gap", str(request.gap),
                "--max-frames", str(request.max_frames), "--start-sec", str(request.start_sec)]
        if request.end_sec is not None:
            argv += ["--end-sec", str(request.end_sec)]
        if not request.reid:
            argv.append("--no-reid")
        track_project.main(track_project.parse_args(argv))
        store.update(job_id, status="completed", stage="completed")
        logger.info("job=%s completed output=%s", job_id, output_dir)
    except Exception as exc:
        logger.exception("job=%s failed", job_id)
        store.update(job_id, status="failed", stage="error", error=str(exc))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": store.all()}


@app.post("/api/jobs")
def create_job(request: JobCreate) -> dict[str, Any]:
    videos = directory(request.videos)
    if request.end_sec is not None and request.end_sec < request.start_sec:
        raise HTTPException(status_code=400, detail="end_sec must be greater than or equal to start_sec")
    experiment_number, output_dir = allocate_experiment()
    job = store.create({"kind": "analysis", "videos": str(videos), "result_dir": str(output_dir),
                        "params": model_data(request), "experiment_number": experiment_number})
    logger.info("job=%s queued name=%s output=%s", job["id"], job["name"], output_dir)
    store.submit(lambda: run_analysis(job["id"], request, output_dir))
    return job


@app.post("/api/jobs/import")
def import_job(request: ImportJob) -> dict[str, Any]:
    result_dir = result_directory(request.result_dir)
    unified = result_dir / "tracks_projection.json"
    if not unified.is_file():
        if (result_dir / "report.json").is_file():
            raise HTTPException(status_code=400, detail="Legacy report.json is not supported; provide tracks_projection.json")
        raise HTTPException(status_code=400, detail="result_dir must contain tracks_projection.json")
    if not (result_dir / "tracked").is_dir():
        raise HTTPException(status_code=400, detail="Unified result must contain tracked/ with bbox videos")
    report = json.loads(unified.read_text(encoding="utf-8"))
    with _runs_lock:
        experiment_number = next_experiment_number()
        job = store.create({"kind": "import", "name": request.name or f"Эксперимент {experiment_number}",
                            "result_dir": str(result_dir), "videos": None,
                            "status": "completed", "stage": "completed",
                            "camera_count": len(report.get("cameras", [])),
                            "experiment_number": experiment_number})
    logger.info("job=%s imported name=%s result=%s cameras=%s", job["id"], job["name"], result_dir, len(report.get("cameras", [])))
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return require_job(job_id)


@app.get("/api/jobs/{job_id}/report")
def get_report(job_id: str) -> Any:
    return read_unified(require_job(job_id))


@app.get("/api/jobs/{job_id}/tracks")
def get_tracks(job_id: str) -> dict[str, Any]:
    report = read_unified(require_job(job_id))
    return {"total": report.get("total_tracks", 0), "tracks": report.get("tracks", [])}


@app.get("/api/jobs/{job_id}/tracks/{track_id}")
def get_track(job_id: str, track_id: int) -> Any:
    report = read_unified(require_job(job_id))
    for track in report.get("tracks", []):
        if int(track.get("global_id", -1)) == track_id:
            return track
    raise HTTPException(status_code=404, detail="Track not found")


@app.get("/api/jobs/{job_id}/embeddings")
def get_embeddings(job_id: str) -> Any:
    path = Path(require_job(job_id)["result_dir"]) / "embeddings.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Embeddings are not available")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/cameras")
def get_cameras(job_id: str) -> dict[str, Any]:
    job = require_job(job_id)
    report = read_unified(job)
    cameras = []
    for index, camera in enumerate(report.get("cameras", []), 1):
        item = dict(camera)
        item["index"] = index
        item["tracked_available"] = tracked_video_path(Path(job["result_dir"]), item, index) is not None
        cameras.append(item)
    return {"cameras": cameras}


@app.get("/api/jobs/{job_id}/videos/{camera_index}")
def get_video(job_id: str, camera_index: int) -> FileResponse:
    job = require_job(job_id)
    report = read_unified(job)
    cameras = report.get("cameras", [])
    if camera_index < 1 or camera_index > len(cameras):
        raise HTTPException(status_code=404, detail="Camera not found")
    path = tracked_video_path(Path(job["result_dir"]), cameras[camera_index - 1], camera_index)
    if path is None:
        raise HTTPException(status_code=404, detail="Tracked camera video not found")
    logger.info("video request job=%s camera=%s path=%s size=%s", job_id, camera_index, path, path.stat().st_size)
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/export/{filename}")
def export_file(job_id: str, filename: str) -> FileResponse:
    path = Path(require_job(job_id)["result_dir"]) / filename
    if filename not in {"tracks_projection.json", "embeddings.json"} or not path.is_file():
        raise HTTPException(status_code=404, detail="Export is not allowed")
    return FileResponse(path, filename=filename)


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")

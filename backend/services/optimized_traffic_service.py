from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
from pathlib import Path
from uuid import uuid4


PROJECT_DIR = Path(__file__).resolve().parents[2]
OPTIMIZED_DIR = PROJECT_DIR / "vendor" / "optimized_traffic"
CTPGR_DIR = OPTIMIZED_DIR / "ctpgr"
ENGINE_PATH = OPTIMIZED_DIR / "engine.py"
CHECKPOINT = OPTIMIZED_DIR / "models" / "gesture_bilstm_multi_video.pt"
UPLOAD_DIR = PROJECT_DIR / "uploads" / "traffic_police"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024

_module = None
_runtime = None
_runtime_lock = asyncio.Lock()
_module_lock = threading.Lock()


def runtime_paths() -> list[Path]:
    return [ENGINE_PATH, CTPGR_DIR / "checkpoints" / "pose_model.pt", CHECKPOINT]


def is_available() -> tuple[bool, str]:
    missing = [str(path) for path in runtime_paths() if not path.is_file()]
    return (not missing, "" if not missing else f"missing: {', '.join(missing)}")


def _load_module():
    global _module
    if _module is not None:
        return _module
    with _module_lock:
        if _module is None:
            available, reason = is_available()
            if not available:
                raise RuntimeError(reason)
            spec = importlib.util.spec_from_file_location("irv_optimized_traffic_engine", ENGINE_PATH)
            if spec is None or spec.loader is None:
                raise RuntimeError("cannot load optimized traffic gesture engine")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _module = module
    return _module


async def get_runtime():
    global _runtime
    if _runtime is not None:
        return _runtime
    async with _runtime_lock:
        if _runtime is None:
            module = _load_module()
            _runtime = await asyncio.to_thread(_create_runtime, module)
    return _runtime


def _create_runtime(module):
    """Load TPHSR while preserving the car-owner gesture module named models."""
    ctpgr_path = str(CTPGR_DIR)
    if ctpgr_path in sys.path:
        sys.path.remove(ctpgr_path)
    sys.path.insert(0, ctpgr_path)

    previous_models = sys.modules.pop("models", None)
    try:
        return module.OptimizedRuntime(CTPGR_DIR, CHECKPOINT)
    finally:
        # TPHSR classes retain their imported module references after startup.
        sys.modules.pop("models", None)
        if previous_models is not None:
            sys.modules["models"] = previous_models


def create_session(runtime):
    return _load_module().LiveGestureSession(runtime)


def create_capture(*, camera_index: int = 0, source_url: str | None = None):
    return _load_module().LatestCameraCapture(camera_index=camera_index, source_url=source_url)


async def save_upload(upload) -> dict:
    suffix = Path(upload.filename or "video.mp4").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise ValueError("仅支持 MP4、AVI、MOV、MKV、WEBM 视频")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    video_id = f"{uuid4().hex}{suffix}"
    path = UPLOAD_DIR / video_id
    written = 0
    try:
        with path.open("wb") as target:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise ValueError("视频不能超过 500 MB")
                target.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"video_id": video_id, "name": upload.filename or video_id, "size_mb": round(written / 1024 / 1024, 1)}


def resolve_upload(video_id: str) -> Path | None:
    candidate = (UPLOAD_DIR / Path(video_id).name).resolve()
    try:
        candidate.relative_to(UPLOAD_DIR.resolve())
    except ValueError:
        return None
    if not candidate.is_file() or candidate.suffix.lower() not in VIDEO_SUFFIXES:
        return None
    return candidate

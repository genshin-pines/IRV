from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("plate")

PROJECT_DIR = Path(__file__).resolve().parents[2]
PLATE_VENDOR = PROJECT_DIR / "vendor" / "plate_hyperlpr"

PLATE_COLOR_MAP = {
    -1: "未知",
    0: "蓝牌",
    1: "黄牌(单层)",
    2: "白牌(单层)",
    3: "绿牌(新能源)",
    4: "黑牌(港澳)",
    5: "香港(单层)",
    6: "香港(双层)",
    7: "澳门(单层)",
    8: "澳门(双层)",
    9: "黄牌(双层)",
}

_catcher = None
_import_error: str | None = None


def get_catcher():
    global _catcher, _import_error
    if _catcher is not None:
        return _catcher
    if _import_error:
        raise RuntimeError(_import_error)
    if str(PLATE_VENDOR) not in sys.path:
        sys.path.insert(0, str(PLATE_VENDOR))
    try:
        from gpu_patch import catcher  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local model/deps
        _import_error = f"车牌识别模块加载失败: {exc}"
        logger.exception(_import_error)
        raise RuntimeError(_import_error) from exc
    _catcher = catcher
    return _catcher


def normalize_plate(raw: Any) -> dict[str, Any]:
    plate_code = raw[0]
    confidence = round(float(raw[1]), 4)
    plate_type = int(raw[2])
    bbox = [int(v) for v in raw[3]]
    return {
        "plate_code": plate_code,
        "confidence": confidence,
        "plate_type": plate_type,
        "plate_color": PLATE_COLOR_MAP.get(plate_type, f"未知({plate_type})"),
        "bbox": bbox,
    }


def recognize_image_bytes(contents: bytes, filename: str = "upload") -> dict[str, Any]:
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        logger.error("plate recognition failed: image decode failed filename=%s", filename)
        raise ValueError("无法解析图片")

    t0 = time.perf_counter()
    results = get_catcher()(img)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    plates = [normalize_plate(item) for item in results]

    logger.info("plate recognition success: filename=%s count=%s latency_ms=%s", filename, len(plates), elapsed_ms)
    for plate in plates:
        if plate["confidence"] < 0.75:
            logger.warning("plate confidence low: code=%s confidence=%s", plate["plate_code"], plate["confidence"])

    return {
        "filename": filename,
        "image_size": f"{img.shape[1]}x{img.shape[0]}",
        "inference_ms": elapsed_ms,
        "plate_count": len(plates),
        "plates": plates,
    }


def recognize_video_file(path: str, filename: str = "video", interval: float = 0.5) -> dict[str, Any]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logger.error("plate video failed: cannot open video filename=%s", filename)
        raise ValueError("无法打开视频文件")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0
    step = max(1, int(fps * interval))
    plate_best: dict[str, dict[str, Any]] = {}
    frame_idx = 0
    processed = 0
    catcher = get_catcher()
    t0 = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                processed += 1
                timestamp = round(frame_idx / fps, 2)
                for raw in catcher(frame):
                    plate = normalize_plate(raw)
                    plate["time_sec"] = timestamp
                    code = plate["plate_code"]
                    if code not in plate_best or plate["confidence"] > plate_best[code]["confidence"]:
                        plate_best[code] = plate
            frame_idx += 1
    finally:
        cap.release()

    plates = sorted(plate_best.values(), key=lambda item: item.get("time_sec", 0))
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info("plate video success: filename=%s processed=%s unique=%s latency_ms=%s", filename, processed, len(plates), elapsed_ms)
    return {
        "filename": filename,
        "fps": round(float(fps), 1),
        "total_frames": total_frames,
        "duration_sec": round(duration, 1),
        "sample_interval_sec": interval,
        "processed_frames": processed,
        "unique_plates": len(plates),
        "inference_ms": elapsed_ms,
        "plates": plates,
    }


def recognize_video_bytes(contents: bytes, filename: str, interval: float = 0.5) -> dict[str, Any]:
    suffix = Path(filename or "video.mp4").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(contents)
        tmp.close()
        return recognize_video_file(tmp.name, filename=filename, interval=interval)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def recognize_stream(rtsp_url: str, duration_sec: float = 8, sample_interval: float = 0.5) -> dict[str, Any]:
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        logger.error("camera stream disconnected: %s", rtsp_url)
        raise ValueError("无法打开 RTSP 视频流")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(fps * sample_interval))
    max_frames = int(max(duration_sec, 1) * fps)
    plate_best: dict[str, dict[str, Any]] = {}
    catcher = get_catcher()
    processed = 0
    frame_idx = 0
    t0 = time.perf_counter()
    try:
        while frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                processed += 1
                timestamp = round(frame_idx / fps, 2)
                for raw in catcher(frame):
                    plate = normalize_plate(raw)
                    plate["time_sec"] = timestamp
                    code = plate["plate_code"]
                    if code not in plate_best or plate["confidence"] > plate_best[code]["confidence"]:
                        plate_best[code] = plate
            frame_idx += 1
    finally:
        cap.release()

    plates = sorted(plate_best.values(), key=lambda item: item.get("time_sec", 0))
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info("plate stream success: url=%s processed=%s unique=%s latency_ms=%s", rtsp_url, processed, len(plates), elapsed_ms)
    return {
        "rtsp_url": rtsp_url,
        "duration_sec": duration_sec,
        "sample_interval_sec": sample_interval,
        "processed_frames": processed,
        "unique_plates": len(plates),
        "inference_ms": elapsed_ms,
        "plates": plates,
    }

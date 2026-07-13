from __future__ import annotations

import logging
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.services.log_service import write_log

logger = logging.getLogger("plate")

PROJECT_DIR = Path(__file__).resolve().parents[2]
PLATE_VENDOR = PROJECT_DIR / "vendor" / "plate_hyperlpr"

MIN_PLATE_CONFIDENCE = 0.98

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


def _ensure_vendor_path() -> None:
    if str(PLATE_VENDOR) not in sys.path:
        sys.path.insert(0, str(PLATE_VENDOR))


def get_catcher():
    global _catcher, _import_error
    if _catcher is not None:
        return _catcher
    if _import_error:
        raise RuntimeError(_import_error)

    _ensure_vendor_path()
    try:
        from gpu_patch import catcher  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local model/deps
        _import_error = f"车牌识别模块加载失败: {exc}"
        logger.exception(_import_error)
        raise RuntimeError(_import_error) from exc
    _catcher = catcher
    return _catcher


def _normalize_confidence(value: Any) -> float:
    raw = float(value)
    return round(raw, 4) if not (math.isnan(raw) or math.isinf(raw)) else 0.0


def normalize_plate(raw: Any) -> dict[str, Any]:
    plate_type = int(raw[2])
    return {
        "plate_code": raw[0],
        "confidence": _normalize_confidence(raw[1]),
        "plate_type": plate_type,
        "plate_color": PLATE_COLOR_MAP.get(plate_type, f"未知({plate_type})"),
        "bbox": [int(v) for v in raw[3]],
        "source": "full",
    }


def _with_plate_color(plate: dict[str, Any]) -> dict[str, Any]:
    item = dict(plate)
    plate_type = int(item.get("plate_type", -1))
    item["plate_color"] = PLATE_COLOR_MAP.get(plate_type, f"未知({plate_type})")
    item["confidence"] = _normalize_confidence(item.get("confidence", 0.0))
    if item.get("bbox") is not None:
        item["bbox"] = [int(v) for v in item["bbox"]]
    return item


def _log_plate_accept(plate: dict[str, Any], *, source: str) -> None:
    write_log(
        "plate",
        "INFO",
        (
            f"plate accepted source={source} code={plate.get('plate_code')} "
            f"score={plate.get('confidence')} origin={plate.get('source', '-')}"
        ),
    )


def _log_plate_reject(plate: dict[str, Any], *, source: str) -> None:
    reasons = ",".join(plate.get("reject_reasons", []) or ["low_confidence_or_invalid"])
    write_log(
        "plate",
        "WARNING",
        (
            f"plate confidence={plate.get('confidence', 0.0)} "
            f"code={plate.get('plate_code', '')} source={source} rejected={reasons}"
        ),
    )


def _log_plate_results(plates: list[dict[str, Any]], rejected: list[dict[str, Any]], *, source: str) -> None:
    for plate in plates:
        _log_plate_accept(plate, source=source)
    for plate in rejected:
        _log_plate_reject(plate, source=source)


def _event_payload(plate: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": plate.get("plate_code", ""),
        "conf": float(plate.get("confidence", 0.0)),
        "color": plate.get("plate_color", "未知"),
        "bbox": plate.get("bbox") or [],
        "plate_type": int(plate.get("plate_type", -1)),
    }


async def publish_plate_events(
    plates: list[dict[str, Any]],
    *,
    camera_id: str = "",
    frame_ts: float | None = None,
) -> int:
    """Publish accepted plate results into the fusion EventBus."""
    if not plates:
        return 0

    try:
        from backend.services.alert_service import get_event_bus
        from fusion.perception_event import PerceptionEvent
    except Exception:
        return 0

    bus = get_event_bus()
    if bus is None:
        return 0

    count = 0
    for plate in plates:
        if not plate.get("plate_code"):
            continue
        event = PerceptionEvent.from_plate(
            _event_payload(plate),
            camera_id=camera_id,
            frame_ts=frame_ts or time.perf_counter(),
        )
        await bus.publish(event)
        count += 1
    return count


def _recognize_image_array(image, *, source: str) -> tuple[list[dict], list, list[dict], float]:
    _ensure_vendor_path()
    from vehicle_lpr import recognize_with_vehicle_crops  # type: ignore

    t0 = time.perf_counter()
    plates, regions, rejected = recognize_with_vehicle_crops(
        image,
        get_catcher(),
        full_frame_mode="fallback",
        min_plate_confidence=MIN_PLATE_CONFIDENCE,
        return_rejected=True,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    plates = [_with_plate_color(plate) for plate in plates]
    rejected = [_with_plate_color(plate) for plate in rejected]
    _log_plate_results(plates, rejected, source=source)
    return plates, regions, rejected, elapsed_ms


def recognize_image_bytes(contents: bytes, filename: str = "upload") -> dict[str, Any]:
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        logger.error("plate recognition failed: image decode failed filename=%s", filename)
        raise ValueError("无法解析图片")

    plates, regions, rejected, elapsed_ms = _recognize_image_array(img, source=filename)
    logger.info(
        "plate image success: filename=%s count=%s rejected=%s latency_ms=%s",
        filename,
        len(plates),
        len(rejected),
        elapsed_ms,
    )

    return {
        "filename": filename,
        "image_size": f"{img.shape[1]}x{img.shape[0]}",
        "inference_ms": elapsed_ms,
        "plate_count": len(plates),
        "vehicle_regions": len([r for r in regions if getattr(r, "source", "") == "vehicle"]),
        "rejected_count": len(rejected),
        "plates": plates,
        "rejected": rejected,
    }


def recognize_video_file(path: str, filename: str = "video", interval: float = 0.5) -> dict[str, Any]:
    """识别视频文件中的车牌 — 与 recognize_stream 使用相同的逐帧 _recognize_image_array 逻辑。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logger.error("plate video failed: cannot open video filename=%s", filename)
        raise ValueError("无法打开视频文件")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0

    step = max(1, int(fps * interval)) if fps > 0 else 1
    plate_best: dict[str, dict[str, Any]] = {}
    rejected_total = 0
    processed = 0
    frame_idx = 0
    t0 = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                processed += 1
                timestamp = round(frame_idx / fps, 2) if fps > 0 else 0
                plates, _regions, rejected, _elapsed = _recognize_image_array(frame, source=filename)
                rejected_total += len(rejected)
                for plate in plates:
                    plate = dict(plate)
                    plate["time_sec"] = timestamp
                    code = plate["plate_code"]
                    if code not in plate_best or plate["confidence"] > plate_best[code]["confidence"]:
                        plate_best[code] = plate
            frame_idx += 1
    finally:
        cap.release()

    plates = sorted(plate_best.values(), key=lambda item: item.get("time_sec", 0))
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(
        "plate video success: source=%s processed=%s unique=%s latency_ms=%s",
        filename,
        processed,
        len(plates),
        elapsed_ms,
    )
    return {
        "source": filename,
        "fps": round(float(fps), 1),
        "total_frames": total_frames,
        "duration_sec": round(duration, 1),
        "sample_interval_sec": interval,
        "processed_frames": processed,
        "rejected_count": rejected_total,
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
    """识别 RTSP 视频流中的车牌 — 与 recognize_video_file 使用相同的逐帧 _recognize_image_array 逻辑。"""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        logger.error("plate stream failed: cannot open stream url=%s", rtsp_url)
        write_log("camera", "ERROR", f"RTSP disconnected Camera timeout url={rtsp_url}")
        raise ValueError("无法打开 RTSP 视频流")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(fps * sample_interval))
    max_frames = int(max(duration_sec, 1) * fps)
    plate_best: dict[str, dict[str, Any]] = {}
    rejected_total = 0
    processed = 0
    frame_idx = 0
    t0 = time.perf_counter()

    try:
        while frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                logger.error("plate stream read failed mid-stream: %s", rtsp_url)
                write_log("camera", "ERROR", f"RTSP disconnected Camera timeout url={rtsp_url}")
                break
            if frame_idx % step == 0:
                processed += 1
                timestamp = round(frame_idx / fps, 2)
                plates, _regions, rejected, _elapsed = _recognize_image_array(frame, source=rtsp_url)
                rejected_total += len(rejected)
                for plate in plates:
                    plate = dict(plate)
                    plate["time_sec"] = timestamp
                    code = plate["plate_code"]
                    if code not in plate_best or plate["confidence"] > plate_best[code]["confidence"]:
                        plate_best[code] = plate
            frame_idx += 1
    finally:
        cap.release()

    plates = sorted(plate_best.values(), key=lambda item: item.get("time_sec", 0))
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(
        "plate stream success: source=%s processed=%s unique=%s latency_ms=%s",
        rtsp_url,
        processed,
        len(plates),
        elapsed_ms,
    )
    return {
        "source": rtsp_url,
        "fps": round(float(fps), 1),
        "total_frames": max_frames,
        "duration_sec": duration_sec,
        "sample_interval_sec": sample_interval,
        "processed_frames": processed,
        "rejected_count": rejected_total,
        "unique_plates": len(plates),
        "inference_ms": elapsed_ms,
        "plates": plates,
    }

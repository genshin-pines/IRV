from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("gesture")
PROJECT_DIR = Path(__file__).resolve().parents[2]
GESTURE_VENDOR = PROJECT_DIR / "vendor" / "web_gesture_backend"

_manager = None
_engine = None
_last_frame_message: dict[str, Any] | None = None
_last_action_message: dict[str, Any] | None = None


def ensure_vendor_path() -> None:
    if str(GESTURE_VENDOR) not in sys.path:
        sys.path.insert(0, str(GESTURE_VENDOR))


def get_stream_manager_class():
    ensure_vendor_path()
    from stream_manager import StreamManager  # type: ignore

    return StreamManager


def get_engine():
    global _engine, _last_frame_message, _last_action_message
    if _engine is not None:
        return _engine
    ensure_vendor_path()
    from gesture_engine import GestureEngine  # type: ignore

    _engine = GestureEngine()
    _engine.on_frame = lambda data: _set_last_frame(data)
    _engine.on_action = lambda data: _set_last_action(data)
    return _engine


def _set_last_frame(data: dict[str, Any]) -> None:
    global _last_frame_message
    _last_frame_message = data


def _set_last_action(data: dict[str, Any]) -> None:
    global _last_action_message
    _last_action_message = data


def start_gesture_stream(src_url: str | None = None, use_webcam: bool = False, camera_index: int = 0, mirror: bool = False) -> dict[str, Any]:
    global _manager
    StreamManager = get_stream_manager_class()
    if _manager and _manager.is_running:
        _manager.stop()
    _manager = StreamManager(src_url=src_url, use_webcam=use_webcam, camera_index=camera_index, mirror=mirror)
    _manager.start()
    if not _manager.is_running:
        raise RuntimeError(_manager.error or "无法启动手势视频流")
    logger.info("gesture stream started: webcam=%s camera_index=%s mirror=%s src=%s", use_webcam, camera_index, mirror, src_url)
    return gesture_status()


def stop_gesture_stream() -> dict[str, Any]:
    global _manager
    if _manager:
        _manager.stop()
        _manager = None
    logger.info("gesture stream stopped")
    return gesture_status()


def gesture_status() -> dict[str, Any]:
    if not _manager:
        return {"running": False, "mjpg_url": "/api/gesture/video-feed", "hls_url": None, "rtsp_url": None}
    return {
        "running": bool(_manager.is_running),
        "mjpg_url": "/api/gesture/video-feed",
        "hls_url": _manager.hls_url,
        "rtsp_url": _manager.dst_url,
        "error": _manager.error,
    }


def latest_frame():
    if not _manager or not _manager.is_running:
        return None
    return _manager.get_latest_frame()


def drain_messages(limit: int = 20) -> list[dict[str, Any]]:
    if not _manager:
        return []
    messages: list[dict[str, Any]] = []
    while len(messages) < limit:
        try:
            _kind, data = _manager.out_queue.get_nowait()
            messages.append(data)
        except Exception:
            break
    return messages


def recognize_frame_bytes(contents: bytes, filename: str = "frame") -> dict[str, Any]:
    global _last_frame_message, _last_action_message
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        logger.error("gesture frame failed: image decode failed filename=%s", filename)
        raise ValueError("无法解析图片")
    _last_frame_message = None
    _last_action_message = None
    annotated = get_engine().process_frame(frame)
    logger.info("gesture frame processed: filename=%s", filename)
    return {
        "filename": filename,
        "image_size": f"{frame.shape[1]}x{frame.shape[0]}",
        "frame": _last_frame_message,
        "action": _last_action_message,
        "annotated_size": f"{annotated.shape[1]}x{annotated.shape[0]}",
    }


def map_event_to_vehicle(event: dict[str, Any]) -> dict[str, Any]:
    gesture_type = str(event.get("gesture_type") or event.get("gesture_action") or "").upper()
    mapping = {
        "SWIPE_LEFT": ("turn", "left", "左转"),
        "SWIPE_RIGHT": ("turn", "right", "右转"),
        "SWIPE_UP": ("window", "up", "关闭车窗"),
        "SWIPE_DOWN": ("window", "down", "打开车窗"),
        "FIST": ("system", "confirm", "确认"),
        "OPEN_PALM": ("system", "home", "主页/唤醒"),
        "CIRCLE": ("media", "volume", "调节音量"),
        "THUMB_UP_DOWN": ("phone", "toggle", "接听/挂断"),
        "ZOOM_IN": ("vehicle", "accelerate", "加速"),
        "ZOOM_OUT": ("vehicle", "decelerate", "减速"),
    }
    target, action, label = mapping.get(gesture_type, ("unknown", "unknown", "未映射"))
    return {"target": target, "action": action, "label": label}

from __future__ import annotations

import logging
import importlib.util
import sys
import time
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
_gesture_control_enabled = False
_last_control_toggle_at = 0.0
_last_turn_action = ""
_last_turn_action_at = 0.0
_last_volume_action_at = 0.0
CONTROL_TOGGLE_SUPPRESS_SEC = 1.5
TURN_ACTION_SUPPRESS_SEC = 1.0
TURN_REVERSE_SUPPRESS_SEC = 1.5
VOLUME_TO_MUSIC_SUPPRESS_SEC = 2.0


def ensure_vendor_path() -> None:
    if str(GESTURE_VENDOR) not in sys.path:
        sys.path.insert(0, str(GESTURE_VENDOR))
    _ensure_vendor_models_module()


def _ensure_vendor_models_module() -> None:
    vendor_models = GESTURE_VENDOR / "models.py"
    current = sys.modules.get("models")
    if getattr(current, "__file__", None) == str(vendor_models):
        return
    sys.modules.pop("models", None)
    spec = importlib.util.spec_from_file_location("models", vendor_models)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载手势模型定义: {vendor_models}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["models"] = module
    spec.loader.exec_module(module)


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


def start_gesture_stream(src_url: str | None = None, use_webcam: bool = False, camera_index: int = 0, mirror: bool = False, user_id: int | None = None) -> dict[str, Any]:
    global _manager
    StreamManager = get_stream_manager_class()
    if _manager and _manager.is_running:
        _manager.stop()
    _manager = StreamManager(src_url=src_url, use_webcam=use_webcam, camera_index=camera_index, mirror=mirror, user_id=user_id)
    _manager.start()
    if not _manager.is_running:
        raise RuntimeError(_manager.error or "无法启动手势视频流")
    logger.info("gesture stream started: user_id=%s webcam=%s camera_index=%s mirror=%s src=%s", user_id, use_webcam, camera_index, mirror, src_url)
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


def latest_recognition(user_id: int | None = None) -> dict[str, Any] | None:
    """Read the current stream's recognition result without consuming its message queue."""
    if not _manager or not _manager.is_running:
        return None
    if user_id is not None and _manager.user_id != user_id:
        raise PermissionError("当前手势摄像头不属于该车主")
    return _manager.get_latest_frame_message()


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
    global _gesture_control_enabled, _last_control_toggle_at, _last_turn_action, _last_turn_action_at, _last_volume_action_at

    gesture_type = str(event.get("gesture_type") or event.get("gesture_action") or "").upper()
    mapping = {
        "SWIPE_LEFT": ("turn", "left", "左转"),
        "SWIPE_RIGHT": ("turn", "right", "右转"),
        "SWIPE_UP": ("media", "volume_up", "音量增加"),
        "SWIPE_DOWN": ("media", "volume_down", "音量降低"),
        "FIST": ("system", "confirm", "确认"),
        "OPEN_PALM": ("system", "home", "主页/唤醒"),
        "CIRCLE": ("media", "volume", "调节音量"),
        "TAP": ("media", "music_toggle", "播放/暂停"),
        "CLICK": ("media", "music_toggle", "播放/暂停"),
        "DOUBLE_TAP": ("media", "music_toggle", "播放/暂停"),
        "DOUBLE_CLICK": ("media", "music_toggle", "播放/暂停"),
        "LIKE": ("vehicle", "lights_on", "开启灯光"),
        "DISLIKE": ("vehicle", "lights_off", "关闭灯光"),
        "CALL": ("phone", "phone_answer", "接听电话"),
        "STOP": ("phone", "phone_hangup", "挂断电话"),
        "STOP_INVERTED": ("phone", "phone_hangup", "挂断电话"),
        "THUMB_UP_DOWN": ("phone", "phone_answer", "接听电话"),
        "DNDV": ("system", "control_toggle", "手势控制开关"),
        "DNDV1": ("system", "control_toggle", "手势控制开关"),
        "CONTROL_TOGGLE": ("system", "control_toggle", "手势控制开关"),
        "ZOOM_IN": ("vehicle", "accelerate", "加速"),
        "ZOOM_OUT": ("vehicle", "decelerate", "减速"),
    }
    target, action, label = mapping.get(gesture_type, ("unknown", "unknown", "未映射"))

    action_applied = True
    suppress_reason = ""
    if action == "control_toggle":
        now = time.time()
        elapsed = now - _last_control_toggle_at
        if elapsed < CONTROL_TOGGLE_SUPPRESS_SEC:
            action_applied = False
            suppress_reason = f"ignore repeated control toggle within {CONTROL_TOGGLE_SUPPRESS_SEC:.1f}s"
        else:
            _last_control_toggle_at = now
            _gesture_control_enabled = not _gesture_control_enabled
            suppress_reason = f"gesture control {'enabled' if _gesture_control_enabled else 'disabled'}"
    elif action not in {"phone_answer", "phone_hangup"} and not _gesture_control_enabled:
        action_applied = False
        suppress_reason = "gesture control disabled"
    elif action == "music_toggle" and time.time() - _last_volume_action_at < VOLUME_TO_MUSIC_SUPPRESS_SEC:
        action_applied = False
        suppress_reason = f"ignore music toggle within {VOLUME_TO_MUSIC_SUPPRESS_SEC:.1f}s after volume action"
    elif action in {"volume_up", "volume_down"}:
        _last_volume_action_at = time.time()
    elif action in {"left", "right"}:
        now = time.time()
        elapsed = now - _last_turn_action_at
        is_reverse = _last_turn_action in {"left", "right"} and _last_turn_action != action
        if is_reverse and elapsed < TURN_REVERSE_SUPPRESS_SEC:
            action_applied = False
            suppress_reason = f"ignore reverse turn action within {TURN_REVERSE_SUPPRESS_SEC:.1f}s"
        elif _last_turn_action == action and elapsed < TURN_ACTION_SUPPRESS_SEC:
            action_applied = False
            suppress_reason = f"ignore repeated turn action within {TURN_ACTION_SUPPRESS_SEC:.1f}s"
        else:
            _last_turn_action = action
            _last_turn_action_at = now

    return {
        "target": target,
        "action": action,
        "label": label,
        "action_applied": action_applied,
        "suppress_reason": suppress_reason,
        "gesture_control_enabled": _gesture_control_enabled,
    }

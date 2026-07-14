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
_last_queued_turn_action = ""
_last_queued_turn_action_at = 0.0
_last_volume_action = ""
_last_volume_action_at = 0.0
_last_queued_volume_action = ""
_last_queued_volume_action_at = 0.0
_last_music_toggle_at = 0.0
_last_queued_music_toggle_at = 0.0
_pending_music_clicks = 0
_pending_music_click_at = 0.0
_pending_command: tuple[str, str, str, str, float] | None = None
CONTROL_TOGGLE_SUPPRESS_SEC = 1.5
TURN_ACTION_SUPPRESS_SEC = 1.0
TURN_REVERSE_SUPPRESS_SEC = 1.5
VOLUME_REVERSE_SUPPRESS_SEC = 1.5
VOLUME_TO_MUSIC_SUPPRESS_SEC = 2.0
MUSIC_TOGGLE_SUPPRESS_SEC = 1.0
MUSIC_CLICK_PAIR_WINDOW_SEC = 1.5
ACTION_CONFIRM_WINDOW_SEC = 5.0


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


def list_available_cameras(max_index: int = 10) -> list[dict[str, Any]]:
    """Probe local camera indexes so the UI never offers unavailable devices."""
    cameras: list[dict[str, Any]] = []
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    for index in range(max_index + 1):
        capture = cv2.VideoCapture(index, backend)
        try:
            if not capture.isOpened():
                continue
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            cameras.append({
                "index": index,
                "label": f"摄像头 {index}",
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            })
        finally:
            capture.release()
    return cameras


def get_engine():
    global _engine, _last_frame_message, _last_action_message
    if _engine is not None:
        return _engine
    ensure_vendor_path()
    from gesture_engine import GestureEngine  # type: ignore

    _engine = GestureEngine()
    _engine.on_frame = _logged_set_last_frame
    _engine.on_action = _logged_set_last_action
    return _engine


def _set_last_frame(data: dict[str, Any]) -> None:
    global _last_frame_message
    _last_frame_message = data


def _set_last_action(data: dict[str, Any]) -> None:
    global _last_action_message
    _last_action_message = data


def _logged_set_last_frame(data: dict[str, Any]) -> None:
    """Write single-frame recognition signals into the Agent log stream."""
    _set_last_frame(data)
    hands = data.get("hands") or []
    if not hands:
        return
    gestures = [hand.get("gesture", "unknown") for hand in hands]
    confidences = [hand.get("confidence", 1.0) for hand in hands]
    min_conf = min(confidences)
    logger.info("gesture frame: type=%s, hands=%d, min_confidence=%.2f", gestures[0], len(hands), min_conf)
    if min_conf < 0.98:
        logger.warning("gesture confidence low: min_confidence=%.2f, type=%s", min_conf, gestures[0])


def _logged_set_last_action(data: dict[str, Any]) -> None:
    """Write recognized actions so Agent rules can diagnose suppressed actions."""
    _set_last_action(data)
    gesture_type = str(data.get("gesture_action") or data.get("gesture") or "unknown")
    action_applied = bool(data.get("action_applied", False))
    reason = str(data.get("suppress_reason") or "")
    hand_id = data.get("hand_id", -1)
    logger.info("gesture action: type=%s, hand_id=%s, applied=%s, reason=%s", gesture_type, hand_id, action_applied, reason)


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
    global _pending_command, _pending_music_clicks, _pending_music_click_at

    gesture_type = str(event.get("gesture_type") or event.get("gesture_action") or "").upper()
    mapping = {
        "SWIPE_LEFT": ("media", "turn_left", "上一首"),
        "SWIPE_LEFT2": ("media", "turn_left", "上一首"),
        "SWIPE_LEFT3": ("media", "turn_left", "上一首"),
        "SWIPE_RIGHT": ("media", "turn_right", "下一首"),
        "SWIPE_RIGHT2": ("media", "turn_right", "下一首"),
        "SWIPE_RIGHT3": ("media", "turn_right", "下一首"),
        "SWIPE_UP": ("media", "volume_up", "音量增加"),
        "SWIPE_DOWN": ("media", "volume_down", "音量降低"),
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
    }
    now = time.time()

    if gesture_type == "OK":
        if _pending_command is None:
            return _command_result("system", "confirm", "确认", False, "检测到 OK 手势，但当前没有待确认操作")
        target, action, label, original_gesture, detected_at = _pending_command
        _pending_command = None
        if now - detected_at > ACTION_CONFIRM_WINDOW_SEC:
            return _command_result(target, action, label, False, f"待确认手势已超过 {ACTION_CONFIRM_WINDOW_SEC:.0f} 秒，请重新操作")
        action_applied, reason = _apply_confirmed_command(action, now)
        message = f"已确认，执行{label}" if action_applied else reason
        result = _command_result(target, action, label, action_applied, message)
        result["confirmed_gesture"] = original_gesture
        return result

    command = mapping.get(gesture_type)
    if command is None:
        return _command_result("unknown", "unknown", "未映射", False, f"检测到{gesture_type or '未知'}手势，但没有可执行操作")
    target, action, label = command

    cooldown_reason = _service_cooldown_reason(action, now)
    if cooldown_reason:
        return _command_result(target, action, label, False, cooldown_reason)

    if _pending_command is not None and _pending_command[1] == action and now - _pending_command[4] <= ACTION_CONFIRM_WINDOW_SEC:
        return _command_result(target, action, label, False, f"{label}业务正在等待 OK 确认")

    if action == "control_toggle":
        _pending_command = None
        action_applied, reason = _apply_confirmed_command(action, now)
        message = f"已识别控制开关，{reason}" if action_applied else reason
        return _command_result(target, action, label, action_applied, message)

    if action == "music_toggle":
        if now - _pending_music_click_at > MUSIC_CLICK_PAIR_WINDOW_SEC:
            _pending_music_clicks = 0
        _pending_music_clicks += 2 if gesture_type in {"DOUBLE_TAP", "DOUBLE_CLICK"} else 1
        _pending_music_click_at = now
        if _pending_music_clicks < 2:
            return _command_result(target, action, label, False, f"检测到第一次点击，请在 {MUSIC_CLICK_PAIR_WINDOW_SEC:.1f} 秒内再次点击")
        _pending_music_clicks = 0

    _record_queued_business(action, now)
    _pending_command = (target, action, label, gesture_type, now)
    return _command_result(target, action, label, False, f"检测到待确认业务：{label}，请做 OK 手势确认")


def _command_result(target: str, action: str, label: str, applied: bool, message: str) -> dict[str, Any]:
    return {
        "target": target,
        "action": action,
        "label": label,
        "action_applied": applied,
        "suppress_reason": message,
        "gesture_control_enabled": _gesture_control_enabled,
    }


def _service_cooldown_reason(action: str, now: float) -> str:
    last_volume_action = _last_volume_action
    last_volume_at = _last_volume_action_at
    if _last_queued_volume_action_at > last_volume_at:
        last_volume_action = _last_queued_volume_action
        last_volume_at = _last_queued_volume_action_at
    if action in {"volume_up", "volume_down"} and last_volume_action in {"volume_up", "volume_down"}:
        if last_volume_action != action and now - last_volume_at < VOLUME_REVERSE_SUPPRESS_SEC:
            return f"{VOLUME_REVERSE_SUPPRESS_SEC:.1f} 秒内忽略反向音量操作"
    last_turn_action = _last_turn_action
    last_turn_at = _last_turn_action_at
    if _last_queued_turn_action_at > last_turn_at:
        last_turn_action = _last_queued_turn_action
        last_turn_at = _last_queued_turn_action_at
    if action in {"turn_left", "turn_right"} and last_turn_action in {"turn_left", "turn_right"}:
        elapsed = now - last_turn_at
        if last_turn_action != action and elapsed < TURN_REVERSE_SUPPRESS_SEC:
            return f"{TURN_REVERSE_SUPPRESS_SEC:.1f} 秒内忽略反向切歌"
        if last_turn_action == action and elapsed < TURN_ACTION_SUPPRESS_SEC:
            return f"{TURN_ACTION_SUPPRESS_SEC:.1f} 秒内忽略重复切歌"
    if action == "music_toggle":
        if now - last_volume_at < VOLUME_TO_MUSIC_SUPPRESS_SEC:
            return f"音量操作后 {VOLUME_TO_MUSIC_SUPPRESS_SEC:.1f} 秒内忽略播放/暂停"
        if now - max(_last_music_toggle_at, _last_queued_music_toggle_at) < MUSIC_TOGGLE_SUPPRESS_SEC:
            return f"{MUSIC_TOGGLE_SUPPRESS_SEC:.1f} 秒内忽略重复播放/暂停"
    return ""


def _record_queued_business(action: str, now: float) -> None:
    global _last_queued_volume_action, _last_queued_volume_action_at, _last_queued_turn_action, _last_queued_turn_action_at, _last_queued_music_toggle_at
    if action in {"volume_up", "volume_down"}:
        _last_queued_volume_action = action
        _last_queued_volume_action_at = now
    elif action in {"turn_left", "turn_right"}:
        _last_queued_turn_action = action
        _last_queued_turn_action_at = now
    elif action == "music_toggle":
        _last_queued_music_toggle_at = now


def _apply_confirmed_command(action: str, now: float) -> tuple[bool, str]:
    global _gesture_control_enabled, _last_control_toggle_at, _last_turn_action, _last_turn_action_at, _last_volume_action, _last_volume_action_at, _last_music_toggle_at

    if action == "control_toggle":
        if now - _last_control_toggle_at < CONTROL_TOGGLE_SUPPRESS_SEC:
            return False, f"{CONTROL_TOGGLE_SUPPRESS_SEC:.1f} 秒内忽略重复控制开关"
        _last_control_toggle_at = now
        _gesture_control_enabled = not _gesture_control_enabled
        return True, f"手势控制已{'开启' if _gesture_control_enabled else '关闭'}"

    if action not in {"phone_answer", "phone_hangup"} and not _gesture_control_enabled:
        return False, "手势控制未开启，本次操作未执行"

    if action == "music_toggle":
        if now - _last_volume_action_at < VOLUME_TO_MUSIC_SUPPRESS_SEC:
            return False, f"音量操作后 {VOLUME_TO_MUSIC_SUPPRESS_SEC:.1f} 秒内忽略播放/暂停"
        if now - _last_music_toggle_at < MUSIC_TOGGLE_SUPPRESS_SEC:
            return False, f"{MUSIC_TOGGLE_SUPPRESS_SEC:.1f} 秒内忽略重复播放/暂停"
        _last_music_toggle_at = now
        return True, ""

    if action in {"volume_up", "volume_down"}:
        _last_volume_action = action
        _last_volume_action_at = now
        return True, ""

    if action in {"turn_left", "turn_right"}:
        elapsed = now - _last_turn_action_at
        is_reverse = _last_turn_action in {"turn_left", "turn_right"} and _last_turn_action != action
        if is_reverse and elapsed < TURN_REVERSE_SUPPRESS_SEC:
            return False, f"{TURN_REVERSE_SUPPRESS_SEC:.1f} 秒内忽略反向切歌"
        if _last_turn_action == action and elapsed < TURN_ACTION_SUPPRESS_SEC:
            return False, f"{TURN_ACTION_SUPPRESS_SEC:.1f} 秒内忽略重复切歌"
        _last_turn_action = action
        _last_turn_action_at = now
    return True, ""

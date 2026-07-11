from __future__ import annotations

import queue
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.config import CTPGR_REFERENCE_DIR


_manager = None
_engine = None
_last_error = ""
_last_frame_message: dict[str, Any] | None = None


TRAFFIC_POLICE_ACTIONS = {
    0: ("none", "无手势", "继续观察"),
    1: ("stop", "停止", "前方交警示意停止"),
    2: ("straight", "直行", "前方交警示意直行"),
    3: ("left_turn", "左转", "前方交警示意左转"),
    4: ("left_wait", "左待转", "前方交警示意左待转"),
    5: ("right_turn", "右转", "前方交警示意右转"),
    6: ("lane_change", "变道", "前方交警示意变道"),
    7: ("slow_down", "减速", "前方交警示意减速"),
    8: ("pull_over", "靠边停车", "前方交警示意靠边停车"),
}


def _reference_root() -> Path:
    return Path(CTPGR_REFERENCE_DIR)


def _ensure_reference_path() -> Path:
    root = _reference_root()
    if not root.exists():
        raise RuntimeError(f"交警手势参考库不存在: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def is_available() -> tuple[bool, str]:
    root = _reference_root()
    if not root.exists():
        return False, f"reference dir missing: {root}"
    missing = [name for name in ("checkpoints/pose_model.pt", "checkpoints/lstm.pt") if not (root / name).exists()]
    if missing:
        return False, f"missing model files: {', '.join(missing)}"
    try:
        import torch  # noqa: F401
    except Exception as exc:
        return False, f"torch unavailable: {exc}"
    return True, "ready"


def _set_last_frame(data: dict[str, Any]) -> None:
    global _last_frame_message
    _last_frame_message = data


def get_engine():
    global _engine, _last_error
    if _engine is not None:
        return _engine
    _ensure_reference_path()
    try:
        from web.gesture_engine import GestureEngine  # type: ignore

        _engine = GestureEngine()
        _engine.on_frame = _set_last_frame
        _last_error = ""
        return _engine
    except Exception as exc:
        _last_error = str(exc)
        raise RuntimeError(f"交警手势模型加载失败: {exc}") from exc


def recognize_frame_bytes(contents: bytes, filename: str = "frame") -> dict[str, Any]:
    global _last_frame_message
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("无法解析图片")
    _last_frame_message = None
    annotated = get_engine().process(frame)
    return {
        "filename": filename,
        "image_size": f"{frame.shape[1]}x{frame.shape[0]}",
        "frame": _last_frame_message,
        "annotated_size": f"{annotated.shape[1]}x{annotated.shape[0]}",
        "available": True,
    }


def start_stream(src_url: str | None = None, use_webcam: bool = False) -> dict[str, Any]:
    global _manager, _last_error
    _ensure_reference_path()
    try:
        from web.stream_manager import StreamManager  # type: ignore

        if _manager and _manager.running:
            _manager.stop()
        _manager = StreamManager(src_url=src_url, use_webcam=use_webcam)
        _manager.start()
        if not _manager.running:
            _last_error = _manager.error or "无法启动交警手势流"
        else:
            _last_error = ""
        return status()
    except Exception as exc:
        _last_error = str(exc)
        raise RuntimeError(f"交警手势流启动失败: {exc}") from exc


def stop_stream() -> dict[str, Any]:
    global _manager
    if _manager:
        _manager.stop()
        _manager = None
    return status()


def status() -> dict[str, Any]:
    available, reason = is_available()
    return {
        "running": bool(_manager and _manager.running),
        "available": available,
        "reason": reason if not available else "",
        "error": _last_error,
        "mjpg_url": "/api/traffic-police/video-feed",
        "labels": [{"id": key, "action": value[0], "label": value[1], "advice": value[2]} for key, value in TRAFFIC_POLICE_ACTIONS.items()],
    }


def latest_frame():
    if not _manager or not _manager.running:
        return None
    return _manager.latest_frame()


def drain_messages(limit: int = 20) -> list[dict[str, Any]]:
    if not _manager:
        return []
    messages: list[dict[str, Any]] = []
    while len(messages) < limit:
        try:
            _kind, data = _manager.out_queue.get_nowait()
            messages.append(data)
        except queue.Empty:
            break
        except Exception:
            break
    return messages


def summarize_gesture_frame(frame_message: dict[str, Any] | None) -> dict[str, Any]:
    gesture = (frame_message or {}).get("gesture") or {}
    gesture_id = int(gesture.get("id") or 0)
    action, label, advice = TRAFFIC_POLICE_ACTIONS.get(gesture_id, TRAFFIC_POLICE_ACTIONS[0])
    return {
        "gesture_id": gesture_id,
        "gesture_action": action,
        "gesture_label": label,
        "confidence": float(gesture.get("confidence") or 0.0),
        "advice": advice,
    }

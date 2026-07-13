from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

import cv2


_ALLOWED_SCHEMES = {"rtsp", "rtmp", "http", "https"}
_LOCK = threading.Lock()
_SOURCE_URL = ""
_LAST_FRAME_AT: datetime | None = None
_LAST_ERROR = ""
_SESSION_ACTIVE = False
_SESSION_CANCELLED = False


def validate_source_url(source_url: str) -> str:
    """Accept only network video streams supported by OpenCV/FFmpeg."""
    value = source_url.strip()
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES or not parsed.netloc:
        raise ValueError("请输入有效的 RTSP、RTMP 或 HTTP(S) 手机视频流地址")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("手机视频流地址端口无效") from exc
    return value


def redact_source_url(source_url: str) -> str:
    """Keep credentials out of responses and logs."""
    parsed = urlsplit(source_url)
    hostname = parsed.hostname or ""
    host = hostname if parsed.port is None else f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def probe_source(source_url: str, timeout_seconds: float = 5.0) -> dict:
    """Open and read one frame with OpenCV/FFmpeg-native timeouts."""
    source_url = validate_source_url(source_url)
    started = time.perf_counter()
    cap = None
    try:
        timeout_ms = max(1, round(timeout_seconds * 1000))
        cap = cv2.VideoCapture(
            source_url,
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms],
        )
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise ValueError("无法打开手机视频源")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("手机视频源未返回画面")
        return {
            "source_url": redact_source_url(source_url),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "probe_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("手机视频源探测失败") from exc
    finally:
        if cap is not None:
            cap.release()


def connect_source(source_url: str) -> dict:
    global _SOURCE_URL, _LAST_ERROR, _LAST_FRAME_AT
    source_url = validate_source_url(source_url)
    probe = probe_source(source_url)
    with _LOCK:
        _SOURCE_URL = source_url
        _LAST_ERROR = ""
        _LAST_FRAME_AT = datetime.now(timezone.utc)
    return {"id": "phone_cam", "name": "答辩手机摄像头", "type": "mobile_stream", **probe}


def disconnect_source() -> None:
    global _SOURCE_URL, _LAST_ERROR, _LAST_FRAME_AT, _SESSION_CANCELLED
    with _LOCK:
        _SOURCE_URL = ""
        _LAST_ERROR = ""
        _LAST_FRAME_AT = None
        _SESSION_CANCELLED = True


def get_connected_source() -> str | None:
    with _LOCK:
        return _SOURCE_URL or None


def acquire_session() -> bool:
    """Allow one mobile inference session so duplicate browser connections do not compete."""
    global _SESSION_ACTIVE, _SESSION_CANCELLED
    with _LOCK:
        if _SESSION_ACTIVE:
            return False
        _SESSION_ACTIVE = True
        _SESSION_CANCELLED = False
        return True


def release_session() -> None:
    global _SESSION_ACTIVE
    with _LOCK:
        _SESSION_ACTIVE = False


def is_session_cancelled() -> bool:
    with _LOCK:
        return _SESSION_CANCELLED


def report_frame_received() -> None:
    global _LAST_FRAME_AT, _LAST_ERROR
    with _LOCK:
        _LAST_FRAME_AT = datetime.now(timezone.utc)
        _LAST_ERROR = ""


def report_stream_error(message: str) -> None:
    global _LAST_ERROR
    with _LOCK:
        _LAST_ERROR = message


def status() -> dict:
    with _LOCK:
        return {
            "id": "phone_cam",
            "name": "答辩手机摄像头",
            "type": "mobile_stream",
            "configured": bool(_SOURCE_URL),
            "source_url": redact_source_url(_SOURCE_URL) if _SOURCE_URL else None,
            "last_frame_at": _LAST_FRAME_AT.isoformat() if _LAST_FRAME_AT else None,
            "last_error": _LAST_ERROR or None,
            "recognizing": _SESSION_ACTIVE,
        }

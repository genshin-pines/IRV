from __future__ import annotations

import time
import threading
from typing import Any

import cv2
import numpy as np

CAMERAS = [
    {"id": "live1", "name": "桥面", "rtsp_url": "rtsp://10.126.59.120:18554/live/live1"},  # TODO-TEST: revert to :8554
    {"id": "live2", "name": "停车场出口", "rtsp_url": "rtsp://10.126.59.120:8554/live/live2"},
    {"id": "live3", "name": "行人检测", "rtsp_url": "rtsp://10.126.59.120:8554/live/live3"},
    {"id": "live4", "name": "消防车识别", "rtsp_url": "rtsp://10.126.59.120:8554/live/live4"},
    {"id": "live5", "name": "桥出口", "rtsp_url": "rtsp://10.126.59.120:8554/live/live5"},
    {"id": "live6", "name": "桥入口", "rtsp_url": "rtsp://10.126.59.120:8554/live/live6"},
    {"id": "live7", "name": "道路2", "rtsp_url": "rtsp://10.126.59.120:8554/live/live7"},
    {"id": "live8", "name": "隧道(事故识别)", "rtsp_url": "rtsp://10.126.59.120:8554/live/live8"},
    {"id": "live9", "name": "隧道(车辆数量)", "rtsp_url": "rtsp://10.126.59.120:8554/live/live9"},
    {"id": "live10", "name": "道路3", "rtsp_url": "rtsp://10.126.59.120:8554/live/live10"},
    {"id": "live11", "name": "停车场入口", "rtsp_url": "rtsp://10.126.59.120:8554/live/live11"},
    {"id": "live12", "name": "道路1", "rtsp_url": "rtsp://10.126.59.120:8554/live/live12"},
]


def list_cameras() -> list[dict[str, str]]:
    return CAMERAS


def get_camera(camera_id: str) -> dict[str, str] | None:
    return next((camera for camera in CAMERAS if camera["id"] == camera_id), None)


def open_capture(rtsp_url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def error_frame(message: str, width: int = 960, height: int = 540) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (24, 26, 30)
    for index, line in enumerate(["Camera stream unavailable", message, time.strftime("%Y-%m-%d %H:%M:%S")]):
        cv2.putText(frame, line, (42, 210 + index * 46), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
    return frame


def encode_jpeg(frame: np.ndarray, width: int | None = None, quality: int = 80) -> bytes:
    if width and frame.shape[1] > width:
        ratio = width / frame.shape[1]
        frame = cv2.resize(frame, (width, max(1, int(frame.shape[0] * ratio))), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to encode jpeg")
    return buffer.tobytes()


def read_snapshot(camera: dict[str, str], width: int = 1280, quality: int = 90) -> bytes:
    cap = open_capture(camera["rtsp_url"])
    try:
        if not cap.isOpened():
            return encode_jpeg(error_frame(f"cannot open {camera['rtsp_url']}"), width, quality)
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = error_frame(f"read failed: {camera['name']} / {camera['id']}")
        return encode_jpeg(frame, width, quality)
    finally:
        cap.release()


def mjpeg_generator(camera: dict[str, str], fps: float = 8, width: int = 960, quality: int = 80):
    delay = 1.0 / max(fps, 0.1)
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    cap = None
    last_open_try = 0.0
    try:
        while True:
            if cap is None or not cap.isOpened():
                now = time.time()
                if now - last_open_try >= 2:
                    last_open_try = now
                    if cap is not None:
                        cap.release()
                    cap = open_capture(camera["rtsp_url"])
                if cap is None or not cap.isOpened():
                    yield boundary + encode_jpeg(error_frame(f"reconnecting: {camera['name']}"), width, quality) + b"\r\n"
                    time.sleep(delay)
                    continue
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                yield boundary + encode_jpeg(error_frame(f"read failed: {camera['name']}"), width, quality) + b"\r\n"
                time.sleep(delay)
                continue
            yield boundary + encode_jpeg(frame, width, quality) + b"\r\n"
            time.sleep(delay)
    finally:
        if cap is not None:
            cap.release()


class LatestFrameCapture:
    """Continuously drain a live stream so preview clients never wait for old frames."""

    def __init__(self, source_url: str):
        self.source_url = source_url
        self._condition = threading.Condition()
        self._latest_frame: np.ndarray | None = None
        self._sequence = 0
        self._running = False
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="mobile-preview-capture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self._running:
            cap = open_capture(self.source_url)
            self._cap = cap
            if not cap.isOpened():
                cap.release()
                self._cap = None
                time.sleep(0.5)
                continue
            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                with self._condition:
                    self._latest_frame = frame
                    self._sequence += 1
                    self._condition.notify_all()
            cap.release()
            self._cap = None
            if self._running:
                time.sleep(0.2)

    def latest(self, after_sequence: int, timeout: float = 1.0) -> tuple[int, np.ndarray | None]:
        deadline = time.perf_counter() + timeout
        with self._condition:
            while self._running and self._sequence <= after_sequence:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._sequence <= after_sequence or self._latest_frame is None:
                return after_sequence, None
            return self._sequence, self._latest_frame.copy()

    def stop(self) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._cap is not None:
            self._cap.release()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        self._thread = None


def latest_mjpeg_generator(camera: dict[str, str], fps: float = 12, width: int = 960, quality: int = 72):
    """MJPEG response that sends only newly captured frames instead of buffered video."""
    delay = 1.0 / max(fps, 0.1)
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    capture = LatestFrameCapture(camera["rtsp_url"])
    sequence = 0
    try:
        capture.start()
        while True:
            started = time.perf_counter()
            sequence, frame = capture.latest(sequence, timeout=1.0)
            if frame is None:
                yield boundary + encode_jpeg(error_frame(f"reconnecting: {camera['name']}"), width, quality) + b"\r\n"
            else:
                yield boundary + encode_jpeg(frame, width, quality) + b"\r\n"
            time.sleep(max(0.0, delay - (time.perf_counter() - started)))
    finally:
        capture.stop()

from __future__ import annotations

import logging
import sys
import time
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.services.log_service import write_log

logger = logging.getLogger("camera")

# ── 车辆检测 vendor path ──
_VENDOR_PLATE = Path(__file__).resolve().parents[2] / "vendor" / "plate_hyperlpr"
if str(_VENDOR_PLATE) not in sys.path:
    sys.path.insert(0, str(_VENDOR_PLATE))

# ── 运动自适应检测间隔 ──
_COOLDOWN_MOVING = 1.0       # 有运动：1 秒检测一次
_COOLDOWN_STATIONARY = 3.0   # 静止：3 秒检测一次
_MOTION_THRESHOLD = 12.0     # 平均像素差异阈值（超过视为运动）

# ── 模型预热（避免首帧阻塞）──
_models_warmed = False

CAMERAS = [
    {"id": "live1", "name": "桥面", "rtsp_url": "rtsp://10.126.59.120:8554/live/live1"},  
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


# FFmpeg 采集选项：TCP 传输 + 10s 超时（含 socket 建连），避免 RTSP 无流时无限阻塞
_FFMPEG_CAPTURE_FLAGS = (
    "?rtsp_transport=tcp"
    "&timeout=10000000"       # 10 s (μs)
    "&stimeout=10000000"      # 10 s (μs) — socket 层超时
    "&rw_timeout=3000000"     # 3 s  — 单次读取超时
)


def open_capture(rtsp_url: str) -> cv2.VideoCapture:
    url = rtsp_url + _FFMPEG_CAPTURE_FLAGS
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def error_frame(message: str, width: int = 960, height: int = 540) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (24, 26, 30)
    for index, line in enumerate(["Camera stream unavailable", message, time.strftime("%Y-%m-%d %H:%M:%S")]):
        cv2.putText(frame, line, (42, 210 + index * 46), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
    return frame


def _draw_labeled_box(frame: np.ndarray, bbox: tuple[int, int, int, int], label: str, color: tuple[int, int, int]) -> None:
    """在帧上绘制带标签的检测框"""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(label, font, 1.0, 3)
    top = max(0, y1 - text_h - baseline - 16)
    right = min(frame.shape[1] - 1, x1 + text_w + 16)
    cv2.rectangle(frame, (x1, top), (right, y1), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 8, y1 - baseline - 5), font, 1.0, color, 3, cv2.LINE_AA)


def _iou(box_a: tuple, box_b: tuple) -> float:
    """计算两个 bbox 的 IoU，用于匹配检测框"""
    xa1, ya1, xa2, ya2 = map(int, box_a)
    xb1, yb1, xb2, yb2 = map(int, box_b)
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = max(1, (xa2 - xa1) * (ya2 - ya1))
    area_b = max(1, (xb2 - xb1) * (yb2 - yb1))
    return inter_area / max(1, area_a + area_b - inter_area)


def _warmup_detection_models() -> None:
    """预加载 YOLO + HyperLPR3 模型，避免首次推流时阻塞首帧"""
    global _models_warmed
    if _models_warmed:
        return
    try:
        from vehicle_lpr import detect_vehicle_regions, get_vehicle_model
        from gpu_patch import catcher  # noqa: F401 — 触发 GPU patch + HyperLPR3 加载

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        detect_vehicle_regions(dummy, conf=0.3)
        catcher(dummy)
        _models_warmed = True
        logger.info("vehicle detection + plate recognition models warmed up")
    except Exception as e:
        write_log("camera", "ERROR", f"model warmup failed, will retry on first stream: {e}")


def _check_motion(
    prev_gray: np.ndarray | None,
    curr_gray: np.ndarray,
    bbox: tuple,
    threshold: float = _MOTION_THRESHOLD,
) -> bool:
    """比较前后两帧在 bbox 区域内的平均像素差异来判断运动状态"""
    if prev_gray is None:
        return True  # 首帧默认视为运动
    x1, y1, x2, y2 = map(int, bbox)
    h, w = prev_gray.shape
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return True
    prev_roi = prev_gray[y1:y2, x1:x2]
    curr_roi = curr_gray[y1:y2, x1:x2]
    diff = cv2.absdiff(prev_roi, curr_roi)
    return float(np.mean(diff)) > threshold


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
    _STALL_TIMEOUT_SEC = 15.0   # 单帧间隔超过此值视为卡顿
    _STALL_CONFIRM_SEC = 30.0   # 持续卡顿超过此值才确认断联
    _SLOW_READ_WARN_SEC = 5.0   # cap.read() 耗时超过此值才告警
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    cap = None
    last_open_try = 0.0
    last_frame_time = time.time()
    disconnect_logged = False
    stall_since: float | None = None  # 卡顿开始时间
    camera_id = camera.get("id", "unknown")
    camera_name = camera.get("name") or camera_id
    rtsp_url = camera.get("rtsp_url", "")

    # ── 车辆检测 + 车牌识别缓存 ──
    _cached_results: list = []              # [{bbox, plate_code, confidence, vehicle_confidence, is_moving}, ...]
    _prev_gray: np.ndarray | None = None    # 上一帧灰度图，用于运动检测
    _last_detect_time = 0.0
    _current_cooldown = _COOLDOWN_STATIONARY  # 初始用静止间隔，首帧检测后自适应
    _models_ready = _models_warmed           # 捕获模块级预热状态
    _first_frame_sent = False                # 首帧已推流标记

    # 标记该摄像头正在被查看，纳入健康检查范围
    from backend.services.camera_health_service import mark_camera_active, mark_camera_inactive
    mark_camera_active(camera_id)

    try:
        while True:
            if cap is None or not cap.isOpened():
                now = time.time()
                if now - last_open_try >= 2:
                    last_open_try = now
                    if cap is not None:
                        cap.release()
                    cap = open_capture(rtsp_url)
                    disconnect_logged = False
                    stall_since = None
                    last_frame_time = time.time()
                if cap is None or not cap.isOpened():
                    yield boundary + encode_jpeg(error_frame(f"reconnecting: {camera_name}"), width, quality) + b"\r\n"
                    time.sleep(delay)
                    continue
            t0 = time.time()
            ok, frame = cap.read()
            elapsed = time.time() - t0
            if elapsed > _SLOW_READ_WARN_SEC:
                logger.warning(
                    "slow read: ok=%s frame=%s elapsed=%.1fs name=%s",
                    ok, type(frame).__name__, elapsed, camera_name,
                )

            # ── 硬断联：cap.read() 返回 False 或空帧 ──
            if not ok or frame is None:
                cap.release()
                cap = None
                if not disconnect_logged:
                    disconnect_logged = True
                    write_log("camera", "ERROR", f"Camera timeout url={rtsp_url} name={camera_name}")
                    logger.warning("camera stream disconnected: %s (%s)", camera_name, rtsp_url)
                yield boundary + encode_jpeg(error_frame(f"read failed: {camera_name}"), width, quality) + b"\r\n"
                time.sleep(delay)
                continue

            # ── 软卡顿：帧到达但间隔异常 → 需持续确认才告警 ──
            gap = elapsed if elapsed > _STALL_TIMEOUT_SEC else time.time() - last_frame_time
            if gap > _STALL_TIMEOUT_SEC:
                if stall_since is None:
                    stall_since = time.time()
                    logger.warning(
                        "camera stream stalling: %s (%s) gap=%.0fs (monitoring)",
                        camera_name, rtsp_url, gap,
                    )
                total_stall = time.time() - stall_since
                if total_stall > _STALL_CONFIRM_SEC and not disconnect_logged:
                    disconnect_logged = True
                    write_log("camera", "ERROR", (
                        f"Camera timeout url={rtsp_url} name={camera_name} "
                        f"gap={gap:.0f}s stall_duration={total_stall:.0f}s"
                    ))
                    logger.warning(
                        "camera stream stalled confirmed: %s (%s) gap=%.0fs duration=%.0fs",
                        camera_name, rtsp_url, gap, total_stall,
                    )
            else:
                # 帧正常到达 → 卡顿自动解除，无需告警
                if stall_since is not None:
                    recovered_gap = time.time() - stall_since
                    logger.info(
                        "camera stream recovered from stall: %s (%s) stall_duration=%.0fs",
                        camera_name, rtsp_url, recovered_gap,
                    )
                    stall_since = None

            last_frame_time = time.time()

            # ── 首帧快速通道：先推流再预热，避免黑屏等待 ──
            if not _first_frame_sent:
                _first_frame_sent = True
                yield boundary + encode_jpeg(frame, width, quality) + b"\r\n"
                _warmup_detection_models()
                _models_ready = True
                _last_detect_time = time.time()
                last_frame_time = time.time()  # 重置，避免预热耗时触发误报卡顿
                time.sleep(delay)
                continue

            # ── 车辆检测 + 车牌识别（运动自适应节流）──
            t_now = time.time()
            if t_now - _last_detect_time >= _current_cooldown:
                _last_detect_time = t_now
                try:
                    from vehicle_lpr import recognize_with_vehicle_crops
                    from gpu_patch import catcher

                    accepted, regions = recognize_with_vehicle_crops(
                        frame, catcher,
                        full_frame_mode="never",
                        vehicle_conf=0.3,
                        min_plate_confidence=0.85,
                    )

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    any_moving = False
                    _cached_results = []

                    # 有车牌的车辆
                    for plate in accepted:
                        vbox = plate.get("vehicle_bbox") or plate["bbox"]
                        is_moving = _check_motion(_prev_gray, gray, vbox)
                        if is_moving:
                            any_moving = True
                        _cached_results.append({
                            "bbox": vbox,
                            "plate_code": plate["plate_code"],
                            "confidence": plate["confidence"],
                            "vehicle_confidence": plate.get("vehicle_confidence"),
                            "is_moving": is_moving,
                        })

                    # ── 低置信度车牌写日志 → PlateLowConfidenceRule 消费 ──
                    _LOW_CONF_ALERT_THRESHOLD = 0.98
                    for plate in accepted:
                        if plate["confidence"] < _LOW_CONF_ALERT_THRESHOLD:
                            write_log(
                                "plate", "WARNING",
                                f"plate low confidence camera={camera_id} "
                                f"code={plate['plate_code']} confidence={plate['confidence']:.4f}"
                            )

                    # 未识别到车牌的车辆（排除已被 plate 覆盖的 region）
                    for region in regions:
                        if any(_iou(region.bbox, r["bbox"]) > 0.5 for r in _cached_results):
                            continue
                        is_moving = _check_motion(_prev_gray, gray, region.bbox)
                        if is_moving:
                            any_moving = True
                        _cached_results.append({
                            "bbox": region.bbox,
                            "plate_code": None,
                            "confidence": None,
                            "vehicle_confidence": region.vehicle_confidence,
                            "is_moving": is_moving,
                        })

                    _prev_gray = gray
                    _current_cooldown = _COOLDOWN_MOVING if any_moving else _COOLDOWN_STATIONARY
                except Exception:
                    pass  # 检测失败静默跳过，不影响推流

            # ── 绘制车辆检测框 + 车牌信息 ──
            for result in _cached_results:
                bbox = result["bbox"]
                plate_code = result["plate_code"]
                vehicle_conf = result.get("vehicle_confidence")
                if plate_code:
                    label = f"{plate_code} {result['confidence']:.0%}"
                    color = (0, 210, 70)      # 绿色：已识别车牌
                else:
                    label = f"vehicle {vehicle_conf:.0%}" if vehicle_conf else "vehicle"
                    color = (255, 180, 60)    # 橙色：车辆（未识别车牌）
                _draw_labeled_box(frame, bbox, label, color)

            yield boundary + encode_jpeg(frame, width, quality) + b"\r\n"
            time.sleep(delay)
    finally:
        mark_camera_inactive(camera_id)
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

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
PLATE_VENDOR = PROJECT_DIR / "vendor" / "plate_hyperlpr"
LOCAL_VIDEO_DIR = PROJECT_DIR / "uploads" / "dashcam"
LOCAL_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

DETECT_INTERVAL_SEC = 0.1
DETECT_EXPAND_RATIO = 0.15
DISPLAY_TRACK_TTL_SEC = 0.45
OCR_TASK_MAX_AGE_SEC = 1.2
LOW_TRAFFIC_MAX_VEHICLES = 2
LOW_TRAFFIC_OCR_COOLDOWN = 0.3
HIGH_TRAFFIC_OCR_COOLDOWN = 2.0

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

_models_warmed = False
_warmup_lock = threading.Lock()


def _ensure_vendor_path() -> None:
    if str(PLATE_VENDOR) not in sys.path:
        sys.path.insert(0, str(PLATE_VENDOR))


def warmup_models() -> None:
    global _models_warmed
    if _models_warmed:
        return
    with _warmup_lock:
        if _models_warmed:
            return
        _ensure_vendor_path()
        from gpu_patch import catcher  # type: ignore
        from vehicle_lpr import get_vehicle_model  # type: ignore

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        get_vehicle_model().predict(dummy, classes=[2, 3, 5, 7], imgsz=640, verbose=False)
        catcher(dummy)
        _models_warmed = True


def resolve_video(video_id: str) -> Path | None:
    if not video_id:
        return None
    candidate = (LOCAL_VIDEO_DIR / Path(video_id).name).resolve()
    try:
        candidate.relative_to(LOCAL_VIDEO_DIR.resolve())
    except ValueError:
        return None
    if not candidate.is_file() or candidate.suffix.lower() not in LOCAL_VIDEO_EXTS:
        return None
    return candidate


async def save_upload(upload) -> dict:
    suffix = Path(upload.filename or "video.mp4").suffix.lower()
    if suffix not in LOCAL_VIDEO_EXTS:
        raise ValueError("仅支持 MP4、AVI、MOV、MKV、WEBM 视频")

    LOCAL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    video_id = f"{uuid4().hex}{suffix}"
    path = LOCAL_VIDEO_DIR / video_id
    size = 0
    try:
        with path.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                output.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if not size:
        path.unlink(missing_ok=True)
        raise ValueError("视频文件为空")
    return {"video_id": video_id, "name": upload.filename or video_id, "size_mb": round(size / 1024 / 1024, 1)}


def delete_video(video_id: str) -> None:
    path = resolve_video(video_id)
    if path:
        path.unlink(missing_ok=True)


def _draw_labeled_box(frame, bbox, label: str, color) -> None:
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(label, font, 1.0, 3)
    top = max(0, y1 - text_h - baseline - 16)
    right = min(frame.shape[1] - 1, x1 + text_w + 16)
    cv2.rectangle(frame, (x1, top), (right, y1), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 8, y1 - baseline - 5), font, 1.0, color, 3, cv2.LINE_AA)


class LocalVideoManager:
    """Realtime YOLO tracking and HyperLPR recognition for one uploaded video."""

    def __init__(self) -> None:
        _ensure_vendor_path()
        from video_plate_tracker import VehiclePlateTracker  # type: ignore

        self.cap = None
        self.running = False
        self.fps = 25.0
        self.tracker = VehiclePlateTracker(max_missed=15)
        self.stream_started_at = time.perf_counter()
        self.last_inference_ms = 0.0
        self._latest_frame = None
        self._ocr_queue: queue.Queue = queue.Queue(maxsize=20)
        self._detect_thread = None
        self._recog_thread = None

    def open(self, path: Path) -> bool:
        self.close()
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        from video_plate_tracker import VehiclePlateTracker  # type: ignore

        self.cap = cap
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        self.running = True
        self.tracker = VehiclePlateTracker(max_missed=15)
        self.stream_started_at = time.perf_counter()
        self.last_inference_ms = 0.0
        self._latest_frame = None
        while not self._ocr_queue.empty():
            try:
                self._ocr_queue.get_nowait()
            except queue.Empty:
                break
        self._detect_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._recog_thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self._detect_thread.start()
        self._recog_thread.start()
        return True

    def read_frame(self):
        if not self.cap or not self.running:
            return None, [], 0.0
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.running = False
            return None, [], 0.0
        self._latest_frame = frame.copy()

        timestamp = round(time.perf_counter() - self.stream_started_at, 2)
        tracks = self.tracker.active_tracks(timestamp=timestamp, max_age=DISPLAY_TRACK_TTL_SEC)
        plates = []
        for track in tracks:
            bbox = [int(value) for value in track["bbox"]]
            code = track["plate_code"] or "???"
            confidence = float(track["plate_conf"])
            has_plate = code != "???"
            color = (0, 210, 70) if has_plate else (36, 88, 235)
            label = f"#{track['track_id']} {code}"
            if has_plate:
                label += f" {confidence:.0%}"
            _draw_labeled_box(frame, bbox, label, color)
            plates.append(
                {
                    "code": code,
                    "conf": confidence,
                    "color": PLATE_COLOR_MAP.get(int(track["plate_type"]), "未知"),
                    "bbox": bbox,
                    "track_id": int(track["track_id"]),
                }
            )
        return frame, plates, self.last_inference_ms

    def _detection_loop(self) -> None:
        from vehicle_lpr import Region, expand_box, get_vehicle_model  # type: ignore

        model = get_vehicle_model()
        while self.running:
            frame = self._latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            started = time.perf_counter()
            timestamp = round(time.perf_counter() - self.stream_started_at, 2)
            result = model.predict(frame, classes=[2, 3, 5, 7], imgsz=640, verbose=False, conf=0.3)[0]
            regions = []
            for box in result.boxes:
                bbox = expand_box(box.xyxy[0].tolist(), frame.shape[1], frame.shape[0], DETECT_EXPAND_RATIO)
                regions.append(Region(source="vehicle", bbox=bbox, vehicle_confidence=float(box.conf[0])))

            is_low_traffic = len(regions) <= LOW_TRAFFIC_MAX_VEHICLES
            candidates = self.tracker.update_regions(
                regions,
                timestamp,
                min_hits_for_ocr=1 if is_low_traffic else 2,
                ocr_cooldown=LOW_TRAFFIC_OCR_COOLDOWN if is_low_traffic else HIGH_TRAFFIC_OCR_COOLDOWN,
                stop_ocr_confidence=0.98,
            )
            for track, bbox in candidates:
                x1, y1, x2, y2 = map(int, bbox)
                crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                if crop.size == 0:
                    self.tracker.cancel_ocr(track.track_id, timestamp)
                    continue
                crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                try:
                    self._ocr_queue.put_nowait((track.track_id, crop, bbox, timestamp))
                except queue.Full:
                    self.tracker.cancel_ocr(track.track_id, timestamp)
            self.last_inference_ms = round((time.perf_counter() - started) * 1000, 1)
            time.sleep(DETECT_INTERVAL_SEC)

    def _recognition_loop(self) -> None:
        from gpu_patch import catcher  # type: ignore

        while self.running:
            try:
                track_id, crop, bbox, task_timestamp = self._ocr_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            timestamp = round(time.perf_counter() - self.stream_started_at, 2)
            if timestamp - task_timestamp > OCR_TASK_MAX_AGE_SEC:
                self.tracker.cancel_ocr(track_id, timestamp)
                continue
            results = catcher(crop)
            if results and results[0][0] and len(results[0][0]) >= 4:
                code, confidence, plate_type = results[0][0], float(results[0][1]), int(results[0][2])
            else:
                code, confidence, plate_type = "", 0.0, -1
            self.tracker.assign_plate(
                track_id,
                code,
                confidence,
                plate_type,
                timestamp,
                task_bbox=bbox,
                task_timestamp=task_timestamp,
                max_task_age=OCR_TASK_MAX_AGE_SEC,
            )

    def close(self) -> None:
        self.running = False
        for thread in (self._detect_thread, self._recog_thread):
            if thread and thread.is_alive():
                thread.join(timeout=1)
        if self.cap:
            self.cap.release()
            self.cap = None

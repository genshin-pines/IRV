"""MediaPipe Hand Landmarker adapter used by the driver gesture stream.

The legacy gesture classifier still provides the gesture names and OCSort IDs.
This module adds a sparse hand geometry overlay without changing that
classifier's behaviour.
"""

from __future__ import annotations

import time
import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# A sparse, real landmark graph is enough for the presentation overlay.
DISPLAY_LANDMARK_INDEXES: tuple[int, ...] = (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 4),
    (0, 5), (5, 8),
    (0, 9), (9, 12),
    (0, 13), (13, 16),
    (0, 17), (17, 20),
)


class HandLandmarkDetector:
    """Run MediaPipe Tasks Hand Landmarker in video mode.

    Import and model failures disable only landmark enrichment so the existing
    gesture classifier can continue to serve users who have not installed the
    optional dependency yet.
    """

    def __init__(self, model_path: str | Path, num_hands: int = 2):
        self.model_path = Path(model_path)
        self.detector = None
        self.error = ""
        self._last_timestamp_ms = 0
        self.detect_interval = 8
        self._last_detection_frame = -self.detect_interval
        self._previous_gray: np.ndarray | None = None
        self._tracked_detections: list[dict[str, Any]] = []
        self._force_detection = False
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if not self.model_path.exists():
                raise FileNotFoundError(f"Hand Landmarker model not found: {self.model_path}")
            # The native MediaPipe loader cannot reliably open non-ASCII paths
            # on Windows, so pass model bytes instead of the filesystem path.
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_buffer=self.model_path.read_bytes()),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=num_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:  # pragma: no cover - depends on local native runtime
            self.error = str(exc)
            print(f"[HandLandmarkDetector] disabled: {self.error}")

    @property
    def enabled(self) -> bool:
        return self.detector is not None

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        if self.detector is None or frame is None or frame.size == 0:
            return []

        try:
            import mediapipe as mp

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # Tasks VIDEO mode requires strictly increasing timestamps.
            timestamp_ms = max(int(time.time() * 1000), self._last_timestamp_ms + 1)
            self._last_timestamp_ms = timestamp_ms
            result = self.detector.detect_for_video(image, timestamp_ms)
        except Exception as exc:  # keep the stream alive if a native call fails
            self.error = str(exc)
            return []

        height, width = frame.shape[:2]
        detections: list[dict[str, Any]] = []
        handedness = getattr(result, "handedness", []) or []
        for index, points in enumerate(result.hand_landmarks or []):
            landmarks = []
            xs: list[int] = []
            ys: list[int] = []
            for landmark_index in DISPLAY_LANDMARK_INDEXES:
                point = points[landmark_index]
                x = max(0, min(width - 1, int(round(point.x * width))))
                y = max(0, min(height - 1, int(round(point.y * height))))
                xs.append(x)
                ys.append(y)
                landmarks.append({
                    "index": landmark_index,
                    "x": x,
                    "y": y,
                    "z": round(float(point.z), 5),
                    "x_norm": round(float(point.x), 6),
                    "y_norm": round(float(point.y), 6),
                })

            category = handedness[index][0] if index < len(handedness) and handedness[index] else None
            detections.append({
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "landmarks": landmarks,
                "handedness": str(getattr(category, "category_name", "") or ""),
                "confidence": float(getattr(category, "score", 0.0) or 0.0),
            })
        return detections

    def process(self, frame: np.ndarray, frame_index: int) -> list[dict[str, Any]]:
        """Refresh MediaPipe periodically and track points on intermediate frames."""
        if self.detector is None or frame is None or frame.size == 0:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        should_detect = (
            self._force_detection
            or self._previous_gray is None
            or frame_index - self._last_detection_frame >= self.detect_interval
        )
        if should_detect:
            detections = self.detect(frame)
            self._last_detection_frame = frame_index
            self._previous_gray = gray
            self._tracked_detections = copy.deepcopy(detections)
            self._force_detection = False
            return detections

        if not self._tracked_detections:
            self._previous_gray = gray
            return []

        tracked: list[dict[str, Any]] = []
        for detection in self._tracked_detections:
            previous = np.array(
                [[float(point["x"]), float(point["y"])] for point in detection["landmarks"]],
                dtype=np.float32,
            ).reshape(-1, 1, 2)
            current, status, _error = cv2.calcOpticalFlowPyrLK(
                self._previous_gray,
                gray,
                previous,
                None,
                winSize=(21, 21),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            minimum_tracked = max(5, int(len(previous) * 0.6))
            if current is None or status is None or int(status.sum()) < minimum_tracked:
                self._force_detection = True
                continue

            height, width = frame.shape[:2]
            points = current.reshape(-1, 2)
            landmarks = copy.deepcopy(detection["landmarks"])
            for index, point in enumerate(points):
                x = max(0, min(width - 1, int(round(float(point[0])))))
                y = max(0, min(height - 1, int(round(float(point[1])))))
                landmarks[index]["x"] = x
                landmarks[index]["y"] = y
                landmarks[index]["x_norm"] = round(x / max(width, 1), 6)
                landmarks[index]["y_norm"] = round(y / max(height, 1), 6)
            updated = copy.deepcopy(detection)
            updated["landmarks"] = landmarks
            updated["bbox"] = [
                min(point["x"] for point in landmarks),
                min(point["y"] for point in landmarks),
                max(point["x"] for point in landmarks),
                max(point["y"] for point in landmarks),
            ]
            tracked.append(updated)

        self._previous_gray = gray
        if not tracked:
            self._force_detection = True
            self._tracked_detections = []
            return []
        self._tracked_detections = tracked
        return tracked

    def close(self) -> None:
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass
        self._previous_gray = None
        self._tracked_detections = []

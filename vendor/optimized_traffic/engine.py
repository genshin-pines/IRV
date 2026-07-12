from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn


NAMES = ("无手势", "停止", "直行", "左转弯", "左转弯待转", "右转弯", "变道", "减速慢行", "靠边停车")
COLORS = (
    (148, 163, 184),
    (239, 68, 68),
    (34, 197, 94),
    (59, 130, 246),
    (14, 165, 233),
    (168, 85, 247),
    (245, 158, 11),
    (234, 179, 8),
    (236, 72, 153),
)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(25, 96, num_layers=2, bidirectional=True, dropout=0.15)
        self.head = nn.Sequential(nn.Linear(192, 96), nn.ReLU(), nn.Dropout(0.15), nn.Linear(96, 9))

    def forward(self, x):
        return self.head(self.rnn(x)[0].reshape(-1, 192))


class OptimizedRuntime:
    def __init__(self, project_dir: Path, checkpoint: Path):
        self.project_dir = project_dir.resolve()
        if str(self.project_dir) not in sys.path:
            sys.path.insert(0, str(self.project_dir))

        from constants.enum_keys import HK, PG
        from constants.keypoints import aic_bones
        from pgdataset.s3_handcraft import BoneLengthAngle
        from pred.human_keypoint_pred import HumanKeypointPredict

        previous_cwd = Path.cwd()
        try:
            os.chdir(self.project_dir)
            self.pose = HumanKeypointPredict()
        finally:
            os.chdir(previous_cwd)

        self.HK = HK
        self.PG = PG
        self.bones = [(start - 1, end - 1) for start, end in aic_bones]
        self.features = BoneLengthAngle()
        self.model = Model()
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
        self.model.eval()
        self.lock = threading.Lock()
        self.font = self._load_font(28)

        # Pay model initialization cost before the first uploaded frame.
        self._pose_coordinates(np.zeros((512, 512, 3), np.uint8))

    @staticmethod
    def _load_font(size: int):
        for candidate in (Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")):
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size=size)
        return ImageFont.load_default()

    @staticmethod
    def resize(frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = frame.shape[:2]
        scale = min(512 / width, 512 / height)
        new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
        canvas = np.zeros((512, 512, 3), np.uint8)
        image = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        left, top = (512 - new_width) // 2, (512 - new_height) // 2
        canvas[top:top + new_height, left:left + new_width] = image
        return canvas, scale, left, top

    def _pose_coordinates(self, image: np.ndarray) -> np.ndarray:
        array = np.transpose(image.astype(np.float32) / 255, (2, 0, 1))[None]
        tensor = torch.from_numpy(array).to(self.pose.device)
        with torch.no_grad():
            heatmaps = self.pose.model_pose(tensor)[self.HK.B1_OUT][0].cpu().numpy()
        joints, height, width = heatmaps.shape
        flat = heatmaps.reshape(joints, -1).argmax(1)
        ys, xs = np.divmod(flat, width)
        return np.stack((xs / width, ys / height), axis=0).astype(np.float32)

    def feature_vector(self, coordinates: np.ndarray) -> np.ndarray:
        data = self.features.handcrafted_features(coordinates[None])
        lengths = data[self.PG.BONE_LENGTH].astype(np.float32)
        torso_scale = np.median(lengths[:, 4:8], axis=1, keepdims=True).clip(1e-3)
        return np.concatenate(
            (lengths / torso_scale, data[self.PG.BONE_ANGLE_COS], data[self.PG.BONE_ANGLE_SIN]),
            axis=1,
        )[0].astype(np.float32)

    def infer(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        image, scale, left, top = self.resize(frame)
        with self.lock:
            coordinates = self._pose_coordinates(image)
        native = coordinates.copy()
        native[0] = np.clip((native[0] * 512 - left) / scale, 0, frame.shape[1] - 1)
        native[1] = np.clip((native[1] * 512 - top) / scale, 0, frame.shape[0] - 1)
        return self.feature_vector(coordinates), native, scale

    def annotate(self, frame: np.ndarray, coordinates: np.ndarray, gesture_id: int, confidence: float) -> np.ndarray:
        output = frame.copy()
        color_rgb = COLORS[gesture_id]
        color_bgr = color_rgb[::-1]
        points = [(int(coordinates[0, i]), int(coordinates[1, i])) for i in range(coordinates.shape[1])]
        for start, end in self.bones:
            cv2.line(output, points[start], points[end], color_bgr, 3, cv2.LINE_AA)
        for point in points:
            cv2.circle(output, point, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(output, point, 7, color_bgr, 2, cv2.LINE_AA)

        text = f"手势：{NAMES[gesture_id]}   置信度：{confidence:.0%}"
        image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        box = draw.textbbox((0, 0), text, font=self.font)
        draw.rounded_rectangle((14, 14, box[2] + 38, box[3] + 34), radius=8, fill=(8, 15, 28))
        draw.text((26, 20), text, font=self.font, fill=color_rgb)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


class LiveGestureSession:
    def __init__(self, runtime: OptimizedRuntime, window_size: int = 64, smoothing_window: int = 5):
        self.runtime = runtime
        self.feature_window: deque[np.ndarray] = deque(maxlen=window_size)
        self.logit_window: deque[np.ndarray] = deque(maxlen=smoothing_window)

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        started = time.perf_counter()
        feature, coordinates, _scale = self.runtime.infer(frame)
        self.feature_window.append(feature)
        sequence = torch.from_numpy(np.stack(self.feature_window)[:, None]).float()
        with torch.no_grad():
            logits = self.runtime.model(sequence)[-1].numpy()
        logits[0] += 1.0
        self.logit_window.append(logits)
        smoothed = np.mean(np.stack(self.logit_window), axis=0)
        probabilities = np.exp(smoothed - smoothed.max())
        probabilities /= probabilities.sum()
        gesture_id = int(probabilities.argmax())
        confidence = float(probabilities[gesture_id])
        annotated = self.runtime.annotate(frame, coordinates, gesture_id, confidence)
        return annotated, {
            "id": gesture_id,
            "name": NAMES[gesture_id],
            "confidence": round(confidence, 4),
            "inference_ms": round((time.perf_counter() - started) * 1000, 1),
        }


class LatestCameraCapture:
    """Continuously capture frames while consumers process only the newest one."""

    def __init__(self, camera_index: int = 0, width: int | None = None, height: int | None = None, source_url: str | None = None):
        self.camera_index = camera_index
        self.source_url = source_url
        self.width = width
        self.height = height
        self.cap = None
        self.running = False
        self.error = ""
        self._condition = threading.Condition()
        self._latest_frame = None
        self._sequence = 0
        self._thread = None

    def start(self) -> None:
        self.stop()
        if self.source_url:
            cap = cv2.VideoCapture(self.source_url, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS, 30)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            source = self.source_url or f"本机摄像头 {self.camera_index}"
            raise RuntimeError(f"无法打开视频源 {source}")
        self.cap = cap
        self.running = True
        self.error = ""
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self.running and self.cap is not None:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.error = "摄像头读取失败"
                self.running = False
                with self._condition:
                    self._condition.notify_all()
                break
            with self._condition:
                self._latest_frame = frame
                self._sequence += 1
                self._condition.notify_all()

    def latest(self, after_sequence: int, timeout: float = 1.0) -> tuple[int, np.ndarray | None]:
        deadline = time.perf_counter() + timeout
        with self._condition:
            while self.running and self._sequence <= after_sequence:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            if self._sequence <= after_sequence or self._latest_frame is None:
                return after_sequence, None
            return self._sequence, self._latest_frame.copy()

    def stop(self) -> None:
        self.running = False
        with self._condition:
            self._condition.notify_all()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1)
        self._thread = None

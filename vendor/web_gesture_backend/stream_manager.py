# stream_manager.py - auto-detect ffmpeg
import sys
import time
import queue
import shutil
import subprocess
import logging
import threading
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from gesture_engine import GestureEngine


def find_ffmpeg():
    # 1. 检查 PATH
    found = shutil.which('ffmpeg')
    if found:
        return found
    # 2. 检查项目根目录的 ffmpeg 文件夹
    root = Path(__file__).parent.parent.parent
    for candidate in [
        root / 'tools' / 'ffmpeg' / 'bin' / 'ffmpeg.exe',
        root / 'ffmpeg-master-latest-win64-gpl' / 'bin' / 'ffmpeg.exe',
        Path.home() / 'ffmpeg' / 'bin' / 'ffmpeg.exe',
    ]:
        if candidate.exists():
            return str(candidate)
    # 3. 兜底
    return 'ffmpeg'


FFMPEG_BIN = find_ffmpeg()
FPS = 25
BITRATE = "2M"
_log = logging.getLogger("gesture")


def _inference_worker(
    input_name,
    frame_shape,
    slots,
    input_lock,
    input_queue,
    result_queue,
    user_id,
    trace_path,
):
    """Run model inference away from camera capture and ffmpeg encoding."""
    input_memory = shared_memory.SharedMemory(name=input_name)
    input_frames = np.ndarray((slots, *frame_shape), dtype=np.uint8, buffer=input_memory.buf)
    engine = None

    try:
        engine = GestureEngine(trace_path=Path(trace_path), reset_trace=True)
        if user_id is not None:
            from backend.services.custom_gesture_service import resolve_runtime_binding
            engine.custom_action_resolver = lambda gesture: resolve_runtime_binding(user_id, gesture)

        frame_message = None
        action_messages = []

        def publish_frame(data):
            nonlocal frame_message
            frame_message = data

        def publish_action(data):
            action_messages.append(data)

        engine.on_frame = publish_frame
        engine.on_action = publish_action

        while True:
            item = input_queue.get()
            if item is None:
                break
            # Drop stale work so inference always follows the newest camera frame.
            while True:
                try:
                    item = input_queue.get_nowait()
                except queue.Empty:
                    break
            sequence, slot = item
            with input_lock:
                frame = input_frames[slot].copy()

            frame_message = None
            action_messages = []
            error = ""
            feedback_jpeg = None
            try:
                engine.process_frame(frame, render=False)
                if engine.drawer.feedback:
                    feedback_frame = engine.drawer.draw(frame.copy())
                    encoded, jpeg = cv2.imencode(".jpg", feedback_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if encoded:
                        feedback_jpeg = jpeg.tobytes()
            except Exception as exc:  # keep the capture process alive on model errors
                error = str(exc)

            if frame_message is not None:
                frame_message = dict(frame_message)
                frame_message["frame_id"] = sequence
            result = (sequence, frame_message, action_messages, feedback_jpeg, error)
            try:
                result_queue.put_nowait(result)
            except queue.Full:
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    result_queue.put_nowait(result)
                except queue.Full:
                    pass
    finally:
        if engine is not None:
            engine.close()
        input_memory.close()


class StreamManager:
    def __init__(self, src_url=None, dst_path="gesture", use_webcam=False, camera_index=0, mirror=False, user_id=None, enable_rtsp=True):
        self.src_url = src_url
        self.dst_path = dst_path
        self.dst_url = f"rtsp://127.0.0.1:8554/{dst_path}"
        self.use_webcam = use_webcam
        self.camera_index = camera_index
        self.mirror = mirror
        self.user_id = user_id
        self.enable_rtsp = enable_rtsp

        self.engine = None
        self.cap = None
        self.ffmpeg_proc = None
        self._running = False
        self._thread = None
        self.out_queue = queue.Queue()
        self._error = None
        self._latest_frame = None
        self._latest_raw_jpeg = None
        self._latest_feedback_jpeg = None
        self._feedback_expires_at = 0.0
        self._latest_frame_message = None
        self._frame_count = 0
        self._recent_actions: list[tuple[float, str, bool]] = []
        self._last_jitter_warn_at = 0.0
        self._last_false_trigger_warn_at = 0.0
        self._last_high_freq_warn_at = 0.0
        self._process_context = None
        self._inference_process = None
        self._input_memory = None
        self._input_lock = None
        self._input_queue = None
        self._result_queue = None
        self._frame_shape = None
        self._shared_slots = 3
        self._frame_sequence = 0
        self._latest_output_sequence = -1

    def start(self):
        if self._running:
            return

        if self.use_webcam or not self.src_url:
            print(f"[StreamManager] Using local webcam index={self.camera_index}")
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            print(f"[StreamManager] Connecting: {self.src_url}")
            self.cap = cv2.VideoCapture(self.src_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            self._error = f"Cannot open: {self.src_url or 'webcam'}"
            _log.error("gesture camera open failed: %s", self._error)
            print(f"[StreamManager] {self._error}")
            return

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        print(f"[StreamManager] Resolution: {width}x{height}")
        print(f"[StreamManager] ffmpeg: {FFMPEG_BIN}")

        self._frame_shape = (height, width, 3)
        self._start_inference_process()
        if self._inference_process is None:
            self._create_serial_engine()

        if self.enable_rtsp:
            try:
                self.ffmpeg_proc = self._start_ffmpeg(width, height)
                print(f"[StreamManager] Pushing: {self.dst_url}")
            except Exception as exc:
                self.ffmpeg_proc = None
                print(f"[StreamManager] ffmpeg disabled: {exc}")
        else:
            print("[StreamManager] ffmpeg disabled for low-latency web mode")

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        _log.info("gesture stream started: user_id=%s webcam=%s src=%s", self.user_id, self.use_webcam, self.src_url)
        print("[StreamManager] Started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.ffmpeg_proc:
            try:
                if self.ffmpeg_proc.poll() is None:
                    self.ffmpeg_proc.stdin.close()
                self.ffmpeg_proc.wait(timeout=5)
            except Exception:
                self.ffmpeg_proc.kill()
            self.ffmpeg_proc = None
        if self.engine:
            self.engine.close()
            self.engine = None
        self._stop_inference_process()
        print("[StreamManager] Stopped")
        _log.info("gesture stream stopped: user_id=%s", self.user_id)

    def _loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                _log.error("gesture frame decode failed: cannot read frame from capture device")
                time.sleep(1)
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)
            self._latest_frame = frame
            encoded, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if encoded:
                self._latest_raw_jpeg = jpeg.tobytes()
            if self._inference_process is not None:
                if not self._inference_process.is_alive():
                    _log.error("gesture inference process exited; falling back to serial mode")
                    self._stop_inference_process()
                    self._create_serial_engine()
                self._submit_frame(frame)
            if self._inference_process is not None:
                self._consume_inference_results()
            else:
                try:
                    self.engine.process_frame(frame, render=False)
                    if self.engine.drawer.feedback:
                        feedback_frame = self.engine.drawer.draw(frame.copy())
                        encoded, jpeg = cv2.imencode(".jpg", feedback_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if encoded:
                            self._latest_feedback_jpeg = jpeg.tobytes()
                            self._feedback_expires_at = time.time() + 0.15
                except Exception as exc:
                    _log.error("gesture inference error: %s", exc)

            if self.ffmpeg_proc is None:
                continue
            if self.ffmpeg_proc.poll() is None:
                try:
                    self.ffmpeg_proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, OSError):
                    print("[StreamManager] ffmpeg pipe broken")
                    break
            else:
                print("[StreamManager] ffmpeg exited")
                break

    def _create_serial_engine(self):
        trace_path = Path(__file__).resolve().parents[2] / "logs" / "gesture_static_trace.log"
        self.engine = GestureEngine(trace_path=trace_path, reset_trace=True)
        if self.user_id is not None:
            from backend.services.custom_gesture_service import resolve_runtime_binding
            self.engine.custom_action_resolver = lambda gesture: resolve_runtime_binding(self.user_id, gesture)
        self.engine.on_frame = self._publish_frame
        self.engine.on_action = self._on_action_logged

    def _start_inference_process(self):
        try:
            self._process_context = mp.get_context("spawn")
            frame_bytes = int(np.prod(self._frame_shape))
            self._input_memory = shared_memory.SharedMemory(create=True, size=frame_bytes * self._shared_slots)
            self._input_lock = self._process_context.Lock()
            self._input_queue = self._process_context.Queue(maxsize=2)
            self._result_queue = self._process_context.Queue(maxsize=2)
            trace_path = Path(__file__).resolve().parents[2] / "logs" / "gesture_static_trace.log"
            self._inference_process = self._process_context.Process(
                target=_inference_worker,
                args=(
                    self._input_memory.name,
                    self._frame_shape,
                    self._shared_slots,
                    self._input_lock,
                    self._input_queue,
                    self._result_queue,
                    self.user_id,
                    str(trace_path),
                ),
                daemon=True,
            )
            self._inference_process.start()
            print("[StreamManager] Inference process started (latest-frame mode)")
        except Exception as exc:
            _log.error("gesture inference process unavailable, using serial mode: %s", exc)
            self._stop_inference_process()

    def _stop_inference_process(self):
        process = self._inference_process
        if process is not None:
            try:
                if self._input_queue is not None:
                    self._input_queue.put_nowait(None)
            except Exception:
                pass
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for q in (self._input_queue, self._result_queue):
            if q is not None:
                try:
                    q.close()
                    q.join_thread()
                except Exception:
                    pass
        for memory in (self._input_memory,):
            if memory is not None:
                try:
                    memory.close()
                    memory.unlink()
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
        self._inference_process = None
        self._input_memory = None
        self._input_queue = None
        self._result_queue = None

    def _submit_frame(self, frame):
        if self._input_memory is None or self._input_queue is None:
            return
        self._frame_sequence += 1
        sequence = self._frame_sequence
        slot = sequence % self._shared_slots
        frames = np.ndarray(
            (self._shared_slots, *self._frame_shape), dtype=np.uint8, buffer=self._input_memory.buf
        )
        with self._input_lock:
            frames[slot][:] = frame
        try:
            self._input_queue.put_nowait((sequence, slot))
        except queue.Full:
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._input_queue.put_nowait((sequence, slot))
            except queue.Full:
                pass

    def _consume_inference_results(self):
        if self._result_queue is None:
            return
        latest = None
        while True:
            try:
                latest = self._result_queue.get_nowait()
            except queue.Empty:
                break
        if latest is None:
            return
        sequence, frame_message, actions, feedback_jpeg, error = latest
        if error:
            _log.error("gesture inference error: %s", error)
        self._latest_output_sequence = sequence
        if feedback_jpeg is not None:
            self._latest_feedback_jpeg = feedback_jpeg
            self._feedback_expires_at = time.time() + 0.15
        if frame_message is not None:
            self._publish_frame(frame_message)
        for action in actions:
            self._on_action_logged(action)

    def _start_ffmpeg(self, width, height):
        cmd = [
            FFMPEG_BIN,
            "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(FPS),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-b:v", BITRATE,
            "-f", "rtsp",
            self.dst_url,
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=sys.stderr)

    def get_latest_frame(self):
        import copy
        f = self._latest_frame
        return copy.deepcopy(f) if f is not None else None

    def get_latest_raw_jpeg(self):
        if self._latest_feedback_jpeg is not None and time.time() < self._feedback_expires_at:
            return self._latest_feedback_jpeg
        return self._latest_raw_jpeg

    def get_latest_frame_message(self):
        import copy
        return copy.deepcopy(self._latest_frame_message) if self._latest_frame_message is not None else None

    def _publish_frame(self, data):
        self._latest_frame_message = data
        self.out_queue.put(("frame", data))
        self._frame_count += 1
        if self._frame_count % 60 != 0:
            return
        hands = data.get("hands") or []
        if not hands:
            return
        gestures = [hand.get("gesture", "unknown") for hand in hands]
        confidences = [hand.get("confidence", 1.0) for hand in hands]
        min_conf = min(confidences)
        _log.info("gesture frame: type=%s, hands=%d, min_confidence=%.2f", gestures[0], len(hands), min_conf)
        if min_conf < 0.98:
            _log.warning("gesture confidence low: min_confidence=%.2f, type=%s", min_conf, gestures[0])

    def _on_action_logged(self, data: dict) -> None:
        """Emit only actionable stream anomalies for Alert Agent consumption."""
        self.out_queue.put(("action", data))
        gesture_type = str(data.get("gesture_action") or data.get("gesture") or "unknown")
        action_applied = bool(data.get("action_applied", False))
        now = time.time()
        self._recent_actions.append((now, gesture_type, action_applied))
        if len(self._recent_actions) > 20:
            self._recent_actions = self._recent_actions[-20:]

        if now - self._last_jitter_warn_at >= 10.0 and len(self._recent_actions) >= 5:
            recent_types = [item[1] for item in self._recent_actions[-5:]]
            changes = sum(previous != current for previous, current in zip(recent_types, recent_types[1:]))
            if changes >= 2:
                self._last_jitter_warn_at = now
                _log.warning("gesture jitter detected: recent_actions=%s, changes=%d/4", recent_types, changes)

        if now - self._last_false_trigger_warn_at >= 10.0 and len(self._recent_actions) >= 10:
            recent = self._recent_actions[-10:]
            suppressed = sum(1 for _, _, applied in recent if not applied)
            if suppressed / len(recent) >= 0.5:
                self._last_false_trigger_warn_at = now
                _log.warning("gesture false trigger risk: stable=false, suppressed=%d/10", suppressed)

        if now - self._last_high_freq_warn_at >= 10.0:
            count = sum(1 for timestamp, _, _ in self._recent_actions if now - timestamp <= 10.0)
            if count >= 8:
                self._last_high_freq_warn_at = now
                _log.warning("gesture unstable high frequency: stable=false, action_count=%d in 10s", count)

    @property
    def is_running(self):
        return self._running

    @property
    def error(self):
        return self._error

    @property
    def hls_url(self):
        return f"http://127.0.0.1:8889/{self.dst_path}" if self.enable_rtsp else None

# stream_manager.py - auto-detect ffmpeg
import sys
import time
import queue
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

import cv2

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


class StreamManager:
    def __init__(self, src_url=None, dst_path="gesture", use_webcam=False, camera_index=0, mirror=False):
        self.src_url = src_url
        self.dst_path = dst_path
        self.dst_url = f"rtsp://127.0.0.1:8554/{dst_path}"
        self.use_webcam = use_webcam
        self.camera_index = camera_index
        self.mirror = mirror

        self.engine = None
        self.cap = None
        self.ffmpeg_proc = None
        self._running = False
        self._thread = None
        self.out_queue = queue.Queue()
        self._error = None
        self._latest_frame = None

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
            print(f"[StreamManager] {self._error}")
            return

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        print(f"[StreamManager] Resolution: {width}x{height}")
        print(f"[StreamManager] ffmpeg: {FFMPEG_BIN}")

        self.engine = GestureEngine()
        self.engine.on_frame = lambda d: self.out_queue.put(("frame", d))
        self.engine.on_action = lambda d: self.out_queue.put(("action", d))

        try:
            self.ffmpeg_proc = self._start_ffmpeg(width, height)
            print(f"[StreamManager] Pushing: {self.dst_url}")
        except Exception as exc:
            self.ffmpeg_proc = None
            print(f"[StreamManager] ffmpeg disabled: {exc}")

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
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
        print("[StreamManager] Stopped")

    def _loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(1)
                continue
            if self.mirror:
                frame = cv2.flip(frame, 1)
            try:
                annotated = self.engine.process_frame(frame)
                self._latest_frame = annotated
            except Exception as e:
                print(f"[StreamManager] Inference error: {e}")
                continue

            if self.ffmpeg_proc is None:
                continue
            if self.ffmpeg_proc and self.ffmpeg_proc.poll() is None:
                try:
                    self.ffmpeg_proc.stdin.write(annotated.tobytes())
                except (BrokenPipeError, OSError):
                    print("[StreamManager] ffmpeg pipe broken")
                    break
            else:
                print("[StreamManager] ffmpeg exited")
                break

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

    @property
    def is_running(self):
        return self._running

    @property
    def error(self):
        return self._error

    @property
    def hls_url(self):
        return f"http://127.0.0.1:8889/{self.dst_path}"

"""
沙盘摄像头实时车牌识别服务
启动: python live_server.py
访问: http://localhost:8004
"""
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|allowed_media_types;video"
os.environ["OPENCV_FFMPEG_THREADS"] = "1"

import asyncio
import base64
import json
import queue
import time
import cv2
import threading
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from pathlib import Path

from gpu_patch import catcher  # GPU 加速版
from vehicle_lpr import get_vehicle_model, Region, expand_box
from video_plate_tracker import VehiclePlateTracker

app = FastAPI(title="实时车牌识别")

LIVE_RECOGNITION_INTERVAL_SEC = 0.5
LIVE_BOX_TTL_SEC = 0.25
LIVE_RESULT_TTL_SEC = 2.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@app.on_event("startup")
async def warmup_models():
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    get_vehicle_model().predict(dummy, classes=[2, 3, 5, 7], imgsz=640, verbose=False)
    catcher(dummy)

# ── 摄像头列表 ────────────────────────────────────────────
CAMERAS = [
    {"id": "0",  "name": "本机摄像头",  "url": "0"},
    # ═══ 沙盘 RTSP（需内网 10.126.59.x）═══
    {"id": "1",  "name": "桥面",       "url": "rtsp://10.126.59.120:8554/live/live1"},
    {"id": "2",  "name": "停车场出口", "url": "rtsp://10.126.59.120:8554/live/live2"},
    {"id": "3",  "name": "行人检测",   "url": "rtsp://10.126.59.120:8554/live/live3"},
    {"id": "4",  "name": "消防车识别", "url": "rtsp://10.126.59.120:8554/live/live4"},
    {"id": "5",  "name": "桥出口",     "url": "rtsp://10.126.59.120:8554/live/live5"},
    {"id": "6",  "name": "桥入口",     "url": "rtsp://10.126.59.120:8554/live/live6"},
    {"id": "7",  "name": "道路2",      "url": "rtsp://10.126.59.120:8554/live/live7"},
    {"id": "8",  "name": "隧道(事故)", "url": "rtsp://10.126.59.120:8554/live/live8"},
    {"id": "9",  "name": "隧道(车载)", "url": "rtsp://10.126.59.120:8554/live/live9"},
    {"id": "10", "name": "道路1",       "url": "rtsp://10.126.59.120:8554/live/live10"},
    {"id": "11", "name": "停车场入口",  "url": "rtsp://10.126.59.120:8554/live/live11"},
    {"id": "12", "name": "道路1",       "url": "rtsp://10.126.59.120:8554/live/live12"},
]

PLATE_COLOR_MAP = {
    -1: "未知", 0: "蓝牌", 1: "黄牌(单层)", 2: "白牌(单层)",
    3: "绿牌(新能源)", 4: "黑牌(港澳)", 5: "香港(单层)",
    6: "香港(双层)", 7: "澳门(单层)", 8: "澳门(双层)", 9: "黄牌(双层)",
}


DETECT_INTERVAL_SEC = 0.1       # YOLO 检测频率
DETECT_EXPAND_RATIO = 0.15      # 裁切时向外扩展比例
DISPLAY_TRACK_TTL_SEC = 0.45    # 检测框太旧就不画，避免滞后框
OCR_TASK_MAX_AGE_SEC = 1.2      # OCR 迟到太久就丢弃，避免错绑
LOW_TRAFFIC_MAX_VEHICLES = 2    # 少车场景更积极 OCR
LOW_TRAFFIC_OCR_COOLDOWN = 0.3
HIGH_TRAFFIC_OCR_COOLDOWN = 2.0
BOX_THICKNESS = 4
LABEL_FONT_SCALE = 1.1
LABEL_THICKNESS = 3


def draw_labeled_box(frame, bbox, label: str, color):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        font,
        LABEL_FONT_SCALE,
        LABEL_THICKNESS,
    )
    pad_x, pad_y = 8, 6
    bg_x1 = x1
    bg_y2 = y1 - 6
    bg_y1 = bg_y2 - text_h - baseline - pad_y * 2
    if bg_y1 < 0:
        bg_y1 = y1 + 6
        bg_y2 = bg_y1 + text_h + baseline + pad_y * 2
    bg_x2 = min(frame.shape[1] - 1, bg_x1 + text_w + pad_x * 2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)

    text_org = (bg_x1 + pad_x, bg_y2 - baseline - pad_y)
    cv2.putText(
        frame,
        label,
        text_org,
        font,
        LABEL_FONT_SCALE,
        (255, 255, 255),
        LABEL_THICKNESS + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        text_org,
        font,
        LABEL_FONT_SCALE,
        color,
        LABEL_THICKNESS,
        cv2.LINE_AA,
    )

class StreamManager:
    """三线程架构：画面线程 + 检测线程(YOLO) + 识别线程(HyperLPR3)"""

    def __init__(self):
        self.cap = None
        self.current_cam = None
        self.running = False
        self.tracker = VehiclePlateTracker(max_missed=15)
        self.stream_started_at = time.perf_counter()
        self.last_inference_ms = 0
        # ── 共享状态 ──
        self._latest_frame = None
        self._ocr_queue = queue.Queue(maxsize=20)
        self._detect_thread = None
        self._recog_thread = None
        self._detect_running = False
        self._recog_running = False

    def open(self, url: str, cam_name: str):
        self.close()
        time.sleep(0.5)

        if url.isdigit():
            cap = cv2.VideoCapture(int(url), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"  [DEBUG] Resolution: {w}x{h}")
                self.cap = cap
                self.current_cam = cam_name
                self.running = True
                self.tracker = VehiclePlateTracker(max_missed=15)
                self.stream_started_at = time.perf_counter()
                self.last_inference_ms = 0
                self._latest_frame = None
                # 清空 OCR 队列
                while not self._ocr_queue.empty():
                    try: self._ocr_queue.get_nowait()
                    except queue.Empty: break
                # 启动两个后台线程
                self._detect_running = True
                self._recog_running = True
                self._detect_thread = threading.Thread(target=self._detection_loop, daemon=True)
                self._recog_thread = threading.Thread(target=self._recognition_loop, daemon=True)
                self._detect_thread.start()
                self._recog_thread.start()
                return True
            cap.release()
        print(f"  [DEBUG] Failed to open stream")
        return False

    # ═══ 画面线程 ═══

    def read_frame(self):
        """只读帧 + 画 tracker 实时位置，不做推理。"""
        if not self.cap or not self.running:
            return None, [], [], 0
        ret, frame = self.cap.read()
        if not ret:
            return None, [], [], 0

        # 喂帧给检测线程
        self._latest_frame = frame.copy()

        # 从 tracker 拿足够新的轨道画框；旧框宁可不画，避免明显滞后。
        timestamp = round(time.perf_counter() - self.stream_started_at, 2)
        tracks = self.tracker.active_tracks(timestamp=timestamp, max_age=DISPLAY_TRACK_TTL_SEC)
        plates = []
        overlay_plates = []
        for t in tracks:
            x1, y1, x2, y2 = map(int, t["bbox"])
            if x2 <= x1 or y2 <= y1:
                continue
            has_plate = t["plate_code"] and len(t["plate_code"]) >= 4
            color = (0, 0, 255) if not has_plate else (0, 255, 0)  # 红=待识别，绿=已有车牌

            label = f"#{t['track_id']}"
            if has_plate:
                label += f" {t['plate_code']} ({t['plate_conf']:.0%})"
            draw_labeled_box(frame, (x1, y1, x2, y2), label, color)

            plates.append({
                "code": t["plate_code"] or "???",
                "conf": t["plate_conf"],
                "color": PLATE_COLOR_MAP.get(t["plate_type"], "未知"),
                "bbox": [x1, y1, x2, y2],
                "track_id": t["track_id"],
            })
            overlay_plates.append({
                "code": t["plate_code"] or "???",
                "conf": t["plate_conf"],
                "bbox": [x1, y1, x2, y2],
                "track_id": t["track_id"],
            })

        return frame, plates, overlay_plates, self.last_inference_ms

    # ═══ 检测线程 (YOLO, ~10Hz) ═══

    def _detection_loop(self):
        """快速循环：YOLO 找车 → 更新 tracker 位置 → 推裁切图到 OCR 队列。"""
        vehicle_model = get_vehicle_model()
        while self._detect_running and self.running:
            frame = self._latest_frame
            if frame is None:
                time.sleep(0.05)
                continue

            t0 = time.perf_counter()
            timestamp = round(time.perf_counter() - self.stream_started_at, 2)

            results = vehicle_model.predict(frame, classes=[2, 3, 5, 7],
                                            imgsz=640, verbose=False, conf=0.3)
            boxes = results[0].boxes

            # 构建 Region 列表送给 tracker
            regions = []

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                x1e, y1e, x2e, y2e = expand_box(
                    (x1, y1, x2, y2),
                    frame.shape[1],
                    frame.shape[0],
                    DETECT_EXPAND_RATIO,
                )
                regions.append(Region(source="vehicle", bbox=(x1e, y1e, x2e, y2e),
                                      vehicle_confidence=conf))

            vehicle_count = len(regions)
            if vehicle_count <= LOW_TRAFFIC_MAX_VEHICLES:
                min_hits_for_ocr = 1
                ocr_cooldown = LOW_TRAFFIC_OCR_COOLDOWN
            else:
                min_hits_for_ocr = 2
                ocr_cooldown = HIGH_TRAFFIC_OCR_COOLDOWN

            ocr_candidates = self.tracker.update_regions(
                regions,
                timestamp,
                min_hits_for_ocr=min_hits_for_ocr,
                ocr_cooldown=ocr_cooldown,
                stop_ocr_confidence=0.95,
            )

            # 需要 OCR 的：裁切2倍放大，丢进队列
            for track, bbox in ocr_candidates:
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                try:
                    self._ocr_queue.put_nowait((track.track_id, crop, bbox, timestamp))
                except queue.Full:
                    self.tracker.cancel_ocr(track.track_id, timestamp)
                    pass  # 队列满了就丢弃

            self.last_inference_ms = round((time.perf_counter() - t0) * 1000, 1)
            time.sleep(DETECT_INTERVAL_SEC)

    # ═══ 识别线程 (HyperLPR3, 队列驱动) ═══

    def _recognition_loop(self):
        """从队列取裁切图，跑 HyperLPR3，结果绑定到 track_id。"""
        while self._recog_running and self.running:
            try:
                track_id, crop, bbox, task_timestamp = self._ocr_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            now = round(time.perf_counter() - self.stream_started_at, 2)
            if now - task_timestamp > OCR_TASK_MAX_AGE_SEC:
                self.tracker.cancel_ocr(track_id, now)
                continue

            results = catcher(crop)
            timestamp = round(time.perf_counter() - self.stream_started_at, 2)
            if results:
                best = results[0]
                code, conf, ptype = best[0], float(best[1]), int(best[2])
                if code and len(code) >= 4:
                    self.tracker.assign_plate(
                        track_id,
                        code,
                        conf,
                        ptype,
                        timestamp,
                        task_bbox=bbox,
                        task_timestamp=task_timestamp,
                        max_task_age=OCR_TASK_MAX_AGE_SEC,
                    )
                else:
                    self.tracker.assign_plate(
                        track_id,
                        "",
                        0.0,
                        -1,
                        timestamp,
                        task_bbox=bbox,
                        task_timestamp=task_timestamp,
                        max_task_age=OCR_TASK_MAX_AGE_SEC,
                    )
            else:
                self.tracker.assign_plate(
                    track_id,
                    "",
                    0.0,
                    -1,
                    timestamp,
                    task_bbox=bbox,
                    task_timestamp=task_timestamp,
                    max_task_age=OCR_TASK_MAX_AGE_SEC,
                )

    # ═══ 清理 ═══

    def close(self):
        self.running = False
        self._detect_running = False
        self._recog_running = False
        for t in [self._detect_thread, self._recog_thread]:
            if t and t.is_alive():
                t.join(timeout=1)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.current_cam = None


# ─── 前端页面 ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>实时车牌识别</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif;
         background: #0f1923; color: #cdd6e0; height: 100vh; display: flex; }
  /* ── 左侧面板 ── */
  .sidebar { width: 260px; background: #1a2733; padding: 16px;
             display: flex; flex-direction: column; gap: 8px;
             border-right: 1px solid #2a3a4a; overflow-y: auto; }
  .sidebar h2 { font-size: 16px; color: #5b9cf5; margin-bottom: 8px; }
  .cam-btn { display: block; width: 100%; padding: 10px 14px;
             border: 1px solid #2a3a4a; border-radius: 8px;
             background: #1e2e3d; color: #bcc8d4; font-size: 13px;
             cursor: pointer; text-align: left; transition: .15s; }
  .cam-btn:hover { background: #253545; border-color: #3a5a7a; }
  .cam-btn.active { background: #1a3a5c; border-color: #5b9cf5; color: #fff; font-weight: 600; }
  .cam-btn .id { color: #5b9cf5; font-weight: 700; margin-right: 6px; }
  .local-live-link { display: block; width: 100%; padding: 12px 14px;
             border: 1px solid #3478d4; border-radius: 8px;
             background: #16324a; color: #e8f2ff; font-size: 13px;
             text-decoration: none; font-weight: 700; transition: .15s; }
  .local-live-link:hover { background: #1c4264; border-color: #5b9cf5; }
  .local-live-link small { display: block; color: #8aa9c8; font-size: 11px;
             font-weight: 400; margin-top: 4px; }
  .status { margin-top: auto; padding: 10px; background: #12202b;
            border-radius: 8px; font-size: 12px; color: #7a8a9a; }
  .status .dot { display: inline-block; width: 8px; height: 8px;
                 border-radius: 50%; margin-right: 6px; }
  .dot.live { background: #4caf50; box-shadow: 0 0 6px #4caf50; }
  .dot.dead { background: #f44336; }

  /* ── 主区域 ── */
  .main { flex: 1; display: flex; flex-direction: column; }
  .topbar { padding: 12px 24px; background: #1a2733;
            display: flex; align-items: center; gap: 16px;
            border-bottom: 1px solid #2a3a4a; font-size: 14px; }
  .topbar .cam-name { font-size: 18px; font-weight: 700; color: #e8edf2; }
  .topbar .fps { color: #7a8a9a; font-size: 12px; }
  .topbar .inference { color: #4caf50; font-size: 13px; margin-left: auto; }

  /* ── 视频画布 ── */
  .video-area { flex: 1; display: flex; align-items: center; justify-content: center;
                background: #0a1219; position: relative; }
  .video-area canvas { max-width: 100%; max-height: 100%; }
  .no-signal { position: absolute; color: #4a5a6a; font-size: 24px; pointer-events: none; }

  /* ── 底部识别列表 ── */
  .plates-bar { background: #1a2733; border-top: 1px solid #2a3a4a;
                padding: 8px 16px; display: flex; gap: 10px; flex-wrap: wrap;
                min-height: 52px; align-items: center; overflow-x: auto; }
  .plates-bar .empty { color: #4a5a6a; font-size: 13px; }
  .plate-tag { background: #1e3a2a; border: 1px solid #2e5a3a;
               padding: 6px 14px; border-radius: 6px; font-size: 14px;
               font-weight: 700; color: #c8e6c9; white-space: nowrap; }
  .plate-tag .conf { font-weight: 400; font-size: 11px; color: #7ab87a; margin-left: 4px; }
</style>
</head>
<body>

<!-- 左侧摄像头列表 -->
<div class="sidebar">
  <h2>沙盘摄像头</h2>
  <div id="camList"></div>
  <a class="local-live-link" href="http://localhost:8003/local-live" target="_blank" rel="noopener">
    本地视频实时检测
    <small>跳转到 8003 调试页</small>
  </a>
  <div class="status">
    <span class="dot" id="statusDot"></span>
    <span id="statusText">等待连接...</span>
  </div>
</div>

<!-- 主区域 -->
<div class="main">
  <div class="topbar">
    <span class="cam-name" id="camName">-</span>
    <span class="fps" id="fpsInfo">-</span>
    <span class="inference" id="inferInfo">-</span>
  </div>
  <div class="video-area" id="videoArea">
    <canvas id="canvas"></canvas>
    <div class="no-signal" id="noSignal">点击左侧摄像头开始</div>
  </div>
  <div class="plates-bar" id="platesBar">
    <span class="empty">等待识别结果...</span>
  </div>
</div>

<script>
const CAMERAS = """ + json.dumps([{"id": c["id"], "name": c["name"]} for c in CAMERAS], ensure_ascii=False) + """;

let ws = null;
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let currentCam = null;
let frameCount = 0;
let lastFpsTime = Date.now();
let fpsVal = 0;

function renderCamList() {
  document.getElementById('camList').innerHTML = CAMERAS.map(c =>
    `<button class="cam-btn" data-id="${c.id}" onclick="switchCam('${c.id}')">
      <span class="id">#${c.id}</span>${c.name}
    </button>`
  ).join('');
}

function switchCam(id) {
  currentCam = id;
  document.querySelectorAll('.cam-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.id === id));
  document.getElementById('noSignal').style.display = 'block';
  document.getElementById('camName').textContent =
    CAMERAS.find(c => c.id === id)?.name || '-';

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: 'switch', camera: id }));
  } else {
    connectWS();
  }
}

function connectWS() {
  if (ws) ws.close();
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`);
  ws.onopen = () => {
    setStatus(true, '已连接');
    if (currentCam) ws.send(JSON.stringify({action:'switch',camera:currentCam}));
  };
  ws.onclose = () => setStatus(false, '已断开');
  ws.onerror = () => setStatus(false, '连接失败');

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'frame') {
      const img = new Image();
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        document.getElementById('noSignal').style.display = 'none';
        frameCount++;
        const now = Date.now();
        if (now - lastFpsTime >= 1000) {
          fpsVal = frameCount; frameCount = 0; lastFpsTime = now;
        }
      };
      img.src = 'data:image/jpeg;base64,' + msg.frame;
      document.getElementById('fpsInfo').textContent = fpsVal + ' FPS';
      document.getElementById('inferInfo').textContent =
        '推理: ' + (msg.inference_ms || '?') + 'ms';

      const bar = document.getElementById('platesBar');
      if (msg.plates && msg.plates.length) {
        bar.innerHTML = msg.plates.map(p =>
          `<span class="plate-tag">#${p.track_id || '-'} ${p.code}
            <span class="conf">${(p.conf*100).toFixed(0)}%</span>
          </span>`
        ).join('');
      } else {
        bar.innerHTML = '<span class="empty">未检测到车牌</span>';
      }
    }
  };
}

function setStatus(live, text) {
  document.getElementById('statusDot').className = 'dot ' + (live ? 'live' : 'dead');
  document.getElementById('statusText').textContent = text;
}

renderCamList();
</script>
</body>
</html>"""

# ─── WebSocket 端点 ───────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    manager = StreamManager()
    send_queue = queue.Queue(maxsize=5)  # 线程安全队列
    reader_thread = None
    stop_event = threading.Event()

    def reader():
        """画面线程：读帧 → JPEG 编码 → WebSocket 推流。"""
        frame_count = 0
        empty_count = 0
        t0 = time.time()
        while not stop_event.is_set() and manager.running:
            frame, plates, overlay_plates, infer_ms = manager.read_frame()
            frame_count += 1

            if frame is None:
                empty_count += 1
                time.sleep(0.3)
                if time.time() - t0 > 5:
                    print(f"  [DEBUG] {frame_count} read attempts, {empty_count} empty frames")
                    t0 = time.time()
                continue

            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(jpeg).decode()

            payload = {
                "type": "frame",
                "frame": b64,
                "plates": plates,
                "inference_ms": infer_ms,
            }
            try:
                send_queue.put_nowait(payload)
            except queue.Full:
                try:
                    send_queue.get_nowait()
                    send_queue.put_nowait(payload)
                except queue.Empty:
                    pass

            if frame_count % 150 == 0 and plates:
                print(f"  [DEBUG] frame #{frame_count}: {len(plates)} plates: {[p['code'] for p in plates]}")

            time.sleep(0.03)

    async def sender():
        """异步任务：发送队列中的帧"""
        while not stop_event.is_set():
            try:
                payload = send_queue.get_nowait()
                await ws.send_json(payload)
            except queue.Empty:
                await asyncio.sleep(0.03)
            except Exception:
                break

    send_task = asyncio.create_task(sender())

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("action") == "switch":
                cam_id = msg["camera"]
                cam = next((c for c in CAMERAS if c["id"] == cam_id), None)
                if cam:
                    print(f"[WS] Switch to #{cam_id}: {cam['name']}")
                    manager.close()
                    if reader_thread and reader_thread.is_alive():
                        reader_thread.join(timeout=1)
                    while not send_queue.empty():
                        try: send_queue.get_nowait()
                        except queue.Empty: break

                    if manager.open(cam["url"], cam["name"]):
                        print(f"  Connected")
                        reader_thread = threading.Thread(target=reader, daemon=True)
                        reader_thread.start()
                        await ws.send_json({"type": "status", "camera": cam["name"], "connected": True})
                    else:
                        await ws.send_json({"type": "error", "msg": f"Can not connect: {cam['name']}"})

    except WebSocketDisconnect:
        pass
    finally:
        stop_event.set()
        manager.close()
        send_task.cancel()
        print("[WS] Disconnected")


if __name__ == "__main__":
    print("\n沙盘实时车牌识别服务")
    print("访问: http://localhost:8004\n")
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="warning")

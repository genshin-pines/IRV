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

from gpu_patch import catcher  # GPU 加速版

app = FastAPI(title="实时车牌识别")

# ── 摄像头列表 ────────────────────────────────────────────
CAMERAS = [
    # ═══ 本地测试 ═══
    {"id": "0",  "name": "本机摄像头",  "url": "0"},
    # ═══ 沙盘 RTSP（需内网 10.126.59.x）═══
    {"id": "1",  "name": "桥面",       "url": "rtsp://10.126.59.120:8554/live/live1"},
    {"id": "2",  "name": "停车场出口", "url": "rtsp://10.126.59.120:8554/live/live2"},
    {"id": "3",  "name": "行人检测",   "url": "rtsp://10.126.59.120:8554/live/live3"},
    {"id": "4",  "name": "消防车识别", "url": "rtsp://10.126.59.120:8554/live/live4"},
    {"id": "5",  "name": "桥出口",     "url": "rtsp://10.126.59.120:8554/live/live5"},
    {"id": "6",  "name": "桥入口",     "url": "rtsp://10.126.59.120:8554/live/live6"},
    {"id": "7",  "name": "道路2",      "url": "rtsp://10.126.59.120:8554/live/live7"},
    {"id": "8",  "name": "隧道(事故)",  "url": "rtsp://10.126.59.120:8554/live/live8"},
    {"id": "9",  "name": "隧道(车载)",  "url": "rtsp://10.126.59.120:8554/live/live9"},
    {"id": "10", "name": "道路1",       "url": "rtsp://10.126.59.120:8554/live/live10"},
    {"id": "11", "name": "停车场入口",  "url": "rtsp://10.126.59.120:8554/live/live11"},
    {"id": "12", "name": "道路1",       "url": "rtsp://10.126.59.120:8554/live/live12"},
]

PLATE_COLOR_MAP = {
    -1: "未知", 0: "蓝牌", 1: "黄牌(单层)", 2: "白牌(单层)",
    3: "绿牌(新能源)", 4: "黑牌(港澳)", 5: "香港(单层)",
    6: "香港(双层)", 7: "澳门(单层)", 8: "澳门(双层)", 9: "黄牌(双层)",
}


class StreamManager:
    """管理每个 WebSocket 连接的流状态"""

    def __init__(self):
        self.cap = None
        self.current_cam = None
        self.running = False
        self.lock = threading.Lock()

    def open(self, url: str, cam_name: str):
        self.close()
        time.sleep(0.5)

        # 本地摄像头 vs RTSP
        if url.isdigit():
            cap = cv2.VideoCapture(int(url), cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"  [DEBUG] Resolution: {w}x{h}, dtype: {frame.dtype}, channels: {frame.shape[2] if len(frame.shape)>2 else 1}")
                self.cap = cap
                self.current_cam = cam_name
                self.running = True
                return True
            else:
                print(f"  [DEBUG] cap opened but read() returned False")
            cap.release()
        else:
            print(f"  [DEBUG] cap.isOpened() = False")
        return False

    def read_frame(self):
        """读取一帧，同时跑识别"""
        if not self.cap or not self.running:
            return None, [], 0
        ret, frame = self.cap.read()
        if not ret:
            return None, [], 0

        # GPU 推理
        t0 = time.perf_counter()
        results = catcher(frame)
        elapsed = round((time.perf_counter() - t0) * 1000, 1)

        # 解析结果
        plates = []
        for r in results:
            plates.append({
                "code": r[0],
                "conf": round(float(r[1]), 3),
                "color": PLATE_COLOR_MAP.get(int(r[2]), "未知"),
                "bbox": [int(v) for v in r[3]],
            })

        # 在帧上画框
        for p in plates:
            x1, y1, x2, y2 = p["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{p['code']} ({p['conf']:.0%})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame, plates, elapsed

    def close(self):
        self.running = False
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

// ── 渲染摄像头列表 ──
function renderCamList() {
  const container = document.getElementById('camList');
  container.innerHTML = CAMERAS.map(c =>
    `<button class="cam-btn" data-id="${c.id}" onclick="switchCam('${c.id}')">
      <span class="id">#${c.id}</span>${c.name}
    </button>`
  ).join('');
}

// ── 切换摄像头 ──
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

// ── WebSocket 连接 ──
function connectWS() {
  if (ws) ws.close();
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${protocol}//${location.host}/ws`);

  ws.onopen = () => {
    setStatus(true, '已连接');
    if (currentCam) {
      ws.send(JSON.stringify({ action: 'switch', camera: currentCam }));
    }
  };

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);

    if (msg.type === 'frame') {
      // 解码并渲染帧
      const img = new Image();
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        document.getElementById('noSignal').style.display = 'none';

        // 计算 FPS
        frameCount++;
        const now = Date.now();
        if (now - lastFpsTime >= 1000) {
          fpsVal = frameCount;
          frameCount = 0;
          lastFpsTime = now;
        }
      };
      img.src = 'data:image/jpeg;base64,' + msg.frame;

      document.getElementById('fpsInfo').textContent = fpsVal + ' FPS';
      document.getElementById('inferInfo').textContent =
        '推理: ' + (msg.inference_ms || '?') + 'ms';

      // 更新底部车牌列表
      const bar = document.getElementById('platesBar');
      if (msg.plates && msg.plates.length) {
        bar.innerHTML = msg.plates.map(p =>
          `<span class="plate-tag">${p.code}
            <span class="conf">${(p.conf*100).toFixed(0)}%</span>
          </span>`
        ).join('');
      } else {
        bar.innerHTML = '<span class="empty">未检测到车牌</span>';
      }
    }
  };

  ws.onclose = () => setStatus(false, '已断开');
  ws.onerror = () => setStatus(false, '连接失败');
}

function setStatus(live, text) {
  document.getElementById('statusDot').className = 'dot ' + (live ? 'live' : 'dead');
  document.getElementById('statusText').textContent = text;
}

// ── 初始化 ──
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
        """在独立线程中读 RTSP 帧、识别、编码"""
        frame_count = 0
        empty_count = 0
        t0 = time.time()
        while not stop_event.is_set() and manager.running:
            frame, plates, infer_ms = manager.read_frame()
            frame_count += 1

            if frame is None:
                empty_count += 1
                time.sleep(0.3)
                # 每 5 秒打印一次状态
                if time.time() - t0 > 5:
                    print(f"  [DEBUG] {frame_count} read attempts, {empty_count} empty frames (no data from stream)")
                    t0 = time.time()
                continue

            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
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

            # 每 5 秒打印识别统计
            if frame_count % 150 == 0 and plates:
                print(f"  [DEBUG] frame #{frame_count}: detected {len(plates)} plates: {[p['code'] for p in plates]}")

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
                    # 清空旧队列
                    while not send_queue.empty():
                        try:
                            send_queue.get_nowait()
                        except queue.Empty:
                            break

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

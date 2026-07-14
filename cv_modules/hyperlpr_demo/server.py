"""
HyperLPR3 车牌识别 FastAPI 服务
启动: python server.py
访问: http://localhost:8003/     (上传页面: 图片 + 视频)
      http://localhost:8003/docs  (Swagger 交互式文档)
"""
import asyncio
import base64
import json
import os
import queue
import threading
import time
import tempfile
import cv2
import numpy as np
from pathlib import Path
from gpu_patch import catcher  # GPU 加速版 HyperLPR3
from plate_track_store import PlateTrackStore
from vehicle_lpr import (
    VEHICLE_CLASS_IDS,
    expand_box,
    get_vehicle_model,
    map_plate_bbox_from_crop,
    recognize_with_vehicle_crops,
)
from live_server import StreamManager as LiveStreamManager
from fastapi import FastAPI, File, UploadFile, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

app = FastAPI(title="车牌识别服务 (GPU)", version="2.0")
BASE_DIR = Path(__file__).resolve().parent
LOCAL_VIDEO_DIR = BASE_DIR / "test_video"
LOCAL_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
TRACKER_CONFIG = "bytetrack.yaml"
SANDBOX_VEHICLE_MODEL = "runs/sandbox_vehicle/yolo26n_clean20/weights/best.pt"
SANDBOX_ALLOWED_PLATES = {"京B6789T", "京E4682Y", "京E7654Z", "京H7912N", "京K9134J"}
VIDEO_SCENES = {
    "dashcam": {"dir": "行车记录仪", "label": "行车记录仪"},
    "sandbox_yolo": {"dir": "沙盘", "label": "沙盘 YOLO"},
}

PLATE_COLOR_MAP = {
    -1: "未知", 0: "蓝牌", 1: "黄牌(单层)", 2: "白牌(单层)",
    3: "绿牌(新能源)", 4: "黑牌(港澳)", 5: "香港(单层)",
    6: "香港(双层)", 7: "澳门(单层)", 8: "澳门(双层)", 9: "黄牌(双层)",
}


def normalize_video_scene(scene: str | None) -> str | None:
    if not scene or scene == "all":
        return None
    return scene if scene in VIDEO_SCENES else None


def list_local_videos(scene: str | None = None):
    if not LOCAL_VIDEO_DIR.exists():
        return []

    normalized_scene = normalize_video_scene(scene)
    roots = []
    if normalized_scene:
        roots = [(normalized_scene, LOCAL_VIDEO_DIR / VIDEO_SCENES[normalized_scene]["dir"])]
    else:
        roots = [
            (scene_key, LOCAL_VIDEO_DIR / scene_cfg["dir"])
            for scene_key, scene_cfg in VIDEO_SCENES.items()
        ]
        roots.append((None, LOCAL_VIDEO_DIR))

    videos = []
    seen = set()
    for scene_key, root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in LOCAL_VIDEO_EXTS:
                continue
            if any(part.startswith(".") for part in path.relative_to(LOCAL_VIDEO_DIR).parts):
                continue
            rel = path.relative_to(LOCAL_VIDEO_DIR).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            inferred_scene = scene_key
            if inferred_scene is None:
                for key, cfg in VIDEO_SCENES.items():
                    try:
                        path.relative_to(LOCAL_VIDEO_DIR / cfg["dir"])
                        inferred_scene = key
                        break
                    except ValueError:
                        pass
            scene_label = VIDEO_SCENES.get(inferred_scene or "", {}).get("label", "未分类")
            videos.append({
                "name": path.name,
                "relative_path": rel,
                "scene": inferred_scene or "uncategorized",
                "scene_label": scene_label,
                "size_mb": round(path.stat().st_size / 1024 / 1024, 1),
            })
    return videos


def resolve_local_video(name: str, scene: str | None = None) -> Path | None:
    if not name:
        return None
    normalized_scene = normalize_video_scene(scene)
    base = LOCAL_VIDEO_DIR
    candidate = (base / name).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    if normalized_scene:
        scene_root = (LOCAL_VIDEO_DIR / VIDEO_SCENES[normalized_scene]["dir"]).resolve()
        try:
            candidate.relative_to(scene_root)
        except ValueError:
            return None
    if not candidate.exists() or candidate.suffix.lower() not in LOCAL_VIDEO_EXTS:
        return None
    return candidate


def recognize_video_path(video_path: str, *, filename: str, interval: float):
    """Offline video debug pipeline: YOLO ByteTrack -> HyperLPR -> per-track voting."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    frame_step = max(1, int(fps * interval)) if fps > 0 else 1
    model = get_vehicle_model()
    store = PlateTrackStore(min_confidence=0.90)

    processed = 0
    tracked_frames = 0
    vehicle_regions = 0
    rejected_count = 0
    started = time.perf_counter()

    stream = model.track(
        source=video_path,
        stream=True,
        tracker=TRACKER_CONFIG,
        classes=VEHICLE_CLASS_IDS,
        conf=0.25,
        imgsz=640,
        verbose=False,
    )

    for frame_idx, result in enumerate(stream):
        frame = result.orig_img
        if frame is None or result.boxes is None or result.boxes.id is None:
            continue

        timestamp = round(frame_idx / fps, 2) if fps > 0 else 0
        boxes = result.boxes
        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        tracked_frames += 1
        should_ocr = frame_idx % frame_step == 0
        if should_ocr:
            processed += 1

        height, width = frame.shape[:2]
        for track_id, box, vehicle_conf in zip(ids, xyxy, confs):
            bbox = expand_box(box, width, height, 0.15)
            store.update_track(track_id, bbox, timestamp, float(vehicle_conf))

            if not should_ocr:
                continue

            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue
            vehicle_regions += 1
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            scale = 2.0
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            found = False
            for raw in catcher(crop):
                found = True
                plate_code = raw[0]
                confidence = float(raw[1])
                plate_type = int(raw[2])
                plate_bbox = map_plate_bbox_from_crop([int(v) for v in raw[3]], bbox, scale)
                before = len(store.tracks.get(track_id).candidates) if store.tracks.get(track_id) else 0
                store.add_plate(
                    track_id,
                    plate_code,
                    confidence,
                    plate_type,
                    timestamp,
                    plate_bbox=plate_bbox,
                )
                after = len(store.tracks.get(track_id).candidates) if store.tracks.get(track_id) else 0
                if after == before:
                    rejected_count += 1
            if not found:
                rejected_count += 1

    plates = store.final_results()
    for plate in plates:
        plate["plate_color"] = PLATE_COLOR_MAP.get(plate.get("plate_type", -1), "未知")

    return {
        "filename": filename,
        "fps": round(fps, 1),
        "total_frames": total_frames,
        "duration_sec": round(duration, 1),
        "sample_interval_sec": interval,
        "processed_frames": processed,
        "tracked_frames": tracked_frames,
        "vehicle_regions": vehicle_regions,
        "rejected_count": rejected_count,
        "unique_plates": len(plates),
        "elapsed_sec": round(time.perf_counter() - started, 1),
        "tracker": "ByteTrack",
        "plates": plates,
    }

# ─── 上传页面 ───────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>车牌识别</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif;
         background: #f0f2f5; min-height: 100vh; }
  .header { background: linear-gradient(135deg, #1a73e8, #0d47a1);
            color: #fff; padding: 30px; text-align: center; }
  .header h1 { font-size: 24px; margin-bottom: 6px; }
  .header p { opacity: .8; font-size: 14px; }
  .container { max-width: 800px; margin: -20px auto 0; padding: 0 16px; }
  .card { background: #fff; border-radius: 12px; box-shadow: 0 2px 12px rgba(0,0,0,.08);
          padding: 24px; margin-bottom: 20px; }
  .tabs { display: flex; gap: 0; border-bottom: 2px solid #e8e8e8; margin-bottom: 20px; }
  .tab { padding: 10px 24px; cursor: pointer; border: none; background: none;
         font-size: 15px; color: #666; border-bottom: 2px solid transparent;
         margin-bottom: -2px; transition: .2s; }
  .tab.active { color: #1a73e8; border-bottom-color: #1a73e8; font-weight: 600; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }
  .drop-zone { border: 2px dashed #d0d5dd; border-radius: 10px; padding: 40px 20px;
               text-align: center; cursor: pointer; transition: .2s; margin-bottom: 16px; }
  .drop-zone:hover, .drop-zone.dragover { border-color: #1a73e8; background: #f0f7ff; }
  .drop-zone .icon { font-size: 40px; margin-bottom: 8px; }
  .drop-zone p { color: #888; font-size: 14px; }
  .drop-zone .hint { color: #aaa; font-size: 12px; margin-top: 6px; }
  input[type=file] { display: none; }
  .preview { text-align: center; margin: 16px 0; display: none; }
  .preview img, .preview video { max-width: 100%; max-height: 300px; border-radius: 8px; }
  .row { display: flex; gap: 12px; align-items: center; margin-bottom: 14px; }
  .row label { font-size: 14px; color: #555; white-space: nowrap; }
  .row input, .row select { flex: 1; padding: 8px 12px; border: 1px solid #d0d5dd;
               border-radius: 6px; font-size: 14px; }
  .btn { display: inline-block; padding: 10px 28px; border: none; border-radius: 8px;
         font-size: 15px; cursor: pointer; font-weight: 600; transition: .2s; }
  .btn-primary { background: #1a73e8; color: #fff; width: 100%; }
  .btn-primary:hover { background: #1557b0; }
  .btn-primary:disabled { background: #a0c4f1; cursor: not-allowed; }
  .result { margin-top: 20px; display: none; }
  .result .summary { font-size: 15px; margin-bottom: 12px; }
  .result table { width: 100%; border-collapse: collapse; font-size: 14px; }
  .result th { background: #f5f7fa; padding: 10px 12px; text-align: left;
               font-weight: 600; color: #555; border-bottom: 2px solid #e8e8e8; }
  .result td { padding: 10px 12px; border-bottom: 1px solid #f0f0f0; }
  .result .loading { text-align: center; padding: 30px; color: #888; }
  .result .spinner { display: inline-block; width: 28px; height: 28px;
    border: 3px solid #e0e0e0; border-top-color: #1a73e8; border-radius: 50%;
    animation: spin .8s linear infinite; margin-right: 10px; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="header">
  <h1>HyperLPR3 车牌识别</h1>
  <p>支持图片上传 &amp; 视频文件上传</p>
</div>
<div class="container">
<div class="card">

  <div class="tabs">
    <button class="tab active" onclick="switchTab('image')">图片识别</button>
    <button class="tab" onclick="switchTab('video')">视频识别</button>
  </div>

  <!-- 图片 Tab -->
  <div id="tab-image" class="tab-panel active">
    <div class="drop-zone" id="dropImage"
         onclick="document.getElementById('fileImage').click()">
      <div class="icon">🖼️</div>
      <p>点击或拖拽上传图片</p>
      <div class="hint">支持 jpg / png / bmp</div>
    </div>
    <input type="file" id="fileImage" accept="image/*"
           onchange="previewImage(this)">
    <div class="preview" id="previewImage">
      <img id="imgPreview" src="" alt="预览">
    </div>
    <button class="btn btn-primary" id="btnImage"
            onclick="recognizeImage()">开始识别</button>
  </div>

  <!-- 视频 Tab -->
  <div id="tab-video" class="tab-panel">
    <div class="drop-zone" id="dropVideo"
         onclick="document.getElementById('fileVideo').click()">
      <div class="icon">🎬</div>
      <p>点击或拖拽上传视频</p>
      <div class="hint">支持 mp4 / avi / mov</div>
    </div>
    <input type="file" id="fileVideo" accept="video/*"
           onchange="previewVideo(this)">
    <div class="preview" id="previewVideo">
      <video id="vidPreview" src="" controls></video>
    </div>
    <div class="row">
      <label>抽帧间隔(秒):</label>
      <input type="number" id="vidInterval" value="0.5" min="0.1" max="5" step="0.1">
    </div>
    <button class="btn btn-primary" id="btnVideo"
            onclick="recognizeVideo()">开始识别</button>
    <div class="row" style="margin-top:16px;">
      <label>本地调试视频:</label>
      <select id="localVideoSelect"></select>
    </div>
    <button class="btn btn-primary" id="btnLocalVideo"
            onclick="recognizeLocalVideo()">识别本地视频</button>
    <a class="btn btn-primary" href="/local-live/dashcam" target="_blank"
       style="text-align:center;text-decoration:none;margin-top:12px;">
      打开行车记录仪实时检测
    </a>
    <a class="btn btn-primary" href="/local-live/sandbox_yolo" target="_blank"
       style="text-align:center;text-decoration:none;margin-top:12px;background:#7b5b00;">
      打开沙盘实时检测（YOLO + OCR确认）
    </a>
  </div>

  <!-- 结果区 -->
  <div class="result" id="resultArea">
    <div id="resultContent"></div>
  </div>

</div>
</div>

<script>
// ── 工具函数 ──
const $ = id => document.getElementById(id);

function switchTab(type) {
  document.querySelectorAll('.tab').forEach((t,i) =>
    t.classList.toggle('active', (i===0 && type==='image') || (i===1 && type==='video')));
  $('tab-image').classList.toggle('active', type==='image');
  $('tab-video').classList.toggle('active', type==='video');
}

// ── 拖拽上传 ──
['dropImage','dropVideo'].forEach(id => {
  const zone = $(id);
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length) {
      const inputId = id==='dropImage' ? 'fileImage' : 'fileVideo';
      $(inputId).files = files;
      id==='dropImage' ? previewImage($(inputId)) : previewVideo($(inputId));
    }
  });
});

// ── 图片预览 ──
function previewImage(input) {
  const file = input.files[0];
  if (!file) return;
  $('previewImage').style.display = 'block';
  $('imgPreview').src = URL.createObjectURL(file);
}

function previewVideo(input) {
  const file = input.files[0];
  if (!file) return;
  $('previewVideo').style.display = 'block';
  $('vidPreview').src = URL.createObjectURL(file);
}

// ── 显示结果 ──
function showResult(html) {
  $('resultArea').style.display = 'block';
  $('resultContent').innerHTML = html;
}

function sourceLabel(source) {
  return source === 'vehicle' ? '车辆裁剪' : '整图';
}

function renderVideoResult(data, elapsed) {
  if (data.error) {
    showResult(`<p style="color:#d32f2f;">${data.error}</p>`);
    return;
  }

  let html = `<div class="summary">📹 ${data.filename} | ⏱ 耗时 ${elapsed}s |
              跟踪器 ${data.tracker || '-'} |
              抽帧 ${data.processed_frames} 张 | 跟踪帧 ${data.tracked_frames || 0} 张 |
              车辆区域 ${data.vehicle_regions || 0} 个 |
              已过滤 ${data.rejected_count || 0} 个候选 |
              输出 <b>${data.unique_plates}</b> 个稳定车牌</div>`;
  if (data.plates.length) {
    html += '<table><tr><th>轨迹</th><th>车牌号</th><th>颜色</th><th>置信度</th><th>候选数</th><th>出现时间</th></tr>';
    data.plates.forEach(p => {
      html += `<tr><td>#${p.track_id || '-'}</td><td><b>${p.plate_code}</b></td><td>${p.plate_color}</td>
               <td>${(p.confidence*100).toFixed(1)}%</td><td>${p.candidate_count || 1}</td>
               <td>${p.first_time ?? '-'}s - ${p.last_time ?? '-'}s</td></tr>`;
    });
    html += '</table>';
  } else { html += '<p style="color:#999;">未识别到车牌</p>'; }
  showResult(html);
}

async function loadLocalVideos() {
  const select = $('localVideoSelect');
  if (!select) return;
  const res = await fetch('/local-videos?scene=all');
  const data = await res.json();
  if (!data.videos.length) {
    select.innerHTML = '<option value="">test_video 目录暂无视频</option>';
    return;
  }
  select.innerHTML = data.videos.map(v =>
    `<option value="${v.relative_path}">${v.scene_label} / ${v.name} (${v.size_mb} MB)</option>`
  ).join('');
}

// ── 图片识别 ──
async function recognizeImage() {
  const file = $('fileImage').files[0];
  if (!file) return alert('请先选择图片');
  $('btnImage').disabled = true;
  showResult('<div class="loading"><span class="spinner"></span>识别中...</div>');

  const fd = new FormData(); fd.append('file', file);
  const res = await fetch('/recognize', { method:'POST', body:fd });
  const data = await res.json();

  let html = `<div class="summary">
              车辆区域 <b>${data.vehicle_regions || 0}</b> 个 |
              检测到 <b>${data.plate_count}</b> 个车牌 |
              已过滤 ${data.rejected_count || 0} 个候选 |
              推理 ${data.inference_ms}ms</div>`;
  if (data.plates.length) {
    html += '<table><tr><th>车牌号</th><th>颜色</th><th>置信度</th><th>来源</th></tr>';
    data.plates.forEach(p => {
      html += `<tr><td><b>${p.plate_code}</b></td><td>${p.plate_color}</td>
               <td>${(p.confidence*100).toFixed(1)}%</td><td>${sourceLabel(p.source)}</td></tr>`;
    });
    html += '</table>';
  } else { html += '<p style="color:#999;">未识别到车牌</p>'; }
  showResult(html);
  $('btnImage').disabled = false;
}

// ── 视频识别 ──
async function recognizeVideo() {
  const file = $('fileVideo').files[0];
  if (!file) return alert('请先选择视频');
  const interval = parseFloat($('vidInterval').value) || 0.5;
  $('btnVideo').disabled = true;
  showResult('<div class="loading"><span class="spinner"></span>视频处理中，请耐心等待...</div>');

  const fd = new FormData(); fd.append('file', file);
  const start = Date.now();
  const res = await fetch(`/recognize-video?interval=${interval}`, { method:'POST', body:fd });
  const data = await res.json();
  const elapsed = ((Date.now()-start)/1000).toFixed(1);

  renderVideoResult(data, elapsed);
  $('btnVideo').disabled = false;
}

async function recognizeLocalVideo() {
  const name = $('localVideoSelect').value;
  if (!name) return alert('请先选择本地视频');
  const interval = parseFloat($('vidInterval').value) || 0.5;
  $('btnLocalVideo').disabled = true;
  showResult('<div class="loading"><span class="spinner"></span>本地视频处理中...</div>');

  const start = Date.now();
  const res = await fetch(`/recognize-local-video?name=${encodeURIComponent(name)}&interval=${interval}`, { method:'POST' });
  const data = await res.json();
  const elapsed = ((Date.now()-start)/1000).toFixed(1);
  renderVideoResult(data, elapsed);
  $('btnLocalVideo').disabled = false;
}

loadLocalVideos();
</script>
</body>
</html>"""


@app.get("/local-live", response_class=HTMLResponse)
async def local_live_selector():
    return """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>8003 本地视频实时检测</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif; background:#0f1923; color:#dce6f0; min-height:100vh; display:flex; align-items:center; justify-content:center; }
  .panel { width:min(720px, calc(100vw - 32px)); }
  h1 { font-size:26px; margin-bottom:10px; }
  p { color:#8ea0b2; margin-bottom:22px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:16px; }
  a { display:block; text-decoration:none; color:#e8edf2; background:#172637; border:1px solid #2d4358; border-radius:10px; padding:22px; transition:.15s; }
  a:hover { border-color:#4b8ee8; transform:translateY(-1px); }
  strong { display:block; font-size:20px; margin-bottom:8px; }
  span { color:#8ea0b2; line-height:1.6; }
</style>
</head>
<body>
<div class="panel">
  <h1>8003 本地视频实时检测</h1>
  <p>请选择调试模式。沙盘模式使用 YOLO 找车 + OCR确认；行车记录仪模式保持通用车牌检测。</p>
  <div class="grid">
    <a href="/local-live/dashcam"><strong>行车记录仪</strong><span>通用道路视频，走原实时检测链路。</span></a>
    <a href="/local-live/sandbox_yolo"><strong>沙盘 YOLO</strong><span>YOLO 找车模型 + OCR确认。</span></a>
  </div>
</div>
</body>
</html>"""


@app.get("/local-live/{scene}", response_class=HTMLResponse)
async def local_live_page(scene: str):
    normalized_scene = normalize_video_scene(scene)
    if normalized_scene is None:
        return JSONResponse({"error": "unknown local-live scene"}, status_code=404)
    scene_label = VIDEO_SCENES[normalized_scene]["label"]
    videos = list_local_videos(normalized_scene)
    return """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>8003 """ + scene_label + """实时检测</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif; background:#0f1923; color:#dce6f0; min-height:100vh; }
  .topbar { padding:14px 18px; background:#1a2733; display:flex; align-items:center; gap:12px; border-bottom:1px solid #2a3a4a; }
  select, button { padding:8px 12px; border-radius:6px; border:1px solid #345; background:#10202e; color:#e8edf2; }
  button { cursor:pointer; background:#1a5fb4; border-color:#3478d4; font-weight:700; }
  .status { margin-left:auto; font-size:13px; color:#89a; }
  .main { height:calc(100vh - 106px); display:flex; align-items:center; justify-content:center; background:#081118; }
  canvas { max-width:100%; max-height:100%; }
  .plates { min-height:52px; padding:8px 14px; display:flex; gap:10px; flex-wrap:wrap; background:#1a2733; border-top:1px solid #2a3a4a; }
  .plate { background:#1e3a2a; border:1px solid #2e5a3a; border-radius:6px; padding:6px 12px; color:#c8e6c9; font-weight:700; }
  .muted { color:#789; }
</style>
</head>
<body>
<div class="topbar">
  <strong>8003 """ + scene_label + """实时检测</strong>
  <select id="videoSelect"></select>
  <button onclick="startVideo()">开始</button>
  <label id="debugWrap" class="muted" style="display:none;align-items:center;gap:4px;">
    <input id="debugToggle" type="checkbox"> 调试框
  </label>
  <span id="fpsInfo" class="muted">-</span>
  <span id="inferInfo" class="muted">-</span>
  <span class="status" id="status">未连接</span>
</div>
<div class="main">
  <canvas id="canvas"></canvas>
</div>
<div class="plates" id="plates"><span class="muted">等待识别结果...</span></div>

<script>
const SCENE = """ + json.dumps(normalized_scene, ensure_ascii=False) + """;
const SCENE_LABEL = """ + json.dumps(scene_label, ensure_ascii=False) + """;
const VIDEOS = """ + json.dumps(videos, ensure_ascii=False) + """;
let ws = null;
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let frameCount = 0;
let fpsStart = Date.now();
let currentVideo = '';

function renderVideos() {
  const select = document.getElementById('videoSelect');
  document.getElementById('debugWrap').style.display = 'none';
  if (!VIDEOS.length) {
    select.innerHTML = '<option value="">test_video 目录暂无视频</option>';
    return;
  }
  select.innerHTML = VIDEOS.map(v =>
    `<option value="${v.relative_path}">${v.name} (${v.size_mb} MB)</option>`
  ).join('');
}

function connectWS() {
  if (ws) ws.close();
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws-local-live`);
  ws.onopen = () => {
    document.getElementById('status').textContent = '已连接';
    if (currentVideo) ws.send(JSON.stringify({
      action:'switch',
      name:currentVideo,
      scene:SCENE,
      debug: document.getElementById('debugToggle')?.checked || false
    }));
  };
  ws.onclose = () => document.getElementById('status').textContent = '已断开';
  ws.onerror = () => document.getElementById('status').textContent = '连接失败';
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'error') {
      document.getElementById('status').textContent = msg.msg;
      return;
    }
    if (msg.type !== 'frame') return;

    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      frameCount++;
      const now = Date.now();
      if (now - fpsStart >= 1000) {
        document.getElementById('fpsInfo').textContent = frameCount + ' FPS';
        frameCount = 0;
        fpsStart = now;
      }
    };
    img.src = 'data:image/jpeg;base64,' + msg.frame;
    document.getElementById('inferInfo').textContent = '推理: ' + (msg.inference_ms || '?') + 'ms';

    const bar = document.getElementById('plates');
    if (msg.plates && msg.plates.length) {
      bar.innerHTML = msg.plates.map(p => {
        const prefix = `#${p.track_id || '-'}`;
        return `<span class="plate">${prefix} ${p.code} ${(p.conf*100).toFixed(0)}%</span>`;
      }).join('');
    } else {
      bar.innerHTML = '<span class="muted">未检测到车牌</span>';
    }
  };
}

function startVideo() {
  currentVideo = document.getElementById('videoSelect').value;
  if (!currentVideo) return alert('请先选择本地视频');
  if (!ws || ws.readyState !== WebSocket.OPEN) connectWS();
  else ws.send(JSON.stringify({
    action:'switch',
    name:currentVideo,
    scene:SCENE,
    debug: document.getElementById('debugToggle')?.checked || false
  }));
}

renderVideos();
</script>
</body>
</html>"""


# ─── 图片识别 API ────────────────────────────────────────

@app.on_event("startup")
async def warmup_models():
    """Move first-request model initialization cost to server startup."""
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    get_vehicle_model().predict(dummy, classes=[2, 3, 5, 7], imgsz=640, verbose=False)
    catcher(dummy)


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    """上传图片，返回车牌识别结果"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "无法解析图片"}, status_code=400)

    t0 = time.perf_counter()
    plates, regions, rejected = recognize_with_vehicle_crops(
        img,
        catcher,
        return_rejected=True,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    for plate in plates:
        plate["plate_color"] = PLATE_COLOR_MAP.get(plate["plate_type"], "未知")

    return {
        "filename": file.filename,
        "image_size": f"{img.shape[1]}x{img.shape[0]}",
        "inference_ms": elapsed_ms,
        "vehicle_regions": sum(1 for region in regions if region.source == "vehicle"),
        "rejected_count": len(rejected),
        "plate_count": len(plates),
        "plates": plates,
    }

# ─── 视频识别 API ────────────────────────────────────────

@app.post("/recognize-video")
async def recognize_video(
    file: UploadFile = File(...),
    interval: float = Query(0.5, ge=0.1, le=10, description="抽帧间隔(秒)")
):
    """上传视频文件，用 ByteTrack 跟踪车辆并按轨迹聚合车牌。"""
    # 写入临时文件
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.close()

        result = recognize_video_path(
            tmp.name,
            filename=file.filename or "video",
            interval=interval,
        )
        if result is None:
            return JSONResponse({"error": "无法打开视频文件"}, status_code=400)
        return result
    finally:
        os.unlink(tmp.name)


@app.get("/local-videos")
async def local_videos(scene: str = Query("all", description="all, dashcam, sandbox_yolo")):
    """列出 test_video 目录中的本地调试视频。"""
    normalized_scene = normalize_video_scene(scene)
    if scene not in ("all", "", None) and normalized_scene is None:
        return JSONResponse({"error": "unknown video scene"}, status_code=400)
    return {"scene": normalized_scene or "all", "videos": list_local_videos(normalized_scene)}


@app.post("/recognize-local-video")
async def recognize_local_video(
    name: str = Query(..., description="test_video 目录下的视频文件名"),
    interval: float = Query(0.5, ge=0.1, le=10, description="抽帧间隔(秒)"),
    scene: str = Query("all", description="all, dashcam, sandbox_yolo")
):
    """直接识别本地 test_video 视频，作为 8003 调试入口。"""
    normalized_scene = normalize_video_scene(scene)
    if scene not in ("all", "", None) and normalized_scene is None:
        return JSONResponse({"error": "unknown video scene"}, status_code=400)
    path = resolve_local_video(name, normalized_scene)
    if path is None:
        return JSONResponse({"error": "本地视频不存在"}, status_code=404)

    result = recognize_video_path(str(path), filename=path.name, interval=interval)
    if result is None:
        return JSONResponse({"error": "无法打开视频文件"}, status_code=400)
    result["local_video"] = True
    result["scene"] = normalized_scene or "all"
    return result


@app.websocket("/ws-local-live")
async def websocket_local_live(ws: WebSocket):
    """8003 本地视频实时检测：复用 8004 的 StreamManager，但只允许 test_video 文件。"""
    await ws.accept()
    manager = None
    send_queue = queue.Queue(maxsize=5)
    reader_thread = None
    stop_event = threading.Event()
    current_scene = "all"

    def reader():
        frame_count = 0
        while not stop_event.is_set() and manager and manager.running:
            frame, plates, _overlay_plates, infer_ms = manager.read_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            frame_count += 1
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue

            payload = {
                "type": "frame",
                "frame": base64.b64encode(jpeg).decode(),
                "plates": plates,
                "inference_ms": infer_ms,
                "scene": current_scene,
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
                print(f"  [LOCAL-LIVE] frame #{frame_count}: {[p['code'] for p in plates]}")
            time.sleep(0.03)

    async def sender():
        while not stop_event.is_set():
            try:
                await ws.send_json(send_queue.get_nowait())
            except queue.Empty:
                await asyncio.sleep(0.03)
            except Exception:
                break

    send_task = asyncio.create_task(sender())

    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("action") != "switch":
                continue

            requested_scene = msg.get("scene", "all")
            normalized_scene = normalize_video_scene(requested_scene)
            if requested_scene not in ("all", "", None) and normalized_scene is None:
                await ws.send_json({"type": "error", "msg": "unknown video scene"})
                continue

            path = resolve_local_video(msg.get("name", ""), normalized_scene)
            if path is None:
                await ws.send_json({"type": "error", "msg": "本地视频不存在"})
                continue

            if manager:
                manager.close()
            if reader_thread and reader_thread.is_alive():
                reader_thread.join(timeout=1)
            while not send_queue.empty():
                try:
                    send_queue.get_nowait()
                except queue.Empty:
                    break

            current_scene = normalized_scene or "all"
            debug = bool(msg.get("debug", False))
            if current_scene == "sandbox_yolo":
                manager = LiveStreamManager(
                    vehicle_model_path=SANDBOX_VEHICLE_MODEL,
                    allowed_plates=SANDBOX_ALLOWED_PLATES,
                    allowed_plate_min_confidence=0.90,
                    suppress_overlapping_tracks=True,
                    detect_interval_sec=0.01,
                )
            else:
                manager = LiveStreamManager()
            print(f"[8003-LOCAL-LIVE] Switch to {path.name} ({current_scene})")
            if manager.open(str(path), path.name):
                reader_thread = threading.Thread(target=reader, daemon=True)
                reader_thread.start()
                await ws.send_json({
                    "type": "status",
                    "video": path.name,
                    "scene": current_scene,
                    "connected": True,
                })
            else:
                await ws.send_json({"type": "error", "msg": f"Can not open: {path.name}"})

    except WebSocketDisconnect:
        pass
    finally:
        stop_event.set()
        if manager:
            manager.close()
        send_task.cancel()
        print("[8003-LOCAL-LIVE] Disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)

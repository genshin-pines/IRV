"""
HyperLPR3 车牌识别 FastAPI 服务
启动: python server.py
访问: http://localhost:8003/     (上传页面: 图片 + 视频)
      http://localhost:8003/docs  (Swagger 交互式文档)
"""
import os
import time
import tempfile
import cv2
import numpy as np
from gpu_patch import catcher  # GPU 加速版 HyperLPR3
from vehicle_lpr import get_vehicle_model, recognize_with_vehicle_crops
from video_plate_tracker import VehiclePlateTracker
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

app = FastAPI(title="车牌识别服务 (GPU)", version="2.0")

PLATE_COLOR_MAP = {
    -1: "未知", 0: "蓝牌", 1: "黄牌(单层)", 2: "白牌(单层)",
    3: "绿牌(新能源)", 4: "黑牌(港澳)", 5: "香港(单层)",
    6: "香港(双层)", 7: "澳门(单层)", 8: "澳门(双层)", 9: "黄牌(双层)",
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

  let html = `<div class="summary">📹 ${data.filename} | ⏱ 耗时 ${elapsed}s |
              抽帧 ${data.processed_frames} 张 | 车辆区域 ${data.vehicle_regions || 0} 个 |
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
  $('btnVideo').disabled = false;
}
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
    """上传视频文件，抽帧识别车牌并去重"""
    # 写入临时文件
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return JSONResponse({"error": "无法打开视频文件"}, status_code=400)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        tracker = VehiclePlateTracker()
        frame_idx = 0
        processed = 0
        vehicle_regions = 0
        rejected_count = 0
        frame_step = max(1, int(fps * interval)) if fps > 0 else 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                processed += 1
                timestamp = round(frame_idx / fps, 2) if fps > 0 else 0
                plates, regions, rejected = recognize_with_vehicle_crops(
                    frame,
                    catcher,
                    return_rejected=True,
                )
                vehicle_regions += sum(1 for region in regions if region.source == "vehicle")
                rejected_count += len(rejected)
                for plate in plates:
                    plate["plate_color"] = PLATE_COLOR_MAP.get(plate["plate_type"], "未知")
                tracker.update(regions, plates, timestamp)
            frame_idx += 1
        cap.release()

        plates = tracker.final_results()

        return {
            "filename": file.filename,
            "fps": round(fps, 1),
            "total_frames": total_frames,
            "duration_sec": round(duration, 1),
            "sample_interval_sec": interval,
            "processed_frames": processed,
            "vehicle_regions": vehicle_regions,
            "rejected_count": rejected_count,
            "unique_plates": len(plates),
            "plates": plates,
        }
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)

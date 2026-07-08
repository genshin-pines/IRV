# IRV — Intelligent Road Vision

车载摄像头视觉感知与人机交互系统。软件工程本科实践项目（2026）。

## 模块

| 模块 | 状态 | 说明 |
|------|------|------|
| 道路车辆车牌识别 | ✅ 完成 | HyperLPR3 + DirectML GPU 加速，支持图片/视频/RTSP 实时流 |
| 交警手势识别 | 🔜 待开发 | MediaPipe Pose + 骨骼关键点分类 |
| 车主手势控车 | 🔜 待开发 | MediaPipe Hands，浏览器端实时识别 |
| 日志监控与告警智能体 | 🔜 待开发 | LLM Agent + WebSocket 推送 |

## 车牌识别

### 环境

```bash
cd cv_modules/hyperlpr_demo
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install onnxruntime-directml  # GPU (NVIDIA)
pip install websockets matplotlib   # 实时流 + 显示
```

### 使用

```bash
# CLI 图片识别
python recognize.py path/to/car.jpg

# CLI 视频识别
python recognize_video.py path/to/video.mp4 --interval 0.5

# Web 服务 (图片/视频上传)
python server.py
# → http://localhost:8003

# 实时流监控 (RTSP + WebSocket)
python live_server.py
# → http://localhost:8004
```

### 关键技术

- **HyperLPR3**: 中国车牌检测+OCR 一体化 Pipeline
- **DirectML**: Windows GPU 加速，RTX 4060 上约 10x 提速（~70ms/帧）
- **WebSocket**: 实时推送编码帧 + 识别结果
- **FastAPI**: 图片/视频上传 API + Swagger 文档

## 技术栈

- **前端**: Vue 3 + Element Plus (待搭建)
- **后端**: Python FastAPI + SpringBoot
- **CV**: HyperLPR3、MediaPipe、OpenCV
- **LLM**: DeepSeek API
- **GPU**: ONNX Runtime DirectML
- **数据库**: MySQL

## 团队

5 人协作，7 天开发周期。

## 运行命令

以下命令默认在项目根目录 `D:\Software_Engineering_Semester_Project_2` 执行。

### 1. 启动流媒体服务器 MediaMTX

```powershell
.\tools\mediamtx\mediamtx.exe
```

默认 RTSP 服务地址：

```text
rtsp://127.0.0.1:8554
```

### 2. 使用 FFmpeg 推送本地视频

将测试视频循环推送到 MediaMTX：

```powershell
.\tools\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe `
  -re -stream_loop -1 `
  -i .\cv_modules\hyperlpr_demo\test_video\test12.mp4 `
  -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p `
  -an -rtsp_transport tcp -f rtsp rtsp://127.0.0.1:8554/live/test12
```

实时识别模块可读取：

```text
rtsp://127.0.0.1:8554/live/test12
```

### 3. 启动车牌图片/视频识别 Web 服务

```powershell
cd .\cv_modules\hyperlpr_demo
..\..\.venv\Scripts\python.exe server.py
```

访问：

```text
http://localhost:8003
```

### 4. 启动车牌实时识别 WebSocket 服务

```powershell
cd .\cv_modules\hyperlpr_demo
..\..\.venv\Scripts\python.exe live_server.py
```

访问：

```text
http://localhost:8004
```


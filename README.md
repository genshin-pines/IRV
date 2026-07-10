# IRV 智能车载视觉系统

IRV 是一个面向沙盘/车载交互演示的多模块视觉系统，整合了车牌识别、车主端手势控车、摄像头视频流、Agent 告警与统一 Web 控制台。当前发布版本为 `v0.9`，重点完成了车主端手势控制的业务开关、误触发抑制、动作反馈 HUD 和日志追踪。

## 功能概览

- 管理者端：查看沙盘摄像头视频流、抽帧识别车牌、生成并查看 Agent 告警。
- 车主端：使用本机摄像头或外部 RTSP 流识别手势，控制音乐、音量、灯光、电话和控制开关。
- 手势控制总开关：默认关闭，只有完成 `palm -> grabbing -> fist` 组合后才允许非电话类手势生效。
- 电话例外：控制关闭时仍允许 `call` 接听和 `stop / stop_inverted` 挂断。
- 业务层抑制：对点击、左右滑、音量反向、控制开关重复触发做冷却，减少一次动作被识别成多次反馈。
- 视频反馈 HUD：只有业务动作真正生效时才在视频画面上显示提示，避免识别层误动作造成“看起来成功”的假反馈。
- 日志追踪：手势帧、动态动作、业务映射和抑制原因写入 `logs/gesture_static_trace.log`，便于复盘。

## 项目结构

```text
IRV_main/
├─ backend/                    # FastAPI 统一后端
│  ├─ routers/                 # plate / gesture / cameras / alerts API
│  ├─ services/                # 车牌、手势、摄像头、日志、LLM 服务
│  └─ main.py                  # 应用入口
├─ frontend/
│  ├─ index.html               # 单页控制台
│  └─ vendor/                  # 前端离线依赖
├─ alert_agent/                # 告警 Agent 与规则调度
├─ fusion/                     # 多源事件融合实验模块
├─ vendor/
│  ├─ plate_hyperlpr/          # HyperLPR 车牌识别适配
│  └─ web_gesture_backend/     # 手势识别引擎与车控映射
├─ docs/                       # 进度记录、接口与部署文档
├─ logs/                       # 运行日志
├─ uploads/                    # 上传文件临时目录
├─ run.ps1                     # Windows 一键启动脚本
└─ requirements.txt            # 根依赖
```

## 环境要求

- Windows 10/11，建议使用 PowerShell。
- Python 3.10+。
- 摄像头或可访问的 RTSP 视频源。
- 可选：DeepSeek API Key，用于 Agent 摘要能力；不配置也可运行基础功能。

## 快速启动

推荐使用项目自带脚本：

```powershell
cd D:\download\git\IRV\IRV_main
.\run.ps1
```

脚本会自动创建 `.venv`、安装依赖并启动 FastAPI。启动后访问：

- 控制台：http://127.0.0.1:8000/
- Swagger：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/api/health

如需手动启动：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

## 配置

复制 `.env.example` 为 `.env` 后按需修改：

```text
APP_ENV=dev
DATABASE_URL=sqlite:///./irv.db
LOG_LEVEL=INFO
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

数据库默认使用项目根目录下的 `irv.db`。日志默认写入 `logs/`。

## 主要 API

| 模块 | 方法与路径 | 说明 |
| --- | --- | --- |
| 健康检查 | `GET /api/health` | 查看后端、Agent 与模块入口 |
| 摄像头列表 | `GET /api/cameras` | 获取沙盘摄像头配置 |
| 摄像头画面 | `GET /api/cameras/{id}/stream` | MJPEG 视频流 |
| 摄像头快照 | `GET /api/cameras/{id}/snapshot.jpg` | 抽取单帧图片 |
| 图片车牌识别 | `POST /api/plate/recognize-image` | 上传图片识别车牌 |
| 视频车牌识别 | `POST /api/plate/recognize-video` | 上传视频按间隔抽帧识别 |
| RTSP 车牌识别 | `POST /api/plate/recognize-stream` | 对视频流采样识别 |
| 启动手势流 | `POST /api/gesture/start` | 启动本机摄像头或 RTSP 手势识别 |
| 停止手势流 | `POST /api/gesture/stop` | 停止手势识别 |
| 手势消息 | `GET /api/gesture/messages` | 拉取帧消息与动作消息 |
| 手势视频 | `GET /api/gesture/video-feed` | 查看带识别框和 HUD 的视频 |
| 模拟手势 | `POST /api/gesture/events` | 前端快捷按钮使用的模拟入口 |
| 告警列表 | `GET /api/alerts` | 查看 Agent 告警 |
| 模拟告警 | `POST /api/logs/simulate` | 生成测试日志与告警 |
| 告警 WS | `WS /ws/alerts` | 告警实时推送 |

## v0.9 手势规则

### 控制开关

系统启动后手势控制默认关闭。关闭状态下仍会识别、画框、写日志、上报动作消息，但业务层不会执行非电话类动作。

唯一的控制开关手势为：

```text
palm(张掌) -> grabbing(抓握) -> fist(握拳)
```

该组合会触发 `DNDV1 -> control_toggle`。其他能在识别层产生圆形或拖拽效果的动作，例如 `DRAG / DRAG2 / DRAG3`，不会再开关控制。

### 控制关闭时的例外

- `CALL`：接听电话。
- `STOP / STOP_INVERTED`：挂断电话。

电话动作不受控制开关影响，方便驾驶场景下优先处理来电。

### 动作映射

| 手势事件 | 业务动作 | 说明 |
| --- | --- | --- |
| `DNDV1` | `control_toggle` | 开启/关闭手势控制 |
| `TAP / DOUBLE_TAP` | `music_toggle` | 播放/暂停 |
| `SWIPE_LEFT*` | `turn_left` | 上一首 |
| `SWIPE_RIGHT*` | `turn_right` | 下一首 |
| `SWIPE_UP* / FAST_SWIPE_UP` | `volume_up` | 音量增加 |
| `SWIPE_DOWN* / FAST_SWIPE_DOWN` | `volume_down` | 音量降低 |
| `LIKE` | `lights_on` | 开灯 |
| `DISLIKE` | `lights_off` | 关灯 |
| `CALL` | `phone_answer` | 接听电话 |
| `STOP / STOP_INVERTED` | `phone_hangup` | 挂断电话 |

### 冷却与抑制

| 规则 | 时间 | 目的 |
| --- | --- | --- |
| 控制开关重复触发 | `1.5s` | 避免一次 DNDV1 抖成开后又关 |
| 播放/暂停重复触发 | `1.0s` | 避免一次点击触发多次播放暂停 |
| 左右滑同向重复 | `1.0s` | 避免一次滑动跳多首 |
| 左右滑反向抖动 | `1.5s` | 避免左右误识别来回切歌 |
| 音量上下反向抖动 | `2.0s` | 避免音量上下抖动 |

被抑制的动作会返回 `action_applied=false`，并在 `suppress_reason` 中说明原因。视频 HUD 只在 `action_applied=true` 时显示。

## 前端使用说明

打开控制台后可切换两个视角：

- 管理者端：选择摄像头，查看实时画面、快照、自动车牌识别和告警分布。
- 车主端：启动本机摄像头或外部视频流，查看手势识别画面和模拟车辆状态。

车主端顶部会显示 `控制开启 / 控制关闭`。控制关闭时视频中央会提示：

```text
手势控制未开启
依次做出张掌、抓握、握拳后启用
```

## 日志与排查

重点日志：

- `logs/gesture_static_trace.log`：手势帧、静态标签、动态动作、业务映射和抑制原因。
- `server.out.log` / `server.err.log`：历史启动输出。
- `docs/工作进度记录_2026-07-09.md`：本轮迭代问题、修正和验证记录。

排查手势误触发时优先看 `ACTION` 行，例如：

```text
ACTION    ...    DNDV1    vehicle=control_toggle    applied=True
ACTION    ...    DRAG     vehicle=                  applied=
ACTION    ...    SWIPE_RIGHT vehicle=turn_right      applied=False reason=ignore reverse turn action within 1.5s
```

## 验证命令

语法检查：

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  backend\services\gesture_service.py `
  vendor\web_gesture_backend\models.py `
  vendor\web_gesture_backend\gesture_engine.py `
  vendor\web_gesture_backend\dgcore\utils\action_controller.py `
  vendor\web_gesture_backend\dgcore\utils\drawer.py `
  vendor\web_gesture_backend\dgcore\utils\enums.py
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

模拟控制开关：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/gesture/events `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"source":"demo","gesture_type":"dndv1","gesture_label":"控制开关","confidence":0.92,"stable":true}'
```

## v0.9 发布重点

- 完成车主端手势控制总开关。
- `DOUBLE_TAP` 与 `TAP` 均等效为播放/暂停。
- 增加点击、左右滑、音量、控制开关的业务层冷却。
- 将 `DNDV1` 收窄为 `palm -> grabbing -> fist`，移除 `DRAG*` 对控制开关的误映射。
- 控制关闭时保留电话接听/挂断能力。
- 视频反馈只在业务真正执行时显示，并替换为更清晰的 HUD 样式。
- 增强手势日志，方便从原始标签、动态动作和业务执行结果三层定位问题。

## 注意事项

- `vendor/web_gesture_backend/dgcore/models/*.onnx` 为手势识别模型文件，运行手势识别时必须保留。
- 摄像头占用、驱动权限或 RTSP 不可达会导致手势/视频流启动失败。
- `onnxruntime-directml` 面向 Windows GPU 加速环境；如环境不支持，可根据本机情况替换为合适的 ONNX Runtime 包。
- 本项目用于演示与教学场景，车控动作均为模拟状态，不直接连接真实车辆控制器。

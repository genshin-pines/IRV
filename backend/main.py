from __future__ import annotations

import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"       # 抑制 FFmpeg h264 解码噪音

from contextlib import asynccontextmanager
from pathlib import Path
import sys

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.schemas.common import ok, fail, SanitizedJSONResponse

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alert_agent.scheduler import agent_status, start_scheduler, stop_scheduler
from backend.config import EVENT_BUS_WINDOW_SECONDS, FUSION_DEDUP_MS, FUSION_LLM_ENABLED
from backend.database import init_db
from backend.middleware.logging_mw import RequestLoggingMiddleware
from backend.routers import alerts_router, auth_router, cameras_router, gesture_router, music_router, plate_router, preferences_router, traffic_police_router, ws_manager
from backend.services.log_service import setup_log_collector

FRONTEND_DIR = PROJECT_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    setup_log_collector()
    start_scheduler(broadcast=ws_manager.broadcast)
    # 初始化融合引擎（EventBus + FusionAgent）
    from backend.services.alert_service import setup_fusion_engine
    await setup_fusion_engine(
        ws_broadcast=ws_manager.broadcast,
        use_llm=FUSION_LLM_ENABLED,
        window_seconds=EVENT_BUS_WINDOW_SECONDS,
        dedup_ms=FUSION_DEDUP_MS,
    )
    yield
    from backend.services.alert_service import stop_fusion_engine
    await stop_fusion_engine()
    await stop_scheduler()


app = FastAPI(
    title="IRV Intelligent Road Vision",
    description="车牌识别 + 手势控车 + Agent 预警 + 沙盘摄像头统一后端",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=SanitizedJSONResponse,
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 异常处理器：统一错误格式 ──────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        loc = " → ".join(str(x) for x in err["loc"])
        errors.append(f"{loc}: {err['msg']}")
    return JSONResponse(
        status_code=422,
        content=fail(message="请求参数校验失败", data={"detail": "; ".join(errors)}),
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content=fail(message=f"接口不存在: {request.method} {request.url.path}"),
    )


# 注册路由
app.include_router(alerts_router)
app.include_router(music_router)
app.include_router(preferences_router)
app.include_router(auth_router)
app.include_router(plate_router)
app.include_router(gesture_router)
app.include_router(traffic_police_router)
app.include_router(cameras_router)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>IRV Backend</h1><p>Open <a href='/docs'>/docs</a></p>")


@app.get("/api/health")
def health():
    result = {"ok": True, "status": "healthy", "agent": agent_status()}
    try:
        from backend.services.alert_service import get_fusion_agent, get_event_bus
        fusion = get_fusion_agent()
        bus = get_event_bus()
        result["fusion_agent"] = fusion.status if fusion else None
        result["event_bus"] = bus.stats if bus else None
    except Exception:
        pass
    result["modules"] = {
        "plate": "/api/plate/recognize-image",
        "gesture": "/api/gesture/status",
        "traffic_police": "/api/traffic-police/status",
        "auth": "/api/auth/security",
        "alerts": "/api/alerts",
        "cameras": "/api/cameras",
    }
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

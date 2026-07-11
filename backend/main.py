from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alert_agent.scheduler import agent_status, start_scheduler, stop_scheduler
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
    yield
    await stop_scheduler()


app = FastAPI(
    title="IRV Intelligent Road Vision",
    description="车牌识别 + 手势控车 + Agent 预警 + 沙盘摄像头统一后端",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return {
        "ok": True,
        "status": "healthy",
        "agent": agent_status(),
        "modules": {
            "plate": "/api/plate/recognize-image",
            "gesture": "/api/gesture/status",
            "traffic_police": "/api/traffic-police/status",
            "auth": "/api/auth/security",
            "alerts": "/api/alerts",
            "cameras": "/api/cameras",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

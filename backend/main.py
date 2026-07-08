from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from alert_agent.scheduler import agent_status, start_scheduler, stop_scheduler
from backend.database import init_db
from backend.middleware.logging_mw import RequestLoggingMiddleware
from backend.routers import alerts_router, ws_manager
from backend.services.log_service import setup_log_collector


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    setup_log_collector()
    start_scheduler(broadcast=ws_manager.broadcast)
    yield
    await stop_scheduler()


app = FastAPI(
    title="IRV Intelligent Road Vision",
    description="Agent 日志监控与智能预警模块",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(alerts_router)


@app.get("/api/health")
def health():
    return {"ok": True, "status": "healthy", "agent": agent_status()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

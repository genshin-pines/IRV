"""
FastAPI 应用入口
@owner 成员D (主) + 成员E (告警/日志部分)

启动:
  cd 项目根目录
  uvicorn backend.main:app --reload
  → http://localhost:8000
  → Swagger 文档: http://localhost:8000/docs
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，以便 import alert_agent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, create_engine

from backend.config import DATABASE_URL, LOG_LEVEL
from backend.routers import alerts_router, broadcast_alert
from backend.middleware.logging_mw import RequestLoggingMiddleware
from backend.services.log_service import setup_log_collector
from backend.services.alert_service import setup_alert_agent, stop_alert_agent

# ── 日志 ──────────────────────────────────────────────
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))

# ── 数据库 ────────────────────────────────────────────
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})


# ── 生命周期 ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭"""
    # Startup
    SQLModel.metadata.create_all(engine)
    setup_log_collector()
    await setup_alert_agent(engine, ws_broadcast=broadcast_alert)
    logging.getLogger("backend").info("Backend 已启动 — Swagger: http://localhost:8000/docs")
    yield
    # Shutdown
    await stop_alert_agent()


# ── App ───────────────────────────────────────────────
app = FastAPI(
    title="IRV — Intelligent Road Vision",
    description="智能车载交互与监控系统 API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 请求日志中间件（成员E）— 最外层，记录所有请求
app.add_middleware(RequestLoggingMiddleware)

# CORS（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(alerts_router)


# ── 健康检查 ──────────────────────────────────────────
@app.get("/api/health")
def health():
    from backend.services.alert_service import get_agent
    agent = get_agent()
    return {
        "status": "ok",
        "agent": agent.status if agent else None,
    }


# ── 入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

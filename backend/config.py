"""
全局配置 — 数据库 / JWT / LLM / 日志
@owner 成员D (主) + 成员E (LLM + 日志部分)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ── 数据库 ──────────────────────────────────────────
# 先用 SQLite 快速启动，Day 5 后可切换 MySQL
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/irv.db")

# ── JWT 认证 ────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 小时

# ── LLM ────────────────────────────────────────────
# @owner 成员E
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")

# ── 日志 ────────────────────────────────────────────
# @owner 成员E
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_COLLECTOR_CAPACITY = int(os.getenv("LOG_COLLECTOR_CAPACITY", "500"))
AGENT_POLL_INTERVAL_SEC = int(os.getenv("AGENT_POLL_INTERVAL_SEC", "30"))

# ── 融合推理 ────────────────────────────────────────
# @owner 成员E
EVENT_BUS_WINDOW_SECONDS = float(os.getenv("EVENT_BUS_WINDOW_SECONDS", "2.0"))
FUSION_DEDUP_MS = int(os.getenv("FUSION_DEDUP_MS", "500"))
FUSION_LLM_ENABLED = os.getenv("FUSION_LLM_ENABLED", "true").lower() == "true"

# ── 告警通知 ────────────────────────────────────────
# @owner 成员E
FEISHU_WEBHOOK_URL = os.getenv("FEISHU_WEBHOOK_URL", "")
DINGTALK_WEBHOOK_URL = os.getenv("DINGTALK_WEBHOOK_URL", "")

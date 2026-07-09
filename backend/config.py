from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
load_dotenv(BASE_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_DIR / 'irv.db'}")
LOG_DIR = PROJECT_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

AGENT_POLL_INTERVAL_SEC = int(os.getenv("AGENT_POLL_INTERVAL_SEC", "3"))
LOG_COLLECTOR_CAPACITY = int(os.getenv("LOG_COLLECTOR_CAPACITY", "2000"))

# ── LLM（告警 & 融合推理）────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "15"))

# ── 融合推理 ──────────────────────────────────────────
EVENT_BUS_WINDOW_SECONDS = float(os.getenv("EVENT_BUS_WINDOW_SECONDS", "2.0"))
FUSION_DEDUP_MS = int(os.getenv("FUSION_DEDUP_MS", "500"))
FUSION_LLM_ENABLED = os.getenv("FUSION_LLM_ENABLED", "true").lower() == "true"

# ── 告警通知 ────────────────────────────────────────
# @owner 成员E
# 方式一：自定义机器人 Webhook（最简单，只需 URL）
FEISHU_WEBHOOK_URL = os.getenv("FEISHU_WEBHOOK_URL", "")
DINGTALK_WEBHOOK_URL = os.getenv("DINGTALK_WEBHOOK_URL", "")
# 方式二：应用机器人 API（支持 @all、卡片消息等高级功能）
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
FEISHU_CHAT_ID = os.getenv("FEISHU_CHAT_ID", "")  # 目标群 ID
FEISHU_NOTIFY_ENABLED = os.getenv("FEISHU_NOTIFY_ENABLED", "false").lower() == "true"

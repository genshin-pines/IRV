from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
# 优先加载 backend/.env，不存在则回退到项目根 .env
load_dotenv(PROJECT_DIR / ".env")
load_dotenv(BASE_DIR / ".env", override=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_DIR / 'irv.db'}")
LOG_DIR = PROJECT_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

AGENT_POLL_INTERVAL_SEC = int(os.getenv("AGENT_POLL_INTERVAL_SEC", "3"))
LOG_COLLECTOR_CAPACITY = int(os.getenv("LOG_COLLECTOR_CAPACITY", "2000"))
CAMERA_HEALTH_CHECK_INTERVAL_SEC = float(os.getenv("CAMERA_HEALTH_CHECK_INTERVAL_SEC", "30"))
CAMERA_HEALTH_FAIL_THRESHOLD = int(os.getenv("CAMERA_HEALTH_FAIL_THRESHOLD", "5"))
CAMERA_HEALTH_PROBE_TIMEOUT_SEC = float(os.getenv("CAMERA_HEALTH_PROBE_TIMEOUT_SEC", "10"))

# ── LLM 通用配置（支持任何 OpenAI 兼容接口） ──────────
# 通用变量（LLM_API_KEY / LLM_BASE_URL）优先，兼容旧版 DEEPSEEK_* 变量
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "15"))
# 保留旧版兼容
DEEPSEEK_API_KEY = LLM_API_KEY
DEEPSEEK_BASE_URL = LLM_BASE_URL

# ── 融合推理 ──────────────────────────────────────────
EVENT_BUS_WINDOW_SECONDS = float(os.getenv("EVENT_BUS_WINDOW_SECONDS", "2.0"))
FUSION_DEDUP_MS = int(os.getenv("FUSION_DEDUP_MS", "500"))
FUSION_LLM_ENABLED = os.getenv("FUSION_LLM_ENABLED", "true").lower() == "true"

# ── 认证 ──────────────────────────────────────────────
AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "irv-local-demo-secret")
AUTH_CODE_TTL_SEC = int(os.getenv("AUTH_CODE_TTL_SEC", "300"))
AUTH_CODE_COOLDOWN_SEC = int(os.getenv("AUTH_CODE_COOLDOWN_SEC", "60"))

# ── 短信 / 邮箱验证码 ─────────────────────────────────
SMS_WEBHOOK_URL = os.getenv("SMS_WEBHOOK_URL", "")
SMS_WEBHOOK_TOKEN = os.getenv("SMS_WEBHOOK_TOKEN", "")

EMAIL_SMTP_HOST = os.getenv("EMAIL_SMTP_HOST", "")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "465"))
EMAIL_SMTP_USER = os.getenv("EMAIL_SMTP_USER", "")
EMAIL_SMTP_PASSWORD = os.getenv("EMAIL_SMTP_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", EMAIL_SMTP_USER)

# ── 飞书通知 ──────────────────────────────────────────
FEISHU_NOTIFY_ENABLED = os.getenv("FEISHU_NOTIFY_ENABLED", "false").lower() == "true"
FEISHU_WEBHOOK_URL = os.getenv("FEISHU_WEBHOOK_URL", "")
FEISHU_APP_ID = os.getenv("FEISHU_APP_ID", "")
FEISHU_APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
FEISHU_CHAT_ID = os.getenv("FEISHU_CHAT_ID", "")

# ── 交警手势识别 ──────────────────────────────────────
CTPGR_REFERENCE_DIR = os.getenv(
    "CTPGR_REFERENCE_DIR",
    str(PROJECT_DIR.parent.parent / "参考库" / "TPHSR" / "ctpgr-publish"),
)

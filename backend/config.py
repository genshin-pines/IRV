from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
load_dotenv(PROJECT_DIR / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_DIR / 'irv.db'}")
LOG_DIR = PROJECT_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

AGENT_POLL_INTERVAL_SEC = int(os.getenv("AGENT_POLL_INTERVAL_SEC", "3"))
LOG_COLLECTOR_CAPACITY = int(os.getenv("LOG_COLLECTOR_CAPACITY", "2000"))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "15"))

AUTH_SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "irv-local-demo-secret")
AUTH_CODE_TTL_SEC = int(os.getenv("AUTH_CODE_TTL_SEC", "300"))
AUTH_CODE_COOLDOWN_SEC = int(os.getenv("AUTH_CODE_COOLDOWN_SEC", "60"))
SMS_WEBHOOK_URL = os.getenv("SMS_WEBHOOK_URL", "")
SMS_WEBHOOK_TOKEN = os.getenv("SMS_WEBHOOK_TOKEN", "")
HUAWEI_SMS_ENDPOINT = os.getenv("HUAWEI_SMS_ENDPOINT", "")
HUAWEI_SMS_APP_KEY = os.getenv("HUAWEI_SMS_APP_KEY", "")
HUAWEI_SMS_APP_SECRET = os.getenv("HUAWEI_SMS_APP_SECRET", "")
HUAWEI_SMS_SENDER = os.getenv("HUAWEI_SMS_SENDER", "")
HUAWEI_SMS_TEMPLATE_ID = os.getenv("HUAWEI_SMS_TEMPLATE_ID", "")
HUAWEI_SMS_SIGNATURE = os.getenv("HUAWEI_SMS_SIGNATURE", "")
HUAWEI_SMS_STATUS_CALLBACK = os.getenv("HUAWEI_SMS_STATUS_CALLBACK", "")
EMAIL_SMTP_HOST = os.getenv("EMAIL_SMTP_HOST", "")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", "465"))
EMAIL_SMTP_USER = os.getenv("EMAIL_SMTP_USER", "")
EMAIL_SMTP_PASSWORD = os.getenv("EMAIL_SMTP_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", EMAIL_SMTP_USER)

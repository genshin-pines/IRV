from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from backend.config import FEISHU_NOTIFY_ENABLED, FEISHU_WEBHOOK_URL


logger = logging.getLogger("alert_agent")
_BEIJING_TZ = timezone(timedelta(hours=8))


def _make_card(alert: dict[str, Any]) -> dict[str, Any]:
    level = str(alert.get("level") or "INFO").upper()
    color = {"CRITICAL": "red", "ERROR": "orange", "WARNING": "yellow", "INFO": "blue"}.get(level, "blue")
    icon = {"CRITICAL": "🔴", "ERROR": "🔴", "WARNING": "🟡", "INFO": "🔵"}.get(level, "🔵")
    title = alert.get("title") or "IRV 告警"
    summary = alert.get("summary") or ""
    detail = alert.get("detail") or ""
    source = alert.get("source_module") or "system"
    generated = "规则引擎 + LLM" if alert.get("ai_generated") else "规则引擎"
    now_str = datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d %H:%M 北京")
    elements = [
        {"tag": "markdown", "content": f"**摘要**\n{summary}"},
        {"tag": "hr"},
        {"tag": "markdown", "content": f"**来源模块**: {source}\n**生成方式**: {generated}"},
        {"tag": "note", "elements": [{"tag": "plain_text", "content": f"IRV 告警系统 · {now_str}"}]},
    ]
    if detail:
        elements.insert(2, {"tag": "markdown", "content": f"**详情**\n{detail[:1200]}"})
        elements.insert(2, {"tag": "hr"})
    return {
        "msg_type": "interactive",
        "card": {
            "header": {"title": {"tag": "plain_text", "content": f"{icon} {level}: {title}"}, "template": color},
            "elements": elements,
        },
    }


async def send_alert_notification(alert: dict[str, Any]) -> bool:
    if not FEISHU_NOTIFY_ENABLED:
        return False
    if str(alert.get("level") or "").upper() not in {"CRITICAL", "ERROR", "WARNING"}:
        return False
    if not FEISHU_WEBHOOK_URL:
        logger.warning("FEISHU_NOTIFY_ENABLED=true but FEISHU_WEBHOOK_URL is empty")
        return False
    try:
        response = requests.post(FEISHU_WEBHOOK_URL, data=json.dumps(_make_card(alert)), headers={"Content-Type": "application/json"}, timeout=10)
        if response.status_code >= 400:
            logger.error("feishu webhook failed: %s", response.text[:300])
            return False
        return True
    except requests.RequestException as exc:
        logger.error("feishu webhook exception: %s", exc)
        return False

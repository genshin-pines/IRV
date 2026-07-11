"""
告警通知服务 — 飞书/钉钉消息推送
@owner 成员E

支持两种模式（按优先级）:
  1. 应用机器人 API (FEISHU_APP_ID + FEISHU_APP_SECRET) — 支持 @all、卡片消息
  2. 自定义机器人 Webhook (FEISHU_WEBHOOK_URL) — 最简单，只需一个 URL

用法:
    from backend.services.notifier import send_alert_notification, send_test_message

    await send_alert_notification(alert_dict)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict

import requests

# 北京时间 UTC+8
_BEIJING_TZ = timezone(timedelta(hours=8))

from backend.config import (
    FEISHU_WEBHOOK_URL,
    FEISHU_APP_ID,
    FEISHU_APP_SECRET,
    FEISHU_CHAT_ID,
    FEISHU_NOTIFY_ENABLED,
)

logger = logging.getLogger(__name__)

# ── Token 缓存 ──────────────────────────────────────────
_cached_token: str = ""
_cached_token_expires_at: float = 0.0


def _get_tenant_access_token() -> str:
    """获取飞书 tenant_access_token（带缓存，有效期约 2h）"""
    global _cached_token, _cached_token_expires_at

    now = time.time()
    if _cached_token and now < _cached_token_expires_at - 60:
        return _cached_token

    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    resp = requests.post(
        url,
        json={"app_id": FEISHU_APP_ID, "app_secret": FEISHU_APP_SECRET},
        timeout=10,
    )
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"获取 tenant_access_token 失败: {data}")

    _cached_token = data["tenant_access_token"]
    _cached_token_expires_at = now + data.get("expire", 7200)
    logger.info(f"飞书 token 已刷新, 有效 {data.get('expire')}s")
    return _cached_token


# ── 消息构建 ────────────────────────────────────────────

# 级别 → 卡片颜色模板
_COLOR_MAP = {
    "CRITICAL": "red",
    "ERROR": "orange",
    "WARNING": "yellow",
    "INFO": "blue",
}
_ICON_MAP = {
    "CRITICAL": "🔴",
    "ERROR": "🔴",
    "WARNING": "🟡",
    "INFO": "🔵",
}
_LABEL_MAP = {
    "CRITICAL": "严重告警",
    "ERROR": "错误告警",
    "WARNING": "警告",
    "INFO": "信息",
}


def _make_card(alert: Dict) -> Dict:
    """构建飞书 Lark MD 卡片消息"""
    level = alert.get("level", "INFO")
    title = alert.get("title", "告警")
    summary = alert.get("summary", "")
    detail = alert.get("detail", "")
    source_module = alert.get("source_module", "")
    affected = alert.get("affected_modules", [])
    ai_generated = alert.get("ai_generated", False)

    color = _COLOR_MAP.get(level, "blue")
    icon = _ICON_MAP.get(level, "🔵")
    label = _LABEL_MAP.get(level, "信息")
    now_str = datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d %H:%M 北京")

    # 构建 Markdown 内容块
    elements = [
        {
            "tag": "markdown",
            "content": f"**摘要**\n{summary}",
        },
    ]

    if detail:
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "markdown",
            "content": f"**详情**\n{detail}",
        })

    elements.append({"tag": "hr"})
    meta_parts = []
    if source_module:
        meta_parts.append(f"**来源模块**: {source_module}")
    if affected:
        meta_parts.append(f"**影响范围**: {', '.join(affected)}")
    meta_parts.append(f"**生成方式**: ⚙️ 规则引擎检测{' + 🤖 LLM 深度分析' if ai_generated else '（LLM 未参与）'}")
    elements.append({
        "tag": "markdown",
        "content": "\n".join(meta_parts),
    })

    # 底部时间戳
    elements.append({
        "tag": "note",
        "elements": [
            {"tag": "plain_text", "content": f"IRV 告警系统 · {now_str}"},
        ],
    })

    # CRITICAL 时 @所有人
    if level == "CRITICAL":
        elements.append({
            "tag": "markdown",
            "content": "<at id=all></at>",
        })

    card = {
        "header": {
            "title": {
                "tag": "plain_text",
                "content": f"{icon} {label}: {title}",
            },
            "template": color,
        },
        "elements": elements,
    }

    return {
        "msg_type": "interactive",
        "card": card,
    }


# ── 消息发送 ────────────────────────────────────────────


async def send_alert_notification(alert: Dict) -> bool:
    """发送告警通知到飞书群（Lark MD 卡片格式）"""
    if not FEISHU_NOTIFY_ENABLED:
        logger.debug("飞书通知未启用 (FEISHU_NOTIFY_ENABLED=false)")
        return False

    level = alert.get("level", "info")

    # 仅 CRITICAL / ERROR / WARNING 发通知
    if level.upper() not in ("CRITICAL", "ERROR", "WARNING"):
        return False

    card = _make_card(alert)

    # 优先使用应用机器人 API
    if FEISHU_APP_ID and FEISHU_APP_SECRET and FEISHU_CHAT_ID:
        return await _send_via_app_api(card)

    # 降级为自定义机器人 Webhook
    if FEISHU_WEBHOOK_URL:
        return await _send_via_webhook(card)

    logger.warning("飞书通知未配置: 缺少 FEISHU_APP_ID 或 FEISHU_WEBHOOK_URL")
    return False


async def send_test_message(message: str = "🧪 飞书通知连通性测试") -> bool:
    """发送一条测试消息到飞书群（用于验证配置）"""
    card = {
        "header": {
            "title": {"tag": "plain_text", "content": "🧪 通知测试"},
            "template": "blue",
        },
        "elements": [
            {"tag": "markdown", "content": message},
            {
                "tag": "note",
                "elements": [
                    {"tag": "plain_text", "content": f"IRV 告警系统 · {datetime.now(_BEIJING_TZ).strftime('%Y-%m-%d %H:%M 北京')}"},
                ],
            },
        ],
    }

    if FEISHU_APP_ID and FEISHU_APP_SECRET and FEISHU_CHAT_ID:
        return await _send_via_app_api({"msg_type": "interactive", "card": card})
    if FEISHU_WEBHOOK_URL:
        return await _send_via_webhook({"msg_type": "interactive", "card": card})

    logger.warning("飞书未配置，无法发送测试消息")
    return False


async def _send_via_app_api(content: Dict) -> bool:
    """通过飞书应用机器人 API 发送卡片消息"""
    try:
        token = _get_tenant_access_token()
        url = "https://open.feishu.cn/open-apis/im/v1/messages"
        params = {"receive_id_type": "chat_id"}

        card_json = json.dumps(content["card"])
        body = {
            "receive_id": FEISHU_CHAT_ID,
            "msg_type": "interactive",
            "content": card_json,
        }

        resp = requests.post(
            url,
            params=params,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            json=body,
            timeout=10,
        )
        data = resp.json()
        if data.get("code") == 0:
            logger.info(f"飞书卡片消息已发送: message_id={data['data']['message_id']}")
            return True
        else:
            logger.error(f"飞书消息发送失败: code={data.get('code')}, msg={data.get('msg')}")
            return False

    except Exception as e:
        logger.error(f"飞书通知异常: {e}")
        return False


async def _send_via_webhook(content: Dict) -> bool:
    """通过飞书自定义机器人 Webhook 发送卡片消息"""
    try:
        body = content
        resp = requests.post(FEISHU_WEBHOOK_URL, json=body, timeout=10)
        data = resp.json()
        if data.get("code") == 0 or data.get("StatusCode") == 0:
            logger.info("飞书 Webhook 卡片消息已发送")
            return True
        else:
            logger.error(f"飞书 Webhook 发送失败: {data}")
            return False

    except Exception as e:
        logger.error(f"飞书 Webhook 异常: {e}")
        return False

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

import asyncio
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


async def _get_tenant_access_token() -> str:
    """获取飞书 tenant_access_token（带缓存，有效期约 2h）"""
    global _cached_token, _cached_token_expires_at

    now = time.time()
    if _cached_token and now < _cached_token_expires_at - 60:
        return _cached_token

    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    resp = await asyncio.to_thread(
        requests.post,
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

# 结构化摘要的四段标签（顺序固定，与 LLM prompt 和 templates.py 对齐）
_SECTION_LABELS = [
    "异常类型与现象",
    "影响范围",
    "告警原因分析",
    "建议处置措施",
]


def _parse_summary(summary: str) -> list[str]:
    """将 LLM/规则引擎输出的多段摘要拆分为各段内容。

    LLM 输出四段纯文本，用空行分隔；规则引擎的模板也使用相同的 \\n\\n 格式。
    返回恰好 4 个元素的列表，空段用占位文本填充。
    """
    parts = [p.strip() for p in summary.strip().split("\n\n") if p.strip()]
    while len(parts) < 4:
        parts.append("（暂无详细信息）")
    return parts[:4]


def _make_card(alert: Dict) -> Dict:
    """构建飞书 Lark MD 卡片消息 — 结构化文本模板。"""
    level = alert.get("level", "INFO")
    title = alert.get("title", "告警")
    summary = alert.get("summary", "")
    source_module = alert.get("source_module", "")
    ai_generated = alert.get("ai_generated", False)

    color = _COLOR_MAP.get(level, "blue")
    icon = _ICON_MAP.get(level, "🔵")
    label = _LABEL_MAP.get(level, "信息")
    now_str = datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d %H:%M 北京")
    gen_method = "🤖 LLM 深度分析" if ai_generated else "⚙️ 规则引擎检测"

    # ── Body：四段式结构化文本 ──
    sections = _parse_summary(summary)
    body_lines = []
    for index, heading in enumerate(_SECTION_LABELS):
        body_lines.append(f"**{heading}**")
        body_lines.append(sections[index])
        if index < 3:
            body_lines.append("")

    body_lines.append("")
    body_lines.append("---")
    body_lines.append(f"📋 来源模块：{source_module}　|　🔧 {gen_method}")
    body_lines.append(f"IRV 告警系统 · {now_str}")

    body_md = "\n".join(body_lines)

    # ── CRITICAL 时 @所有人 ──
    elements: list[dict] = [{"tag": "markdown", "content": body_md}]
    if level == "CRITICAL":
        elements.append({"tag": "markdown", "content": "<at id=all></at>"})

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
    now_str = datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d %H:%M 北京")
    body_lines = [
        "**异常类型与现象**",
        message,
        "",
        "**影响范围**",
        "测试消息，无实际影响",
        "",
        "**告警原因分析**",
        "人工触发连通性测试",
        "",
        "**建议处置措施**",
        "无需处置，此为测试消息",
        "",
        "---",
        "📋 来源模块：system　|　🔧 手动测试",
        f"IRV 告警系统 · {now_str}",
    ]
    body_md = "\n".join(body_lines)

    card = {
        "header": {
            "title": {"tag": "plain_text", "content": "🧪 通知测试"},
            "template": "blue",
        },
        "elements": [
            {"tag": "markdown", "content": body_md},
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
        token = await _get_tenant_access_token()
        url = "https://open.feishu.cn/open-apis/im/v1/messages"
        params = {"receive_id_type": "chat_id"}

        card_json = json.dumps(content["card"])
        body = {
            "receive_id": FEISHU_CHAT_ID,
            "msg_type": "interactive",
            "content": card_json,
        }

        resp = await asyncio.to_thread(
            requests.post,
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
        resp = await asyncio.to_thread(requests.post, FEISHU_WEBHOOK_URL, json=body, timeout=10)
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

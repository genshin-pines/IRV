"""
告警通知服务 — 飞书/钉钉消息推送
@owner 成员E

支持两种模式（按优先级）:
  1. 应用机器人 API (FEISHU_APP_ID + FEISHU_APP_SECRET) — 支持 @all、卡片消息
  2. 自定义机器人 Webhook (FEISHU_WEBHOOK_URL) — 最简单，只需一个 URL

用法:
    from backend.services.notifier import send_alert_notification

    await send_alert_notification(alert_dict)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, Optional

import requests

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
    logger.debug(f"飞书 token 已刷新, 有效 {data.get('expire')}s")
    return _cached_token


# ── 消息发送 ────────────────────────────────────────────


async def send_alert_notification(alert: Dict) -> bool:
    """
    发送告警通知到飞书群。

    根据配置自动选择: 应用机器人 API > Webhook URL。
    非 CRITICAL 级别的告警默认不发送（避免打扰）。

    Args:
        alert: Agent 生成的告警字典，包含 level/title/summary/webhook_markdown 等

    Returns:
        True 如果成功发送，False 如果未配置或发送失败
    """
    if not FEISHU_NOTIFY_ENABLED:
        return False

    level = alert.get("level", "info")

    # 仅 CRITICAL 和 WARNING 发通知，INFO 不发
    if level == "info":
        return False

    # 构建消息内容
    content = _build_message(alert, level)

    # 优先使用应用机器人 API
    if FEISHU_APP_ID and FEISHU_APP_SECRET and FEISHU_CHAT_ID:
        return await _send_via_app_api(content)

    # 降级为自定义机器人 Webhook
    if FEISHU_WEBHOOK_URL:
        return await _send_via_webhook(content)

    logger.debug("飞书通知未配置 (缺少 FEISHU_WEBHOOK_URL 或 FEISHU_APP_ID)")
    return False


def _build_message(alert: Dict, level: str) -> Dict:
    """构建飞书消息内容"""
    title = alert.get("title", "告警")
    summary = alert.get("summary", "")
    detail = alert.get("detail", "")
    source_module = alert.get("source_module", "")
    affected_modules = alert.get("affected_modules", [])
    ai_generated = alert.get("ai_generated", False)

    # 级别对应的颜色和图标
    level_config = {
        "critical": {"color": "red", "icon": "🔴", "label": "严重"},
        "warning": {"color": "yellow", "icon": "🟡", "label": "警告"},
    }
    cfg = level_config.get(level, level_config["warning"])

    # 文本消息体（用于 text 类型，简单快速）
    text_lines = [
        f"{cfg['icon']} **{cfg['label']}告警**: {title}",
        f"",
        f"**摘要**: {summary}" if summary else "",
        f"**详情**: {detail}" if detail else "",
        f"**来源模块**: {source_module}" if source_module else "",
        f"**影响范围**: {', '.join(affected_modules)}" if affected_modules else "",
        f"**生成方式**: {'AI 分析' if ai_generated else '规则引擎'}",
    ]
    text_content = "\n".join(line for line in text_lines if line)

    # CRITICAL 时 @所有人
    if level == "critical":
        text_content += "\n\n<at user_id=\"all\"></at>"

    return {
        "msg_type": "text",
        "text_content": text_content,
    }


async def _send_via_app_api(content: Dict) -> bool:
    """通过飞书应用机器人 API 发送消息"""
    try:
        token = _get_tenant_access_token()
        url = "https://open.feishu.cn/open-apis/im/v1/messages"
        params = {"receive_id_type": "chat_id"}

        msg_content = json.dumps({"text": content["text_content"]})
        body = {
            "receive_id": FEISHU_CHAT_ID,
            "msg_type": "text",
            "content": msg_content,
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
            logger.info(f"飞书通知已发送: message_id={data['data']['message_id']}")
            return True
        else:
            logger.error(f"飞书通知发送失败: code={data.get('code')}, msg={data.get('msg')}")
            return False

    except Exception as e:
        logger.error(f"飞书通知异常: {e}")
        return False


async def _send_via_webhook(content: Dict) -> bool:
    """通过飞书自定义机器人 Webhook 发送消息"""
    try:
        body = {
            "msg_type": "text",
            "content": {
                "text": content["text_content"],
            },
        }

        resp = requests.post(FEISHU_WEBHOOK_URL, json=body, timeout=10)
        data = resp.json()
        if data.get("code") == 0 or data.get("StatusCode") == 0:
            logger.info("飞书 Webhook 通知已发送")
            return True
        else:
            logger.error(f"飞书 Webhook 发送失败: {data}")
            return False

    except Exception as e:
        logger.error(f"飞书 Webhook 异常: {e}")
        return False

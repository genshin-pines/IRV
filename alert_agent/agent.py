from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from sqlalchemy.orm import Session

from alert_agent.rules import RuleEngine, RuleResult
from backend.models.alert_event import AlertStatus
from backend.schemas.alerts import AlertCreate
from backend.services.alert_service import create_alert
from backend.services.llm_service import llm_service
from backend.services.log_service import get_collector


logger = logging.getLogger("alert_agent")
Broadcast = Callable[[dict], Awaitable[None]]

# 级别 → 控制台颜色/图标
_LEVEL_ICON = {"CRITICAL": "🔴", "ERROR": "🔴", "WARNING": "🟡", "INFO": "🔵"}


class AlertAgent:
    def __init__(self, broadcast: Broadcast | None = None) -> None:
        self.rule_engine = RuleEngine()
        self.broadcast = broadcast

    async def run_once(self, db: Session) -> list[dict]:
        logs = get_collector().get_new_logs()
        if not logs:
            return []
        results = self.rule_engine.analyze(logs)
        if not results:
            return []

        alerts = []
        for result in results:
            alert = create_alert(db, self._to_create_payload(result))
            message = {
                "id": alert.id,
                "level": alert.level,
                "title": alert.title,
                "summary": alert.summary,
                "source_module": alert.source_module,
                "created_at": alert.created_at.isoformat(),
            }
            alerts.append(message)

            # 控制台即时输出
            icon = _LEVEL_ICON.get(alert.level, "⚪")
            print(f"\n{icon} [巡检告警] {alert.level} | {alert.title}")
            print(f"   摘要: {alert.summary}")
            print(f"   来源: {alert.source_module} | ID: {alert.id}")

            if self.broadcast:
                await self.broadcast(message)

            # 飞书通知（后台发送，不阻塞巡检）
            asyncio.create_task(self._notify(alert))

        logger.info("alert generated count=%s", len(alerts))
        return alerts

    async def _notify(self, alert) -> None:
        try:
            from backend.services.notifier import send_alert_notification
            result = await send_alert_notification({
                "level": alert.level,
                "title": alert.title,
                "summary": alert.summary,
                "detail": alert.raw_log or "",
                "source_module": alert.source_module,
                "affected_modules": [alert.source_module],
                "ai_generated": bool(alert.llm_summary and alert.llm_summary != alert.summary),
            })
            if result:
                logger.info(f"飞书通知已发送: [{alert.level}] {alert.title}")
            else:
                logger.debug(f"飞书通知跳过: [{alert.level}] {alert.title} (未配置或非告警级别)")
        except Exception:
            logger.exception("飞书通知异常")

    def _to_create_payload(self, result: RuleResult) -> AlertCreate:
        raw_log = "\n".join(result.raw_logs[-20:])
        fallback = f"{result.title}：{result.summary}"
        llm_summary = llm_service.summarize(result.raw_logs, fallback)
        return AlertCreate(
            level=result.level,
            title=result.title,
            summary=fallback,
            source_module=result.source_module,
            raw_log=raw_log,
            llm_summary=llm_summary,
            status=AlertStatus.UNREAD,
        )

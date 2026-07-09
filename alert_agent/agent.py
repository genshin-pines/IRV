from __future__ import annotations

import asyncio
import logging
import time
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

_LEVEL_ICON = {"CRITICAL": "🔴", "ERROR": "🔴", "WARNING": "🟡", "INFO": "🔵"}

# 事件驱动模式默认防抖（同规则 N 秒内不重复触发）
_DEFAULT_DEBOUNCE_SEC = 10


class AlertAgent:
    def __init__(self, broadcast: Broadcast | None = None, *, debounce_sec: float = _DEFAULT_DEBOUNCE_SEC) -> None:
        self.rule_engine = RuleEngine()
        self.broadcast = broadcast
        self.debounce_sec = debounce_sec
        self._last_fired: dict[str, float] = {}  # rule_id → last trigger time

    # ── 事件驱动入口 ────────────────────────────────────────

    async def trigger(self) -> list[dict]:
        """
        事件驱动触发 — 日志写入时立即调用。
        内置防抖：同一 rule_id 在 debounce_sec 内不重复触发。
        """
        logs = get_collector().get_new_logs()
        if not logs:
            return []
        results = self.rule_engine.analyze(logs)
        if not results:
            return []

        # 防抖过滤
        now = time.monotonic()
        fresh = []
        for r in results:
            last = self._last_fired.get(r.rule_id, 0)
            if now - last >= self.debounce_sec:
                self._last_fired[r.rule_id] = now
                fresh.append(r)

        if not fresh:
            logger.debug(f"防抖跳过 {len(results)} 条告警 (debounce={self.debounce_sec}s)")
            return []

        # 创建告警
        from backend.database import SessionLocal
        alerts = []
        with SessionLocal() as db:
            for result in fresh:
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

                icon = _LEVEL_ICON.get(alert.level, "⚪")
                print(f"\n{icon} [事件触发] {alert.level} | {alert.title}")
                print(f"   摘要: {alert.summary}")
                print(f"   来源: {alert.source_module} | ID: {alert.id}")

                if self.broadcast:
                    await self.broadcast(message)

                asyncio.create_task(self._notify(alert))

        if alerts:
            logger.info("alert triggered count=%s", len(alerts))
        return alerts

    # ── 定时兜底（兼容旧行为，清理过期数据） ────────────────

    async def run_once(self, db: Session) -> list[dict]:
        """定时巡检（兜底 + 数据清理），30s 一次"""
        logs = get_collector().get_new_logs()
        if not logs:
            return []
        results = self.rule_engine.analyze(logs)
        alerts = []
        for result in results:
            alert = create_alert(db, self._to_create_payload(result))
            alerts.append({
                "id": alert.id,
                "level": alert.level,
                "title": alert.title,
                "summary": alert.summary,
                "source_module": alert.source_module,
                "created_at": alert.created_at.isoformat(),
            })
            asyncio.create_task(self._notify(alert))
        if alerts:
            logger.info("poll fallback: count=%s", len(alerts))
        return alerts

    # ── 通知 ────────────────────────────────────────────────

    async def _notify(self, alert) -> None:
        try:
            from backend.services.notifier import send_alert_notification
            await send_alert_notification({
                "level": alert.level,
                "title": alert.title,
                "summary": alert.summary,
                "detail": alert.raw_log or "",
                "source_module": alert.source_module,
                "affected_modules": [alert.source_module],
                "ai_generated": alert.ai_generated,
            })
        except Exception:
            logger.exception("飞书通知异常")

    # ── 内部 ────────────────────────────────────────────────

    def _to_create_payload(self, result: RuleResult) -> AlertCreate:
        raw_log = "\n".join(result.raw_logs[-20:])
        fallback = f"{result.title}：{result.summary}"
        llm_summary = llm_service.summarize(result.raw_logs, fallback)
        # LLM 参与判断：返回值不等于原始模板文字说明 LLM 确实加工了
        ai_generated = bool(llm_summary and llm_summary.strip() != fallback.strip())
        return AlertCreate(
            level=result.level,
            title=result.title,
            summary=llm_summary,   # LLM 输出（不可用时自动降级为 fallback）
            source_module=result.source_module,
            raw_log=raw_log,
            llm_summary=llm_summary,
            ai_generated=ai_generated,
            status=AlertStatus.UNREAD,
        )

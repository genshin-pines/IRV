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
_DEFAULT_DEBOUNCE_SEC = 10.0


class AlertAgent:
    def __init__(self, broadcast: Broadcast | None = None, *, debounce_sec: float = _DEFAULT_DEBOUNCE_SEC) -> None:
        self.rule_engine = RuleEngine()
        self.broadcast = broadcast
        self.debounce_sec = debounce_sec
        self._last_fired: dict[str, float] = {}

    async def trigger(self) -> list[dict]:
        logs = get_collector().get_new_logs()
        if not logs:
            return []
        results = self.rule_engine.analyze(logs)
        if not results:
            return []

        now = time.monotonic()
        fresh = []
        for result in results:
            last = self._last_fired.get(result.rule_id, 0.0)
            if now - last >= self.debounce_sec:
                self._last_fired[result.rule_id] = now
                fresh.append(result)

        if not fresh:
            logger.debug("alert trigger debounced count=%s", len(results))
            return []

        from backend.database import SessionLocal

        alerts = []
        with SessionLocal() as db:
            for result in fresh:
                alert = create_alert(db, self._to_create_payload(result))
                message = self._to_message(alert)
                alerts.append(message)
                if self.broadcast:
                    await self.broadcast(message)
                asyncio.create_task(self._notify(alert))
        if alerts:
            logger.info("alert triggered count=%s", len(alerts))
        return alerts

    async def run_once(self, db: Session) -> list[dict]:
        logs = get_collector().get_new_logs()
        if not logs:
            return []
        results = self.rule_engine.analyze(logs)
        alerts = []
        for result in results:
            alert = create_alert(db, self._to_create_payload(result))
            message = self._to_message(alert)
            alerts.append(message)
            if self.broadcast:
                await self.broadcast(message)
            asyncio.create_task(self._notify(alert))
        if alerts:
            logger.info("alert generated count=%s", len(alerts))
        return alerts

    def _to_create_payload(self, result: RuleResult) -> AlertCreate:
        raw_log = "\n".join(result.raw_logs[-20:])
        fallback = f"{result.title}：{result.summary}"
        llm_summary = llm_service.summarize(result.raw_logs, fallback)
        ai_generated = bool(llm_summary and llm_summary.strip() != fallback.strip())
        return AlertCreate(
            level=result.level,
            title=result.title,
            summary=llm_summary or fallback,
            source_module=result.source_module,
            raw_log=raw_log,
            llm_summary=llm_summary,
            ai_generated=ai_generated,
            status=AlertStatus.UNREAD,
        )

    @staticmethod
    def _to_message(alert) -> dict:
        return {
            "id": alert.id,
            "level": alert.level,
            "title": alert.title,
            "summary": alert.summary,
            "source_module": alert.source_module,
            "created_at": alert.created_at.isoformat(),
        }

    async def _notify(self, alert) -> None:
        try:
            from backend.services.notifier import send_alert_notification

            await send_alert_notification(
                {
                    "level": alert.level,
                    "title": alert.title,
                    "summary": alert.summary,
                    "detail": alert.raw_log or "",
                    "source_module": alert.source_module,
                    "affected_modules": [alert.source_module],
                    "ai_generated": alert.ai_generated,
                }
            )
        except Exception:
            logger.exception("alert notification failed")

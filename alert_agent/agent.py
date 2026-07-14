from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import deque
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
_DEFAULT_DEBOUNCE_SEC = 15.0
_ANALYZE_WINDOW = 200

# 按规则差异化冷却（秒）—— 减少同类型告警的重复发送
# 高频规则（持久性条件、突发性强）给予更长冷却，避免告警风暴
_RULE_DEBOUNCE: dict[str, float] = {
    # ── 安全事件 / 快速感知（60s = 1min）──
    "login_fail": 60.0,
    "unauthorized_access": 60.0,
    "gesture_jitter": 60.0,
    "plate_low_conf": 60.0,
    # ── 持久性故障（300s = 5min）──
    "camera_disconnect": 300.0,
    "gesture_false_trigger": 300.0,
    "network_exception": 300.0,
    "llm_degradation": 300.0,
    # ── 复合告警（600s = 10min）──
    "mixed": 600.0,
    # ── 半持久性（180s = 3min）──
    "api_timeout": 180.0,
    # ── 管线/组件故障（120s = 2min）──
    "plate_pipeline_failure": 120.0,
    "database_exception": 120.0,
    "fusion_exception": 120.0,
    "traffic_police_anomaly": 120.0,
    "driver_assist_risk": 120.0,
    "gesture_low_conf": 120.0,
}

# 全局限流：每分钟最多发送多少条告警
_GLOBAL_RATE_LIMIT = 10
_GLOBAL_RATE_WINDOW = 60.0


class AlertAgent:
    def __init__(self, broadcast: Broadcast | None = None, *, debounce_sec: float = _DEFAULT_DEBOUNCE_SEC) -> None:
        self.rule_engine = RuleEngine()
        self.broadcast = broadcast
        self.debounce_sec = debounce_sec
        self._last_fired: dict[str, float] = {}          # rule_id → last fire time
        self._fingerprints: dict[str, float] = {}         # content hash → last fire time
        self._fire_times: deque[float] = deque()          # sliding window for global rate limit

    # ── 去重 / 限流 ──────────────────────────────────

    def _is_dup_by_fingerprint(self, rule_id: str, raw_logs: list[str]) -> bool:
        """内容指纹去重：同一 rule + 相同日志摘要视为重复告警"""
        seed = rule_id + "\n".join(sorted(raw_logs[:5]))
        fp = hashlib.sha256(seed.encode()).hexdigest()[:16]
        cooldown = _RULE_DEBOUNCE.get(rule_id, self.debounce_sec)
        now = time.monotonic()
        last = self._fingerprints.get(fp, 0.0)
        if now - last < cooldown:
            return True
        self._fingerprints[fp] = now
        self._cleanup_fingerprints(now)
        return False

    def _is_rate_limited(self) -> bool:
        """全局限流：滑动窗口内超过上限则丢弃"""
        now = time.monotonic()
        cutoff = now - _GLOBAL_RATE_WINDOW
        while self._fire_times and self._fire_times[0] < cutoff:
            self._fire_times.popleft()
        if len(self._fire_times) >= _GLOBAL_RATE_LIMIT:
            return True
        self._fire_times.append(now)
        return False

    def _cleanup_fingerprints(self, now: float) -> None:
        """定期清理过期指纹，避免内存泄漏"""
        cutoff = now - max(_RULE_DEBOUNCE.values()) * 3
        stale = [fp for fp, ts in self._fingerprints.items() if ts < cutoff]
        for fp in stale:
            del self._fingerprints[fp]

    # ── 核心逻辑 ──────────────────────────────────────

    async def trigger(self) -> list[dict]:
        try:
            return await self._do_trigger()
        except Exception:
            logger.exception("alert trigger failed")
            return []

    async def _do_trigger(self) -> list[dict]:
        new_logs = get_collector().get_new_logs()
        if not new_logs:
            return []
        logs, _ = get_collector().query(page_size=_ANALYZE_WINDOW)
        results = self.rule_engine.analyze(logs)
        if not results:
            return []

        now = time.monotonic()
        fresh = []
        for result in results:
            cooldown = _RULE_DEBOUNCE.get(result.rule_id, self.debounce_sec)
            last = self._last_fired.get(result.rule_id, 0.0)
            if now - last < cooldown:
                continue
            if self._is_dup_by_fingerprint(result.rule_id, result.raw_logs):
                continue
            self._last_fired[result.rule_id] = now
            fresh.append(result)

        if not fresh:
            logger.debug("alert trigger debounced count=%s rule_ids=%s",
                         len(results), [r.rule_id for r in results])
            return []

        from backend.database import SessionLocal

        alerts = []
        with SessionLocal() as db:
            for result in fresh:
                if self._is_rate_limited():
                    logger.warning("alert rate limit hit, dropping rule=%s", result.rule_id)
                    continue
                alert = create_alert(db, self._to_create_payload(result))
                message = self._to_message(alert)
                alerts.append(message)
                if self.broadcast:
                    await self.broadcast(message)
                asyncio.create_task(self._notify(alert))
        if alerts:
            logger.info("alert triggered count=%s rules=%s",
                        len(alerts), [a["title"] for a in alerts])
        return alerts

    async def run_once(self, db: Session) -> list[dict]:
        new_logs = get_collector().get_new_logs()
        if not new_logs:
            return []
        logs, _ = get_collector().query(page_size=_ANALYZE_WINDOW)
        results = self.rule_engine.analyze(logs)

        now = time.monotonic()
        fresh = []
        for result in results:
            cooldown = _RULE_DEBOUNCE.get(result.rule_id, self.debounce_sec)
            last = self._last_fired.get(result.rule_id, 0.0)
            if now - last < cooldown:
                continue
            if self._is_dup_by_fingerprint(result.rule_id, result.raw_logs):
                continue
            self._last_fired[result.rule_id] = now
            fresh.append(result)

        if not fresh:
            return []

        alerts = []
        for result in fresh:
            if self._is_rate_limited():
                logger.warning("alert rate limit hit, dropping rule=%s", result.rule_id)
                continue
            alert = create_alert(db, self._to_create_payload(result))
            message = self._to_message(alert)
            alerts.append(message)
            if self.broadcast:
                await self.broadcast(message)
            asyncio.create_task(self._notify(alert))
        if alerts:
            logger.info("alert generated count=%s rules=%s",
                        len(alerts), [a["title"] for a in alerts])
        return alerts

    _SKIP_LLM_RULES = frozenset({"llm_degradation"})

    def _to_create_payload(self, result: RuleResult) -> AlertCreate:
        raw_log = "\n".join(result.raw_logs[-20:])
        fallback = result.summary  # 规则模板已包含四段结构化文本
        if result.rule_id in self._SKIP_LLM_RULES:
            llm_summary = ""
            ai_generated = False
        else:
            llm_summary = llm_service.summarize(result.raw_logs, fallback)
            ai_generated = bool(llm_summary and llm_summary.strip() != fallback.strip())
        return AlertCreate(
            level=result.level,
            title=result.title,
            summary=fallback,
            source_module=result.source_module,
            raw_log=raw_log,
            llm_summary=llm_summary if ai_generated else "",
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

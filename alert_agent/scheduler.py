from __future__ import annotations

import asyncio
from contextlib import suppress
import logging

from alert_agent.agent import AlertAgent, Broadcast
from backend.config import AGENT_POLL_INTERVAL_SEC
from backend.database import SessionLocal
from backend.services.alert_service import cleanup_old_alerts
from backend.services.log_service import get_collector


logger = logging.getLogger("alert_agent")
_task: asyncio.Task | None = None
_agent: AlertAgent | None = None
_event_loop: asyncio.AbstractEventLoop | None = None


async def _poll_loop(agent: AlertAgent) -> None:
    """兜底轮询：处理事件驱动遗漏的日志 + 定期清理过期告警"""
    while True:
        try:
            with SessionLocal() as db:
                await agent.run_once(db)
                cleanup_old_alerts(db)
        except Exception:
            logger.exception("alert agent poll failed")
        await asyncio.sleep(AGENT_POLL_INTERVAL_SEC)


def _on_log_emitted() -> None:
    """日志写入回调 — 跨线程安全调度到事件循环"""
    if _agent is not None and _event_loop is not None and _event_loop.is_running():
        asyncio.run_coroutine_threadsafe(_agent.trigger(), _event_loop)


def start_scheduler(broadcast: Broadcast | None = None) -> None:
    global _task, _agent, _event_loop
    if _task and not _task.done():
        return

    _agent = AlertAgent(broadcast=broadcast)
    _event_loop = asyncio.get_running_loop()

    # 兜底轮询（低频，处理漏网之鱼 + 数据清理）
    _task = asyncio.create_task(_poll_loop(_agent), name="alert-agent-poll")

    # 核心：事件驱动 — 日志写入立即触发
    get_collector().on_emit = _on_log_emitted

    logger.info("alert agent started (event-driven + poll fallback every %ss)", AGENT_POLL_INTERVAL_SEC)


async def stop_scheduler() -> None:
    global _task
    get_collector().on_emit = None
    if _task:
        _task.cancel()
        with suppress(asyncio.CancelledError):
            await _task
        _task = None
        logger.info("alert agent scheduler stopped")


def agent_status() -> dict:
    return {
        "running": bool(_task and not _task.done()),
        "interval_seconds": AGENT_POLL_INTERVAL_SEC,
    }

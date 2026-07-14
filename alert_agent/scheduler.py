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


async def _loop(agent: AlertAgent) -> None:
    while True:
        try:
            with SessionLocal() as db:
                await agent.run_once(db)
                cleanup_old_alerts(db)
        except Exception:
            logger.exception("alert agent loop failed")
        await asyncio.sleep(AGENT_POLL_INTERVAL_SEC)


def _on_log_emitted() -> None:
    if _agent is not None and _event_loop is not None and _event_loop.is_running():
        future = asyncio.run_coroutine_threadsafe(_agent.trigger(), _event_loop)
        # 捕获异步异常，避免静默丢失
        def _log_exc(fut):
            exc = fut.exception()
            if exc is not None:
                logger.error("alert trigger coroutine failed: %s", exc)
        future.add_done_callback(_log_exc)


def start_scheduler(broadcast: Broadcast | None = None) -> None:
    global _task, _agent, _event_loop
    if _task and not _task.done():
        return
    _agent = AlertAgent(broadcast=broadcast)
    _event_loop = asyncio.get_running_loop()
    get_collector().on_emit = _on_log_emitted
    _task = asyncio.create_task(_loop(_agent), name="alert-agent-scheduler")
    logger.info("alert agent scheduler started with event trigger")


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

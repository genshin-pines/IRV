from __future__ import annotations

import asyncio
from contextlib import suppress
import logging

from alert_agent.agent import AlertAgent, Broadcast
from backend.config import AGENT_POLL_INTERVAL_SEC
from backend.database import SessionLocal
from backend.services.alert_service import cleanup_old_alerts


logger = logging.getLogger("alert_agent")
_task: asyncio.Task | None = None
_agent: AlertAgent | None = None


async def _loop(agent: AlertAgent) -> None:
    while True:
        try:
            with SessionLocal() as db:
                await agent.run_once(db)
                cleanup_old_alerts(db)
        except Exception:
            logger.exception("alert agent loop failed")
        await asyncio.sleep(AGENT_POLL_INTERVAL_SEC)


def start_scheduler(broadcast: Broadcast | None = None) -> None:
    global _task, _agent
    if _task and not _task.done():
        return
    _agent = AlertAgent(broadcast=broadcast)
    _task = asyncio.create_task(_loop(_agent), name="alert-agent-scheduler")
    logger.info("alert agent scheduler started")


async def stop_scheduler() -> None:
    global _task
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

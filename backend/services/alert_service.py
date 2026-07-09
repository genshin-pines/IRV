"""
告警服务 — 桥接 AlertAgent 与 FastAPI
@owner 成员E

职责:
  1. 创建并管理 AlertAgent 生命周期
  2. 创建并管理 FusionAgent + EventBus 生命周期
  3. 告警回调：将 Agent 产生的告警写入数据库 + WebSocket 广播
  4. 融合回调：将 FusionAgent 产生的驾驶建议推送 WebSocket
  5. 提供查询接口（供 API router 调用）
"""

import asyncio
import logging
from typing import List, Dict, Optional, Callable

from sqlmodel import Session, select

from alert_agent.agent import AlertAgent, create_agent
from alert_agent.llm_client import LLMClient
from backend.config import (
    DEEPSEEK_API_KEY, LLM_PROVIDER,
    AGENT_POLL_INTERVAL_SEC, LOG_COLLECTOR_CAPACITY,
)
from backend.models.alert_event import AlertEvent
from backend.services.log_service import get_collector

logger = logging.getLogger(__name__)

# 模块级单例
_agent: AlertAgent | None = None
_event_bus: object | None = None
_fusion_agent: object | None = None
_ws_broadcast: Callable[[Dict], None] | None = None


async def setup_alert_agent(
    engine,
    ws_broadcast: Optional[Callable[[Dict], None]] = None,
):
    """
    初始化告警 Agent 并启动后台巡检。

    在 FastAPI startup 事件中调用。

    Args:
        engine: SQLModel engine（用于写入告警事件）
        ws_broadcast: WebSocket 广播函数（签名: async fn(alert_dict)）
    """
    global _agent, _ws_broadcast
    _ws_broadcast = ws_broadcast

    # 创建 LLM 客户端
    client: Optional[LLMClient] = None
    if DEEPSEEK_API_KEY:
        try:
            from alert_agent.llm_client import create_client
            client = create_client(LLM_PROVIDER, api_key=DEEPSEEK_API_KEY)
            logger.info(f"LLM 客户端就绪: {LLM_PROVIDER}")
        except Exception as e:
            logger.warning(f"LLM 客户端创建失败，降级为纯规则模式: {e}")

    # 创建 Agent
    collector = get_collector()
    _agent = create_agent(
        client=client,
        collector=collector,
        use_llm=client is not None,
        alert_callback=make_alert_callback(engine),
    )
    _agent.start(interval=AGENT_POLL_INTERVAL_SEC)
    logger.info(f"Alert Agent 已启动, 巡检间隔={AGENT_POLL_INTERVAL_SEC}s, LLM={'启用' if client else '禁用'}")


async def stop_alert_agent():
    """停止 Agent（FastAPI shutdown 事件）"""
    global _agent
    if _agent:
        _agent.stop()
        _agent = None


def get_agent() -> AlertAgent | None:
    """获取当前 Agent 实例"""
    return _agent


def get_alert_history(
    session: Session,
    hours: int = 24,
    level: Optional[str] = None,
    limit: int = 50,
) -> List[AlertEvent]:
    """查询告警历史"""
    stmt = select(AlertEvent).order_by(AlertEvent.created_at.desc()).limit(limit)
    if level:
        stmt = stmt.where(AlertEvent.level == level)
    return list(session.exec(stmt).all())


def get_alert_stats(session: Session) -> Dict:
    """告警统计"""
    alerts = list(session.exec(select(AlertEvent)).all())
    by_level = {"info": 0, "warning": 0, "critical": 0}
    by_module: Dict[str, int] = {}
    for a in alerts:
        by_level[a.level] = by_level.get(a.level, 0) + 1
        for mod in a.affected_modules.split(","):
            mod = mod.strip()
            if mod:
                by_module[mod] = by_module.get(mod, 0) + 1

    return {
        "total": len(alerts),
        "by_level": by_level,
        "by_module": by_module,
        "acknowledged": sum(1 for a in alerts if a.acknowledged),
        "unacknowledged": sum(1 for a in alerts if not a.acknowledged),
    }


def make_alert_callback(engine):
    """
    创建告警回调：写入数据库 + WebSocket 广播。
    每条 Agent 产生的告警都会经过这里。

    注意：Agent 在 daemon 线程中运行，回调可能在不同线程被调用，
    因此返回的是同步函数，内部通过 asyncio.run_coroutine_threadsafe
    将异步操作调度到主事件循环。
    """
    # 捕获 FastAPI 主事件循环
    try:
        main_loop = asyncio.get_running_loop()
    except RuntimeError:
        main_loop = None
        logger.warning("无法获取运行中的事件循环，将创建新循环")

    async def _async_callback(alert: Dict):
        # 1. 写入数据库
        try:
            with Session(engine) as session:
                record = AlertEvent(
                    level=alert.get("level", "info"),
                    title=alert.get("title", ""),
                    summary=alert.get("summary", ""),
                    detail=alert.get("detail", ""),
                    source_module=alert.get("affected_modules", [None])[0] or "",
                    affected_modules=",".join(alert.get("affected_modules", [])),
                    ai_generated=alert.get("ai_generated", False),
                    fingerprint=alert.get("_fingerprint", ""),
                    webhook_markdown=alert.get("webhook_markdown", ""),
                    ttl_minutes=alert.get("ttl_minutes", 60),
                )
                session.add(record)
                session.commit()
        except Exception as e:
            logger.error(f"告警落库失败: {e}")

        # 2. WebSocket 广播（如果前端已连接）
        if _ws_broadcast:
            try:
                ws_msg = alert.get("websocket") or {
                    "type": "alert",
                    "level": alert.get("level"),
                    "title": alert.get("title"),
                    "message": alert.get("summary"),
                }
                await _ws_broadcast(ws_msg)
            except Exception as e:
                logger.error(f"WebSocket 广播失败: {e}")

        # 3. 飞书通知（CRITICAL/WARNING → 群消息）
        try:
            from backend.services.notifier import send_alert_notification
            await send_alert_notification(alert)
        except Exception as e:
            logger.error(f"飞书通知失败: {e}")

    def callback(alert: Dict):
        """同步回调 — Agent 线程安全调用"""
        if main_loop and main_loop.is_running():
            # 在主事件循环中调度执行
            asyncio.run_coroutine_threadsafe(_async_callback(alert), main_loop)
        else:
            # 降级：新事件循环执行（不涉及 WebSocket 的场景）
            try:
                asyncio.run(_async_callback(alert))
            except RuntimeError:
                logger.warning("事件循环冲突，跳过 WebSocket 推送")
                # 至少尝试写数据库
                try:
                    with Session(engine) as session:
                        record = AlertEvent(
                            level=alert.get("level", "info"),
                            title=alert.get("title", ""),
                            summary=alert.get("summary", ""),
                            detail=alert.get("detail", ""),
                            source_module=alert.get("affected_modules", [None])[0] or "",
                            affected_modules=",".join(alert.get("affected_modules", [])),
                            ai_generated=alert.get("ai_generated", False),
                            fingerprint=alert.get("_fingerprint", ""),
                            webhook_markdown=alert.get("webhook_markdown", ""),
                            ttl_minutes=alert.get("ttl_minutes", 60),
                        )
                        session.add(record)
                        session.commit()
                except Exception as e:
                    logger.error(f"告警落库失败: {e}")

    return callback


# ═══════════════════════════════════════════════════════════
# 融合引擎生命周期管理
# ═══════════════════════════════════════════════════════════

async def setup_fusion_engine(
    engine,
    ws_broadcast=None,
    *,
    use_llm: bool = True,
    window_seconds: float = 2.0,
    dedup_ms: int = 500,
):
    """
    初始化 FusionAgent + EventBus。

    在 FastAPI startup 事件中调用（setup_alert_agent 之后）。

    Args:
        engine: SQLModel engine
        ws_broadcast: WebSocket 广播函数
        use_llm: 是否启用 LLM 融合推理
        window_seconds: 滑动窗口大小（秒）
        dedup_ms: 防抖间隔（毫秒）
    """
    global _event_bus, _fusion_agent

    from fusion import AsyncEventBus, FusionAgent

    # 创建事件总线
    _event_bus = AsyncEventBus(window_seconds=window_seconds)
    logger.info(f"EventBus 已创建: window={window_seconds}s")

    # 创建 LLM 客户端
    llm_client = None
    if use_llm and DEEPSEEK_API_KEY:
        try:
            from alert_agent.llm_client import create_client
            llm_client = create_client(LLM_PROVIDER, api_key=DEEPSEEK_API_KEY)
            logger.info(f"Fusion LLM 客户端就绪: {LLM_PROVIDER}")
        except Exception as e:
            logger.warning(f"Fusion LLM 客户端创建失败，降级为纯规则模式: {e}")

    # 创建融合结果回调（WebSocket 推送 + 可选 DB 写入）
    async def on_fusion_result(result):
        """融合结果回调：WebSocket 推送驾驶建议"""
        if ws_broadcast:
            try:
                await ws_broadcast(result.to_websocket())
            except Exception as e:
                logger.error(f"融合结果 WebSocket 推送失败: {e}")

        # 如果有融合告警，也推送
        alert = result.to_alert()
        if alert and ws_broadcast:
            try:
                await ws_broadcast(alert)
            except Exception as e:
                logger.error(f"融合告警 WebSocket 推送失败: {e}")

    # 创建 FusionAgent
    _fusion_agent = FusionAgent(
        event_bus=_event_bus,
        llm_client=llm_client,
        use_llm=use_llm and llm_client is not None,
        dedup_interval_ms=dedup_ms,
        result_callback=on_fusion_result,
    )
    await _fusion_agent.start()
    logger.info(
        f"FusionAgent 已启动: LLM={'启用' if use_llm and llm_client else '禁用'}, "
        f"window={window_seconds}s, dedup={dedup_ms}ms"
    )


async def stop_fusion_engine():
    """停止融合引擎（FastAPI shutdown 事件）"""
    global _fusion_agent, _event_bus
    if _fusion_agent:
        await _fusion_agent.stop()
        _fusion_agent = None
    _event_bus = None


def get_event_bus():
    """获取事件总线实例（供 API router 使用）"""
    return _event_bus


def get_fusion_agent():
    """获取融合引擎实例（供 API router 使用）"""
    return _fusion_agent

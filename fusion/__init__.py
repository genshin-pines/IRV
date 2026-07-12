"""
融合推理模块 — 跨模块感知事件融合与驾驶建议生成

职责:
  1. 接收三路感知事件（车牌识别 + 交警手势 + 车主手势）
  2. 事件驱动融合推理（LLM + 规则引擎 fallback）
  3. 生成综合驾驶建议（driving_suggestion）+ 融合告警（fusion_alert）
  4. 延迟追踪与实时性监控

用法:
    from fusion import (
        AsyncEventBus, PerceptionEvent, FusionAgent,
        EventType, Module,
    )

    event_bus = AsyncEventBus(window_seconds=2.0)
    agent = FusionAgent(event_bus, llm_client, result_callback=on_result)
    await agent.start()
"""

from .perception_event import (
    PerceptionEvent,
    EventType,
    Module,
    FusionResult,
    SuggestionAction,
    Urgency,
)
from .event_bus import AsyncEventBus
from .fusion_agent import FusionAgent
from .latency_tracker import PipelineLatency, LatencyTracker

__all__ = [
    # ── 事件模型 ──
    "PerceptionEvent",
    "EventType",
    "Module",
    "FusionResult",
    "SuggestionAction",
    "Urgency",
    # ── 事件总线 ──
    "AsyncEventBus",
    # ── 融合引擎 ──
    "FusionAgent",
    # ── 延迟追踪 ──
    "PipelineLatency",
    "LatencyTracker",
]

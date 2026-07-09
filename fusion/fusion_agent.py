"""
融合推理引擎 — 事件驱动的跨模块感知融合 Agent

核心职责:
  1. 订阅 AsyncEventBus 的 perception.* 主题
  2. 每收到事件 → 收集滑动窗口 → 防抖 → 融合推理
  3. LLM 路径：调用 DeepSeek/Kimi/GPT-4o 进行语义级融合推理
  4. 规则路径：LLM 不可用时降级为 rule_fusion.py 规则引擎
  5. 输出 FusionResult → 通过 result_callback 推送

延迟优化:
  - 防抖机制：500ms 内最多一次 LLM 调用
  - 异步执行：LLM 调用通过 asyncio.to_thread 不阻塞事件循环
  - 规则先行：每次先跑规则引擎，结果立即可用
  - LLM 增强：LLM 结果到达后覆盖规则结果（如果更优）

用法:
    from fusion import AsyncEventBus, FusionAgent
    from alert_agent.llm_client import create_client

    bus = AsyncEventBus(window_seconds=2.0)
    client = create_client("deepseek")
    agent = FusionAgent(bus, client, result_callback=on_result)
    await agent.start()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Callable, Awaitable, Dict, List, Optional

from .event_bus import AsyncEventBus
from .perception_event import (
    PerceptionEvent, FusionResult, Module,
    SuggestionAction, Urgency,
)
from .rule_fusion import rule_based_fusion
from .latency_tracker import LatencyTracker

logger = logging.getLogger(__name__)

# 结果回调签名: async fn(result: FusionResult) -> None
ResultCallback = Callable[[FusionResult], Awaitable[None]]


class FusionAgent:
    """
    事件驱动的融合推理引擎。

    工作流程:
      1. 订阅 EventBus ("perception.*")
      2. on_event → 收集窗口 → 检查防抖 → asyncio.create_task(fuse)
      3. fuse → rule_fusion (立即可用) → [LLM fusion (异步增强)]
      4. 结果通过 result_callback 推送

    Args:
        event_bus: 感知事件总线
        llm_client: LLM 客户端（为 None 时纯规则模式）
        use_llm: 是否启用 LLM 融合推理
        dedup_interval_ms: 防抖间隔（毫秒），避免 LLM 过度调用
        result_callback: 融合结果回调（异步），用于 WebSocket 推送 + DB 写入
    """

    def __init__(
        self,
        event_bus: AsyncEventBus,
        llm_client: Optional[object] = None,
        *,
        use_llm: bool = True,
        dedup_interval_ms: int = 500,
        result_callback: Optional[ResultCallback] = None,
    ):
        self._bus = event_bus
        self._llm_client = llm_client
        self._use_llm = use_llm and llm_client is not None
        self._dedup_ms = dedup_interval_ms
        self._result_callback = result_callback

        # 状态
        self._running = False
        self._last_fusion_at: float = 0.0
        self._fusion_count: int = 0
        self._llm_error_count: int = 0
        self._llm_available: bool = True
        self._pending_task: Optional[asyncio.Task] = None

        # 延迟追踪
        self._latency_tracker = LatencyTracker(window_size=200)

        # 最近一次结果（供 API 查询）
        self._latest_result: Optional[FusionResult] = None

    # ── 生命周期 ──────────────────────────────────────────────

    async def start(self):
        """订阅 EventBus 并开始监听"""
        if self._running:
            return
        self._running = True
        self._bus.subscribe("perception.*", self._on_event)
        logger.info(
            f"FusionAgent 已启动: LLM={'启用' if self._use_llm else '禁用'}, "
            f"防抖={self._dedup_ms}ms"
        )

    async def stop(self):
        """停止监听"""
        self._running = False
        if self._pending_task:
            self._pending_task.cancel()
        logger.info("FusionAgent 已停止")

    # ── 事件处理 ──────────────────────────────────────────────

    async def _on_event(self, event: PerceptionEvent):
        """
        接收单个感知事件（由 EventBus 回调）。

        防抖逻辑:
          - 距上次融合 < dedup_ms → 跳过（防止 LLM 过载）
          - 否则 → 异步启动融合任务
        """
        if not self._running:
            return

        # 记录事件发布到融合开始的时间
        self._latency_tracker.start(event.event_id, event.frame_timestamp)
        self._latency_tracker.record(event.event_id, "event_published")
        self._latency_tracker.record(event.event_id, "fusion_started")

        # 防抖检查
        now = time.monotonic()
        since_last = (now - self._last_fusion_at) * 1000
        if since_last < self._dedup_ms:
            logger.debug(f"防抖: 距上次融合 {since_last:.0f}ms < {self._dedup_ms}ms, 跳过")
            return

        self._last_fusion_at = now

        # 异步启动融合（fire-and-forget，不阻塞事件循环）
        self._pending_task = asyncio.create_task(self._fuse_and_publish(event.event_id))

    # ── 融合核心 ──────────────────────────────────────────────

    async def _fuse_and_publish(self, trigger_event_id: str):
        """执行融合推理并发布结果"""
        try:
            # 收集滑动窗口上下文
            context = await self._bus.get_context()

            if context["window_size"] == 0:
                logger.debug("滑动窗口为空，跳过融合")
                return

            # Step 1: 规则引擎 — 立即执行
            events = await self._bus.get_window()
            rule_result = rule_based_fusion(events)

            # Step 2: LLM 增强 — 如果可用且场景有意义
            llm_result: Optional[FusionResult] = None
            if self._use_llm and self._llm_available:
                try:
                    llm_result = await self._llm_fuse(context)
                except Exception as e:
                    logger.warning(f"LLM 融合推理失败: {e}")
                    self._llm_error_count += 1
                    if self._llm_error_count >= 3:
                        self._llm_available = False
                        logger.warning("LLM 连续失败 ≥3 次，降级为纯规则模式")

            # Step 3: 选择最优结果 — LLM > 规则 > 默认
            result = llm_result or rule_result

            if result is None:
                # 有事件但无匹配场景，生成默认正常结果
                result = self._default_result(context)

            result.latency_ms = self._latency_tracker.total_latency_ms if hasattr(
                self._latency_tracker, 'total_latency_ms'
            ) else 0
            self._fusion_count += 1
            self._latest_result = result

            # 完成延迟追踪
            self._latency_tracker.record(trigger_event_id, "fusion_completed")
            self._latency_tracker.finish(trigger_event_id)

            # 发布结果
            if self._result_callback:
                try:
                    await self._result_callback(result)
                    self._latency_tracker.record(trigger_event_id, "ws_broadcast")
                except Exception as e:
                    logger.error(f"结果回调异常: {e}")

            logger.info(
                f"融合 #{self._fusion_count}: action={result.action.value}, "
                f"urgency={result.urgency.value}, confidence={result.confidence:.0%}, "
                f"LLM={result.ai_generated}"
            )

        except asyncio.CancelledError:
            logger.debug("融合任务被取消")
        except Exception as e:
            logger.error(f"融合推理异常: {e}", exc_info=True)

    # ── LLM 融合 ──────────────────────────────────────────────

    async def _llm_fuse(self, context: Dict) -> Optional[FusionResult]:
        """
        调用 LLM 进行语义级融合推理。

        通过 asyncio.to_thread 将同步 LLM 调用包装为异步，
        避免阻塞 FastAPI 事件循环。
        """
        if not self._llm_client:
            return None

        from .fusion_prompts import FUSION_REASONING_PROMPT

        # 格式化感知状态为 JSON 字符串
        perception_state = self._format_for_llm(context)

        t0 = time.perf_counter()
        self._llm_call_started = t0

        try:
            # 异步包装同步 LLM 调用（避免阻塞事件循环）
            llm_response = await asyncio.to_thread(
                self._llm_client.chat_json,
                user_message=json.dumps(perception_state, ensure_ascii=False),
                system_prompt=FUSION_REASONING_PROMPT.replace(
                    "{perception_state}",
                    json.dumps(perception_state, ensure_ascii=False, indent=2),
                ),
                temperature=0.3,
                max_tokens=2048,
            )
        except Exception:
            self._llm_available = False
            raise

        llm_elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"LLM 融合推理完成: {llm_elapsed:.0f}ms")

        # 解析 LLM 响应为 FusionResult
        return self._parse_llm_response(llm_response, context)

    def _parse_llm_response(self, response: Dict, context: Dict) -> Optional[FusionResult]:
        """解析 LLM 返回的 JSON 为 FusionResult"""
        suggestion = response.get("suggestion", {})
        if not suggestion:
            return None

        action_str = suggestion.get("action", "normal")
        try:
            action = SuggestionAction(action_str)
        except ValueError:
            action = SuggestionAction.NORMAL

        urgency_str = suggestion.get("urgency", "low")
        try:
            urgency = Urgency(urgency_str)
        except ValueError:
            urgency = Urgency.LOW

        # 收集相关事件 ID
        related_events = self._collect_related_event_ids(context)

        return FusionResult(
            action=action,
            reasoning=suggestion.get("reasoning", ""),
            confidence=float(suggestion.get("confidence", 0.5)),
            urgency=urgency,
            related_events=related_events,
            scene_summary=response.get("scene_summary", ""),
            alerts=response.get("alerts", []),
            ai_generated=True,
        )

    # ── 辅助方法 ──────────────────────────────────────────────

    def _format_for_llm(self, context: Dict) -> Dict:
        """将融合上下文格式化为 LLM 可读的 JSON"""
        formatted = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "window_events": context["window_size"],
            "modules": {},
        }

        for key in ("plate", "traffic_gesture", "driver_gesture"):
            ctx = context.get(key, {})
            latest = ctx.get("latest")
            formatted["modules"][key] = {
                "has_data": latest is not None,
                "latest_event": latest.to_dict() if latest else None,
                "recent_count_2s": ctx.get("count_2s", 0),
                "avg_confidence": ctx.get("avg_confidence", 0),
                "stable": ctx.get("stable_1s", False),
            }
            if key == "plate":
                formatted["modules"][key]["all_plates"] = ctx.get("all_plates", [])

        return formatted

    @staticmethod
    def _collect_related_event_ids(context: Dict) -> List[str]:
        """收集所有模块最新事件的 ID"""
        ids = []
        for key in ("plate", "traffic_gesture", "driver_gesture"):
            latest = context.get(key, {}).get("latest")
            if latest:
                ids.append(latest.event_id)
        return ids

    @staticmethod
    def _default_result(context: Dict) -> FusionResult:
        """生成默认正常结果（有事件但无匹配场景时）"""
        desc_parts = []
        for key, label in [("plate", "车牌检测"), ("traffic_gesture", "交警手势"), ("driver_gesture", "车主手势")]:
            if context.get(key, {}).get("latest"):
                desc_parts.append(label)
        scene = "、".join(desc_parts) + "正常" if desc_parts else "系统运行正常"
        return FusionResult(
            action=SuggestionAction.NORMAL,
            reasoning=f"感知模块运行正常，当前场景无异常。{scene}",
            confidence=0.9,
            urgency=Urgency.LOW,
            related_events=[],
            scene_summary=scene,
            alerts=[],
            ai_generated=False,
        )

    # ── 属性 ──────────────────────────────────────────────────

    @property
    def status(self) -> Dict:
        """融合引擎状态摘要"""
        return {
            "running": self._running,
            "fusion_count": self._fusion_count,
            "llm_available": self._llm_available,
            "llm_enabled": self._use_llm,
            "llm_error_count": self._llm_error_count,
            "dedup_interval_ms": self._dedup_ms,
            "last_fusion_at": datetime.fromtimestamp(
                self._last_fusion_at, tz=timezone.utc
            ).isoformat() if self._last_fusion_at else None,
            "latest_result": self._latest_result.to_websocket() if self._latest_result else None,
        }

    @property
    def latency_stats(self) -> Dict:
        """延迟统计"""
        return self._latency_tracker.get_stats()

    @property
    def latest_result(self) -> Optional[FusionResult]:
        """最近一次融合结果"""
        return self._latest_result

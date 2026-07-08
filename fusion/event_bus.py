"""
异步事件总线 — 感知事件的发布/订阅 + 滑动窗口

设计:
  - 基于 asyncio.Queue 的事件分发
  - 滑动窗口自动维护最近 N 秒的感知上下文
  - 支持通配符主题订阅 (perception.* / perception.plate 等)
  - 线程安全：通过 asyncio.run_coroutine_threadsafe 支持同步线程发布
  - 并发控制：asyncio.Lock 保护窗口操作

用法:
    bus = AsyncEventBus(window_seconds=2.0)

    async def on_plate(event: PerceptionEvent):
        print(f"检测到车牌: {event.plate_code}")

    bus.subscribe("perception.plate", on_plate)
    await bus.publish(event)
    recent = await bus.get_window(Module.PLATE_RECOGNITION)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Callable, Awaitable, Dict, List, Optional, Set

from .perception_event import PerceptionEvent, Module, EventType

logger = logging.getLogger(__name__)

# 订阅者回调签名: async fn(event: PerceptionEvent) -> None
Subscriber = Callable[[PerceptionEvent], Awaitable[None]]


class AsyncEventBus:
    """
    基于 asyncio 的感知事件发布/订阅总线。

    特性:
      - 主题订阅：支持 "perception.*" / "perception.plate" 等通配符
      - 滑动窗口：维护最近 window_seconds 秒的所有事件
      - 并发发布：多个协程可同时 publish，窗口操作有锁保护
      - Fan-out：每个事件并发推送给所有匹配订阅者
    """

    def __init__(self, window_seconds: float = 2.0):
        self._window_seconds = window_seconds
        self._subscribers: Dict[str, List[Subscriber]] = {}
        self._window: deque[PerceptionEvent] = deque()
        self._lock = asyncio.Lock()
        self._event_count: int = 0
        self._started_at: float = time.perf_counter()

    # ── 订阅管理 ──────────────────────────────────────────────

    def subscribe(self, topic: str, callback: Subscriber) -> None:
        """
        订阅指定主题的感知事件。

        Args:
            topic: 主题模式
                "perception.*"          — 所有感知事件
                "perception.plate"      — 仅车牌事件
                "perception.traffic_gesture" — 仅交警手势
                "perception.driver_gesture"  — 仅车主手势
            callback: 异步回调函数，签名为 async fn(event) -> None
        """
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)
        logger.info(f"订阅者已注册: topic={topic}, 当前订阅数={len(self._subscribers[topic])}")

    def unsubscribe(self, topic: str, callback: Subscriber) -> None:
        """取消订阅"""
        if topic in self._subscribers:
            try:
                self._subscribers[topic].remove(callback)
            except ValueError:
                pass

    # ── 事件发布 ──────────────────────────────────────────────

    async def publish(self, event: PerceptionEvent) -> None:
        """
        发布一个感知事件到总线。

        1. 加入滑动窗口（锁保护）
        2. 匹配订阅者主题
        3. 并发推送给所有匹配的订阅者

        线程安全：可以从同步线程调用（见 publish_threadsafe）
        """
        # Step 1: 加入滑动窗口
        async with self._lock:
            self._window.append(event)
            self._event_count += 1
            self._trim_window()

        # Step 2: 找到匹配的订阅者
        matched = self._match_subscribers(event)

        if not matched:
            return

        # Step 3: 并发推送给所有匹配订阅者
        results = await asyncio.gather(
            *(self._safe_invoke(sub, event) for sub in matched),
            return_exceptions=True,
        )

        for r in results:
            if isinstance(r, Exception):
                logger.error(f"订阅者回调异常: {r}")

    def publish_threadsafe(self, event: PerceptionEvent,
                           loop: asyncio.AbstractEventLoop) -> None:
        """
        从同步线程发布事件（fire-and-forget）。

        Args:
            event: 感知事件
            loop: FastAPI 主事件循环（asyncio.get_running_loop()）
        """
        asyncio.run_coroutine_threadsafe(self.publish(event), loop)

    # ── 滑动窗口查询 ──────────────────────────────────────────

    async def get_window(self, module: Optional[Module] = None,
                         event_type: Optional[EventType] = None,
                         max_age_seconds: Optional[float] = None) -> List[PerceptionEvent]:
        """
        获取滑动窗口内的感知事件。

        Args:
            module: 按模块筛选（None = 全部）
            event_type: 按事件类型筛选
            max_age_seconds: 最大事件年龄（默认使用配置的 window_seconds）

        Returns:
            按时间排序的事件列表（最早的在前）
        """
        async with self._lock:
            self._trim_window()
            events = list(self._window)

        max_age = max_age_seconds or self._window_seconds
        cutoff = time.monotonic() - max_age

        result = []
        for e in events:
            # 按年龄过滤
            age = time.monotonic() - e.frame_timestamp if e.frame_timestamp else 0
            if e.frame_timestamp and age > max_age:
                continue
            # 按模块过滤
            if module and e.module != module:
                continue
            # 按事件类型过滤
            if event_type and e.event_type != event_type:
                continue
            result.append(e)

        return result

    def get_window_snapshot(self) -> List[PerceptionEvent]:
        """
        获取滑动窗口快照（同步版本，用于测试/调试）。
        注意：不加锁，可能返回不完整快照。生产环境请用 get_window()。
        """
        self._trim_window_sync()
        return list(self._window)

    async def get_context(self) -> Dict[str, Dict]:
        """
        获取融合上下文 — 窗口内各模块的感知状态摘要。

        Returns:
            {
              "plate": {
                "latest": PerceptionEvent | None,
                "recent": [...],         # 最近3个事件
                "count_2s": N,           # 2秒内事件数
                "avg_confidence": float,
                "all_plates": ["京A12345", ...],
              },
              "traffic_gesture": { ... },
              "driver_gesture": { ... },
              "window_size": N,          # 窗口内事件总数
            }
        """
        async with self._lock:
            self._trim_window()
            events = list(self._window)

        return self._build_context(events)

    # ── 统计 ──────────────────────────────────────────────────

    @property
    def stats(self) -> Dict:
        """事件总线统计"""
        return {
            "event_count": self._event_count,
            "window_size": len(self._window),
            "window_seconds": self._window_seconds,
            "subscribers": {k: len(v) for k, v in self._subscribers.items()},
            "uptime_seconds": round(time.perf_counter() - self._started_at, 1),
        }

    # ── 内部方法 ──────────────────────────────────────────────

    def _trim_window(self) -> None:
        """移除窗口外的事件（需在锁内调用）"""
        cutoff = time.monotonic() - self._window_seconds
        while self._window:
            oldest = self._window[0]
            ts = oldest.frame_timestamp or (time.monotonic() - oldest.age_seconds)
            if ts < cutoff:
                self._window.popleft()
            else:
                break

    def _trim_window_sync(self) -> None:
        """同步版本的窗口裁剪"""
        cutoff = time.monotonic() - self._window_seconds
        while self._window:
            oldest = self._window[0]
            ts = oldest.frame_timestamp or (time.monotonic() - oldest.age_seconds)
            if ts < cutoff:
                self._window.popleft()
            else:
                break

    def _match_subscribers(self, event: PerceptionEvent) -> List[Subscriber]:
        """找到匹配事件的所有订阅者"""
        matched: List[Subscriber] = []
        seen: Set[int] = set()  # 用 id 去重

        # 生成事件可能的主题标签
        topics = self._event_topics(event)

        for topic in topics:
            for sub in self._subscribers.get(topic, []):
                if id(sub) not in seen:
                    seen.add(id(sub))
                    matched.append(sub)

        return matched

    @staticmethod
    def _event_topics(event: PerceptionEvent) -> List[str]:
        """生成事件匹配的主题列表"""
        topics = ["perception.*"]

        if event.module == Module.PLATE_RECOGNITION:
            topics.append("perception.plate")
        elif event.module == Module.TRAFFIC_GESTURE:
            topics.append("perception.traffic_gesture")
        elif event.module == Module.DRIVER_GESTURE:
            topics.append("perception.driver_gesture")

        return topics

    async def _safe_invoke(self, callback: Subscriber, event: PerceptionEvent) -> None:
        """安全调用订阅者，异常不会影响其他订阅者"""
        try:
            await callback(event)
        except Exception as e:
            logger.error(f"订阅者回调异常 ({callback.__name__}): {e}", exc_info=True)

    def _build_context(self, events: List[PerceptionEvent]) -> Dict[str, Dict]:
        """从事件列表构建融合上下文"""
        now = time.monotonic()
        context: Dict[str, Dict] = {
            "plate": {"latest": None, "recent": [], "count_2s": 0,
                       "avg_confidence": 0.0, "all_plates": []},
            "traffic_gesture": {"latest": None, "recent": [], "count_2s": 0,
                                 "avg_confidence": 0.0, "stable_1s": False},
            "driver_gesture": {"latest": None, "recent": [], "count_2s": 0,
                                "avg_confidence": 0.0, "stable_1s": False},
            "window_size": len(events),
        }

        for e in events:
            age = now - (e.frame_timestamp or now)
            module_key = _module_to_key(e.module)

            if module_key not in context:
                continue

            ctx = context[module_key]

            # 最近 2 秒计数
            if age <= 2.0:
                ctx["count_2s"] += 1

            # 最近 3 个事件
            ctx["recent"].append(e)
            if len(ctx["recent"]) > 3:
                ctx["recent"] = ctx["recent"][-3:]

            # 最新事件
            ctx["latest"] = e

            # 车牌去重收集
            if e.module == Module.PLATE_RECOGNITION and e.plate_code:
                if e.plate_code not in ctx["all_plates"]:
                    ctx["all_plates"].append(e.plate_code)

        # 计算各模块平均置信度
        for key in ("plate", "traffic_gesture", "driver_gesture"):
            recent = context[key]["recent"]
            if recent:
                context[key]["avg_confidence"] = round(
                    sum(e.confidence for e in recent) / len(recent), 3
                )

        # 手势稳定性检查（1 秒内同一手势不变 → stable）
        for key in ("traffic_gesture", "driver_gesture"):
            recent_1s = [e for e in context[key]["recent"]
                         if now - (e.frame_timestamp or now) <= 1.0]
            if len(recent_1s) >= 2:
                gestures = [e.gesture_name for e in recent_1s]
                context[key]["stable_1s"] = len(set(gestures)) == 1

        return context


def _module_to_key(module: Module) -> str:
    """模块枚举 → 上下文字段名"""
    if module == Module.PLATE_RECOGNITION:
        return "plate"
    elif module == Module.TRAFFIC_GESTURE:
        return "traffic_gesture"
    elif module == Module.DRIVER_GESTURE:
        return "driver_gesture"
    return ""

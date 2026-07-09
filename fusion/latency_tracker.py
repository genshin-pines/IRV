"""
延迟追踪器 — 记录全链路各阶段时间戳，暴露统计 API

追踪的阶段:
  frame_captured      — cv2.VideoCapture.read() 时间戳
  recognition_done    — 识别模型返回结果
  event_published     — EventBus.publish() 完成
  fusion_started      — FusionAgent 开始处理
  llm_call_started    — LLM API 调用开始 (如有)
  llm_call_ended      — LLM API 调用完成
  fusion_completed    — 融合结果生成
  ws_broadcast        — WebSocket 推送完成

统计指标:
  - P50 / P95 / P99 端到端延迟
  - 各阶段耗时占比
  - 超阈值 (>1s) 事件计数和告警
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class PipelineLatency:
    """单条感知事件的全链路延迟记录"""

    event_id: str = ""
    frame_captured_at: float = 0.0
    recognition_done_at: float = 0.0
    event_published_at: float = 0.0
    fusion_started_at: float = 0.0
    llm_call_started_at: float = 0.0
    llm_call_ended_at: float = 0.0
    fusion_completed_at: float = 0.0
    ws_broadcast_at: float = 0.0
    use_llm: bool = False

    # ── 阶段耗时 (ms) ────────────────────────────────────────

    @property
    def capture_to_recognition_ms(self) -> float:
        return _ms(self.frame_captured_at, self.recognition_done_at)

    @property
    def recognition_to_publish_ms(self) -> float:
        return _ms(self.recognition_done_at, self.event_published_at)

    @property
    def publish_to_fusion_ms(self) -> float:
        return _ms(self.event_published_at, self.fusion_started_at)

    @property
    def fusion_to_llm_ms(self) -> float:
        return _ms(self.fusion_started_at, self.llm_call_started_at) if self.llm_call_started_at else 0

    @property
    def llm_call_ms(self) -> float:
        return _ms(self.llm_call_started_at, self.llm_call_ended_at) if self.use_llm else 0

    @property
    def llm_to_complete_ms(self) -> float:
        end = self.llm_call_ended_at if self.use_llm else self.fusion_started_at
        return _ms(end, self.fusion_completed_at)

    @property
    def complete_to_ws_ms(self) -> float:
        return _ms(self.fusion_completed_at, self.ws_broadcast_at)

    @property
    def total_latency_ms(self) -> float:
        """端到端总延迟（帧采集 → WebSocket 推送）"""
        return _ms(self.frame_captured_at, self.ws_broadcast_at)

    @property
    def exceeds_threshold(self) -> bool:
        """是否超过 1 秒延迟阈值"""
        return self.total_latency_ms > 1000.0

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "total_ms": round(self.total_latency_ms, 1),
            "stages": {
                "capture→recognition": round(self.capture_to_recognition_ms, 1),
                "recognition→publish": round(self.recognition_to_publish_ms, 1),
                "publish→fusion": round(self.publish_to_fusion_ms, 1),
                "llm_call": round(self.llm_call_ms, 1),
                "fusion→complete": round(self.llm_to_complete_ms, 1),
                "complete→ws": round(self.complete_to_ws_ms, 1),
            },
            "use_llm": self.use_llm,
            "over_threshold": self.exceeds_threshold,
        }


class LatencyTracker:
    """
    全链路延迟追踪器。

    用法:
        tracker = LatencyTracker(window_size=100)

        # 在 pipeline 各阶段记录时间戳
        tracker.start("evt_001", frame_ts=1234567890.123)
        tracker.record("evt_001", "recognition_done")
        tracker.record("evt_001", "fusion_completed")
        tracker.finish("evt_001")

        # 获取统计
        stats = tracker.get_stats()  # {"p50_ms": ..., "p95_ms": ..., "p99_ms": ...}
    """

    def __init__(self, window_size: int = 200):
        self._records: deque[PipelineLatency] = deque(maxlen=window_size)
        self._active: Dict[str, PipelineLatency] = {}
        self._lock = threading.Lock()
        self._exceeded_count: int = 0
        self._total_count: int = 0

    # ── 记录 ──────────────────────────────────────────────────

    def start(self, event_id: str, frame_ts: Optional[float] = None) -> PipelineLatency:
        """开始追踪一个事件（在帧采集时调用）"""
        record = PipelineLatency(
            event_id=event_id,
            frame_captured_at=frame_ts or time.perf_counter(),
        )
        with self._lock:
            self._active[event_id] = record
        return record

    def record(self, event_id: str, stage: str) -> None:
        """
        记录 pipeline 阶段完成的时间戳。

        Args:
            event_id: 事件 ID
            stage: 阶段名
                "recognition_done" / "event_published" / "fusion_started" /
                "llm_call_started" / "llm_call_ended" / "fusion_completed" /
                "ws_broadcast"
        """
        now = time.perf_counter()
        with self._lock:
            r = self._active.get(event_id)
            if r is None:
                return

            if stage == "recognition_done":
                r.recognition_done_at = now
            elif stage == "event_published":
                r.event_published_at = now
            elif stage == "fusion_started":
                r.fusion_started_at = now
            elif stage == "llm_call_started":
                r.llm_call_started_at = now
                r.use_llm = True
            elif stage == "llm_call_ended":
                r.llm_call_ended_at = now
            elif stage == "fusion_completed":
                r.fusion_completed_at = now
            elif stage == "ws_broadcast":
                r.ws_broadcast_at = now

    def finish(self, event_id: str) -> Optional[PipelineLatency]:
        """
        完成追踪，移入历史记录。

        Returns:
            完成的 PipelineLatency 记录，或 None（未知事件）
        """
        with self._lock:
            r = self._active.pop(event_id, None)
            if r is None:
                return None
            if r.ws_broadcast_at == 0:
                r.ws_broadcast_at = time.perf_counter()
            self._records.append(r)
            self._total_count += 1
            if r.exceeds_threshold:
                self._exceeded_count += 1
        return r

    # ── 统计 ──────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """
        获取延迟统计。

        Returns:
            {
              "count": N,
              "exceeded_1s": N,
              "exceed_rate": 0.0,
              "p50_ms": float,
              "p95_ms": float,
              "p99_ms": float,
              "min_ms": float,
              "max_ms": float,
              "avg_ms": float,
              "stage_breakdown": { ... },  # 各阶段平均耗时
              "recent": [...]              # 最近 10 条记录
            }
        """
        with self._lock:
            records = list(self._records)

        if not records:
            return {"count": 0, "exceeded_1s": 0, "note": "暂无数据"}

        latencies = sorted(r.total_latency_ms for r in records)
        n = len(latencies)

        def percentile(p: int) -> float:
            idx = max(0, min(n - 1, int(n * p / 100)))
            return latencies[idx]

        # 各阶段平均耗时
        stage_sums = {
            "capture→recognition": 0.0, "recognition→publish": 0.0,
            "publish→fusion": 0.0, "llm_call": 0.0,
            "fusion→complete": 0.0, "complete→ws": 0.0,
        }
        llm_count = 0
        for r in records:
            stage_sums["capture→recognition"] += r.capture_to_recognition_ms
            stage_sums["recognition→publish"] += r.recognition_to_publish_ms
            stage_sums["publish→fusion"] += r.publish_to_fusion_ms
            if r.use_llm:
                stage_sums["llm_call"] += r.llm_call_ms
                llm_count += 1
            stage_sums["fusion→complete"] += r.llm_to_complete_ms
            stage_sums["complete→ws"] += r.complete_to_ws_ms

        stage_avg = {
            k: round(v / n, 1) for k, v in stage_sums.items()
        }
        if llm_count > 0:
            stage_avg["llm_call"] = round(stage_sums["llm_call"] / llm_count, 1)
            stage_avg["llm_samples"] = llm_count

        return {
            "count": n,
            "exceeded_1s": self._exceeded_count,
            "exceed_rate": round(self._exceeded_count / max(1, self._total_count), 3),
            "p50_ms": round(percentile(50), 1),
            "p95_ms": round(percentile(95), 1),
            "p99_ms": round(percentile(99), 1),
            "min_ms": round(latencies[0], 1),
            "max_ms": round(latencies[-1], 1),
            "avg_ms": round(sum(latencies) / n, 1),
            "stage_breakdown": stage_avg,
            "recent": [r.to_dict() for r in records[-10:]],
        }

    @property
    def exceeded_count(self) -> int:
        return self._exceeded_count

    @property
    def total_count(self) -> int:
        return self._total_count


def _ms(start: float, end: float) -> float:
    """计算两个时间戳之间的毫秒差"""
    if start == 0 or end == 0:
        return 0.0
    return max(0.0, (end - start) * 1000)

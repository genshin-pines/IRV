"""
感知事件数据模型 — 统一三路感知模块的结构化事件格式

三路感知:
  - plate_recognition: 车牌检测结果（车牌号/颜色/置信度/位置）
  - traffic_gesture:   交警手势识别（8种中国标准指挥手势）
  - driver_gesture:    车主手势识别（6种预定义控车手势）

每个感知事件包含标准化的元数据 + 模块特有的 data 载荷。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════
# 枚举定义
# ═══════════════════════════════════════════════════════════════════

class Module(str, Enum):
    """感知模块标识"""
    PLATE_RECOGNITION = "plate_recognition"
    TRAFFIC_GESTURE = "traffic_gesture"
    DRIVER_GESTURE = "driver_gesture"


class EventType(str, Enum):
    """感知事件类型"""
    PLATE_DETECTED = "plate_detected"
    TRAFFIC_GESTURE = "traffic_gesture"
    DRIVER_GESTURE = "driver_gesture"


class SuggestionAction(str, Enum):
    """驾驶建议动作类型"""
    NORMAL = "normal"               # 正常行驶
    CAUTION = "caution"             # 注意观察
    SLOW_DOWN = "slow_down"         # 减速
    STOP = "stop"                   # 停车等待
    LANE_CHANGE = "lane_change"     # 变道
    FOLLOW_GESTURE = "follow_gesture"  # 按交警手势行驶


class Urgency(str, Enum):
    """紧急程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ═══════════════════════════════════════════════════════════════════
# 交警手势常量
# ═══════════════════════════════════════════════════════════════════

TRAFFIC_GESTURES = {
    "停止": "stop",
    "直行": "go_straight",
    "左转弯": "turn_left",
    "左转弯待转": "wait_left",
    "右转弯": "turn_right",
    "变道": "change_lane",
    "减速慢行": "slow_down",
    "靠边停车": "pull_over",
}

# 交警手势 → 驾驶建议的优先级映射（数字越大优先级越高）
GESTURE_PRIORITY = {
    "stop": 10,
    "pull_over": 9,
    "turn_left": 5,
    "turn_right": 5,
    "wait_left": 4,
    "change_lane": 3,
    "slow_down": 2,
    "go_straight": 1,
}


# ═══════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PerceptionEvent:
    """
    统一感知事件 — 所有识别模块的标准输出格式。

    Attributes:
        event_id: 全局唯一事件 ID (UUID hex)
        timestamp: 识别完成时间 (UTC)
        module: 来源模块
        event_type: 事件类别
        data: 模块特有的结构化识别结果
        confidence: 识别置信度 [0.0, 1.0]
        frame_timestamp: 原始帧采集时间 (perf_counter 或 ISO8601)
        camera_id: 摄像头流标识 (如 "live1")
    """

    event_id: str
    timestamp: datetime
    module: Module
    event_type: EventType
    data: Dict[str, Any]
    confidence: float
    frame_timestamp: Optional[float] = None
    camera_id: Optional[str] = None

    # ── 工厂方法 ──────────────────────────────────────────────

    @classmethod
    def from_plate(cls, plate_data: dict, camera_id: str = "",
                   frame_ts: Optional[float] = None) -> "PerceptionEvent":
        """
        从车牌识别结果创建事件。

        Args:
            plate_data: 车牌识别结果 dict，至少包含 code, conf, color
                格式与 live_server.py 输出一致:
                {"code": "京A12345", "conf": 0.95, "color": "蓝牌", "bbox": [...]}
        """
        return cls(
            event_id=uuid.uuid4().hex,
            timestamp=datetime.now(timezone.utc),
            module=Module.PLATE_RECOGNITION,
            event_type=EventType.PLATE_DETECTED,
            data={
                "plate_code": plate_data.get("code", ""),
                "plate_color": plate_data.get("color", "未知"),
                "confidence": plate_data.get("conf", 0.0),
                "bbox": plate_data.get("bbox", []),
                "plate_type": plate_data.get("plate_type", -1),
            },
            confidence=float(plate_data.get("conf", 0.0)),
            camera_id=camera_id,
            frame_timestamp=frame_ts,
        )

    @classmethod
    def from_gesture(cls, gesture_type: str, gesture_name: str,
                     confidence: float, module: Module,
                     camera_id: str = "",
                     frame_ts: Optional[float] = None,
                     keypoints: Optional[list] = None) -> "PerceptionEvent":
        """
        从手势识别结果创建事件。

        Args:
            gesture_type: 手势英文标识
            gesture_name: 手势中文名称
            confidence: 识别置信度
            module: Module.TRAFFIC_GESTURE 或 Module.DRIVER_GESTURE
        """
        return cls(
            event_id=uuid.uuid4().hex,
            timestamp=datetime.now(timezone.utc),
            module=module,
            event_type=(EventType.TRAFFIC_GESTURE if module == Module.TRAFFIC_GESTURE
                        else EventType.DRIVER_GESTURE),
            data={
                "gesture": gesture_name,
                "gesture_type": gesture_type,
                "confidence": confidence,
                "keypoints": keypoints or [],
            },
            confidence=confidence,
            camera_id=camera_id,
            frame_timestamp=frame_ts,
        )

    # ── 便利属性 ──────────────────────────────────────────────

    @property
    def plate_code(self) -> Optional[str]:
        """车牌号（仅 plate_detected 事件有效）"""
        if self.event_type == EventType.PLATE_DETECTED:
            return self.data.get("plate_code")
        return None

    @property
    def plate_color(self) -> Optional[str]:
        """车牌颜色（仅 plate_detected 事件有效）"""
        if self.event_type == EventType.PLATE_DETECTED:
            return self.data.get("plate_color")
        return None

    @property
    def gesture_name(self) -> Optional[str]:
        """手势中文名（仅 gesture 事件有效）"""
        if self.event_type in (EventType.TRAFFIC_GESTURE, EventType.DRIVER_GESTURE):
            return self.data.get("gesture")
        return None

    @property
    def gesture_type(self) -> Optional[str]:
        """手势英文标识（仅 gesture 事件有效）"""
        if self.event_type in (EventType.TRAFFIC_GESTURE, EventType.DRIVER_GESTURE):
            return self.data.get("gesture_type")
        return None

    @property
    def age_seconds(self) -> float:
        """事件产生至今的秒数"""
        return (datetime.now(timezone.utc) - self.timestamp).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """转为可序列化的 dict（用于 JSON / WebSocket 推送）"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "module": self.module.value,
            "event_type": self.event_type.value,
            "data": self.data,
            "confidence": self.confidence,
            "camera_id": self.camera_id or "",
        }


@dataclass
class FusionResult:
    """
    融合推理结果 — 综合三路感知后生成的驾驶建议。

    Attributes:
        action: 建议动作类型
        reasoning: 融合推理链条（自然语言解释）
        confidence: 综合置信度 [0.0, 1.0]
        urgency: 紧急程度
        related_events: 参与融合推理的事件 ID 列表
        scene_summary: 一句话场景描述
        alerts: 融合告警列表
        ai_generated: 是否由 LLM 生成
        generated_at: 生成时间
        latency_ms: 融合处理延迟（从最早感知事件到生成结果）
    """

    action: SuggestionAction
    reasoning: str
    confidence: float
    urgency: Urgency
    related_events: List[str] = field(default_factory=list)
    scene_summary: str = ""
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    ai_generated: bool = False
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float = 0.0

    def to_websocket(self) -> Dict[str, Any]:
        """转为 WebSocket 推送格式"""
        return {
            "type": "fusion_suggestion",
            "action": self.action.value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "urgency": self.urgency.value,
            "scene_summary": self.scene_summary,
            "alerts": self.alerts,
            "ai_generated": self.ai_generated,
            "generated_at": self.generated_at.isoformat(),
            "latency_ms": round(self.latency_ms, 1),
        }

    def to_alert(self) -> Optional[Dict[str, Any]]:
        """如果有融合告警，转为 alert 格式（兼容现有 WebSocket）"""
        if not self.alerts:
            return None
        primary = self.alerts[0]
        return {
            "type": "alert",
            "level": primary.get("level", "warning"),
            "title": primary.get("message", "融合告警"),
            "message": self.reasoning,
            "timestamp": self.generated_at.isoformat(),
            "source_module": "fusion_agent",
            "suggested_action": self.action.value,
            "dismissible": True,
        }

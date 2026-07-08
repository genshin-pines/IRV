"""
融合推理规则引擎 — LLM 不可用时的降级方案

基于预定义规则的场景匹配：
  1. 交警手势 × 车牌检测 → 具体驾驶建议
  2. 手势优先级 → 交警 > 车主
  3. 传感器置信度 → 稳定性检查

规则优先级从高到低排列，首次匹配即返回。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .perception_event import (
    PerceptionEvent, Module, EventType,
    FusionResult, SuggestionAction, Urgency,
    TRAFFIC_GESTURES,
)


def rule_based_fusion(events: List[PerceptionEvent]) -> Optional[FusionResult]:
    """
    基于规则的融合推理（LLM fallback）。

    按优先级匹配场景规则，首次命中即返回。

    Args:
        events: 滑动窗口内的感知事件列表

    Returns:
        FusionResult 或 None（无匹配场景）
    """
    if not events:
        return None

    # 分离各模块事件
    plates = [e for e in events if e.module == Module.PLATE_RECOGNITION]
    traffic = [e for e in events if e.module == Module.TRAFFIC_GESTURE]
    driver = [e for e in events if e.module == Module.DRIVER_GESTURE]

    # 获取各模块的最新稳定结果
    latest_traffic = _latest_stable(traffic, window=1.0)
    latest_driver = _latest_stable(driver, window=1.0)
    latest_plate = plates[-1] if plates else None

    recent_plates = [e.plate_code for e in plates[-5:] if e.plate_code]
    unique_plates = list(set(recent_plates))

    # ═══ 规则匹配（优先级从高到低） ════════════════════════════════

    # R1: 交警停止 + 前方有车 → 立即减速停车
    if _match_gesture(latest_traffic, "停止"):
        if latest_plate:
            return FusionResult(
                action=SuggestionAction.STOP,
                reasoning=f"检测到交警停止手势（置信度 {latest_traffic.confidence:.0%}），"
                          f"同时检测到前方车辆 {_format_plates(unique_plates)}，建议立即减速停车",
                confidence=min(latest_traffic.confidence, latest_plate.confidence),
                urgency=Urgency.CRITICAL,
                related_events=[e.event_id for e in [latest_traffic, latest_plate] if e],
                scene_summary=f"交警停止 + 前方{len(unique_plates)}辆车 → 停车避让",
                alerts=[{
                    "type": "fusion_alert",
                    "level": "critical",
                    "message": f"交警停止手势，前方{len(unique_plates)}辆车，需立即停车",
                }],
                ai_generated=False,
            )
        else:
            return FusionResult(
                action=SuggestionAction.STOP,
                reasoning=f"检测到交警停止手势（置信度 {latest_traffic.confidence:.0%}），建议停车等待",
                confidence=latest_traffic.confidence,
                urgency=Urgency.HIGH,
                related_events=[latest_traffic.event_id],
                scene_summary="交警停止手势 → 停车",
                alerts=[{
                    "type": "fusion_alert",
                    "level": "warning",
                    "message": "交警停止手势，请停车等待",
                }],
                ai_generated=False,
            )

    # R2: 交警靠边停车 + 前方有车 → 减速靠边
    if _match_gesture(latest_traffic, "靠边停车"):
        return FusionResult(
            action=SuggestionAction.SLOW_DOWN,
            reasoning=f"检测到交警靠边停车手势（置信度 {latest_traffic.confidence:.0%}），"
                      f"{'前方有车辆 ' + _format_plates(unique_plates) + '，' if unique_plates else ''}"
                      f"建议减速靠边",
            confidence=latest_traffic.confidence,
            urgency=Urgency.HIGH,
            related_events=[e.event_id for e in [latest_traffic, latest_plate] if e],
            scene_summary="交警靠边停车手势 → 减速靠边",
            alerts=[{
                "type": "fusion_alert",
                "level": "warning",
                "message": "交警靠边停车手势，请减速靠边",
            }],
            ai_generated=False,
        )

    # R3: 交警减速慢行 + 前方有车 → 减速
    if _match_gesture(latest_traffic, "减速慢行"):
        plate_info = f"前方有 {len(unique_plates)} 辆车" if unique_plates else "前方道路"
        return FusionResult(
            action=SuggestionAction.SLOW_DOWN,
            reasoning=f"检测到交警减速慢行手势（置信度 {latest_traffic.confidence:.0%}），"
                      f"{plate_info}，建议减速慢行",
            confidence=latest_traffic.confidence,
            urgency=Urgency.MEDIUM,
            related_events=[e.event_id for e in [latest_traffic, latest_plate] if e],
            scene_summary=f"交警减速慢行 + {plate_info} → 减速",
            alerts=[],
            ai_generated=False,
        )

    # R4: 交警转弯（左/右）+ 目标方向有车 → 等待让行
    if _match_gesture(latest_traffic, "左转弯") or _match_gesture(latest_traffic, "右转弯"):
        direction = "左" if _match_gesture(latest_traffic, "左转弯") else "右"
        if latest_plate:
            return FusionResult(
                action=SuggestionAction.CAUTION,
                reasoning=f"检测到交警{direction}转弯手势（置信度 {latest_traffic.confidence:.0%}），"
                          f"前方有车辆 {_format_plates(unique_plates)}，建议注意观察，等待让行",
                confidence=min(latest_traffic.confidence, latest_plate.confidence),
                urgency=Urgency.MEDIUM,
                related_events=[e.event_id for e in [latest_traffic, latest_plate] if e],
                scene_summary=f"交警{direction}转 + 前方{len(unique_plates)}辆车 → 注意观察",
                alerts=[],
                ai_generated=False,
            )
        else:
            return FusionResult(
                action=SuggestionAction.FOLLOW_GESTURE,
                reasoning=f"检测到交警{direction}转弯手势（置信度 {latest_traffic.confidence:.0%}），"
                          f"前方无车辆，可按手势{direction}转",
                confidence=latest_traffic.confidence,
                urgency=Urgency.LOW,
                related_events=[latest_traffic.event_id],
                scene_summary=f"交警{direction}转 → 按手势行驶",
                alerts=[],
                ai_generated=False,
            )

    # R5: 交警直行 + 道路畅通 → 正常行驶
    if _match_gesture(latest_traffic, "直行"):
        return FusionResult(
            action=SuggestionAction.NORMAL,
            reasoning=f"检测到交警直行手势（置信度 {latest_traffic.confidence:.0%}），可正常通行",
            confidence=latest_traffic.confidence,
            urgency=Urgency.LOW,
            related_events=[latest_traffic.event_id],
            scene_summary="交警直行 → 正常行驶",
            alerts=[],
            ai_generated=False,
        )

    # R6: 交警变道 → 变道建议
    if _match_gesture(latest_traffic, "变道"):
        return FusionResult(
            action=SuggestionAction.LANE_CHANGE,
            reasoning=f"检测到交警变道手势（置信度 {latest_traffic.confidence:.0%}），建议按指示变道",
            confidence=latest_traffic.confidence,
            urgency=Urgency.MEDIUM,
            related_events=[latest_traffic.event_id],
            scene_summary="交警变道手势 → 变道",
            alerts=[],
            ai_generated=False,
        )

    # R7: 交警有手势 + 车主有手势 → 交警优先
    if latest_traffic and latest_driver:
        return FusionResult(
            action=SuggestionAction.FOLLOW_GESTURE,
            reasoning=f"交警手势（{latest_traffic.gesture_name}）与车主手势"
                      f"（{latest_driver.gesture_name}）同时检测到，优先响应交警手势",
            confidence=latest_traffic.confidence,
            urgency=Urgency.MEDIUM,
            related_events=[latest_traffic.event_id, latest_driver.event_id],
            scene_summary=f"交警({latest_traffic.gesture_name}) + 车主({latest_driver.gesture_name}) → 交警优先",
            alerts=[{
                "type": "fusion_alert",
                "level": "info",
                "message": f"交警手势优先：忽略车主'{latest_driver.gesture_name}'手势",
            }],
            ai_generated=False,
        )

    # R8: 仅车牌检测（无交警手势）→ 正常，记录场景
    if plates and not traffic:
        if len(plates) >= 3:
            return FusionResult(
                action=SuggestionAction.CAUTION,
                reasoning=f"检测到多辆车辆（{_format_plates(unique_plates)}），"
                          f"无交警手势，建议注意车距",
                confidence=0.8,
                urgency=Urgency.LOW,
                related_events=[e.event_id for e in plates[-3:]],
                scene_summary=f"前方{len(unique_plates)}辆车，无交警 → 注意观察",
                alerts=[],
                ai_generated=False,
            )
        else:
            return FusionResult(
                action=SuggestionAction.NORMAL,
                reasoning=f"检测到车辆 {_format_plates(unique_plates)}，无交警手势，正常行驶",
                confidence=latest_plate.confidence if latest_plate else 0.5,
                urgency=Urgency.LOW,
                related_events=[e.event_id for e in plates],
                scene_summary="前方检测到车辆 → 正常行驶",
                alerts=[],
                ai_generated=False,
            )

    # R9: 传感器异常检测 — 低置信度持续
    low_conf_events = [e for e in events if e.confidence < 0.3]
    if len(low_conf_events) >= 3:
        modules = set(e.module.value for e in low_conf_events)
        return FusionResult(
            action=SuggestionAction.CAUTION,
            reasoning=f"多个模块置信度偏低（{', '.join(modules)}），"
                      f"建议检查传感器和光线条件",
            confidence=0.3,
            urgency=Urgency.MEDIUM,
            related_events=[e.event_id for e in low_conf_events[-3:]],
            scene_summary=f"传感器异常：{'/'.join(modules)}置信度低",
            alerts=[{
                "type": "fusion_alert",
                "level": "warning",
                "message": f"感知模块置信度持续偏低: {', '.join(modules)}",
            }],
            ai_generated=False,
        )

    # 无匹配规则
    return None


# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════

def _match_gesture(event: Optional[PerceptionEvent], gesture_name: str) -> bool:
    """检查事件是否匹配指定手势名称"""
    if event is None:
        return False
    return event.gesture_name == gesture_name


def _latest_stable(events: List[PerceptionEvent], window: float = 1.0) -> Optional[PerceptionEvent]:
    """
    获取滑动窗口内稳定的最新事件。

    稳定条件：最近 1 秒内同一手势未变化（≥2 次相同结果）。
    如果不满足稳定条件，返回最新的事件（仍不保证稳定）。
    """
    if not events:
        return None

    # 取最近 1 秒内的事件
    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=window)
    recent = [e for e in events if e.timestamp >= cutoff]

    if len(recent) >= 2:
        gestures = [e.gesture_name for e in recent]
        if len(set(gestures)) == 1:
            return recent[-1]  # 稳定的最新结果

    # 不稳定，仍返回最新事件
    return events[-1]


def _format_plates(plate_codes: List[str]) -> str:
    """格式化车牌列表为显示字符串"""
    if not plate_codes:
        return "未知车辆"
    if len(plate_codes) == 1:
        return plate_codes[0]
    return f"{plate_codes[0]} 等{len(plate_codes)}辆车"

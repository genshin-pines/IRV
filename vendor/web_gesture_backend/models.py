"""
数据模型定义 — 手势信息、车辆操控信息
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


# ======================== 动态手势 → 车辆操控映射 ========================

class VehicleAction(Enum):
    """车辆操控指令（8种）"""
    ROLL_DOWN_WINDOW = "roll_down_window"    # 打开车窗
    ROLL_UP_WINDOW   = "roll_up_window"      # 关闭车窗
    TURN_LEFT        = "turn_left"           # 左转
    TURN_RIGHT       = "turn_right"          # 右转
    ACCELERATE       = "accelerate"          # 加速
    DECELERATE       = "decelerate"          # 减速
    HONK             = "honk"                # 鸣笛
    TOGGLE_LIGHTS   = "toggle_lights"      # 切换灯光
    TEMP_UP          = "temp_up"             # 空调升温
    TEMP_DOWN        = "temp_down"           # 空调降温


# 动态手势事件名 → 车辆操控指令映射
GESTURE_TO_VEHICLE: dict[str, VehicleAction] = {
    "SWIPE_DOWN":     VehicleAction.ROLL_DOWN_WINDOW,
    "SWIPE_UP":       VehicleAction.ROLL_UP_WINDOW,
    "SWIPE_LEFT":     VehicleAction.TURN_LEFT,
    "SWIPE_RIGHT":    VehicleAction.TURN_RIGHT,
    "ZOOM_IN":        VehicleAction.ACCELERATE,
    "ZOOM_OUT":       VehicleAction.DECELERATE,
    "TAP":            VehicleAction.HONK,
    "DOUBLE_TAP":     VehicleAction.TOGGLE_LIGHTS,
    "CLOCKWISE":      VehicleAction.TEMP_UP,
    "COUNTERCLOCK":   VehicleAction.TEMP_DOWN,
    "SWIPE_LEFT2":    VehicleAction.TURN_LEFT,
    "SWIPE_RIGHT2":   VehicleAction.TURN_RIGHT,
    "SWIPE_UP2":      VehicleAction.ROLL_UP_WINDOW,
    "SWIPE_DOWN2":    VehicleAction.ROLL_DOWN_WINDOW,
    "SWIPE_LEFT3":    VehicleAction.TURN_LEFT,
    "SWIPE_RIGHT3":   VehicleAction.TURN_RIGHT,
    "SWIPE_UP3":      VehicleAction.ROLL_UP_WINDOW,
    "SWIPE_DOWN3":    VehicleAction.ROLL_DOWN_WINDOW,
    "FAST_SWIPE_UP":  VehicleAction.ACCELERATE,
    "FAST_SWIPE_DOWN":VehicleAction.DECELERATE,
}


# ======================== 推送消息类型 ========================

@dataclass
class HandInfo:
    """单只手的信息"""
    hand_id: int
    bbox: list[int]          # [x1, y1, x2, y2]
    gesture: str             # 静态手势名 (e.g. "fist", "palm")
    gesture_id: int
    center: tuple[float, float]
    confidence: float


@dataclass
class FrameMessage:
    """每帧推送的手势信息"""
    type: str = "frame"
    timestamp: float = 0.0
    fps: float = 0.0
    hands: list[HandInfo] = field(default_factory=list)


@dataclass
class ActionMessage:
    """动态手势动作事件 + 对应车辆操控"""
    type: str = "action"
    timestamp: float = 0.0
    gesture_action: str = ""           # 动态手势名 (e.g. "SWIPE_DOWN")
    vehicle_action: str = ""           # 车辆操控名 (e.g. "roll_down_window")
    vehicle_label: str = ""            # 中文描述 (e.g. "打开车窗")
    hand_id: int = -1


# 车辆操控中文标签
VEHICLE_LABELS: dict[str, str] = {
    "roll_down_window": "打开车窗",
    "roll_up_window":   "关闭车窗",
    "turn_left":        "左转",
    "turn_right":       "右转",
    "accelerate":       "加速",
    "decelerate":       "减速",
    "honk":             "鸣笛",
    "toggle_lights":   "切换灯光",
    "temp_up":          "空调升温",
    "temp_down":        "空调降温",
}


def make_action_message(gesture_event_name: str, hand_id: int = -1) -> Optional[ActionMessage]:
    """根据动态手势事件名生成车辆操控消息"""
    import time
    vehicle = GESTURE_TO_VEHICLE.get(gesture_event_name)
    if vehicle is None:
        return None
    return ActionMessage(
        type="action",
        timestamp=time.time(),
        gesture_action=gesture_event_name,
        vehicle_action=vehicle.value,
        vehicle_label=VEHICLE_LABELS.get(vehicle.value, vehicle.value),
        hand_id=hand_id,
    )


def to_dict(obj) -> dict:
    """dataclass → dict，处理 center 等特殊字段"""
    d = asdict(obj)
    if "hands" in d:
        for h in d["hands"]:
            if "center" in h:
                h["center"] = list(h["center"])
    return d
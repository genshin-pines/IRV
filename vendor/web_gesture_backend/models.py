from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class VehicleAction(Enum):
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    ACCELERATE = "accelerate"
    DECELERATE = "decelerate"
    HONK = "honk"
    MUSIC_TOGGLE = "music_toggle"
    TOGGLE_LIGHTS = "toggle_lights"
    LIGHTS_ON = "lights_on"
    LIGHTS_OFF = "lights_off"
    PHONE_ANSWER = "phone_answer"
    PHONE_HANGUP = "phone_hangup"
    CONTROL_TOGGLE = "control_toggle"
    TEMP_UP = "temp_up"
    TEMP_DOWN = "temp_down"


GESTURE_TO_VEHICLE: dict[str, VehicleAction] = {
    "SWIPE_UP": VehicleAction.VOLUME_UP,
    "SWIPE_DOWN": VehicleAction.VOLUME_DOWN,
    "SWIPE_UP2": VehicleAction.VOLUME_UP,
    "SWIPE_DOWN2": VehicleAction.VOLUME_DOWN,
    "SWIPE_UP3": VehicleAction.VOLUME_UP,
    "SWIPE_DOWN3": VehicleAction.VOLUME_DOWN,
    "FAST_SWIPE_UP": VehicleAction.VOLUME_UP,
    "FAST_SWIPE_DOWN": VehicleAction.VOLUME_DOWN,
    "SWIPE_LEFT": VehicleAction.TURN_LEFT,
    "SWIPE_RIGHT": VehicleAction.TURN_RIGHT,
    "SWIPE_LEFT2": VehicleAction.TURN_LEFT,
    "SWIPE_RIGHT2": VehicleAction.TURN_RIGHT,
    "SWIPE_LEFT3": VehicleAction.TURN_LEFT,
    "SWIPE_RIGHT3": VehicleAction.TURN_RIGHT,
    "LIKE": VehicleAction.LIGHTS_ON,
    "DISLIKE": VehicleAction.LIGHTS_OFF,
    "CALL": VehicleAction.PHONE_ANSWER,
    "STOP": VehicleAction.PHONE_HANGUP,
    "STOP_INVERTED": VehicleAction.PHONE_HANGUP,
    "DNDV": VehicleAction.CONTROL_TOGGLE,
    "DNDV1": VehicleAction.CONTROL_TOGGLE,
    "CONTROL_TOGGLE": VehicleAction.CONTROL_TOGGLE,
    "ZOOM_IN": VehicleAction.ACCELERATE,
    "ZOOM_OUT": VehicleAction.DECELERATE,
    "TAP": VehicleAction.MUSIC_TOGGLE,
    "DOUBLE_TAP": VehicleAction.MUSIC_TOGGLE,
    "CLOCKWISE": VehicleAction.TEMP_UP,
    "COUNTERCLOCK": VehicleAction.TEMP_DOWN,
}


VEHICLE_LABELS: dict[str, str] = {
    "volume_up": "音量增加",
    "volume_down": "音量降低",
    "turn_left": "左转",
    "turn_right": "右转",
    "accelerate": "加速",
    "decelerate": "减速",
    "honk": "鸣笛",
    "music_toggle": "播放/暂停",
    "toggle_lights": "切换灯光",
    "lights_on": "开启灯光",
    "lights_off": "关闭灯光",
    "phone_answer": "接听电话",
    "phone_hangup": "挂断电话",
    "control_toggle": "手势控制开关",
    "temp_up": "空调升温",
    "temp_down": "空调降温",
}


@dataclass
class HandInfo:
    hand_id: int
    bbox: list[int]
    gesture: str
    gesture_id: int
    center: tuple[float, float]
    confidence: float


@dataclass
class FrameMessage:
    type: str = "frame"
    timestamp: float = 0.0
    fps: float = 0.0
    hands: list[HandInfo] = field(default_factory=list)


@dataclass
class ActionMessage:
    type: str = "action"
    timestamp: float = 0.0
    gesture_action: str = ""
    vehicle_action: str = ""
    vehicle_label: str = ""
    hand_id: int = -1
    action_applied: bool = True
    suppress_reason: str = ""
    gesture_control_enabled: bool = False


def make_action_message(gesture_event_name: str, hand_id: int = -1) -> Optional[ActionMessage]:
    import time

    vehicle = GESTURE_TO_VEHICLE.get(gesture_event_name)
    if vehicle is None:
        return None

    now = time.time()
    action_applied, suppress_reason = _action_gate(vehicle, now)
    return ActionMessage(
        type="action",
        timestamp=now,
        gesture_action=gesture_event_name,
        vehicle_action=vehicle.value,
        vehicle_label=VEHICLE_LABELS.get(vehicle.value, vehicle.value),
        hand_id=hand_id,
        action_applied=action_applied,
        suppress_reason=suppress_reason,
        gesture_control_enabled=_gesture_control_enabled,
    )


VOLUME_REVERSE_SUPPRESS_SEC = 2.0
TURN_ACTION_SUPPRESS_SEC = 1.0
TURN_REVERSE_SUPPRESS_SEC = 1.5
MUSIC_TOGGLE_SUPPRESS_SEC = 1.0
VOLUME_TO_MUSIC_SUPPRESS_SEC = 2.0
CONTROL_TOGGLE_SUPPRESS_SEC = 1.5
_last_volume_action: VehicleAction | None = None
_last_volume_action_at = 0.0
_last_turn_action: VehicleAction | None = None
_last_turn_action_at = 0.0
_last_music_toggle_at = 0.0
_last_control_toggle_at = 0.0
_gesture_control_enabled = False


def _action_gate(vehicle: VehicleAction, now: float) -> tuple[bool, str]:
    global _last_volume_action, _last_volume_action_at, _last_turn_action, _last_turn_action_at, _last_music_toggle_at, _last_control_toggle_at, _gesture_control_enabled

    if vehicle == VehicleAction.CONTROL_TOGGLE:
        elapsed = now - _last_control_toggle_at
        if elapsed < CONTROL_TOGGLE_SUPPRESS_SEC:
            state = "enabled" if _gesture_control_enabled else "disabled"
            return False, f"ignore repeated control toggle within {CONTROL_TOGGLE_SUPPRESS_SEC:.1f}s; gesture control {state}"
        _last_control_toggle_at = now
        _gesture_control_enabled = not _gesture_control_enabled
        state = "enabled" if _gesture_control_enabled else "disabled"
        return True, f"gesture control {state}"

    if vehicle in {VehicleAction.PHONE_ANSWER, VehicleAction.PHONE_HANGUP}:
        return True, ""

    if not _gesture_control_enabled:
        return False, "gesture control disabled"

    if vehicle == VehicleAction.MUSIC_TOGGLE:
        volume_elapsed = now - _last_volume_action_at
        if volume_elapsed < VOLUME_TO_MUSIC_SUPPRESS_SEC:
            return False, f"ignore music toggle within {VOLUME_TO_MUSIC_SUPPRESS_SEC:.1f}s after volume action"
        elapsed = now - _last_music_toggle_at
        if elapsed < MUSIC_TOGGLE_SUPPRESS_SEC:
            return False, f"ignore repeated music toggle within {MUSIC_TOGGLE_SUPPRESS_SEC:.1f}s"
        _last_music_toggle_at = now
        return True, ""

    turn_actions = {VehicleAction.TURN_LEFT, VehicleAction.TURN_RIGHT}
    if vehicle in turn_actions:
        elapsed = now - _last_turn_action_at
        is_reverse = _last_turn_action in turn_actions and _last_turn_action != vehicle
        if is_reverse and elapsed < TURN_REVERSE_SUPPRESS_SEC:
            return False, f"ignore reverse turn action within {TURN_REVERSE_SUPPRESS_SEC:.1f}s"
        if _last_turn_action == vehicle and elapsed < TURN_ACTION_SUPPRESS_SEC:
            return False, f"ignore repeated turn action within {TURN_ACTION_SUPPRESS_SEC:.1f}s"
        _last_turn_action = vehicle
        _last_turn_action_at = now
        return True, ""

    volume_actions = {VehicleAction.VOLUME_UP, VehicleAction.VOLUME_DOWN}
    if vehicle not in volume_actions:
        return True, ""

    is_reverse = _last_volume_action in volume_actions and _last_volume_action != vehicle
    elapsed = now - _last_volume_action_at
    if is_reverse and elapsed < VOLUME_REVERSE_SUPPRESS_SEC:
        return False, f"ignore reverse volume action within {VOLUME_REVERSE_SUPPRESS_SEC:.1f}s"

    _last_volume_action = vehicle
    _last_volume_action_at = now
    return True, ""


def to_dict(obj) -> dict:
    data = asdict(obj)
    if "hands" in data:
        for hand in data["hands"]:
            if "center" in hand:
                hand["center"] = list(hand["center"])
    return data

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
    "CLICK": VehicleAction.MUSIC_TOGGLE,
    "DOUBLE_CLICK": VehicleAction.MUSIC_TOGGLE,
    "LIKE": VehicleAction.LIGHTS_ON,
    "DISLIKE": VehicleAction.LIGHTS_OFF,
    "CALL": VehicleAction.PHONE_ANSWER,
    "STOP": VehicleAction.PHONE_HANGUP,
    "STOP_INVERTED": VehicleAction.PHONE_HANGUP,
    "DNDV": VehicleAction.CONTROL_TOGGLE,
    "DNDV1": VehicleAction.CONTROL_TOGGLE,
    "CONTROL_TOGGLE": VehicleAction.CONTROL_TOGGLE,
    "TAP": VehicleAction.MUSIC_TOGGLE,
    "DOUBLE_TAP": VehicleAction.MUSIC_TOGGLE,
    "CLOCKWISE": VehicleAction.TEMP_UP,
    "COUNTERCLOCK": VehicleAction.TEMP_DOWN,
}


VEHICLE_LABELS: dict[str, str] = {
    "volume_up": "音量增加",
    "volume_down": "音量降低",
    "turn_left": "上一首",
    "turn_right": "下一首",
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

    gesture_event_name = gesture_event_name.upper()
    now = time.time()
    if gesture_event_name == "OK":
        return _confirm_pending_action(now, hand_id)

    vehicle = GESTURE_TO_VEHICLE.get(gesture_event_name)
    if vehicle is None:
        return None

    queued = _queue_pending_action(vehicle, now, gesture_event_name)
    if queued is None:
        return None
    action_applied, suppress_reason = queued
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


def make_custom_action_message(gesture_key: str, vehicle_action: str, display_name: str, hand_id: int = -1) -> Optional[ActionMessage]:
    try:
        vehicle = VehicleAction(vehicle_action)
    except ValueError:
        return None

    import time

    event_name = f"CUSTOM:{gesture_key}"
    queued = _queue_pending_action(vehicle, time.time(), event_name)
    if queued is None:
        return None
    action_applied, suppress_reason = queued
    return ActionMessage(
        type="action",
        timestamp=time.time(),
        gesture_action=event_name,
        vehicle_action=vehicle.value,
        vehicle_label=display_name or VEHICLE_LABELS.get(vehicle.value, vehicle.value),
        hand_id=hand_id,
        action_applied=action_applied,
        suppress_reason=suppress_reason,
        gesture_control_enabled=_gesture_control_enabled,
    )


VOLUME_REVERSE_SUPPRESS_SEC = 1.5
TURN_ACTION_SUPPRESS_SEC = 1.0
TURN_REVERSE_SUPPRESS_SEC = 1.5
MUSIC_TOGGLE_SUPPRESS_SEC = 1.0
VOLUME_TO_MUSIC_SUPPRESS_SEC = 2.0
MUSIC_CLICK_PAIR_WINDOW_SEC = 1.5
ACTION_CONFIRM_WINDOW_SEC = 5.0
CONTROL_TOGGLE_SUPPRESS_SEC = 1.5
_last_volume_action: VehicleAction | None = None
_last_volume_action_at = 0.0
_last_queued_volume_action: VehicleAction | None = None
_last_queued_volume_action_at = 0.0
_last_turn_action: VehicleAction | None = None
_last_turn_action_at = 0.0
_last_queued_turn_action: VehicleAction | None = None
_last_queued_turn_action_at = 0.0
_last_music_toggle_at = 0.0
_last_queued_music_toggle_at = 0.0
_pending_music_clicks = 0
_pending_music_click_at = 0.0
_pending_action: tuple[VehicleAction, str, float] | None = None
_last_control_toggle_at = 0.0
_gesture_control_enabled = False


def _queue_pending_action(vehicle: VehicleAction, now: float, gesture_event_name: str) -> tuple[bool, str] | None:
    global _pending_music_clicks, _pending_music_click_at, _pending_action

    if vehicle == VehicleAction.CONTROL_TOGGLE:
        _pending_action = None
        action_applied, reason = _action_gate(vehicle, now)
        return action_applied, f"已识别控制开关，{reason}" if action_applied else reason

    cooldown_reason = _business_cooldown_reason(vehicle, now)
    if cooldown_reason:
        return False, cooldown_reason

    if _pending_action is not None:
        pending_vehicle, _pending_gesture, pending_at = _pending_action
        if pending_vehicle == vehicle and now - pending_at <= ACTION_CONFIRM_WINDOW_SEC:
            return None

    if vehicle == VehicleAction.MUSIC_TOGGLE and not gesture_event_name.startswith("CUSTOM:"):
        if now - _pending_music_click_at > MUSIC_CLICK_PAIR_WINDOW_SEC:
            _pending_music_clicks = 0
        _pending_music_clicks += 2 if gesture_event_name in {"DOUBLE_TAP", "DOUBLE_CLICK"} else 1
        _pending_music_click_at = now
        if _pending_music_clicks < 2:
            return False, f"检测到第一次点击，请在 {MUSIC_CLICK_PAIR_WINDOW_SEC:.1f} 秒内再次点击"
        _pending_music_clicks = 0

    _record_queued_business(vehicle, now)
    _pending_action = (vehicle, gesture_event_name, now)
    label = VEHICLE_LABELS.get(vehicle.value, vehicle.value)
    return False, f"检测到待确认业务：{label}，请做 OK 手势确认"


def _business_cooldown_reason(vehicle: VehicleAction, now: float) -> str:
    last_volume_action = _last_volume_action
    last_volume_at = _last_volume_action_at
    if _last_queued_volume_action_at > last_volume_at:
        last_volume_action = _last_queued_volume_action
        last_volume_at = _last_queued_volume_action_at
    if vehicle in {VehicleAction.VOLUME_UP, VehicleAction.VOLUME_DOWN}:
        if last_volume_action in {VehicleAction.VOLUME_UP, VehicleAction.VOLUME_DOWN}:
            if last_volume_action != vehicle and now - last_volume_at < VOLUME_REVERSE_SUPPRESS_SEC:
                return f"{VOLUME_REVERSE_SUPPRESS_SEC:.1f} 秒内忽略反向音量操作"
    last_turn_action = _last_turn_action
    last_turn_at = _last_turn_action_at
    if _last_queued_turn_action_at > last_turn_at:
        last_turn_action = _last_queued_turn_action
        last_turn_at = _last_queued_turn_action_at
    if vehicle in {VehicleAction.TURN_LEFT, VehicleAction.TURN_RIGHT}:
        if last_turn_action in {VehicleAction.TURN_LEFT, VehicleAction.TURN_RIGHT}:
            elapsed = now - last_turn_at
            if last_turn_action != vehicle and elapsed < TURN_REVERSE_SUPPRESS_SEC:
                return f"{TURN_REVERSE_SUPPRESS_SEC:.1f} 秒内忽略反向切歌"
            if last_turn_action == vehicle and elapsed < TURN_ACTION_SUPPRESS_SEC:
                return f"{TURN_ACTION_SUPPRESS_SEC:.1f} 秒内忽略重复切歌"
    if vehicle == VehicleAction.MUSIC_TOGGLE:
        if now - last_volume_at < VOLUME_TO_MUSIC_SUPPRESS_SEC:
            return f"音量操作后 {VOLUME_TO_MUSIC_SUPPRESS_SEC:.1f} 秒内忽略播放/暂停"
        if now - max(_last_music_toggle_at, _last_queued_music_toggle_at) < MUSIC_TOGGLE_SUPPRESS_SEC:
            return f"{MUSIC_TOGGLE_SUPPRESS_SEC:.1f} 秒内忽略重复播放/暂停"
    return ""


def _record_queued_business(vehicle: VehicleAction, now: float) -> None:
    global _last_queued_volume_action, _last_queued_volume_action_at, _last_queued_turn_action, _last_queued_turn_action_at, _last_queued_music_toggle_at
    if vehicle in {VehicleAction.VOLUME_UP, VehicleAction.VOLUME_DOWN}:
        _last_queued_volume_action = vehicle
        _last_queued_volume_action_at = now
    elif vehicle in {VehicleAction.TURN_LEFT, VehicleAction.TURN_RIGHT}:
        _last_queued_turn_action = vehicle
        _last_queued_turn_action_at = now
    elif vehicle == VehicleAction.MUSIC_TOGGLE:
        _last_queued_music_toggle_at = now


def _confirm_pending_action(now: float, hand_id: int) -> ActionMessage:
    global _pending_action

    if _pending_action is None:
        return ActionMessage(
            timestamp=now,
            gesture_action="OK",
            hand_id=hand_id,
            action_applied=False,
            suppress_reason="检测到 OK 手势，但当前没有待确认操作",
            gesture_control_enabled=_gesture_control_enabled,
        )

    vehicle, original_event, detected_at = _pending_action
    _pending_action = None
    if now - detected_at > ACTION_CONFIRM_WINDOW_SEC:
        return ActionMessage(
            timestamp=now,
            gesture_action="OK",
            vehicle_action=vehicle.value,
            vehicle_label=VEHICLE_LABELS.get(vehicle.value, vehicle.value),
            hand_id=hand_id,
            action_applied=False,
            suppress_reason=f"待确认手势已超过 {ACTION_CONFIRM_WINDOW_SEC:.0f} 秒，请重新操作",
            gesture_control_enabled=_gesture_control_enabled,
        )

    action_applied, reason = _action_gate(vehicle, now)
    label = VEHICLE_LABELS.get(vehicle.value, vehicle.value)
    return ActionMessage(
        timestamp=now,
        gesture_action=f"OK_CONFIRM_{original_event}",
        vehicle_action=vehicle.value,
        vehicle_label=label,
        hand_id=hand_id,
        action_applied=action_applied,
        suppress_reason=f"已确认，执行{label}" if action_applied else reason,
        gesture_control_enabled=_gesture_control_enabled,
    )


def _action_gate(vehicle: VehicleAction, now: float) -> tuple[bool, str]:
    global _last_volume_action, _last_volume_action_at, _last_turn_action, _last_turn_action_at, _last_music_toggle_at, _last_control_toggle_at, _gesture_control_enabled

    if vehicle == VehicleAction.CONTROL_TOGGLE:
        elapsed = now - _last_control_toggle_at
        if elapsed < CONTROL_TOGGLE_SUPPRESS_SEC:
            state = "开启" if _gesture_control_enabled else "关闭"
            return False, f"{CONTROL_TOGGLE_SUPPRESS_SEC:.1f} 秒内忽略重复控制开关，当前控制已{state}"
        _last_control_toggle_at = now
        _gesture_control_enabled = not _gesture_control_enabled
        state = "开启" if _gesture_control_enabled else "关闭"
        return True, f"手势控制已{state}"

    if vehicle in {VehicleAction.PHONE_ANSWER, VehicleAction.PHONE_HANGUP}:
        return True, ""

    if not _gesture_control_enabled:
        return False, "手势控制未开启，本次操作未执行"

    if vehicle == VehicleAction.MUSIC_TOGGLE:
        volume_elapsed = now - _last_volume_action_at
        if volume_elapsed < VOLUME_TO_MUSIC_SUPPRESS_SEC:
            return False, f"音量操作后 {VOLUME_TO_MUSIC_SUPPRESS_SEC:.1f} 秒内忽略播放/暂停"
        elapsed = now - _last_music_toggle_at
        if elapsed < MUSIC_TOGGLE_SUPPRESS_SEC:
            return False, f"{MUSIC_TOGGLE_SUPPRESS_SEC:.1f} 秒内忽略重复播放/暂停"
        _last_music_toggle_at = now
        return True, ""

    turn_actions = {VehicleAction.TURN_LEFT, VehicleAction.TURN_RIGHT}
    if vehicle in turn_actions:
        elapsed = now - _last_turn_action_at
        is_reverse = _last_turn_action in turn_actions and _last_turn_action != vehicle
        if is_reverse and elapsed < TURN_REVERSE_SUPPRESS_SEC:
            return False, f"{TURN_REVERSE_SUPPRESS_SEC:.1f} 秒内忽略反向切歌"
        if _last_turn_action == vehicle and elapsed < TURN_ACTION_SUPPRESS_SEC:
            return False, f"{TURN_ACTION_SUPPRESS_SEC:.1f} 秒内忽略重复切歌"
        _last_turn_action = vehicle
        _last_turn_action_at = now
        return True, ""

    volume_actions = {VehicleAction.VOLUME_UP, VehicleAction.VOLUME_DOWN}
    if vehicle not in volume_actions:
        return True, ""

    is_reverse = _last_volume_action in volume_actions and _last_volume_action != vehicle
    elapsed = now - _last_volume_action_at
    if is_reverse and elapsed < VOLUME_REVERSE_SUPPRESS_SEC:
        return False, f"{VOLUME_REVERSE_SUPPRESS_SEC:.1f} 秒内忽略反向音量操作"

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

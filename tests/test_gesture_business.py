import time
from types import SimpleNamespace

from backend.services import gesture_service
from vendor.web_gesture_backend.gesture_engine import GestureEngine
from vendor.web_gesture_backend import models as gesture_models


def _reset_service_state() -> None:
    gesture_service._gesture_control_enabled = True
    gesture_service._last_turn_action = ""
    gesture_service._last_turn_action_at = 0.0
    gesture_service._last_queued_turn_action = ""
    gesture_service._last_queued_turn_action_at = 0.0
    gesture_service._last_volume_action = ""
    gesture_service._last_volume_action_at = 0.0
    gesture_service._last_queued_volume_action = ""
    gesture_service._last_queued_volume_action_at = 0.0
    gesture_service._last_music_toggle_at = 0.0
    gesture_service._last_queued_music_toggle_at = 0.0
    gesture_service._pending_music_clicks = 0
    gesture_service._pending_music_click_at = 0.0
    gesture_service._pending_command = None


def _reset_vendor_state() -> None:
    gesture_models._gesture_control_enabled = True
    gesture_models._last_turn_action = None
    gesture_models._last_turn_action_at = 0.0
    gesture_models._last_queued_turn_action = None
    gesture_models._last_queued_turn_action_at = 0.0
    gesture_models._last_volume_action = None
    gesture_models._last_volume_action_at = 0.0
    gesture_models._last_queued_volume_action = None
    gesture_models._last_queued_volume_action_at = 0.0
    gesture_models._last_music_toggle_at = 0.0
    gesture_models._last_queued_music_toggle_at = 0.0
    gesture_models._pending_music_clicks = 0
    gesture_models._pending_music_click_at = 0.0
    gesture_models._pending_action = None


def test_simulated_swipe_queues_business_then_applies_turn_cooldowns(monkeypatch):
    _reset_service_state()

    monkeypatch.setattr(gesture_service.time, "time", lambda: 100.0)
    detected = gesture_service.map_event_to_vehicle({"gesture_type": "swipe_left"})
    assert detected["action"] == "turn_left"
    assert detected["label"] == "上一首"
    assert detected["action_applied"] is False
    assert detected["suppress_reason"].startswith("检测到待确认业务：上一首")

    monkeypatch.setattr(gesture_service.time, "time", lambda: 100.1)
    confirmed = gesture_service.map_event_to_vehicle({"gesture_type": "ok"})
    assert confirmed["action_applied"] is True
    assert confirmed["suppress_reason"] == "已确认，执行上一首"

    monkeypatch.setattr(gesture_service.time, "time", lambda: 100.5)
    repeated = gesture_service.map_event_to_vehicle({"gesture_type": "swipe_left"})
    assert repeated["action_applied"] is False
    assert repeated["suppress_reason"] == "1.0 秒内忽略重复切歌"
    assert gesture_service._pending_command is None

    monkeypatch.setattr(gesture_service.time, "time", lambda: 101.2)
    reversed_too_soon = gesture_service.map_event_to_vehicle({"gesture_type": "swipe_right"})
    assert reversed_too_soon["suppress_reason"] == "1.5 秒内忽略反向切歌"
    assert gesture_service._pending_command is None

    monkeypatch.setattr(gesture_service.time, "time", lambda: 101.7)
    next_business = gesture_service.map_event_to_vehicle({"gesture_type": "swipe_right"})
    assert next_business["action"] == "turn_right"
    assert next_business["suppress_reason"].startswith("检测到待确认业务：下一首")


def test_real_gesture_swipe_cooldown_runs_before_pending_business(monkeypatch):
    _reset_vendor_state()

    monkeypatch.setattr(time, "time", lambda: 200.0)
    detected = gesture_models.make_action_message("SWIPE_LEFT")
    assert detected is not None
    assert detected.vehicle_action == "turn_left"
    assert detected.vehicle_label == "上一首"
    assert detected.suppress_reason.startswith("检测到待确认业务：上一首")

    monkeypatch.setattr(time, "time", lambda: 200.1)
    confirmed = gesture_models.make_action_message("OK")
    assert confirmed is not None and confirmed.action_applied is True

    monkeypatch.setattr(time, "time", lambda: 200.5)
    repeated = gesture_models.make_action_message("SWIPE_LEFT2")
    assert repeated is not None
    assert repeated.suppress_reason == "1.0 秒内忽略重复切歌"
    assert gesture_models._pending_action is None

    monkeypatch.setattr(time, "time", lambda: 201.2)
    reversed_too_soon = gesture_models.make_action_message("SWIPE_RIGHT3")
    assert reversed_too_soon is not None
    assert reversed_too_soon.suppress_reason == "1.5 秒内忽略反向切歌"
    assert gesture_models._pending_action is None


def test_music_business_stays_blocked_for_two_seconds_after_volume(monkeypatch):
    _reset_vendor_state()
    gesture_models._last_volume_action = gesture_models.VehicleAction.VOLUME_UP
    gesture_models._last_volume_action_at = 300.0

    monkeypatch.setattr(time, "time", lambda: 301.0)
    blocked = gesture_models.make_action_message("DOUBLE_TAP")
    assert blocked is not None
    assert blocked.suppress_reason == "音量操作后 2.0 秒内忽略播放/暂停"
    assert gesture_models._pending_action is None

    monkeypatch.setattr(time, "time", lambda: 302.1)
    detected = gesture_models.make_action_message("DOUBLE_TAP")
    assert detected is not None
    assert detected.suppress_reason.startswith("检测到待确认业务：播放/暂停")


def test_volume_return_cannot_replace_pending_volume_business(monkeypatch):
    _reset_vendor_state()

    monkeypatch.setattr(time, "time", lambda: 400.0)
    volume_up = gesture_models.make_action_message("FAST_SWIPE_UP")
    assert volume_up is not None
    assert volume_up.suppress_reason.startswith("检测到待确认业务：音量增加")

    monkeypatch.setattr(time, "time", lambda: 401.0)
    return_motion = gesture_models.make_action_message("FAST_SWIPE_DOWN")
    assert return_motion is not None
    assert return_motion.suppress_reason == "1.5 秒内忽略反向音量操作"
    assert gesture_models._pending_action is not None
    assert gesture_models._pending_action[0] == gesture_models.VehicleAction.VOLUME_UP

    monkeypatch.setattr(time, "time", lambda: 401.1)
    confirmed = gesture_models.make_action_message("OK")
    assert confirmed is not None
    assert confirmed.action_applied is True
    assert confirmed.vehicle_action == "volume_up"


def test_simulated_return_motion_keeps_original_pending_business(monkeypatch):
    _reset_service_state()

    monkeypatch.setattr(gesture_service.time, "time", lambda: 500.0)
    gesture_service.map_event_to_vehicle({"gesture_type": "swipe_up"})

    monkeypatch.setattr(gesture_service.time, "time", lambda: 501.0)
    return_motion = gesture_service.map_event_to_vehicle({"gesture_type": "swipe_down"})
    assert return_motion["suppress_reason"] == "1.5 秒内忽略反向音量操作"
    assert gesture_service._pending_command is not None
    assert gesture_service._pending_command[1] == "volume_up"


def test_unrelated_business_replaces_pending_business_without_wait(monkeypatch):
    _reset_vendor_state()

    monkeypatch.setattr(time, "time", lambda: 600.0)
    lights = gesture_models.make_action_message("LIKE")
    assert lights is not None
    assert lights.vehicle_action == "lights_on"

    monkeypatch.setattr(time, "time", lambda: 600.1)
    volume = gesture_models.make_action_message("FAST_SWIPE_DOWN")
    assert volume is not None
    assert volume.suppress_reason.startswith("检测到待确认业务：音量降低")
    assert gesture_models._pending_action is not None
    assert gesture_models._pending_action[0] == gesture_models.VehicleAction.VOLUME_DOWN


def test_thumb_index_is_one_alias_only_for_vertical_volume_business():
    engine = GestureEngine.__new__(GestureEngine)
    engine._vertical_alias_states = {}
    engine._vertical_alias_min_frames = 5
    engine._vertical_alias_max_frames = 30

    engine._frame_index = 10
    assert engine._business_vertical_alias_actions([SimpleNamespace(hand_id=1, gesture="point")]) == []
    engine._frame_index = 16
    assert engine._business_vertical_alias_actions([SimpleNamespace(hand_id=1, gesture="thumb_index")]) == [("FAST_SWIPE_UP", 1)]

    engine._frame_index = 20
    assert engine._business_vertical_alias_actions([SimpleNamespace(hand_id=1, gesture="thumb_index")]) == []
    engine._frame_index = 26
    assert engine._business_vertical_alias_actions([SimpleNamespace(hand_id=1, gesture="point")]) == [("FAST_SWIPE_DOWN", 1)]


def test_reverse_volume_business_is_available_after_one_point_five_seconds(monkeypatch):
    _reset_vendor_state()

    monkeypatch.setattr(time, "time", lambda: 700.0)
    gesture_models.make_action_message("FAST_SWIPE_UP")

    monkeypatch.setattr(time, "time", lambda: 701.4)
    blocked = gesture_models.make_action_message("FAST_SWIPE_DOWN")
    assert blocked is not None
    assert blocked.suppress_reason == "1.5 秒内忽略反向音量操作"

    monkeypatch.setattr(time, "time", lambda: 701.6)
    detected = gesture_models.make_action_message("FAST_SWIPE_DOWN")
    assert detected is not None
    assert detected.suppress_reason.startswith("检测到待确认业务：音量降低")

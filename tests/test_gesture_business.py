from backend.services import gesture_service
from vendor.web_gesture_backend import models as gesture_models


def test_music_toggle_is_suppressed_after_volume_action(monkeypatch):
    gesture_service._gesture_control_enabled = True
    gesture_service._last_volume_action_at = 0.0

    monkeypatch.setattr(gesture_service.time, "time", lambda: 100.0)
    volume = gesture_service.map_event_to_vehicle({"gesture_type": "swipe_up"})
    assert volume["action"] == "volume_up"
    assert volume["action_applied"] is True

    monkeypatch.setattr(gesture_service.time, "time", lambda: 101.0)
    tap = gesture_service.map_event_to_vehicle({"gesture_type": "tap"})
    assert tap["action"] == "music_toggle"
    assert tap["action_applied"] is False
    assert "2.0s after volume action" in tap["suppress_reason"]

    monkeypatch.setattr(gesture_service.time, "time", lambda: 102.1)
    tap = gesture_service.map_event_to_vehicle({"gesture_type": "tap"})
    assert tap["action_applied"] is True


def test_phone_action_is_not_blocked_after_volume_action(monkeypatch):
    gesture_service._gesture_control_enabled = True
    gesture_service._last_volume_action_at = 200.0
    monkeypatch.setattr(gesture_service.time, "time", lambda: 200.5)

    call = gesture_service.map_event_to_vehicle({"gesture_type": "call"})
    assert call["action"] == "phone_answer"
    assert call["action_applied"] is True


def test_real_gesture_gate_suppresses_tap_after_volume_action():
    gesture_models._gesture_control_enabled = True
    gesture_models._last_volume_action_at = 0.0
    gesture_models._last_music_toggle_at = 0.0

    applied, _ = gesture_models._action_gate(gesture_models.VehicleAction.VOLUME_UP, 300.0)
    assert applied is True

    applied, reason = gesture_models._action_gate(gesture_models.VehicleAction.MUSIC_TOGGLE, 301.0)
    assert applied is False
    assert "2.0s after volume action" in reason

    applied, _ = gesture_models._action_gate(gesture_models.VehicleAction.MUSIC_TOGGLE, 302.1)
    assert applied is True

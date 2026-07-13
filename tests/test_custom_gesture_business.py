import time

from backend.services import custom_gesture_service as custom
from vendor.web_gesture_backend import models as gesture_models


def _frame(gesture_key: str, confidence: float = 0.95):
    return {
        "hands": [{
            "gesture_key": gesture_key,
            "confidence": confidence,
            "reserved": False,
            "reason": "",
        }]
    }


def test_reserved_gestures_cannot_be_bound():
    assert "palm" in custom.RESERVED_GESTURES
    assert "like" in custom.RESERVED_GESTURES
    assert "four" not in custom.RESERVED_GESTURES


def test_image_draft_returns_reserved_label_for_user_feedback(monkeypatch):
    monkeypatch.setattr(custom, "detect_frame", lambda _: {
        "hands": [{
            "gesture_key": "like",
            "confidence": 0.91,
            "detector_confidence": 0.98,
            "classifier_confidence": 0.91,
            "reserved": True,
            "reason": custom.RESERVED_REASON,
        }]
    })

    result = custom.create_image_draft(902, "图片测试", "lights_on", b"frame")
    assert result["gesture_key"] == "like"
    assert result["can_bind"] is False
    assert result["reason"] == custom.RESERVED_REASON


def test_camera_second_capture_requires_same_gesture(monkeypatch):
    session = custom.create_camera_session(901, "自定义灯光", "lights_on")
    session_id = session["session_id"]
    monkeypatch.setattr(custom, "detect_frame", lambda _: _frame("four"))

    for _ in range(custom.STABLE_FRAME_COUNT):
        result = custom.process_camera_frame(None, 901, session_id, b"frame")
    assert result["phase"] == "first_ready"

    assert custom.confirm_camera_first(901, session_id)["phase"] == "await_hand_removed"
    monkeypatch.setattr(custom, "detect_frame", lambda _: {"hands": []})
    custom.process_camera_frame(None, 901, session_id, b"empty")
    assert custom.process_camera_frame(None, 901, session_id, b"empty")["phase"] == "second_capture"

    monkeypatch.setattr(custom, "detect_frame", lambda _: _frame("rock"))
    result = custom.process_camera_frame(None, 901, session_id, b"different")
    assert result["phase"] == "second_capture"
    assert "不一致" in result["message"]


def _reset_vendor_business_state():
    gesture_models._gesture_control_enabled = True
    gesture_models._pending_action = None
    gesture_models._pending_music_clicks = 0
    gesture_models._pending_music_click_at = 0.0
    gesture_models._last_queued_music_toggle_at = 0.0
    gesture_models._last_music_toggle_at = 0.0


def test_custom_static_gesture_uses_existing_ok_confirmation(monkeypatch):
    _reset_vendor_business_state()
    monkeypatch.setattr(time, "time", lambda: 1000.0)

    detected = gesture_models.make_custom_action_message("four", "lights_on", "氛围灯")
    assert detected is not None
    assert detected.action_applied is False
    assert detected.vehicle_label == "氛围灯"
    assert detected.suppress_reason.startswith("检测到待确认业务：开启灯光")

    monkeypatch.setattr(time, "time", lambda: 1000.1)
    confirmed = gesture_models.make_action_message("OK")
    assert confirmed is not None
    assert confirmed.action_applied is True
    assert confirmed.vehicle_action == "lights_on"


def test_custom_music_binding_is_not_treated_as_a_single_click(monkeypatch):
    _reset_vendor_business_state()
    monkeypatch.setattr(time, "time", lambda: 1100.0)

    detected = gesture_models.make_custom_action_message("four", "music_toggle", "媒体开关")
    assert detected is not None
    assert detected.suppress_reason.startswith("检测到待确认业务：播放/暂停")
    assert "第一次点击" not in detected.suppress_reason


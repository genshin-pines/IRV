from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.database import SessionLocal
from backend.models.custom_gesture_binding import CustomGestureBinding


ACTION_CATALOG = {
    "volume_up": "音量增加",
    "volume_down": "音量降低",
    "turn_left": "上一首",
    "turn_right": "下一首",
    "music_toggle": "播放/暂停",
    "lights_on": "开启灯光",
    "lights_off": "关闭灯光",
    "temp_up": "空调升温",
    "temp_down": "空调降温",
}

# These labels participate in existing static controls or dynamic-state recognition.
RESERVED_GESTURES = {
    "hand_down", "hand_right", "hand_left", "thumb_index", "thumb_left", "thumb_right",
    "thumb_down", "half_right", "part_hand_heart", "part_hand_heart2", "fist_inverted",
    "two_left", "two_right", "two_down", "grabbing", "grip", "point", "call",
    "little_finger", "middle_finger", "dislike", "fist", "like", "mute", "ok", "one",
    "palm", "peace", "peace_inverted", "stop", "stop_inverted", "three2", "two_up",
    "three_gun",
}
RESERVED_REASON = "该静态手势已被系统控车或动态手势识别占用"
MIN_CONFIDENCE = 0.85
IMAGE_MIN_DETECTOR_CONFIDENCE = 0.30
IMAGE_MIN_CLASSIFIER_CONFIDENCE = 0.20
STABLE_FRAME_COUNT = 8
DRAFT_TTL_SEC = 300
CUSTOM_COOLDOWN_SEC = 1.5


@dataclass
class Draft:
    user_id: int
    display_name: str
    action_code: str
    gesture_key: str
    confidence: float
    source: str
    expires_at: float
    can_bind: bool = True
    reject_reason: str = ""


@dataclass
class CameraSession:
    user_id: int
    display_name: str
    action_code: str
    expires_at: float
    phase: str = "first_capture"
    candidate: str | None = None
    first_gesture: str | None = None
    confidence: float = 0.0
    stable_count: int = 0
    absent_frames: int = 0


_drafts: dict[str, Draft] = {}
_camera_sessions: dict[str, CameraSession] = {}
_runtime_last_action: dict[tuple[int, int], float] = {}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _clean_expired() -> None:
    now = time.monotonic()
    for collection in (_drafts, _camera_sessions):
        for key, value in list(collection.items()):
            if value.expires_at <= now:
                collection.pop(key, None)


def _validate_details(display_name: str, action_code: str) -> tuple[str, str]:
    name = display_name.strip()
    if not 1 <= len(name) <= 64:
        raise ValueError("事件名称长度应为 1 到 64 个字符")
    if action_code not in ACTION_CATALOG:
        raise ValueError("不支持的控车动作")
    return name, action_code


def catalog() -> dict[str, Any]:
    from vendor.web_gesture_backend.dgcore.utils.enums import targets

    return {
        "actions": [{"code": code, "label": label} for code, label in ACTION_CATALOG.items()],
        "gestures": [
            {"key": key, "label": key.replace("_", " "), "reserved": key in RESERVED_GESTURES,
             "reason": RESERVED_REASON if key in RESERVED_GESTURES else ""}
            for key in targets
        ],
        "minimum_confidence": MIN_CONFIDENCE,
        "stable_frame_count": STABLE_FRAME_COUNT,
    }


def serialize_binding(row: CustomGestureBinding) -> dict[str, Any]:
    return {
        "id": row.id,
        "display_name": row.display_name,
        "gesture_key": row.gesture_key,
        "action_code": row.action_code,
        "action_label": ACTION_CATALOG.get(row.action_code, row.action_code),
        "enabled": row.enabled,
        "source": row.source,
        "confidence": round(row.confidence, 3),
        "trigger_count": row.trigger_count,
        "last_triggered_at": row.last_triggered_at.isoformat() if row.last_triggered_at else None,
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
    }


def list_bindings(db: Session, user_id: int) -> list[dict[str, Any]]:
    rows = db.scalars(
        select(CustomGestureBinding)
        .where(CustomGestureBinding.user_id == user_id)
        .order_by(CustomGestureBinding.updated_at.desc())
    ).all()
    return [serialize_binding(row) for row in rows]


def _ensure_bindable(db: Session, user_id: int, gesture_key: str) -> None:
    if gesture_key in RESERVED_GESTURES:
        raise ValueError(RESERVED_REASON)
    if db.scalar(select(CustomGestureBinding.id).where(
        CustomGestureBinding.user_id == user_id,
        CustomGestureBinding.gesture_key == gesture_key,
    )):
        raise ValueError("该手势已绑定到当前账户的其他控车事件")


def create_binding(
    db: Session, user_id: int, display_name: str, action_code: str, gesture_key: str,
    confidence: float, source: str,
) -> dict[str, Any]:
    display_name, action_code = _validate_details(display_name, action_code)
    _ensure_bindable(db, user_id, gesture_key)
    row = CustomGestureBinding(
        user_id=user_id, display_name=display_name, action_code=action_code,
        gesture_key=gesture_key, confidence=confidence, source=source,
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ValueError("事件名称或手势已存在") from exc
    db.refresh(row)
    return serialize_binding(row)


def update_binding(db: Session, user_id: int, binding_id: int, changes: dict[str, Any]) -> dict[str, Any]:
    row = db.scalar(select(CustomGestureBinding).where(
        CustomGestureBinding.id == binding_id, CustomGestureBinding.user_id == user_id,
    ))
    if row is None:
        raise LookupError("未找到该手势绑定")
    if "display_name" in changes and changes["display_name"] is not None:
        row.display_name, _ = _validate_details(changes["display_name"], row.action_code)
    if "action_code" in changes and changes["action_code"] is not None:
        _, row.action_code = _validate_details(row.display_name, changes["action_code"])
    if "enabled" in changes and changes["enabled"] is not None:
        row.enabled = bool(changes["enabled"])
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ValueError("事件名称已存在") from exc
    db.refresh(row)
    return serialize_binding(row)


def delete_binding(db: Session, user_id: int, binding_id: int) -> None:
    row = db.scalar(select(CustomGestureBinding).where(
        CustomGestureBinding.id == binding_id, CustomGestureBinding.user_id == user_id,
    ))
    if row is None:
        raise LookupError("未找到该手势绑定")
    db.delete(row)
    db.commit()


def detect_frame(contents: bytes) -> dict[str, Any]:
    """Return one frame's static labels without feeding the dynamic tracker."""
    from backend.services.gesture_service import get_engine
    from vendor.web_gesture_backend.dgcore.utils.enums import targets

    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解析图片")
    controller = get_engine().controller
    boxes, detection_probs = controller.detection_model(image)
    if boxes is None or len(boxes) == 0:
        return {"hands": []}
    labels, confidences = controller.classification_model.predict(image, boxes)
    detector_scores = np.asarray(detection_probs).reshape(-1)
    hands = []
    for index, label in enumerate(labels):
        key = targets[int(label)] if 0 <= int(label) < len(targets) else "unknown"
        detector_conf = float(detector_scores[index]) if index < len(detector_scores) else 0.0
        classifier_conf = float(confidences[index]) if index < len(confidences) else 0.0
        hands.append({
            "gesture_key": key,
            # The classifier confidence is the semantic confidence. Detection and
            # classification are exposed separately instead of discarding a valid label.
            "confidence": round(classifier_conf, 3),
            "detector_confidence": round(detector_conf, 3),
            "classifier_confidence": round(classifier_conf, 3),
            "reserved": key in RESERVED_GESTURES,
            "reason": RESERVED_REASON if key in RESERVED_GESTURES else "",
        })
    return {"hands": hands}


def _one_valid_hand(result: dict[str, Any]) -> tuple[str, float]:
    hands = result["hands"]
    if len(hands) != 1:
        raise ValueError("请确保画面中仅有一只完整手势")
    hand = hands[0]
    if hand["gesture_key"] == "unknown":
        raise ValueError("未识别到可用静态手势")
    if hand["reserved"]:
        raise ValueError(RESERVED_REASON)
    if hand["confidence"] < MIN_CONFIDENCE:
        raise ValueError("手势置信度不足，请调整光线并保持手势清晰")
    return hand["gesture_key"], hand["confidence"]


def create_image_draft(user_id: int, display_name: str, action_code: str, contents: bytes) -> dict[str, Any]:
    name, action = _validate_details(display_name, action_code)
    result = detect_frame(contents)
    hands = result["hands"]
    if len(hands) != 1:
        return {
            "draft_id": "",
            "can_bind": False,
            "reason": "未检测到唯一手势，请上传仅包含一只清晰手势的图片",
            "detected_hands": hands,
            "action_label": ACTION_CATALOG[action],
        }

    hand = hands[0]
    gesture_key = hand["gesture_key"]
    confidence = hand["confidence"]
    reason = ""
    if gesture_key == "unknown":
        reason = "未识别到可用静态手势"
    elif hand["reserved"]:
        reason = RESERVED_REASON
    elif hand["detector_confidence"] < IMAGE_MIN_DETECTOR_CONFIDENCE:
        reason = "手部检测置信度较低，请使用光线充足、手部完整的图片"
    elif hand["classifier_confidence"] < IMAGE_MIN_CLASSIFIER_CONFIDENCE:
        reason = "手势分类不够稳定，请更换更清晰、背景更简单的图片"
    can_bind = not reason
    _clean_expired()
    draft_id = uuid4().hex
    _drafts[draft_id] = Draft(
        user_id, name, action, gesture_key, confidence, "image", time.monotonic() + DRAFT_TTL_SEC,
        can_bind=can_bind, reject_reason=reason,
    )
    return {
        "draft_id": draft_id,
        "gesture_key": gesture_key,
        "confidence": confidence,
        "detector_confidence": hand["detector_confidence"],
        "classifier_confidence": hand["classifier_confidence"],
        "can_bind": can_bind,
        "reason": reason,
        "action_label": ACTION_CATALOG[action],
        "expires_in": DRAFT_TTL_SEC,
    }


def confirm_image_draft(db: Session, user_id: int, draft_id: str) -> dict[str, Any]:
    _clean_expired()
    draft = _drafts.pop(draft_id, None)
    if draft is None or draft.user_id != user_id:
        raise LookupError("绑定草稿不存在或已过期")
    if not draft.can_bind:
        raise ValueError(draft.reject_reason or "该图片中的手势不可绑定")
    return create_binding(db, user_id, draft.display_name, draft.action_code, draft.gesture_key, draft.confidence, draft.source)


def create_camera_session(user_id: int, display_name: str, action_code: str) -> dict[str, Any]:
    name, action = _validate_details(display_name, action_code)
    _clean_expired()
    session_id = uuid4().hex
    _camera_sessions[session_id] = CameraSession(user_id, name, action, time.monotonic() + DRAFT_TTL_SEC)
    return {"session_id": session_id, "phase": "first_capture", "expires_in": DRAFT_TTL_SEC}


def process_camera_observation(
    db: Session, user_id: int, session_id: str, detected: dict[str, Any],
) -> dict[str, Any]:
    _clean_expired()
    session = _camera_sessions.get(session_id)
    if session is None or session.user_id != user_id:
        raise LookupError("摄像头验证会话不存在或已过期")
    hands = detected["hands"]
    if session.phase == "await_hand_removed":
        if not hands:
            session.absent_frames += 1
            if session.absent_frames >= 2:
                session.phase = "second_capture"
        return {"phase": session.phase, "message": "请将手移出画面后再次作出同一手势"}
    try:
        gesture_key, confidence = _one_valid_hand(detected)
    except ValueError as exc:
        session.stable_count = 0
        return {"phase": session.phase, "message": str(exc), "hands": hands}
    if session.phase == "second_capture" and gesture_key != session.first_gesture:
        session.stable_count = 0
        return {"phase": session.phase, "gesture_key": gesture_key, "confidence": confidence,
                "message": "二次手势与首次确认结果不一致，请重新作出首次手势"}
    if session.candidate != gesture_key:
        session.candidate, session.confidence, session.stable_count = gesture_key, confidence, 1
    else:
        session.confidence = min(session.confidence, confidence)
        session.stable_count += 1
    if session.stable_count < STABLE_FRAME_COUNT:
        return {"phase": session.phase, "gesture_key": gesture_key, "confidence": confidence,
                "stable_count": session.stable_count, "required_stable_count": STABLE_FRAME_COUNT}
    if session.phase == "first_capture":
        session.phase = "first_ready"
        session.first_gesture = gesture_key
        return {"phase": session.phase, "gesture_key": gesture_key, "confidence": session.confidence,
                "message": "已识别到稳定手势，请确认后进行二次验证"}
    if session.phase == "second_capture":
        binding = create_binding(db, user_id, session.display_name, session.action_code, gesture_key, session.confidence, "camera")
        _camera_sessions.pop(session_id, None)
        return {"phase": "bound", "binding": binding, "message": "二次手势一致，绑定成功"}
    return {"phase": session.phase, "gesture_key": gesture_key, "confidence": session.confidence}


def process_camera_frame(db: Session, user_id: int, session_id: str, contents: bytes) -> dict[str, Any]:
    return process_camera_observation(db, user_id, session_id, detect_frame(contents))


def process_active_stream_frame(db: Session, user_id: int, session_id: str) -> dict[str, Any]:
    """Use the annotated server camera stream so the browser never opens a competing camera."""
    from backend.services.gesture_service import latest_recognition

    frame = latest_recognition(user_id)
    if frame is None:
        return {"phase": "first_capture", "message": "正在等待摄像头识别结果"}
    hands = []
    for hand in frame.get("hands", []):
        key = str(hand.get("gesture") or "unknown")
        hands.append({
            "gesture_key": key,
            "confidence": float(hand.get("confidence") or 0.0),
            "reserved": key in RESERVED_GESTURES,
            "reason": RESERVED_REASON if key in RESERVED_GESTURES else "",
        })
    return process_camera_observation(db, user_id, session_id, {"hands": hands})


def confirm_camera_first(user_id: int, session_id: str) -> dict[str, Any]:
    _clean_expired()
    session = _camera_sessions.get(session_id)
    if session is None or session.user_id != user_id or session.phase != "first_ready":
        raise LookupError("当前会话尚未获得可确认的手势")
    session.phase = "await_hand_removed"
    session.candidate = None
    session.stable_count = 0
    session.absent_frames = 0
    return {"phase": session.phase, "message": "请先将手移出画面，再次作出相同手势"}


def resolve_runtime_binding(user_id: int | None, gesture_key: str) -> dict[str, str] | None:
    if user_id is None or gesture_key in RESERVED_GESTURES:
        return None
    with SessionLocal() as db:
        row = db.scalar(select(CustomGestureBinding).where(
            CustomGestureBinding.user_id == user_id,
            CustomGestureBinding.gesture_key == gesture_key,
            CustomGestureBinding.enabled.is_(True),
        ))
        if row is None:
            return None
        now = time.monotonic()
        cooldown_key = (user_id, row.id)
        if now - _runtime_last_action.get(cooldown_key, 0.0) < CUSTOM_COOLDOWN_SEC:
            return None
        _runtime_last_action[cooldown_key] = now
        row.trigger_count += 1
        row.last_triggered_at = _utc_now()
        db.commit()
        return {"gesture_key": gesture_key, "action_code": row.action_code, "display_name": row.display_name}



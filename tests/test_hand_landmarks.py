import numpy as np
import cv2

from vendor.web_gesture_backend.gesture_engine import GestureEngine
from vendor.web_gesture_backend.hand_landmark_detector import HandLandmarkDetector
from vendor.web_gesture_backend.models import FrameMessage, HandInfo, to_dict

DISPLAY_INDEXES = (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)


def _landmarks(offset_x=0, offset_y=0):
    return [
        {"index": landmark_index, "x": offset_x + index * 2, "y": offset_y + index * 2,
         "z": 0.0, "x_norm": 0.1, "y_norm": 0.1}
        for index, landmark_index in enumerate(DISPLAY_INDEXES)
    ]


def test_landmarks_are_matched_to_tracked_hand_and_serialized():
    engine = GestureEngine.__new__(GestureEngine)
    hand = HandInfo(
        hand_id=7,
        bbox=[10, 10, 80, 80],
        gesture="like",
        gesture_id=1,
        center=(45.0, 45.0),
        confidence=1.0,
    )
    detections = [{
        "bbox": [12, 12, 76, 76],
        "landmarks": _landmarks(15, 15),
        "handedness": "Right",
        "confidence": 0.98765,
    }]

    engine._attach_landmarks([hand], detections)
    payload = to_dict(FrameMessage(hands=[hand]))

    assert len(payload["hands"][0]["landmarks"]) == 10
    assert payload["hands"][0]["handedness"] == "Right"
    assert payload["hands"][0]["landmark_confidence"] == 0.9877


def test_unmatched_mediapipe_hand_does_not_change_legacy_recognition_results():
    engine = GestureEngine.__new__(GestureEngine)
    hands = []
    detections = [{
        "bbox": [20, 20, 90, 90],
        "landmarks": _landmarks(20, 20),
        "handedness": "Left",
        "confidence": 0.9,
    }]

    engine._attach_landmarks(hands, detections)

    assert hands == []


def test_landmark_skeleton_is_drawn_on_video_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hand = HandInfo(
        hand_id=1,
        bbox=[5, 5, 90, 90],
        gesture="stop",
        gesture_id=1,
        center=(45.0, 45.0),
        confidence=1.0,
        landmarks=_landmarks(10, 10),
    )

    GestureEngine._draw_landmarks(frame, hand)

    assert np.count_nonzero(frame) > 0


def test_mediapipe_refresh_is_skipped_while_optical_flow_tracks_points():
    detector = HandLandmarkDetector.__new__(HandLandmarkDetector)
    detector.detector = object()
    detector.detect_interval = 3
    detector._last_detection_frame = -3
    detector._previous_gray = None
    detector._tracked_detections = []
    detector._force_detection = False
    calls = []

    base_points = [(20 + (index % 7) * 8, 20 + (index // 7) * 12) for index in range(21)]
    detection = {
        "bbox": [20, 20, 68, 44],
        "landmarks": [
            {"x": x, "y": y, "z": 0.0, "x_norm": x / 100, "y_norm": y / 100}
            for x, y in base_points
        ],
        "handedness": "Right",
        "confidence": 0.9,
    }
    detector.detect = lambda frame: calls.append(frame) or [detection]

    def frame_with_offset(offset):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for x, y in base_points:
            cv2.circle(frame, (x + offset, y), 3, (255, 255, 255), -1)
        return frame

    detector.process(frame_with_offset(0), frame_index=1)
    tracked = detector.process(frame_with_offset(2), frame_index=2)

    assert len(calls) == 1
    assert len(tracked) == 1
    assert tracked[0]["landmarks"][0]["x"] >= base_points[0][0] + 1

    detector.process(frame_with_offset(4), frame_index=4)
    assert len(calls) == 2

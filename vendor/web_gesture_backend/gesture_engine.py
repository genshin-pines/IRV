# gesture_engine.py - relative path version
import sys
import time
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dgcore.main_controller import MainController
from dgcore.utils import targets, Event, Drawer

from models import HandInfo, FrameMessage, make_action_message, to_dict


class GestureEngine:
    def __init__(self, detector_path=None, classifier_path=None):
        base = Path(__file__).parent / 'dgcore' / 'models'
        detector_path = detector_path or str(base / 'hand_detector.onnx')
        classifier_path = classifier_path or str(base / 'crops_classifier.onnx')

        print(f'[GestureEngine] detector: {detector_path}')
        print(f'[GestureEngine] classifier: {classifier_path}')
        self.controller = MainController(detector_path, classifier_path)

        self.on_action = None
        self.on_frame = None
        self._last_time = time.time()
        self._fps = 0.0
        self.drawer = Drawer()
        self._last_vertical_action = None
        self._last_vertical_action_at = 0.0
        self._opposite_vertical_cooldown_sec = 6.0

    def _should_emit_action(self, event_name, now):
        up_events = {"SWIPE_UP", "SWIPE_UP2", "SWIPE_UP3", "FAST_SWIPE_UP"}
        down_events = {"SWIPE_DOWN", "SWIPE_DOWN2", "SWIPE_DOWN3", "FAST_SWIPE_DOWN"}
        if event_name not in up_events and event_name not in down_events:
            return True

        direction = "up" if event_name in up_events else "down"
        opposite = self._last_vertical_action in {"up", "down"} and self._last_vertical_action != direction
        if opposite and now - self._last_vertical_action_at < self._opposite_vertical_cooldown_sec:
            return False

        self._last_vertical_action = direction
        self._last_vertical_action_at = now
        return True

    def process_frame(self, frame):
        now = time.time()
        self._fps = 0.9 * self._fps + 0.1 / max(now - self._last_time, 0.001)
        self._last_time = now

        bboxes, ids, labels = self.controller(frame)

        hands = []
        if bboxes is not None and len(bboxes) > 0:
            bboxes_i = bboxes.astype(np.int32)
            for i in range(bboxes_i.shape[0]):
                box = bboxes_i[i]
                label = labels[i] if labels is not None and i < len(labels) else -1
                hands.append(HandInfo(
                    hand_id=int(ids[i]) if ids is not None else -1,
                    bbox=[int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    gesture=targets[label] if 0 <= label < len(targets) else 'unknown',
                    gesture_id=int(label),
                    center=(float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)),
                    confidence=1.0,
                ))

        if self.on_frame:
            self.on_frame(to_dict(FrameMessage(timestamp=now, fps=round(self._fps, 1), hands=hands)))

        count_of_zoom = 0
        thumb_boxes = []
        if len(self.controller.tracks) > 0:
            for trk in self.controller.tracks:
                if trk['tracker'].time_since_update < 1:
                    if len(trk['hands']):
                        count_of_zoom += (trk['hands'][-1].gesture == 3)
                        thumb_boxes.append(trk['hands'][-1].bbox)

                if trk['hands'].action is not None:
                    event_name = trk['hands'].action.name
                    if not self._should_emit_action(event_name, now):
                        trk['hands'].action = None
                        continue
                    self.drawer.set_action(trk['hands'].action)
                    msg = make_action_message(event_name)
                    if msg and self.on_action:
                        self.on_action(to_dict(msg))
                    trk['hands'].action = None

        if count_of_zoom == 2:
            self.drawer.draw_two_hands(frame, thumb_boxes)

        annotated = frame.copy()
        for hand in hands:
            b = hand.bbox
            cv2.rectangle(annotated, (b[0], b[1]), (b[2], b[3]), (255, 255, 0), 3)
            cv2.putText(annotated, f'ID{hand.hand_id}:{hand.gesture}',
                        (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(annotated, f'FPS:{self._fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        annotated = self.drawer.draw(annotated)
        return annotated

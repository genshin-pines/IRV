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

from models import HandInfo, FrameMessage, make_action_message, make_custom_action_message, to_dict


class GestureEngine:
    def __init__(self, detector_path=None, classifier_path=None, trace_path=None, reset_trace=False):
        base = Path(__file__).parent / 'dgcore' / 'models'
        detector_path = detector_path or str(base / 'hand_detector.onnx')
        classifier_path = classifier_path or str(base / 'crops_classifier.onnx')
        project_dir = Path(__file__).resolve().parents[2]
        self.trace_path = Path(trace_path) if trace_path else project_dir / "logs" / "gesture_static_trace.log"
        self._frame_index = 0

        print(f'[GestureEngine] detector: {detector_path}')
        print(f'[GestureEngine] classifier: {classifier_path}')
        self._init_trace(reset_trace=reset_trace)
        self.controller = MainController(detector_path, classifier_path)

        self.on_action = None
        self.on_frame = None
        self.custom_action_resolver = None
        self._last_time = time.time()
        self._fps = 0.0
        self.drawer = Drawer()
        self._last_static_action = None
        self._last_static_action_at = 0.0
        self._static_action_interval = 1.5
        self._static_action_candidate = None
        self._static_action_candidate_center = None
        self._static_action_candidate_count = 0
        self._static_action_min_frames = 8
        self._static_action_max_move = 35.0
        self._vertical_alias_states = {}
        self._vertical_alias_min_frames = 5
        self._vertical_alias_max_frames = 30
        self._custom_static_candidate = None
        self._custom_static_candidate_center = None
        self._custom_static_candidate_count = 0

    def _init_trace(self, reset_trace=False):
        try:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "w" if reset_trace else "a"
            with self.trace_path.open(mode, encoding="utf-8") as fp:
                fp.write(f"# gesture static trace start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                fp.write("# frame\ttime\tfps\thand_id\tgesture_id\tgesture\tbbox\tcenter\n")
        except Exception as exc:
            print(f"[GestureEngine] trace disabled: {exc}")
            self.trace_path = None

    def _trace_frame(self, now, hands):
        if not self.trace_path:
            return
        if not hands:
            return
        try:
            with self.trace_path.open("a", encoding="utf-8") as fp:
                for hand in hands:
                    fp.write(
                        f"{self._frame_index}\t{now:.3f}\t{self._fps:.1f}\t"
                        f"{hand.hand_id}\t{hand.gesture_id}\t{hand.gesture}\t"
                        f"{hand.bbox}\t{[round(v, 1) for v in hand.center]}\n"
                    )
        except Exception as exc:
            print(f"[GestureEngine] trace frame failed: {exc}")
            self.trace_path = None

    def _trace_action(self, now, event_name, msg):
        if not self.trace_path:
            return
        try:
            with self.trace_path.open("a", encoding="utf-8") as fp:
                fp.write(
                    f"ACTION\t{now:.3f}\t{event_name}\t"
                    f"vehicle={msg.vehicle_action if msg else ''}\t"
                    f"applied={msg.action_applied if msg else ''}\t"
                    f"reason={msg.suppress_reason if msg else ''}\n"
                )
        except Exception as exc:
            print(f"[GestureEngine] trace action failed: {exc}")
            self.trace_path = None

    def process_frame(self, frame):
        self._frame_index += 1
        now = time.time()
        self._fps = 0.9 * self._fps + 0.1 / max(now - self._last_time, 0.001)
        self._last_time = now

        bboxes, ids, labels = self.controller(frame)

        hands = []
        if bboxes is not None and len(bboxes) > 0:
            bboxes_i = bboxes.astype(np.int32)
            for i in range(bboxes_i.shape[0]):
                box = bboxes_i[i]
                label = labels[i] if labels is not None and i < len(labels) and labels[i] is not None else -1
                hands.append(HandInfo(
                    hand_id=int(ids[i]) if ids is not None else -1,
                    bbox=[int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    gesture=targets[label] if 0 <= label < len(targets) else 'unknown',
                    gesture_id=int(label),
                    center=(float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)),
                    confidence=1.0,
                ))

        self._trace_frame(now, hands)
        if self.on_frame:
            self.on_frame(to_dict(FrameMessage(timestamp=now, fps=round(self._fps, 1), hands=hands)))

        for event_name, hand_id in self._business_vertical_alias_actions(hands):
            msg = make_action_message(event_name, hand_id=hand_id)
            self._trace_action(now, event_name, msg)
            if msg and msg.action_applied:
                self.drawer.set_feedback(
                    msg.vehicle_action,
                    msg.vehicle_label,
                    control_enabled=msg.gesture_control_enabled,
                )
            if msg and self.on_action:
                self.on_action(to_dict(msg))

        static_action = self._static_action(hands)
        if static_action:
            msg = make_action_message(static_action)
            self._trace_action(now, static_action, msg)
            if msg and msg.action_applied:
                self.drawer.set_feedback(
                    msg.vehicle_action,
                    msg.vehicle_label,
                    control_enabled=msg.gesture_control_enabled,
                )
            if msg and self.on_action:
                self.on_action(to_dict(msg))

        custom_gesture = self._custom_static_gesture(hands)
        if custom_gesture and self.custom_action_resolver:
            try:
                binding = self.custom_action_resolver(custom_gesture)
            except Exception as exc:
                print(f"[GestureEngine] custom gesture lookup failed: {exc}")
                binding = None
            if binding:
                msg = make_custom_action_message(
                    binding["gesture_key"], binding["action_code"], binding["display_name"],
                )
                self._trace_action(now, f"CUSTOM:{custom_gesture}", msg)
                if msg and msg.action_applied:
                    self.drawer.set_feedback(
                        msg.vehicle_action, msg.vehicle_label, control_enabled=msg.gesture_control_enabled,
                    )
                if msg and self.on_action:
                    self.on_action(to_dict(msg))

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
                    msg = make_action_message(event_name)
                    self._trace_action(now, event_name, msg)
                    if msg and msg.action_applied:
                        self.drawer.set_feedback(
                            msg.vehicle_action,
                            msg.vehicle_label,
                            control_enabled=msg.gesture_control_enabled,
                        )
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

    def _business_vertical_alias_actions(self, hands):
        """Complete the point/one alias groups for vertical-swipe business events."""
        point_aliases = {"point", "fist", "fist_inverted"}
        one_aliases = {"one", "mute", "little_finger", "thumb_index"}
        events = []
        active_ids = set()
        for hand in hands:
            hand_id = hand.hand_id
            active_ids.add(hand_id)
            if hand.gesture in point_aliases:
                kind = "point_group"
            elif hand.gesture in one_aliases:
                kind = "one_group"
            else:
                continue

            previous = self._vertical_alias_states.get(hand_id)
            if previous is None:
                self._vertical_alias_states[hand_id] = (kind, self._frame_index, hand.gesture)
                continue

            previous_kind, start_frame, previous_gesture = previous
            if previous_kind == kind:
                continue

            duration = self._frame_index - start_frame
            if self._vertical_alias_min_frames <= duration <= self._vertical_alias_max_frames:
                # Native point/one transitions are already emitted by the original
                # controller. This adapter fills only business-layer aliases.
                business_aliases = {"fist", "fist_inverted", "thumb_index"}
                contains_business_alias = previous_gesture in business_aliases or hand.gesture in business_aliases
                if contains_business_alias:
                    event_name = "FAST_SWIPE_UP" if previous_kind == "point_group" else "FAST_SWIPE_DOWN"
                    events.append((event_name, hand_id))
                self._vertical_alias_states.pop(hand_id, None)
            else:
                self._vertical_alias_states[hand_id] = (kind, self._frame_index, hand.gesture)

        expired_before = self._frame_index - self._vertical_alias_max_frames
        for hand_id, (_kind, start_frame, _gesture) in list(self._vertical_alias_states.items()):
            if start_frame < expired_before or hand_id not in active_ids:
                self._vertical_alias_states.pop(hand_id, None)
        return events

    def _static_action(self, hands):
        action = None
        center = None
        if any(hand.gesture == "like" for hand in hands):
            action = "LIKE"
            center = next(hand.center for hand in hands if hand.gesture == "like")
        elif any(hand.gesture == "dislike" for hand in hands):
            action = "DISLIKE"
            center = next(hand.center for hand in hands if hand.gesture == "dislike")
        elif any(hand.gesture == "call" for hand in hands):
            action = "CALL"
            center = next(hand.center for hand in hands if hand.gesture == "call")
        elif any(hand.gesture in {"stop", "stop_inverted"} for hand in hands):
            action = "STOP"
            center = next(hand.center for hand in hands if hand.gesture in {"stop", "stop_inverted"})
        elif any(hand.gesture == "ok" for hand in hands):
            action = "OK"
            center = next(hand.center for hand in hands if hand.gesture == "ok")

        if not action:
            self._static_action_candidate = None
            self._static_action_candidate_center = None
            self._static_action_candidate_count = 0
            return None

        if action != self._static_action_candidate or self._static_action_candidate_center is None:
            self._static_action_candidate = action
            self._static_action_candidate_center = center
            self._static_action_candidate_count = 1
            return None

        dx = center[0] - self._static_action_candidate_center[0]
        dy = center[1] - self._static_action_candidate_center[1]
        if (dx * dx + dy * dy) ** 0.5 > self._static_action_max_move:
            self._static_action_candidate_center = center
            self._static_action_candidate_count = 1
            return None

        self._static_action_candidate_count += 1
        self._static_action_candidate_center = center
        required_frames = 2 if action == "OK" else self._static_action_min_frames
        if self._static_action_candidate_count < required_frames:
            return None

        now = time.time()
        elapsed = now - self._last_static_action_at
        if action == self._last_static_action and elapsed < self._static_action_interval:
            return None

        self._last_static_action = action
        self._last_static_action_at = now
        return action

    def _custom_static_gesture(self, hands):
        if len(hands) != 1 or hands[0].gesture == "unknown":
            self._custom_static_candidate = None
            self._custom_static_candidate_center = None
            self._custom_static_candidate_count = 0
            return None

        gesture = hands[0].gesture
        center = hands[0].center
        if gesture != self._custom_static_candidate or self._custom_static_candidate_center is None:
            self._custom_static_candidate = gesture
            self._custom_static_candidate_center = center
            self._custom_static_candidate_count = 1
            return None

        dx = center[0] - self._custom_static_candidate_center[0]
        dy = center[1] - self._custom_static_candidate_center[1]
        if (dx * dx + dy * dy) ** 0.5 > self._static_action_max_move:
            self._custom_static_candidate_center = center
            self._custom_static_candidate_count = 1
            return None

        self._custom_static_candidate_count += 1
        self._custom_static_candidate_center = center
        if self._custom_static_candidate_count < self._static_action_min_frames:
            return None
        return gesture

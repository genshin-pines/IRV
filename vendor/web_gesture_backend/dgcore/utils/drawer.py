import cv2
import numpy as np

from .enums import Event


class Drawer:
    def __init__(self):
        self.height = self.width = None
        self.feedback = None
        self.show_delay = 0
        self.max_show_delay = 18

    def set_action(self, action):
        event_to_action = {
            Event.SWIPE_LEFT: "turn_left",
            Event.SWIPE_LEFT2: "turn_left",
            Event.SWIPE_LEFT3: "turn_left",
            Event.SWIPE_RIGHT: "turn_right",
            Event.SWIPE_RIGHT2: "turn_right",
            Event.SWIPE_RIGHT3: "turn_right",
            Event.SWIPE_UP: "volume_up",
            Event.SWIPE_UP2: "volume_up",
            Event.SWIPE_UP3: "volume_up",
            Event.FAST_SWIPE_UP: "volume_up",
            Event.SWIPE_DOWN: "volume_down",
            Event.SWIPE_DOWN2: "volume_down",
            Event.SWIPE_DOWN3: "volume_down",
            Event.FAST_SWIPE_DOWN: "volume_down",
            Event.TAP: "music_toggle",
            Event.DOUBLE_TAP: "music_toggle",
            Event.ZOOM_IN: "accelerate",
            Event.ZOOM_OUT: "decelerate",
            Event.DNDV1: "control_toggle",
        }
        vehicle_action = event_to_action.get(action)
        if vehicle_action:
            self.set_feedback(vehicle_action)

    def set_feedback(self, vehicle_action, label="", control_enabled=None):
        self.feedback = {
            "vehicle_action": vehicle_action,
            "label": label,
            "control_enabled": control_enabled,
        }
        self.show_delay = 0

    def draw_two_hands(self, frame, bboxes):
        self.height, self.width, _ = frame.shape
        center_x1, center_y1 = bboxes[0][0] + (bboxes[0][2] - bboxes[0][0]) // 2, bboxes[0][1] + (bboxes[0][3] - bboxes[0][1]) // 2
        center_x2, center_y2 = bboxes[1][0] + (bboxes[1][2] - bboxes[1][0]) // 2, bboxes[1][1] + (bboxes[1][3] - bboxes[1][1]) // 2
        diff = int(center_x1 - center_x2)
        return cv2.rectangle(
            frame,
            (int(center_x1), int(center_y1 - diff * 0.3)),
            (int(center_x2), int(center_y2 + diff * 0.3)),
            (255, 0, 0),
            5,
        )

    def draw(self, frame):
        if self.height is None:
            self.height, self.width, _ = frame.shape
        if not self.feedback:
            return frame

        self.height, self.width, _ = frame.shape
        self._draw_feedback_card(frame)
        self.show_delay += 1
        if self.show_delay > self.max_show_delay:
            self.show_delay = 0
            self.feedback = None
        return frame

    def _draw_feedback_card(self, frame):
        style = self._feedback_style()
        card_w = min(430, max(300, int(self.width * 0.46)))
        card_h = 96
        x1 = (self.width - card_w) // 2
        y1 = max(28, int(self.height * 0.12))
        x2 = x1 + card_w
        y2 = y1 + card_h

        overlay = frame.copy()
        self._rounded_rect(overlay, (x1, y1), (x2, y2), 18, (21, 28, 42), -1)
        self._rounded_rect(overlay, (x1, y1), (x2, y2), 18, style["accent"], 2)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        icon_center = (x1 + 54, y1 + card_h // 2)
        cv2.circle(frame, icon_center, 28, style["accent"], -1)
        cv2.circle(frame, icon_center, 29, (255, 255, 255), 1)
        self._draw_icon(frame, style["icon"], icon_center, (255, 255, 255))

        title_org = (x1 + 100, y1 + 41)
        subtitle_org = (x1 + 100, y1 + 68)
        cv2.putText(frame, style["title"], title_org, cv2.FONT_HERSHEY_SIMPLEX, 0.74, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, style["subtitle"], subtitle_org, cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 204, 220), 1, cv2.LINE_AA)

    def _feedback_style(self):
        action = self.feedback.get("vehicle_action")
        enabled = self.feedback.get("control_enabled")
        styles = {
            "volume_up": ("VOL +", "Volume increased", "volume_up", (72, 188, 139)),
            "volume_down": ("VOL -", "Volume decreased", "volume_down", (72, 188, 139)),
            "turn_left": ("PREV TRACK", "Previous song", "left", (88, 144, 232)),
            "turn_right": ("NEXT TRACK", "Next song", "right", (88, 144, 232)),
            "music_toggle": ("PLAY / PAUSE", "Music toggled", "music", (96, 190, 232)),
            "lights_on": ("LIGHT ON", "Cabin light enabled", "light", (42, 182, 150)),
            "lights_off": ("LIGHT OFF", "Cabin light disabled", "light_off", (92, 112, 132)),
            "phone_answer": ("CALL ANSWER", "Phone connected", "phone", (50, 190, 120)),
            "phone_hangup": ("CALL END", "Phone disconnected", "phone_off", (82, 96, 116)),
            "temp_up": ("TEMP +", "Temperature increased", "temp_up", (232, 151, 75)),
            "temp_down": ("TEMP -", "Temperature decreased", "temp_down", (74, 154, 224)),
            "accelerate": ("ACCELERATE", "Vehicle action accepted", "plus", (74, 154, 224)),
            "decelerate": ("DECELERATE", "Vehicle action accepted", "minus", (92, 112, 132)),
        }
        if action == "control_toggle":
            if enabled:
                return {"title": "CONTROL ON", "subtitle": "Gesture control enabled", "icon": "power", "accent": (42, 182, 150)}
            return {"title": "CONTROL OFF", "subtitle": "Gesture control disabled", "icon": "power", "accent": (71, 116, 224)}

        title, subtitle, icon, accent = styles.get(action, ("ACTION OK", "Gesture accepted", "check", (88, 144, 232)))
        return {"title": title, "subtitle": subtitle, "icon": icon, "accent": accent}

    def _draw_icon(self, frame, icon, center, color):
        x, y = center
        if icon == "power":
            cv2.circle(frame, center, 13, color, 3)
            cv2.line(frame, (x, y - 20), (x, y - 5), color, 3, cv2.LINE_AA)
        elif icon == "music":
            pts = np.array([[x - 8, y - 13], [x - 8, y + 13], [x + 12, y]], np.int32)
            cv2.fillConvexPoly(frame, pts, color)
            cv2.rectangle(frame, (x + 16, y - 12), (x + 20, y + 12), color, -1)
            cv2.rectangle(frame, (x + 24, y - 12), (x + 28, y + 12), color, -1)
        elif icon in {"volume_up", "volume_down"}:
            pts = np.array([[x - 18, y - 8], [x - 9, y - 8], [x + 2, y - 17], [x + 2, y + 17], [x - 9, y + 8], [x - 18, y + 8]], np.int32)
            cv2.fillConvexPoly(frame, pts, color)
            cv2.ellipse(frame, (x + 4, y), (10, 14), 0, -45, 45, color, 2)
            if icon == "volume_up":
                cv2.line(frame, (x + 21, y - 7), (x + 21, y + 7), color, 2, cv2.LINE_AA)
                cv2.line(frame, (x + 14, y), (x + 28, y), color, 2, cv2.LINE_AA)
            else:
                cv2.line(frame, (x + 14, y), (x + 28, y), color, 2, cv2.LINE_AA)
        elif icon in {"left", "right"}:
            if icon == "left":
                cv2.arrowedLine(frame, (x + 16, y), (x - 17, y), color, 4, tipLength=0.45)
            else:
                cv2.arrowedLine(frame, (x - 16, y), (x + 17, y), color, 4, tipLength=0.45)
        elif icon in {"light", "light_off"}:
            cv2.circle(frame, (x, y - 4), 10, color, 3)
            cv2.line(frame, (x - 8, y + 10), (x + 8, y + 10), color, 3, cv2.LINE_AA)
            cv2.line(frame, (x - 5, y + 17), (x + 5, y + 17), color, 2, cv2.LINE_AA)
            if icon == "light_off":
                cv2.line(frame, (x - 18, y - 18), (x + 18, y + 18), color, 3, cv2.LINE_AA)
        elif icon in {"phone", "phone_off"}:
            cv2.putText(frame, "CALL", (x - 23, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2, cv2.LINE_AA)
            if icon == "phone_off":
                cv2.line(frame, (x - 18, y - 18), (x + 18, y + 18), color, 3, cv2.LINE_AA)
        elif icon == "temp_up":
            cv2.putText(frame, "+", (x - 10, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
        elif icon == "temp_down":
            cv2.putText(frame, "-", (x - 10, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
        elif icon == "plus":
            cv2.line(frame, (x - 13, y), (x + 13, y), color, 4, cv2.LINE_AA)
            cv2.line(frame, (x, y - 13), (x, y + 13), color, 4, cv2.LINE_AA)
        elif icon == "minus":
            cv2.line(frame, (x - 14, y), (x + 14, y), color, 4, cv2.LINE_AA)
        else:
            cv2.line(frame, (x - 14, y), (x - 2, y + 12), color, 4, cv2.LINE_AA)
            cv2.line(frame, (x - 2, y + 12), (x + 16, y - 12), color, 4, cv2.LINE_AA)

    @staticmethod
    def _rounded_rect(frame, pt1, pt2, radius, color, thickness):
        x1, y1 = pt1
        x2, y2 = pt2
        radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
        if thickness < 0:
            cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
            cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
            cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1)
            cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1)
            cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1)
            cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1)
            return

        cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)

from scipy.spatial import distance
from collections import deque

from .enums import Event, HandPosition, targets
from .hand import Hand


class Deque:
    def __init__(self, maxlen=30, min_frames=20):
        self.maxlen = maxlen
        self._deque = []
        self.action = None
        self.min_absolute_distance = 1.5
        self.min_frames = min_frames
        self.fast_swipe_min_frames = 5
        self.action_deque = deque(maxlen=5)
        self._frame_counter = 0
        self._last_tap_action_frame = -999
        self.tap_action_min_gap_frames = 14
        self.fast_swipe_point_gestures = {19}
        self.fast_swipe_one_gestures = {30, 28, 22}
        self.three_gun_gesture = 41
        self.three_gun_swipe_distance = 0.45
        self.three_gun_min_frames = 10
        self.motion_swipe_min_frames = 6
        self.motion_swipe_distance = 0.75
        self.motion_swipe_axis_ratio = 1.35
        self.right_swipe_start_alias_gestures = {-1, 14, 20, 27, 41, None}
        self.right_swipe_end_alias_gestures = {-1, 1, 9, 14, 17, 20, 23, 27, 32, 33, 41, None}
        self.right_swipe_alias_gestures = (
            self.right_swipe_start_alias_gestures | self.right_swipe_end_alias_gestures
        )
        self.right_swipe_alias_distance = 0.30
        self.right_swipe_alias_axis_ratio = 0.8
        self.right_swipe_alias_min_frames = 3
        self.three_gun_smooth_window = 6
        self.three_gun_smooth_min_frames = 3
        self._gesture_raw_history = deque(maxlen=self.three_gun_smooth_window)
        self.dndv1_max_frames = 30
        self.dndv1_grabbing_min_frames = 2

    def __len__(self):
        return len(self._deque)

    def index_position(self, x):
        for i in range(len(self._deque)):
            if self._deque[i].position == x:
                return i
        return -1

    def index_gesture(self, x):
        for i in range(len(self._deque)):
            if self._deque[i].gesture == x:
                return i
        return -1

    def __getitem__(self, index):
        return self._deque[index]

    def __setitem__(self, index, value):
        self._deque[index] = value

    def __delitem__(self, index):
        del self._deque[index]

    def __iter__(self):
        return iter(self._deque)

    def __reversed__(self):
        return reversed(self._deque)

    def append(self, x):
        self._frame_counter += 1
        if self.maxlen is not None and len(self) >= self.maxlen:
            self._deque.pop(0)
        raw_gesture = x.gesture
        self.smooth_three_gun_unknown_jitter(x)
        if x.bbox is not None:
            self._gesture_raw_history.append(raw_gesture)
        self.set_hand_position(x)
        self._deque.append(x)
        self.check_is_action(x)

    def check_duration(self, start_index, min_frames=None):
        """
        Check duration of swipe.

        Parameters
        ----------
        start_index : int
            Index of start position of swipe.

        Returns
        -------
        bool
            True if duration of swipe is more than min_frames.
        """
        if min_frames == None:
            min_frames = self.min_frames
        if start_index is not None and start_index >= 0 and len(self) - start_index >= min_frames:
            return True
        else:
            return False
        
    def check_duration_max(self, start_index, max_frames=10):
        """
        Check duration of swipe.

        Parameters
        ----------
        start_index : int
            Index of start position of swipe.

        Returns
        -------
        bool
            True if duration of swipe is more than min_frames.
        """
        if start_index is not None and start_index >= 0 and len(self) - start_index <= max_frames:
            return True
        else:
            return False
        
    def check_is_action(self, x):
        """
        Check if gesture is action.

        Parameters
        ----------
        x : Hand
            Hand object.

        Returns
        -------
        bool
            True if gesture is action.
        """
        if self.detect_dndv1_sequence(x):
            self.action = Event.DNDV1
            self.clear()
            return True

        if x.position == HandPosition.LEFT_END and HandPosition.RIGHT_START in self:
            start_index = self.index_position(HandPosition.RIGHT_START)
            if (
                self.swipe_distance(self._deque[start_index], x, min_absolute_distance=self.swipe_distance_threshold(x))
                and self.check_duration(start_index, min_frames=self.swipe_duration_threshold(x))
                and self.check_horizontal_swipe(self._deque[start_index], x)
            ):
                self.action = Event.SWIPE_LEFT
                self.clear()
                return True
            
        elif x.position == HandPosition.RIGHT_END and HandPosition.LEFT_START in self:
            start_index = self.index_position(HandPosition.LEFT_START)
            if (
                self.swipe_distance(self._deque[start_index], x, min_absolute_distance=self.swipe_distance_threshold(x))
                and self.check_duration(start_index, min_frames=self.right_swipe_duration_threshold(x))
                and self.check_horizontal_swipe(self._deque[start_index], x)
            ):
                self.action = Event.SWIPE_RIGHT
                self.clear()
                return True

        elif x.position == HandPosition.UP_END and HandPosition.DOWN_START in self:
            start_index = self.index_position(HandPosition.DOWN_START)
            if (
                self.swipe_distance(self._deque[start_index], x)
                and self.check_duration(start_index)
                and self.check_vertical_swipe(self._deque[start_index], x)
            ):
                self.action = Event.SWIPE_UP
                self.clear()
                return True
            else:
                self.clear()

        elif x.position == HandPosition.DOWN_END and HandPosition.UP_START in self:
            start_index = self.index_position(HandPosition.UP_START)
            if (
                self.swipe_distance(self._deque[start_index], x)
                and self.check_duration(start_index)
                and self.check_vertical_swipe(self._deque[start_index], x)
            ):
                self.action = Event.SWIPE_DOWN
                self.clear()
                return True
            else:
                self.clear()

        elif x.gesture == 18: # grip
            if self.action is None:
                start_index = self.index_gesture(18) 
                if self.check_duration(start_index):
                    self.action = Event.DRAG2
                    return True
                
        elif self.action == Event.DRAG2 and x.gesture in [11, 12]: # hand heart
            self.action = Event.DROP2
            self.clear()
            return True
        
        elif x.gesture == 29: # ok
            if self.action is None:
                start_index = self.index_gesture(29)
                if self.check_duration(start_index):
                    self.action = Event.DRAG3
                    return True
                 
        elif self.action == Event.DRAG3 and x.gesture in [11, 12]: # hand heart
            self.action = Event.DROP3
            self.clear()
            return True
        
        elif x.position == HandPosition.FAST_SWIPE_UP_END and HandPosition.FAST_SWIPE_UP_START in self:
            start_index = self.index_position(HandPosition.FAST_SWIPE_UP_START)
            if (
                self.check_duration(start_index, min_frames=self.fast_swipe_min_frames)
            ):
                self.action = Event.FAST_SWIPE_UP
                self.clear()
                return True
            else:
                self.clear()

        elif x.position == HandPosition.FAST_SWIPE_DOWN_END and HandPosition.FAST_SWIPE_DOWN_START in self:
            start_index = self.index_position(HandPosition.FAST_SWIPE_DOWN_START)
            if (
                self.check_duration(start_index, min_frames=self.fast_swipe_min_frames)
            ):
                self.action = Event.FAST_SWIPE_DOWN
                self.clear()
                return True
            else:
                self.clear()

        elif x.position == HandPosition.ZOOM_IN_END and HandPosition.ZOOM_IN_START in self:
            start_index = self.index_position(HandPosition.ZOOM_IN_START)
            if (
                    self.check_duration(start_index, min_frames=20)
                    and self.check_vertical_swipe(self._deque[start_index], x)
                    and self.check_horizontal_swipe(self._deque[start_index], x)
                ):
                    self.action = Event.ZOOM_IN
                    self.clear()
                    return True
        
        elif x.position == HandPosition.ZOOM_OUT_END and HandPosition.ZOOM_OUT_START in self:
            start_index = self.index_position(HandPosition.ZOOM_OUT_START)
            if (
                    self.check_duration(start_index, min_frames=20)
                    and self.check_vertical_swipe(self._deque[start_index], x)
                    and self.check_horizontal_swipe(self._deque[start_index], x)
                ):
                    self.action = Event.ZOOM_OUT
                    self.clear()
                    return True
            else:
                self.clear()

        elif x.position == HandPosition.LEFT_END2 and HandPosition.RIGHT_START2 in self:
            
            start_index = self.index_position(HandPosition.RIGHT_START2)
            if (
                self.swipe_distance(self._deque[start_index], x, min_absolute_distance=self.swipe_distance_threshold(x))
                and self.check_duration(start_index, min_frames=self.swipe_duration_threshold(x))
                and self.check_horizontal_swipe(self._deque[start_index], x)
            ):
                self.action = Event.SWIPE_LEFT2
                self.clear()
                return True
            else:
                self.clear()
            
        elif x.position == HandPosition.RIGHT_END2 and HandPosition.LEFT_START2 in self:
            start_index = self.index_position(HandPosition.LEFT_START2)
            if (
                self.swipe_distance(self._deque[start_index], x, min_absolute_distance=self.swipe_distance_threshold(x))
                and self.check_duration(start_index, min_frames=self.right_swipe_duration_threshold(x))
                and self.check_horizontal_swipe(self._deque[start_index], x)
            ):
                self.action = Event.SWIPE_RIGHT2
                self.clear()
                return True
            else:
                self.clear()

        elif x.position == HandPosition.UP_END2 and HandPosition.DOWN_START2 in self:
            start_index = self.index_position(HandPosition.DOWN_START2)
            if (
                self.swipe_distance(self._deque[start_index], x)
                and self.check_duration(start_index)
                and self.check_vertical_swipe(self._deque[start_index], x)
            ):
                self.action = Event.SWIPE_UP2
                self.clear()
                return True
            else:
                self.clear()

        elif x.position == HandPosition.LEFT_END3 and HandPosition.RIGHT_START3 in self:
            start_index = self.index_position(HandPosition.RIGHT_START3)
            if (
                self.swipe_distance(self._deque[start_index], x, min_absolute_distance=self.swipe_distance_threshold(x))
                and self.check_duration(start_index, min_frames=self.swipe_duration_threshold(x))
                and self.check_horizontal_swipe(self._deque[start_index], x)
            ):
                self.action = Event.SWIPE_LEFT3 # two
                self.clear()
                return True
            else:
                self.clear()
            
        elif x.position == HandPosition.RIGHT_END3 and HandPosition.LEFT_START3 in self:
            start_index = self.index_position(HandPosition.LEFT_START3)
            if (
                self.swipe_distance(self._deque[start_index], x, min_absolute_distance=self.swipe_distance_threshold(x))
                and self.check_duration(start_index, min_frames=self.right_swipe_duration_threshold(x))
                and self.check_horizontal_swipe(self._deque[start_index], x)
            ):
                self.action = Event.SWIPE_RIGHT3
                self.clear()
                return True
            else:
                self.clear()

        elif x.position == HandPosition.UP_END3 and HandPosition.DOWN_START3 in self:
            start_index = self.index_position(HandPosition.DOWN_START3)
            if (
                self.check_duration(start_index, min_frames=15)
                and self.check_vertical_swipe(self._deque[start_index], x)
            ):
                self.action = Event.SWIPE_UP3
                self.clear()
                return True
            else:
                self.clear()

        elif x.position == HandPosition.DOWN_END3 and HandPosition.UP_START3 in self:
            start_index = self.index_position(HandPosition.UP_START3)
            if (
                self.check_duration(start_index, min_frames=15)
                and self.check_vertical_swipe(self._deque[start_index], x)
            ):
                self.action = Event.SWIPE_DOWN3
                self.clear()
                return True
            else:
                self.clear()

        elif HandPosition.DRAG_START in self and x.gesture == 25: # fist
            if self.action is None:
                start_index = self.index_gesture(17) # grabbing
                
                if self.check_duration(start_index, min_frames=8):
                    self.action = Event.DRAG
                    self.clear()
                    return True
                else:
                    self.clear()
        
        elif HandPosition.ZOOM_IN_START in self and x.gesture == 19: # point
            start_index = self.index_position(HandPosition.ZOOM_IN_START)
            if (
                self.check_duration(start_index, min_frames=8)
                and self.check_vertical_swipe(self._deque[start_index], x)
                and self.check_horizontal_swipe(self._deque[start_index], x)
            ):
                if not self.can_emit_tap_action():
                    self.clear()
                    return False
                self.action = Event.TAP
                self.mark_tap_action_emitted()
                self.clear()
                return True
            elif (
                self.check_duration(start_index, min_frames=2)
                and self.check_duration_max(start_index, max_frames=8)
                and self.check_vertical_swipe(self._deque[start_index], x)
                and self.check_horizontal_swipe(self._deque[start_index], x)
            ):
                self.action_deque.append(Event.TAP)
                if len(self.action_deque) >= 2 and self.action_deque[-1] == Event.TAP and self.action_deque[-2] == Event.TAP:
                    self.action_deque.pop()
                    self.action_deque.pop()
                    if not self.can_emit_tap_action():
                        self.clear()
                        return False
                    self.action = Event.DOUBLE_TAP
                    self.mark_tap_action_emitted()
                    self.clear()
                    return True
            else:
                self.clear()

        elif x.position == HandPosition.DOWN_END2 and HandPosition.ZOOM_OUT_START in self:
            start_index = self.index_position(HandPosition.ZOOM_OUT_START)
            if (
                self.swipe_distance(self._deque[start_index], x)
                and self.check_vertical_swipe(self._deque[start_index], x)
            ):
                self.action = Event.SWIPE_DOWN2
                self.clear()
                return True
            else:
                self.clear()
                
        elif x.position == HandPosition.ZOOM_OUT_START and HandPosition.UP_START2 in self:
            start_index = self.index_position(HandPosition.UP_START2)
            if (
                self.swipe_distance(self._deque[start_index], x)
                and self.check_vertical_swipe(self._deque[start_index], x)
            ):
                self.action = Event.SWIPE_UP2
                self.clear()
                return True
            else:
                self.clear()

        elif self.action == Event.DRAG and x.gesture in [35, 31, 36, 17]: # [stop, palm, stop_inverted, grabbing]
            self.action = Event.DROP
            self.clear()
            return True

        motion_action = self.detect_horizontal_motion_swipe(x)
        if motion_action is not None:
            self.action = motion_action
            self.clear()
            return True

        return False

    @staticmethod
    def check_horizontal_swipe(start_hand, x):
        """
        Check if swipe is horizontal.

        Parameters
        ----------
        start_hand : Hand
            Hand object of start position of swipe.

        x : Hand
            Hand object of end position of swipe.

        Returns
        -------
        bool
            True if swipe is horizontal.

        """
        boundary = [start_hand.bbox[1], start_hand.bbox[3]]
        if boundary[0] < x.center[1] < boundary[1]:
            return True
        else:
            return False

    @staticmethod
    def check_vertical_swipe(start_hand, x):
        """
        Check if swipe is vertical.

        Parameters
        ----------
        start_hand : Hand
            Hand object of start position of swipe.

        x : Hand
            Hand object of end position of swipe.

        Returns
        -------
        bool
            True if swipe is vertical.

        """
        boundary = [start_hand.bbox[0], start_hand.bbox[2]]
        if boundary[0] < x.center[0] < boundary[1]:
            return True
        else:
            return False

    def can_emit_tap_action(self):
        return self._frame_counter - self._last_tap_action_frame >= self.tap_action_min_gap_frames

    def mark_tap_action_emitted(self):
        self._last_tap_action_frame = self._frame_counter

    def __contains__(self, item):
        for x in self._deque:
            if x.position == item:
                return True

    def latest_position(self, *positions):
        for item in reversed(self._deque):
            if item.position in positions:
                return item.position
        return None

    def latest_hand_with_position(self, *positions):
        for item in reversed(self._deque):
            if item.position in positions:
                return item
        return None

    def latest_index_and_hand_with_position(self, *positions):
        for index in range(len(self._deque) - 1, -1, -1):
            item = self._deque[index]
            if item.position in positions:
                return index, item
        return None, None

    @staticmethod
    def is_unknown_gesture(gesture):
        return gesture is None or gesture == -1

    def smooth_three_gun_unknown_jitter(self, hand: Hand):
        if hand.bbox is None:
            return
        if not self.is_unknown_gesture(hand.gesture):
            return

        recent = list(self._gesture_raw_history)[-(self.three_gun_smooth_window - 1):]
        recent.append(hand.gesture)
        if len(recent) < self.three_gun_smooth_min_frames:
            return

        allowed = {self.three_gun_gesture, -1, None}
        if any(gesture not in allowed for gesture in recent):
            return

        three_gun_count = sum(1 for gesture in recent if gesture == self.three_gun_gesture)
        if three_gun_count > len(recent) / 2:
            hand.gesture = self.three_gun_gesture

    def swipe_distance_threshold(self, hand: Hand):
        if hand.gesture == self.three_gun_gesture:
            return self.three_gun_swipe_distance
        if hand.gesture in self.right_swipe_alias_gestures:
            return self.right_swipe_alias_distance
        return self.min_absolute_distance

    def swipe_duration_threshold(self, hand: Hand):
        if hand.gesture == self.three_gun_gesture:
            return self.three_gun_min_frames
        if hand.gesture in self.right_swipe_alias_gestures:
            return self.right_swipe_alias_min_frames
        return self.min_frames

    def right_swipe_duration_threshold(self, hand: Hand):
        if hand.gesture in self.right_swipe_end_alias_gestures:
            return self.right_swipe_alias_min_frames
        return self.swipe_duration_threshold(hand)

    def detect_horizontal_motion_swipe(self, hand: Hand):
        if hand.bbox is None or hand.center is None:
            return None
        if len(self) < self.motion_swipe_min_frames:
            return None

        current_index = len(self) - 1
        for start_index, start_hand in enumerate(self._deque[:-1]):
            if current_index - start_index + 1 < self.motion_swipe_min_frames:
                continue
            if start_hand.bbox is None or start_hand.center is None:
                continue

            dx = hand.center[0] - start_hand.center[0]
            dy = hand.center[1] - start_hand.center[1]
            hand_size = (start_hand.size + hand.size) / 2
            if hand_size <= 0:
                continue
            horizontal_distance = abs(dx) / hand_size
            if horizontal_distance < self.motion_swipe_distance:
                continue
            if abs(dx) < abs(dy) * self.motion_swipe_axis_ratio:
                continue
            return Event.SWIPE_RIGHT if dx > 0 else Event.SWIPE_LEFT
        return None

    def detect_dndv1_sequence(self, hand: Hand):
        if hand.gesture != 25:  # fist
            return False
        if len(self) < 3:
            return False

        current_index = len(self) - 1
        min_index = max(0, current_index - self.dndv1_max_frames + 1)
        grabbing_index = None
        for index in range(current_index - 1, min_index - 1, -1):
            if self._deque[index].gesture == 17:  # grabbing
                grabbing_index = index
                break

        if grabbing_index is None:
            return False
        if current_index - grabbing_index + 1 < self.dndv1_grabbing_min_frames:
            return False

        for index in range(grabbing_index - 1, min_index - 1, -1):
            if self._deque[index].gesture == 31:  # palm
                return True
        return False

    def right_swipe_alias_position(self, hand: Hand):
        if hand.gesture not in self.right_swipe_alias_gestures:
            return None
        if hand.bbox is None or hand.center is None:
            return None

        if hand.gesture in self.right_swipe_end_alias_gestures:
            starts = [
                (HandPosition.LEFT_START, HandPosition.RIGHT_END),
                (HandPosition.LEFT_START2, HandPosition.RIGHT_END2),
                (HandPosition.LEFT_START3, HandPosition.RIGHT_END3),
            ]
            for start_position, end_position in starts:
                start_index, start_hand = self.latest_index_and_hand_with_position(start_position)
                if start_hand is None or start_hand.bbox is None or start_hand.center is None:
                    continue
                if len(self) - start_index + 1 < self.right_swipe_alias_min_frames:
                    return HandPosition.UNKNOWN
                dx = hand.center[0] - start_hand.center[0]
                dy = hand.center[1] - start_hand.center[1]
                hand_size = (start_hand.size + hand.size) / 2
                if hand_size <= 0:
                    continue
                if dx / hand_size < self.right_swipe_alias_distance:
                    continue
                if dx < abs(dy) * self.right_swipe_alias_axis_ratio:
                    continue
                return end_position
            if self.latest_position(HandPosition.LEFT_START, HandPosition.LEFT_START2, HandPosition.LEFT_START3):
                return HandPosition.UNKNOWN

        if not self.latest_position(
            HandPosition.LEFT_START,
            HandPosition.LEFT_START2,
            HandPosition.LEFT_START3,
            HandPosition.RIGHT_START,
            HandPosition.RIGHT_START2,
            HandPosition.RIGHT_START3,
        ) and hand.gesture in self.right_swipe_start_alias_gestures:
            return HandPosition.LEFT_START
        return None

    def set_hand_position(self, hand: Hand):
        """
        Set hand position.

        Parameters
        ----------
        hand : Hand
            Hand object.
        """
        right_alias_position = self.right_swipe_alias_position(hand)
        if right_alias_position is not None:
            hand.position = right_alias_position
            return

        if hand.gesture in [31, 35, 36]: # [palm, stop, stop_inv]
            if HandPosition.DOWN_START in self:
                hand.position = HandPosition.UP_END
            else:
                hand.position = HandPosition.UP_START

        elif hand.gesture == 0: # hand_down
            if HandPosition.UP_START in self:
                hand.position = HandPosition.DOWN_END
            else:
                hand.position = HandPosition.DOWN_START

        elif hand.gesture == 1: # hand_right
            if HandPosition.LEFT_START in self:
                hand.position = HandPosition.RIGHT_END
            elif HandPosition.LEFT_START2 in self:
                hand.position = HandPosition.RIGHT_END2
            elif HandPosition.LEFT_START3 in self:
                hand.position = HandPosition.RIGHT_END3
            else:
                hand.position = HandPosition.RIGHT_START

        elif hand.gesture == 2: # hand_left
            if HandPosition.RIGHT_START in self:
                hand.position = HandPosition.LEFT_END
            elif HandPosition.RIGHT_START2 in self:
                hand.position = HandPosition.LEFT_END2
            elif HandPosition.RIGHT_START3 in self:
                hand.position = HandPosition.LEFT_END3
            else:
                hand.position = HandPosition.LEFT_START

        elif hand.gesture in self.fast_swipe_one_gestures:
            if HandPosition.FAST_SWIPE_UP_START in self:
                hand.position = HandPosition.FAST_SWIPE_UP_END
            else:
                hand.position = HandPosition.FAST_SWIPE_DOWN_START

        elif hand.gesture in self.fast_swipe_point_gestures:
            if HandPosition.FAST_SWIPE_DOWN_START in self:
                hand.position = HandPosition.FAST_SWIPE_DOWN_END
            else:
                hand.position = HandPosition.FAST_SWIPE_UP_START

        elif hand.gesture == self.three_gun_gesture:
            latest = self.latest_position(
                HandPosition.RIGHT_START,
                HandPosition.RIGHT_START2,
                HandPosition.RIGHT_START3,
                HandPosition.LEFT_START,
                HandPosition.LEFT_START2,
                HandPosition.LEFT_START3,
            )
            if latest == HandPosition.RIGHT_START:
                hand.position = HandPosition.LEFT_END
            elif latest == HandPosition.RIGHT_START2:
                hand.position = HandPosition.LEFT_END2
            elif latest == HandPosition.RIGHT_START3:
                hand.position = HandPosition.LEFT_END3
            elif latest == HandPosition.LEFT_START:
                hand.position = HandPosition.RIGHT_END
            elif latest == HandPosition.LEFT_START2:
                hand.position = HandPosition.RIGHT_END2
            elif latest == HandPosition.LEFT_START3:
                hand.position = HandPosition.RIGHT_END3
            else:
                hand.position = HandPosition.LEFT_START

        elif hand.gesture == 17: # grabbing
            hand.position = HandPosition.DRAG_START
        
        elif hand.gesture in [13, 25]: # fist / fist_inverted
            if HandPosition.ZOOM_OUT_START in self:
                hand.position = HandPosition.ZOOM_OUT_END
            else:
                hand.position = HandPosition.ZOOM_IN_START
        
        elif hand.gesture == 3: # thumb_index
            if HandPosition.ZOOM_IN_START in self:
                hand.position = HandPosition.ZOOM_IN_END
            else:
                hand.position = HandPosition.ZOOM_OUT_START

        elif hand.gesture == 38: # three2
            if HandPosition.ZOOM_IN_START in self:
                hand.position = HandPosition.ZOOM_IN_END
            else:
                hand.position = HandPosition.ZOOM_OUT_START

        elif hand.gesture == 5: # thumb_right
            if HandPosition.LEFT_START2 in self:
                hand.position = HandPosition.RIGHT_END2
            elif HandPosition.LEFT_START in self:
                hand.position = HandPosition.RIGHT_END
            elif HandPosition.LEFT_START3 in self:
                hand.position = HandPosition.RIGHT_END3
            else:
                hand.position = HandPosition.RIGHT_START2

        elif hand.gesture == 4: # thumb_left
            if HandPosition.RIGHT_START2 in self:
                hand.position = HandPosition.LEFT_END2
            elif HandPosition.RIGHT_START in self:
                hand.position = HandPosition.LEFT_END
            elif HandPosition.RIGHT_START3 in self:
                hand.position = HandPosition.LEFT_END3
            else:
                hand.position = HandPosition.LEFT_START2

        elif hand.gesture == 15: # two_right
            if HandPosition.LEFT_START3 in self:
                hand.position = HandPosition.RIGHT_END3
            elif HandPosition.LEFT_START in self:
                hand.position = HandPosition.RIGHT_END
            elif HandPosition.LEFT_START2 in self:
                hand.position = HandPosition.RIGHT_END2
            else:
                hand.position = HandPosition.RIGHT_START3

        elif hand.gesture == 14: # two_left
            if HandPosition.RIGHT_START3 in self:
                hand.position = HandPosition.LEFT_END3
            elif HandPosition.RIGHT_START in self:
                hand.position = HandPosition.LEFT_END
            elif HandPosition.RIGHT_START2 in self:
                hand.position = HandPosition.LEFT_END2
            else:
                hand.position = HandPosition.LEFT_START3
        
        elif hand.gesture == 39: # two_up
            if HandPosition.DOWN_START3 in self:
                hand.position = HandPosition.UP_END3
            else:
                hand.position = HandPosition.UP_START3

        elif hand.gesture == 16: # two_down
            if HandPosition.UP_START3 in self:
                hand.position = HandPosition.DOWN_END3
            else:
                hand.position = HandPosition.DOWN_START3

        elif hand.gesture == 6: # thumb_down
            if HandPosition.ZOOM_OUT_START in self:
                hand.position = HandPosition.DOWN_END2
            else:
                hand.position = HandPosition.UP_START2
        else:
            hand.position = HandPosition.UNKNOWN

    def swipe_distance(
        self,
        first_hand: Hand,
        last_hand: Hand,
        min_absolute_distance=None,
    ):
        """
        Check if swipe distance is more than min_distance.

        Parameters
        ----------
        first_hand : Hand
            Hand object of start position of swipe.

        last_hand : Hand
            Hand object of end position of swipe.

        Returns
        -------
        bool
            True if swipe distance is more than min_distance.

        """
        if min_absolute_distance is None:
            min_absolute_distance = self.min_absolute_distance
        hand_dist = distance.euclidean(first_hand.center, last_hand.center)
        hand_size = (first_hand.size + last_hand.size) / 2
        return hand_dist / hand_size > min_absolute_distance

    def clear(self):
        self._deque.clear()

    def copy(self):
        return self._deque.copy()

    def count(self, x):
        return self._deque.count(x)

    def extend(self, iterable):
        self._deque.extend(iterable)

    def insert(self, i, x):
        self._deque.insert(i, x)

    def pop(self):
        return self._deque.pop()

    def remove(self, value):
        self._deque.remove(value)

    def reverse(self):
        self._deque.reverse()

    def __str__(self):
        return f"Deque({[hand.gesture for hand in self._deque]})"

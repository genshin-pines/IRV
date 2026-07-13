from vendor.web_gesture_backend.dgcore.utils.action_controller import Deque
from vendor.web_gesture_backend.dgcore.utils.enums import Event, HandPosition
from vendor.web_gesture_backend.dgcore.utils.hand import Hand


def test_right_swipe_alias_accepts_moderate_distance_and_vertical_drift():
    controller = Deque()
    start = Hand([0, 0, 100, 100], hand_id=1, gesture=41)
    start.position = HandPosition.LEFT_START
    middle = Hand([12, 10, 112, 110], hand_id=1, gesture=41)
    middle.position = HandPosition.UNKNOWN
    controller._deque.extend([start, middle])

    end = Hand([30, 25, 130, 125], hand_id=1, gesture=1)
    assert controller.right_swipe_alias_position(end) == HandPosition.RIGHT_END


def test_right_swipe_alias_still_rejects_tiny_jitter():
    controller = Deque()
    start = Hand([0, 0, 100, 100], hand_id=1, gesture=41)
    start.position = HandPosition.LEFT_START
    middle = Hand([5, 2, 105, 102], hand_id=1, gesture=41)
    middle.position = HandPosition.UNKNOWN
    controller._deque.extend([start, middle])

    end = Hand([15, 5, 115, 105], hand_id=1, gesture=1)
    assert controller.right_swipe_alias_position(end) == HandPosition.UNKNOWN


def test_right_swipe_keeps_start_when_first_end_frame_has_vertical_drift():
    controller = Deque()
    start = Hand([0, 0, 100, 100], hand_id=1, gesture=41)
    start.position = HandPosition.LEFT_START
    middle = Hand([20, 20, 120, 120], hand_id=1, gesture=41)
    middle.position = HandPosition.UNKNOWN
    controller._deque.extend([start, middle])

    drifted_end = Hand([60, 60, 160, 160], hand_id=1, gesture=1)
    drifted_end.position = HandPosition.RIGHT_END
    controller._deque.append(drifted_end)
    assert controller.check_is_action(drifted_end) is False
    assert HandPosition.LEFT_START in controller

    recovered_end = Hand([90, 35, 190, 135], hand_id=1, gesture=1)
    recovered_end.position = HandPosition.RIGHT_END
    controller._deque.append(recovered_end)
    assert controller.check_is_action(recovered_end) is True
    assert controller.action == Event.SWIPE_RIGHT


def test_mirrored_horizontal_tracks_both_emit_swipes():
    right_controller = Deque()
    right_start = Hand([0, 0, 100, 100], hand_id=1, gesture=41)
    right_start.position = HandPosition.LEFT_START
    right_middle = Hand([45, 10, 145, 110], hand_id=1, gesture=41)
    right_middle.position = HandPosition.UNKNOWN
    right_end = Hand([90, 20, 190, 120], hand_id=1, gesture=1)
    right_end.position = HandPosition.RIGHT_END
    right_controller._deque.extend([right_start, right_middle, right_end])

    left_controller = Deque()
    left_start = Hand([90, 0, 190, 100], hand_id=1, gesture=1)
    left_start.position = HandPosition.RIGHT_START
    left_middle = Hand([45, 10, 145, 110], hand_id=1, gesture=41)
    left_middle.position = HandPosition.UNKNOWN
    left_end = Hand([0, 20, 100, 120], hand_id=1, gesture=14)
    left_end.position = HandPosition.LEFT_END
    left_controller._deque.extend([left_start, left_middle, left_end])

    assert right_controller.check_is_action(right_end) is True
    assert right_controller.action == Event.SWIPE_RIGHT
    assert left_controller.check_is_action(left_end) is True
    assert left_controller.action == Event.SWIPE_LEFT

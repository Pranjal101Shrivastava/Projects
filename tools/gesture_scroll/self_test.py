#!/usr/bin/env python3
"""Tests for the gesture logic, with synthetic landmarks instead of a camera.

The geometry in gesture_scroll.py is the part that decides whether the thing is
usable, and it is also the part that is easy to get backwards (image y grows
downward, and a fist looks a lot like a pinch). None of it needs OpenCV or
MediaPipe, so it can be checked on any machine:

    python3 self_test.py          # no pytest needed
    python3 -m pytest self_test.py -q
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gesture_scroll import (  # noqa: E402
    DOWN, DRAG, IDLE, UP,
    Calibrator, GestureFilter, Scroller,
    classify, fingers_up, hand_span, index_reach, is_pinch, pinch_ratio,
)


@dataclass
class P:
    """One landmark. MediaPipe returns objects with .x/.y/.z; only x and y are used."""
    x: float
    y: float
    z: float = 0.0


# A hand held upright in front of the camera. Wrist low in the frame, knuckles
# above it, so one hand span is 0.30 of the frame height.
WRIST_XY = (0.50, 0.90)
MCP_Y = 0.60
FINGER_X = {"index": 0.44, "middle": 0.50, "ring": 0.56, "pinky": 0.62}


def build_hand(*, up: tuple[str, ...] = (), thumb=(0.30, 0.68), index_tip=None):
    """Assemble the 21 landmarks for a given finger configuration.

    `up` names the extended fingers; the rest are curled. A curled finger is not
    simply a lowered one: the middle joint becomes the highest point of the
    finger and the tip comes back down against the palm, roughly halfway between
    the knuckles and the wrist. Getting that right matters, because the distance
    from wrist to fingertip is what separates a fist from a pinch.

    `index_tip` overrides the index fingertip, which is how the pinch poses are
    built.
    """
    lm = [P(0.0, 0.0) for _ in range(21)]
    lm[0] = P(*WRIST_XY)                       # wrist
    lm[1], lm[2], lm[3] = P(0.40, 0.82), P(0.35, 0.76), P(0.32, 0.72)
    lm[4] = P(*thumb)                          # thumb tip

    for name, (mcp, pip, dip, tip) in {
        "index": (5, 6, 7, 8),
        "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16),
        "pinky": (17, 18, 19, 20),
    }.items():
        x = FINGER_X[name]
        lm[mcp] = P(x, MCP_Y)                  # knuckles roughly in a line
        if name in up:
            lm[pip] = P(x, 0.50)
            lm[dip] = P(x, 0.40)
            lm[tip] = P(x, 0.32)
        else:
            # Curled, with the geometry measured off real photographs: the
            # middle joint becomes the highest point of the finger and the tip
            # comes back down to about the knuckle line -- roughly 0.95 hand
            # spans from the wrist, not into the middle of the palm. Getting
            # this wrong is what let a fist read as a pinch.
            curled_x = x + (0.50 - x) * 0.3    # tips converge on the palm
            lm[pip] = P(x, 0.52)               # middle joint: highest point
            lm[dip] = P(curled_x, 0.57)
            lm[tip] = P(curled_x, 0.615)
    if index_tip is not None:
        lm[8] = P(*index_tip)
    return lm


OPEN_PALM = build_hand(up=("index", "middle", "ring", "pinky"))
ONE_FINGER = build_hand(up=("index",))
TWO_FINGERS = build_hand(up=("index", "middle"))
# A fist: every finger curled and the thumb folded across them, which puts the
# thumb tip close to the index fingertip. This is the pose that a naive
# distance-only pinch test misreads.
FIST = build_hand(thumb=(0.518, 0.615))
# A real pinch: thumb and index tip meeting out in front of the palm.
# A real pinch: the index bent so its tip meets the thumb out in front of
# the palm, about 1.4 hand spans from the wrist -- between a fist (0.95) and a
# finger pointing straight out (1.9).
PINCH = build_hand(up=("middle", "ring", "pinky"),
                   index_tip=(0.44, 0.484), thumb=(0.455, 0.497))


def test_hand_span_is_wrist_to_knuckle():
    assert abs(hand_span(OPEN_PALM) - 0.30) < 1e-9


def test_fingers_up_reads_the_four_curling_fingers():
    assert fingers_up(OPEN_PALM) == (True, True, True, True)
    assert fingers_up(ONE_FINGER) == (True, False, False, False)
    assert fingers_up(TWO_FINGERS) == (True, True, False, False)
    assert fingers_up(FIST) == (False, False, False, False)


def test_margin_suppresses_a_borderline_finger():
    # Tip 0.01 above the joint: real enough to pass a bare comparison, not
    # enough to survive the dead band that stops per-frame flicker.
    lm = build_hand()
    lm[8] = P(FINGER_X["index"], lm[6].y - 0.01)
    assert fingers_up(lm, margin=0.02)[0] is False
    assert fingers_up(lm, margin=0.005)[0] is True


def test_pinch_is_distinguished_from_a_fist():
    # Both poses put the thumb tip right next to the index tip ...
    assert pinch_ratio(PINCH) < 0.22
    assert pinch_ratio(FIST) < 0.22
    # ... so distance alone cannot tell them apart. Reach is what does: the
    # pinch happens out in front of the palm, the fist against it.
    assert 1.3 < index_reach(PINCH) < 1.5
    assert 0.9 < index_reach(FIST) < 1.0
    assert is_pinch(PINCH) is True
    assert is_pinch(FIST) is False


def hand_from_ratios(pinch: float, reach: float):
    """A hand placed to hit exact pinch/reach values, for the measurements below.

    Only the four landmarks is_pinch reads are meaningful here, so this is not a
    usable hand for anything else -- fingers_up would be reading zeros.
    """
    lm = [P(0.0, 0.0) for _ in range(21)]
    span = 0.30
    lm[0] = P(0.50, 0.90)                              # wrist
    lm[9] = P(0.50, 0.60)                              # middle MCP: the span
    lm[8] = P(0.50, 0.90 - reach * span)               # index tip
    lm[4] = P(0.50 + pinch * span, lm[8].y)            # thumb tip
    return lm


# Measured by running the landmarker over MediaPipe's own published photographs
# of hands (see image_test.py), as (thumb-index distance, index reach) in hand
# spans. The fist is the whole reason is_pinch has two conditions: a 0.75 reach
# threshold passed every synthetic test and still called this fist a pinch.
MEASURED = {
    "fist": (0.20, 0.94),
    "thumb_up": (0.97, 0.80),
    "pointing_up": (1.06, 1.92),
    "victory": (1.05, 1.84),
}


def test_no_measured_real_hand_is_read_as_a_pinch():
    for name, (pinch, reach) in MEASURED.items():
        lm = hand_from_ratios(pinch, reach)
        assert abs(pinch_ratio(lm) - pinch) < 1e-9, name
        assert abs(index_reach(lm) - reach) < 1e-9, name
        assert is_pinch(lm) is False, f"{name} must not be a pinch"


def test_the_old_threshold_would_have_failed_on_a_real_fist():
    # Kept as a tripwire: this is the exact regression the photographs caught.
    fist = hand_from_ratios(*MEASURED["fist"])
    assert is_pinch(fist, reach_threshold=0.75) is True    # the bug
    assert is_pinch(fist, reach_threshold=1.35) is False   # the fix


def test_classify_maps_each_pose():
    assert classify(ONE_FINGER) == UP
    assert classify(TWO_FINGERS) == DOWN
    assert classify(PINCH) == DRAG
    assert classify(OPEN_PALM) == IDLE
    assert classify(FIST) == IDLE


def test_classify_without_drag_ignores_a_pinch():
    # The pinch pose has three fingers extended, so with drag off it must fall
    # through to IDLE rather than to a finger command.
    assert classify(PINCH, allow_drag=False) == IDLE


def test_thresholds_are_scale_invariant():
    # Same pose, hand twice as far from the camera: every ratio is unchanged, so
    # the gesture must not depend on where you sit.
    far = [P(0.5 + (p.x - 0.5) * 0.5, 0.9 + (p.y - 0.9) * 0.5) for p in ONE_FINGER]
    assert abs(hand_span(far) - 0.15) < 1e-9
    assert classify(far) == UP


def test_filter_requires_a_sustained_gesture():
    f = GestureFilter(hold=3)
    assert f.update(UP) == IDLE          # one frame is not a command
    assert f.update(UP) == IDLE
    assert f.update(UP) == UP            # three in a row is
    assert f.update(DOWN) == IDLE        # a changed frame breaks the run
    assert f.update(DOWN) == IDLE
    assert f.update(DOWN) == DOWN


def test_filter_rejects_a_gesture_flickering_through():
    f = GestureFilter(hold=5)
    for g in (IDLE, UP, UP, DOWN, IDLE, UP):   # a hand passing through frame
        assert f.update(g) == IDLE


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_step_scroll_is_rate_limited():
    clock, sent = FakeClock(), []
    s = Scroller(sent.append, step=2, interval=0.1, clock=clock)
    s.step_scroll(+1)
    s.step_scroll(+1)          # same instant: suppressed
    clock.t = 0.05
    s.step_scroll(+1)          # still inside the interval
    clock.t = 0.11
    s.step_scroll(-1)
    assert sent == [2, -2]


def test_drag_pulls_the_page_with_the_hand():
    sent = []
    s = Scroller(sent.append, drag_gain=12.0, clock=FakeClock())
    s.drag(0.5)                # hand moves down half a span -> reveal content above
    assert sent == [6]
    s.drag(-0.5)               # and back up
    assert sent == [6, -6]


def test_drag_accumulates_sub_click_movement():
    sent = []
    s = Scroller(sent.append, drag_gain=10.0, clock=FakeClock())
    for _ in range(4):
        s.drag(0.03)           # 0.3 of a click per frame: never rounds to zero
    assert sent == [1]
    for _ in range(4):
        s.drag(0.03)
    assert sent == [1, 1]


def test_reset_drops_the_pending_remainder():
    sent = []
    s = Scroller(sent.append, drag_gain=10.0, clock=FakeClock())
    s.drag(0.09)               # 0.9 of a click, held back
    s.reset()
    s.drag(0.09)               # must not combine with the pre-reset remainder
    assert sent == []


def test_invert_drag_flips_the_direction():
    sent = []
    s = Scroller(sent.append, drag_gain=12.0, invert_drag=True, clock=FakeClock())
    s.drag(0.5)
    assert sent == [-6]


def test_calibrator_suggests_a_threshold_between_the_poses():
    c = Calibrator()
    c.record("pinch", hand_from_ratios(0.08, 1.40))
    c.record("fist", hand_from_ratios(0.20, 0.96))
    out = c.report()
    assert "--reach-threshold 1.18" in out, out     # midway between 0.96 and 1.40
    assert "--pinch-threshold 0.11" in out, out     # just clear of the pinch samples


def test_calibrator_refuses_to_invent_a_threshold_when_poses_overlap():
    c = Calibrator()
    c.record("pinch", hand_from_ratios(0.08, 0.90))
    c.record("fist", hand_from_ratios(0.20, 1.00))
    out = c.report()
    assert "overlap" in out
    assert "--reach-threshold" not in out


def test_calibrator_needs_both_poses():
    c = Calibrator()
    c.record("pinch", hand_from_ratios(0.08, 1.40))
    assert "Not enough samples" in c.report()


def _run() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc or 'assertion failed'}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())

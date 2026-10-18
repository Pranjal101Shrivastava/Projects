#!/usr/bin/env python3
"""Tests for the gesture logic, with synthetic landmarks instead of a camera.

The geometry in gesture_scroll.py is the part that decides whether the thing is
usable, and it is also the part that is easy to get backwards (image y grows
downward, and a fist looks a lot like a pinch). None of it needs OpenCV or
MediaPipe, so it can be checked on any machine:

    python3 self_test.py          # no pytest needed
    python3 -m pytest self_test.py -q

Two kinds of test live here. The named cases pin specific behaviour. The sweeps
check properties across a range of inputs -- scale, position, rotation, jitter --
because "works on the one hand I imagined" is exactly the assumption that let a
real fist read as a pinch. image_test.py then checks the same logic against
photographs, which is where the assumptions get audited for real.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gesture_scroll import (  # noqa: E402
    DOWN, DRAG, IDLE, UP,
    MARGIN_SPANS, PINCH_THRESHOLD, REACH_THRESHOLD,
    Calibrator, GestureFilter, Scroller,
    classify, fingers_up, hand_span, index_reach, is_pinch, pinch_ratio,
)

# ---------------------------------------------------------------------------
# Harness. check() counts every assertion so the suite can report how much it
# actually verified, rather than how many functions it happens to be split into.
# ---------------------------------------------------------------------------
CHECKS = 0


def check(condition, message: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message or "check failed")


def equal(got, want, message: str = "") -> None:
    check(got == want, f"{message}: got {got!r}, want {want!r}" if message
          else f"got {got!r}, want {want!r}")


# ---------------------------------------------------------------------------
# Synthetic hands
# ---------------------------------------------------------------------------
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
    finger and the tip comes back down against the palm, roughly at the knuckle
    line. Getting that right matters, because the distance from wrist to
    fingertip is what separates a fist from a pinch.

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
            # Curled, with the geometry measured off real photographs: tip back
            # down at about 0.95 hand spans from the wrist, not into the middle
            # of the palm. Getting this wrong is what let a fist read as a pinch.
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
# thumb tip right beside the index fingertip. This is the pose that a naive
# distance-only pinch test misreads.
FIST = build_hand(thumb=(0.518, 0.615))
# A real pinch: the index bent so its tip meets the thumb out in front of the
# palm, about 1.4 hand spans from the wrist -- between a fist (0.95) and a
# finger pointing straight out (1.9).
PINCH = build_hand(up=("middle", "ring", "pinky"),
                   index_tip=(0.44, 0.484), thumb=(0.455, 0.497))

POSES = {UP: ONE_FINGER, DOWN: TWO_FINGERS, DRAG: PINCH}
ALL_POSES = {"one finger": (ONE_FINGER, UP), "two fingers": (TWO_FINGERS, DOWN),
             "pinch": (PINCH, DRAG), "fist": (FIST, IDLE), "open palm": (OPEN_PALM, IDLE)}


def transform(lm, scale: float = 1.0, dx: float = 0.0, dy: float = 0.0,
              deg: float = 0.0, about=WRIST_XY):
    """Move, resize or rotate a hand in normalised coordinates, about the wrist."""
    r = math.radians(deg)
    ca, sa = math.cos(r), math.sin(r)
    ox, oy = about
    out = []
    for p in lm:
        x, y = (p.x - ox) * scale, (p.y - oy) * scale
        out.append(P(ox + x * ca - y * sa + dx, oy + x * sa + y * ca + dy))
    return out


def jitter(lm, amplitude_spans: float, seed: int):
    """Add bounded random noise to every landmark, in hand spans."""
    rng = random.Random(seed)
    amp = amplitude_spans * hand_span(lm)
    return [P(p.x + rng.uniform(-amp, amp), p.y + rng.uniform(-amp, amp)) for p in lm]


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


# ---------------------------------------------------------------------------
# Named cases
# ---------------------------------------------------------------------------
def test_hand_span_is_wrist_to_knuckle():
    check(abs(hand_span(OPEN_PALM) - 0.30) < 1e-9)


def test_fingers_up_reads_the_four_curling_fingers():
    equal(fingers_up(OPEN_PALM), (True, True, True, True), "open palm")
    equal(fingers_up(ONE_FINGER), (True, False, False, False), "one finger")
    equal(fingers_up(TWO_FINGERS), (True, True, False, False), "two fingers")
    equal(fingers_up(FIST), (False, False, False, False), "fist")


def test_margin_suppresses_a_borderline_finger():
    # Tip barely above the joint: real enough to pass a bare comparison, not
    # enough to survive the dead band that stops per-frame flicker.
    lm = build_hand()
    lm[8] = P(FINGER_X["index"], lm[6].y - 0.01)
    check(fingers_up(lm, margin=0.07)[0] is False, "0.01 should be inside the dead band")
    check(fingers_up(lm, margin=0.01)[0] is True, "a smaller dead band should let it through")


def test_pinch_is_distinguished_from_a_fist():
    # Both poses put the thumb tip right next to the index tip ...
    check(pinch_ratio(PINCH) < PINCH_THRESHOLD, "pinch is close")
    check(pinch_ratio(FIST) < PINCH_THRESHOLD, "so is a fist")
    # ... so distance alone cannot tell them apart. Reach is what does: the
    # pinch happens out in front of the palm, the fist against it.
    check(1.3 < index_reach(PINCH) < 1.5, "pinch reaches forward")
    check(0.9 < index_reach(FIST) < 1.0, "a fist does not")
    check(is_pinch(PINCH) is True)
    check(is_pinch(FIST) is False)


def test_classify_maps_each_pose():
    for name, (pose, want) in ALL_POSES.items():
        equal(classify(pose), want, name)


def test_classify_without_drag_ignores_a_pinch():
    # The pinch pose has three fingers extended, so with drag off it must fall
    # through to IDLE rather than to a finger command.
    equal(classify(PINCH, allow_drag=False), IDLE)


def test_no_measured_real_hand_is_read_as_a_pinch():
    for name, (pinch, reach) in MEASURED.items():
        lm = hand_from_ratios(pinch, reach)
        check(abs(pinch_ratio(lm) - pinch) < 1e-9, name)
        check(abs(index_reach(lm) - reach) < 1e-9, name)
        check(is_pinch(lm) is False, f"{name} must not be a pinch")


def test_the_old_threshold_would_have_failed_on_a_real_fist():
    # Kept as a tripwire: this is the exact regression the photographs caught.
    fist = hand_from_ratios(*MEASURED["fist"])
    check(is_pinch(fist, reach_threshold=0.75) is True, "the bug")
    check(is_pinch(fist, reach_threshold=REACH_THRESHOLD) is False, "the fix")


# ---------------------------------------------------------------------------
# Sweeps: properties that must hold across a range, not at one point
# ---------------------------------------------------------------------------
def test_scale_invariance_across_apparent_hand_size():
    """A hand further from the camera fills less of the frame. Nothing may change.

    This is what caught the finger margin being an absolute distance while every
    other threshold was in hand spans: below about a tenth of the original size,
    no finger could clear the fixed dead band and every gesture went idle.
    """
    for name, (pose, want) in ALL_POSES.items():
        for factor in (0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7,
                       1.0, 1.3, 1.6, 2.0, 2.5, 3.0):
            equal(classify(transform(pose, scale=factor)), want, f"{name} at x{factor}")


def test_position_invariance_across_the_frame():
    """Where the hand sits in frame is not information. Only its shape is."""
    for name, (pose, want) in ALL_POSES.items():
        for dx, dy in product((-0.3, -0.15, 0.0, 0.15), (-0.4, -0.2, 0.0)):
            equal(classify(transform(pose, dx=dx, dy=dy)), want,
                  f"{name} at ({dx}, {dy})")


def test_rotation_tolerance():
    """Gestures survive a tilted hand, which is how people actually hold them."""
    for name, (pose, want) in ALL_POSES.items():
        for deg in range(-80, 81, 10):
            equal(classify(transform(pose, deg=deg)), want, f"{name} at {deg} deg")


def test_rotation_limit_is_where_the_finger_stops_being_up():
    """The documented edge of that envelope, asserted rather than left implicit.

    "Up" means the fingertip is above its middle joint, so a hand rotated to
    horizontal has no fingers up by definition. image_test.py pins the same
    boundary on a real photograph of a hand pointing sideways.
    """
    equal(classify(transform(ONE_FINGER, deg=90)), IDLE, "sideways is not a command")
    equal(classify(transform(TWO_FINGERS, deg=90)), IDLE, "nor is sideways two fingers")
    # A pinch is measured by distances alone, so it has no orientation at all.
    for deg in (-180, -90, 0, 90, 180):
        equal(classify(transform(PINCH, deg=deg)), DRAG, f"pinch at {deg} deg")


def test_landmark_jitter_does_not_change_the_gesture():
    """Model output wobbles frame to frame. Small wobble must not flip a gesture."""
    for name, (pose, want) in ALL_POSES.items():
        for seed in range(30):
            equal(classify(jitter(pose, 0.015, seed)), want, f"{name} seed {seed}")


def test_pinch_is_the_gesture_most_sensitive_to_jitter():
    """Documents why the hold filter matters most for drag.

    A pinch is defined by a distance of a fifth of a hand span, so noise of a
    comparable size can break it. The finger gestures compare joints much
    further apart and barely move.
    """
    def failures(pose, want, amplitude):
        return sum(classify(jitter(pose, amplitude, seed)) != want
                   for seed in range(40))

    calm = failures(PINCH, DRAG, 0.015)
    noisy = failures(PINCH, DRAG, 0.12)
    check(calm == 0, f"pinch should be stable when calm, got {calm} failures")
    check(noisy > calm, "pinch should degrade as jitter grows")
    for name, pose, want in (("one finger", ONE_FINGER, UP), ("two", TWO_FINGERS, DOWN)):
        equal(failures(pose, want, 0.12), 0, f"{name} should shrug off the same noise")


def test_pinch_decision_over_a_grid_never_fires_in_fist_territory():
    """The guarantee that matters: nothing with a curled finger is ever a drag.

    A real fist measures 0.94 hand spans of reach, so anything at or below 1.0
    is a hand that is not reaching forward, whatever its thumb is doing.
    """
    for i in range(15):
        for j in range(15):
            pinch = 0.02 + i * 0.06          # 0.02 .. 0.86
            reach = 0.40 + j * 0.13          # 0.40 .. 2.22
            lm = hand_from_ratios(pinch, reach)
            verdict = is_pinch(lm)
            if verdict:
                check(pinch < PINCH_THRESHOLD and reach > REACH_THRESHOLD,
                      f"pinch fired outside its own thresholds at {pinch:.2f}/{reach:.2f}")
            check(not (reach <= 1.0 and verdict),
                  f"fist territory fired a drag at {pinch:.2f}/{reach:.2f}")


def test_filter_over_every_short_sequence():
    """Exhaustive: every sequence of three gestures over five frames.

    The filter's whole contract is "report a gesture only when the last N frames
    agree", and there are few enough sequences to check all of them.
    """
    alphabet = (UP, DOWN, IDLE)
    for sequence in product(alphabet, repeat=5):
        f = GestureFilter(hold=5)
        result = [f.update(g) for g in sequence]
        # Only the final frame can have a full window behind it.
        want_last = sequence[0] if len(set(sequence)) == 1 else IDLE
        equal(result[-1], want_last, f"sequence {''.join(s[0] for s in sequence)}")


def test_filter_hold_length_is_respected():
    for hold in (1, 2, 3, 5, 8):
        f = GestureFilter(hold=hold)
        for frame in range(hold - 1):
            equal(f.update(UP), IDLE, f"hold={hold} frame {frame} too early")
        equal(f.update(UP), UP, f"hold={hold} should fire on frame {hold}")


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
    equal(sent, [2, -2])


def test_rate_limit_holds_across_settings():
    """However fast frames arrive, emissions stay at most one per interval."""
    for interval in (0.02, 0.05, 0.1, 0.25):
        for frame_dt in (0.004, 0.01, 0.033, 0.05):
            clock, sent = FakeClock(), []
            s = Scroller(sent.append, step=1, interval=interval, clock=clock)
            for _ in range(200):
                s.step_scroll(+1)
                clock.t += frame_dt
            elapsed = 200 * frame_dt
            check(len(sent) <= math.ceil(elapsed / interval) + 1,
                  f"interval={interval} dt={frame_dt}: {len(sent)} emissions "
                  f"in {elapsed:.2f}s")
            check(len(sent) >= 1, "something should have been emitted")


def test_drag_pulls_the_page_with_the_hand():
    sent = []
    s = Scroller(sent.append, drag_gain=12.0, clock=FakeClock())
    s.drag(0.5)                # hand moves down half a span -> reveal content above
    equal(sent, [6])
    s.drag(-0.5)               # and back up
    equal(sent, [6, -6])


def test_drag_accumulates_sub_click_movement():
    sent = []
    s = Scroller(sent.append, drag_gain=10.0, clock=FakeClock())
    for _ in range(4):
        s.drag(0.03)           # 0.3 of a click per frame: never rounds to zero
    equal(sent, [1])
    for _ in range(4):
        s.drag(0.03)
    equal(sent, [1, 1])


def test_drag_conserves_total_movement():
    """Slow movement must not be lost to rounding, over any frame breakdown.

    A hand moving one span should scroll the same amount whether the loop saw it
    in three frames or ninety.
    """
    for seed in range(40):
        rng = random.Random(seed)
        gain = rng.choice((4.0, 8.0, 12.0, 25.0))
        total = rng.uniform(-3.0, 3.0)
        frames = rng.randint(3, 90)
        sent = []
        s = Scroller(sent.append, drag_gain=gain, clock=FakeClock())
        for _ in range(frames):
            s.drag(total / frames)
        expected = gain * total
        check(abs(sum(sent) - expected) < 1.0,
              f"seed {seed}: {sum(sent)} clicks vs {expected:.2f} expected")


def test_drag_is_symmetric():
    for gain in (4.0, 12.0, 30.0):
        for movement in (0.1, 0.33, 0.5, 1.0, 2.0):
            up, down = [], []
            Scroller(up.append, drag_gain=gain, clock=FakeClock()).drag(movement)
            Scroller(down.append, drag_gain=gain, clock=FakeClock()).drag(-movement)
            equal(sum(up), -sum(down), f"gain={gain} movement={movement}")


def test_invert_drag_flips_the_direction():
    sent = []
    s = Scroller(sent.append, drag_gain=12.0, invert_drag=True, clock=FakeClock())
    s.drag(0.5)
    equal(sent, [-6])


def test_reset_drops_the_pending_remainder():
    sent = []
    s = Scroller(sent.append, drag_gain=10.0, clock=FakeClock())
    s.drag(0.09)               # 0.9 of a click, held back
    s.reset()
    s.drag(0.09)               # must not combine with the pre-reset remainder
    equal(sent, [])


def test_calibrator_suggests_a_threshold_between_the_poses():
    c = Calibrator()
    c.record("pinch", hand_from_ratios(0.08, 1.40))
    c.record("fist", hand_from_ratios(0.20, 0.96))
    out = c.report()
    check("--reach-threshold 1.18" in out, out)     # midway between 0.96 and 1.40
    check("--pinch-threshold 0.11" in out, out)     # just clear of the pinch samples


def test_calibrator_refuses_to_invent_a_threshold_when_poses_overlap():
    c = Calibrator()
    c.record("pinch", hand_from_ratios(0.08, 0.90))
    c.record("fist", hand_from_ratios(0.20, 1.00))
    out = c.report()
    check("overlap" in out)
    check("--reach-threshold" not in out)


def test_calibrator_needs_both_poses():
    c = Calibrator()
    c.record("pinch", hand_from_ratios(0.08, 1.40))
    check("Not enough samples" in c.report())


def test_calibration_suggestions_actually_separate_the_poses():
    """Whatever it suggests must classify the samples it was given correctly."""
    for seed in range(20):
        rng = random.Random(seed)
        fist_reach = rng.uniform(0.80, 1.05)
        pinch_reach = rng.uniform(1.25, 1.70)
        c = Calibrator()
        for _ in range(4):
            c.record("pinch", hand_from_ratios(rng.uniform(0.04, 0.14), pinch_reach))
            c.record("fist", hand_from_ratios(rng.uniform(0.15, 0.21), fist_reach))
        line = [l for l in c.report().splitlines() if "--reach-threshold" in l]
        check(len(line) == 1, f"seed {seed}: expected one suggestion")
        suggested = float(line[0].split("--reach-threshold")[1].split()[0])
        check(fist_reach < suggested < pinch_reach,
              f"seed {seed}: {suggested} does not separate {fist_reach} and {pinch_reach}")


def _run() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        before = CHECKS
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
        else:
            print(f"ok   {name:58} {CHECKS - before:5} checks")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed, {CHECKS} checks total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())

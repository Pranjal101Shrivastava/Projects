#!/usr/bin/env python3
"""Scroll whatever is under the mouse pointer with hand gestures seen through a webcam.

The point of this is hands-free reading: park the mouse pointer over a PDF or a
web page, hold up one finger to go up, two to go down, or pinch and move your
hand to drag the page the way you would on a touchscreen.

How it works, in one loop that runs ~30 times a second:

  1. OpenCV pulls a frame from the webcam.
  2. MediaPipe Hands returns 21 landmarks for the hand in view -- the wrist plus
     four points along each finger, each with an x/y position normalised to the
     image (0..1) and a rough depth z, which this script ignores.
  3. `classify` turns those landmarks into one of four gestures.
  4. A short hold filter rejects anything that does not persist for several
     frames, which is what stops a reach for your coffee from scrolling.
  5. pyautogui emits a real scroll-wheel event, so it works in any application.

Everything above step 4 is pure geometry on the landmark list, so it is unit
tested in self_test.py without a camera.

Usage:
    python3 gesture_scroll.py              # discrete + drag gestures, with preview
    python3 gesture_scroll.py --dry-run    # print what it would do, scroll nothing
    python3 gesture_scroll.py --no-preview # no window; quit with Ctrl-C

Only the standard library is imported at module scope; cv2, mediapipe and
pyautogui are imported inside main() so the gesture logic stays importable (and
testable) on a machine that does not have them.
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from collections import deque
from typing import Callable, Sequence

# ---------------------------------------------------------------------------
# MediaPipe hand landmark indices. The model always returns all 21 in this
# order, so naming the ones we use beats indexing by magic number.
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20

# (tip, pip) for the four fingers that curl in the image plane. The thumb moves
# sideways rather than up and down, so it needs different treatment and is only
# used here as one half of the pinch measurement.
FINGER_JOINTS = (
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
)

# Gestures this script recognises.
UP = "up"
DOWN = "down"
DRAG = "drag"
IDLE = "idle"

# pyautogui's scroll unit is not the same size on every platform, so the number
# of clicks that feels like a comfortable page nudge differs. These are starting
# points; --step overrides them.
_DEFAULT_STEP = {"Darwin": 3, "Windows": 1, "Linux": 2}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _dist(lm: Sequence, a: int, b: int) -> float:
    """Distance between two landmarks in normalised image units."""
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)


def hand_span(lm: Sequence) -> float:
    """Wrist-to-middle-knuckle distance: a proxy for how big the hand looks.

    Every other measurement is divided by this, which is what makes the
    thresholds below independent of how far you are sitting from the camera.
    """
    return max(_dist(lm, WRIST, MIDDLE_MCP), 1e-6)


def fingers_up(lm: Sequence, margin: float = 0.02) -> tuple[bool, bool, bool, bool]:
    """Which of index/middle/ring/pinky are extended.

    A finger counts as extended when its tip sits higher in the image than its
    middle joint. Image y grows downward, so "higher" means a smaller y. The
    margin is a dead band: without it a finger resting exactly at the boundary
    flickers between states from one frame to the next.
    """
    return tuple(lm[tip].y < lm[pip].y - margin for tip, pip in FINGER_JOINTS)


def pinch_ratio(lm: Sequence) -> float:
    """Thumb-tip to index-tip distance, in hand spans. Small means pinched."""
    return _dist(lm, THUMB_TIP, INDEX_TIP) / hand_span(lm)


def index_reach(lm: Sequence) -> float:
    """Wrist to index-tip distance, in hand spans. Small means the finger is curled in."""
    return _dist(lm, WRIST, INDEX_TIP) / hand_span(lm)


def is_pinch(lm: Sequence, pinch_threshold: float, reach_threshold: float) -> bool:
    """True for a deliberate thumb-and-index pinch.

    The distance test alone is not enough: in a closed fist the thumb also ends
    up near the index fingertip, which would read as a pinch and make the page
    lurch every time you relaxed your hand. A real pinch happens out in front of
    the palm, so we additionally require the index tip to be reaching away from
    the wrist, which a fist never is.
    """
    return pinch_ratio(lm) < pinch_threshold and index_reach(lm) > reach_threshold


def classify(
    lm: Sequence,
    pinch_threshold: float = 0.22,
    reach_threshold: float = 0.75,
    margin: float = 0.02,
    allow_drag: bool = True,
) -> str:
    """Map 21 landmarks onto one gesture.

    Pinch is tested first because a pinching hand also has a partly extended
    index finger and would otherwise be read as the one-finger gesture.
    """
    if allow_drag and is_pinch(lm, pinch_threshold, reach_threshold):
        return DRAG
    index, middle, ring, pinky = fingers_up(lm, margin)
    if ring or pinky:
        # An open palm or a three-finger shape is deliberately not a command:
        # it is the gesture you make when waving or picking something up.
        return IDLE
    if index and not middle:
        return UP
    if index and middle:
        return DOWN
    return IDLE


# ---------------------------------------------------------------------------
# Hold filter
# ---------------------------------------------------------------------------
class GestureFilter:
    """Only reports a gesture once it has held for `hold` consecutive frames.

    This is the single most important piece of the script for everyday use.
    Without it, the frames your hand passes through on the way to a mug of
    coffee each fire a scroll event.
    """

    def __init__(self, hold: int = 5) -> None:
        self.hold = max(1, hold)
        self._recent: deque[str] = deque(maxlen=self.hold)

    def update(self, gesture: str) -> str:
        self._recent.append(gesture)
        if len(self._recent) < self.hold:
            return IDLE
        first = self._recent[0]
        return first if all(g == first for g in self._recent) else IDLE


# ---------------------------------------------------------------------------
# Scrolling
# ---------------------------------------------------------------------------
class Scroller:
    """Turns gestures into scroll-wheel clicks.

    `scroll_fn` takes a signed integer click count, positive for up, matching
    pyautogui.scroll. Injecting it keeps this class testable and gives --dry-run
    somewhere to hook in.
    """

    def __init__(
        self,
        scroll_fn: Callable[[int], None],
        step: int = 2,
        interval: float = 0.06,
        drag_gain: float = 12.0,
        invert_drag: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.scroll_fn = scroll_fn
        self.step = step
        self.interval = interval
        self.drag_gain = drag_gain
        self.invert_drag = invert_drag
        self.clock = clock
        self._last_emit = float("-inf")
        # Drag movement is fractional: a frame may be worth 0.4 of a click.
        # Accumulating the remainder instead of rounding it away is what keeps
        # slow hand movement from being swallowed entirely.
        self._accum = 0.0

    def step_scroll(self, direction: int) -> None:
        """Fixed-size scroll while a discrete gesture is held, rate limited."""
        now = self.clock()
        if now - self._last_emit < self.interval:
            return
        self._last_emit = now
        self.scroll_fn(direction * self.step)

    def drag(self, dy_spans: float) -> None:
        """Proportional scroll from hand movement, measured in hand spans.

        Moving the hand down (dy > 0) pulls the page down with it, which reveals
        the content above -- a scroll up, hence the positive sign.
        """
        sign = -1.0 if self.invert_drag else 1.0
        self._accum += sign * self.drag_gain * dy_spans
        clicks = int(self._accum)
        if clicks:
            self._accum -= clicks
            self.scroll_fn(clicks)

    def reset(self) -> None:
        self._accum = 0.0


# ---------------------------------------------------------------------------
# Camera loop
# ---------------------------------------------------------------------------
def _camera_backend(cv2_module) -> int:
    """DirectShow on Windows, where the default backend can take ~10s to open."""
    if platform.system() == "Windows":
        return cv2_module.CAP_DSHOW
    return cv2_module.CAP_ANY


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_step = _DEFAULT_STEP.get(platform.system(), 2)
    p = argparse.ArgumentParser(
        description="Scroll whatever is under the mouse pointer with webcam hand gestures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--step", type=int, default=default_step,
                   help="scroll clicks per emission for the finger gestures")
    p.add_argument("--interval", type=float, default=0.06,
                   help="minimum seconds between finger-gesture scrolls")
    p.add_argument("--drag-gain", type=float, default=12.0,
                   help="scroll clicks per hand-span of pinch-drag movement")
    p.add_argument("--invert-drag", action="store_true",
                   help="move hand down to scroll down instead of dragging the page")
    p.add_argument("--hold", type=int, default=5,
                   help="frames a gesture must persist before it counts")
    p.add_argument("--pinch-threshold", type=float, default=0.22,
                   help="thumb-index distance, in hand spans, that counts as a pinch")
    p.add_argument("--no-drag", action="store_true", help="disable the pinch-drag gesture")
    p.add_argument("--no-preview", action="store_true",
                   help="do not open the camera window (quit with Ctrl-C)")
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the preview")
    p.add_argument("--dry-run", action="store_true",
                   help="print scroll events instead of sending them")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - depends on the local machine
        print(f"Missing dependency: {exc.name}\n"
              f"Install with: pip install -r requirements.txt", file=sys.stderr)
        return 1

    if args.dry_run:
        def scroll_fn(clicks: int) -> None:
            print(f"scroll {clicks:+d}")
    else:
        try:
            import pyautogui
        except ImportError as exc:  # pragma: no cover
            print(f"Missing dependency: {exc.name}", file=sys.stderr)
            return 1
        scroll_fn = pyautogui.scroll

    cap = cv2.VideoCapture(args.camera, _camera_backend(cv2))
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}. On macOS, grant Camera access to\n"
              f"your terminal in System Settings > Privacy & Security > Camera.",
              file=sys.stderr)
        return 1
    # 640x480 is plenty for hand landmarks and keeps the per-frame cost low.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_hands = mp.solutions.hands
    drawing = mp.solutions.drawing_utils

    gestures = GestureFilter(hold=args.hold)
    scroller = Scroller(
        scroll_fn,
        step=args.step,
        interval=args.interval,
        drag_gain=args.drag_gain,
        invert_drag=args.invert_drag,
    )

    prev_palm_y: float | None = None
    fps = 0.0
    last_t = time.monotonic()

    print("Running. Keep the mouse pointer over the document you want to scroll.")
    print("  one finger = up    two fingers = down    pinch and move = drag")
    print("  q or Esc in the preview window, or Ctrl-C here, to quit.")

    try:
        with mp_hands.Hands(
            max_num_hands=1,
            # The lite landmark model. On a laptop CPU it is the difference
            # between comfortably real time and visibly lagging.
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera read failed.", file=sys.stderr)
                    return 1
                if not args.no_mirror:
                    # Mirror so the preview behaves like a mirror. Landmarks are
                    # taken from the flipped frame so they still line up, and
                    # flipping horizontally leaves up/down untouched.
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = hands.process(rgb)

                hand = None
                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]

                raw = IDLE
                if hand is not None:
                    raw = classify(
                        hand.landmark,
                        pinch_threshold=args.pinch_threshold,
                        allow_drag=not args.no_drag,
                    )
                gesture = gestures.update(raw)

                if gesture == DRAG and hand is not None:
                    palm_y = hand.landmark[MIDDLE_MCP].y
                    span = hand_span(hand.landmark)
                    if prev_palm_y is not None:
                        scroller.drag((palm_y - prev_palm_y) / span)
                    prev_palm_y = palm_y
                else:
                    # Leaving drag resets the anchor and the fractional
                    # remainder, so the next pinch starts from a clean slate.
                    prev_palm_y = None
                    scroller.reset()
                    if gesture == UP:
                        scroller.step_scroll(+1)
                    elif gesture == DOWN:
                        scroller.step_scroll(-1)

                now = time.monotonic()
                dt = now - last_t
                last_t = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

                if not args.no_preview:
                    if hand is not None:
                        drawing.draw_landmarks(
                            frame, hand, mp_hands.HAND_CONNECTIONS)
                    colour = (0, 220, 0) if gesture != IDLE else (160, 160, 160)
                    cv2.putText(frame, gesture.upper(), (12, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
                    cv2.putText(frame, f"{fps:4.1f} fps", (12, 466),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    cv2.imshow("gesture scroll", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

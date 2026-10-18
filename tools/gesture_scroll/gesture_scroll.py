#!/usr/bin/env python3
"""Scroll whatever is under the mouse pointer with hand gestures seen through a webcam.

The point of this is hands-free reading: park the mouse pointer over a PDF or a
web page, hold up one finger to go up, two to go down, or pinch and move your
hand to drag the page the way you would on a touchscreen.

How it works, in one loop that runs ~30 times a second:

  1. OpenCV pulls a frame from the webcam.
  2. MediaPipe returns 21 landmarks for the hand in view -- the wrist plus four
     points along each finger, each with an x/y position normalised to the image
     (0..1) and a rough depth z, which this script ignores.
  3. `classify` turns those landmarks into one of four gestures.
  4. A short hold filter rejects anything that does not persist for several
     frames, which is what stops a reach for your coffee from scrolling.
  5. pyautogui emits a real scroll-wheel event, so it works in any application.

MediaPipe has two generations of Python API and this script supports both,
picking whichever is installed:

  * The Tasks API (`mediapipe.tasks`), which is what you get on MediaPipe 1.x.
    It needs a model file, downloaded once to a cache directory on first run.
  * The older `mediapipe.solutions.hands`, which bundles its model. It was
    removed in MediaPipe 1.0, so this path only runs on 0.10.x.

Everything up to step 4 is pure geometry on the landmark list, so it is unit
tested in self_test.py without a camera, and checked against real photographs
of hands in image_test.py.

Usage:
    python3 check_setup.py                 # check this machine first
    python3 gesture_scroll.py --dry-run    # print what it would do, scroll nothing
    python3 gesture_scroll.py              # for real
    python3 gesture_scroll.py --calibrate  # measure your own hand

Only the standard library is imported at module scope; cv2, mediapipe and
pyautogui are imported inside main() so the gesture logic stays importable (and
testable) on a machine that does not have them.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import Callable, Sequence

MAC = platform.system() == "Darwin"
WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# MediaPipe hand landmark indices. Both API generations return all 21 in this
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

# The standard hand skeleton, for drawing the preview. The old API shipped a
# helper for this; the Tasks API does not, and hardcoding the 21-point skeleton
# is both shorter than the compatibility shim and identical on both.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),              # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),              # index
    (9, 10), (10, 11), (11, 12),                 # middle
    (13, 14), (14, 15), (15, 16),                # ring
    (0, 17), (17, 18), (18, 19), (19, 20),       # pinky
    (5, 9), (9, 13), (13, 17),                   # knuckles
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

# The hand landmark bundle for the Tasks API. Downloaded once, ~7.8 MB.
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

# Defaults for the two pinch tests. See is_pinch for why there are two, and
# README.md for the measurements these came from.
PINCH_THRESHOLD = 0.22
REACH_THRESHOLD = 1.35


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
    """Wrist to index-tip distance, in hand spans.

    Around 0.9 when the finger is curled into the palm, around 1.9 when it is
    pointing straight out, and in between when it is bent into a pinch.
    """
    return _dist(lm, WRIST, INDEX_TIP) / hand_span(lm)


def is_pinch(lm: Sequence,
             pinch_threshold: float = PINCH_THRESHOLD,
             reach_threshold: float = REACH_THRESHOLD) -> bool:
    """True for a deliberate thumb-and-index pinch.

    The distance between thumb tip and index tip is the obvious test, and it is
    not sufficient: in a closed fist the thumb folds down right next to the
    curled index fingertip. Measured on a real photograph of a fist, that
    distance is 0.20 hand spans -- inside any sensible pinch threshold. Relying
    on it alone makes the page lurch every time you relax your hand.

    So a pinch also has to be *out in front of the palm*. A fist measures 0.94
    hand spans from wrist to index tip and an extended finger measures about
    1.9, which leaves a wide gap to put a threshold in.
    """
    return (pinch_ratio(lm) < pinch_threshold
            and index_reach(lm) > reach_threshold)


def classify(
    lm: Sequence,
    pinch_threshold: float = PINCH_THRESHOLD,
    reach_threshold: float = REACH_THRESHOLD,
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
# The landmark model, for the Tasks API
# ---------------------------------------------------------------------------
def default_model_path() -> Path:
    """Where to keep the downloaded model: the platform's cache directory.

    Deliberately not inside the repository -- it is a 7.8 MB build artifact, and
    deleting the cache directory is a clean way to start over.
    """
    if WINDOWS:
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif MAC:
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "gesture_scroll" / "hand_landmarker.task"


def ensure_model(path: Path, url: str = MODEL_URL,
                 log: Callable[[str], None] = print) -> Path:
    """Download the landmark model once, if it is not already cached.

    Downloads to a .part file and renames on success, so an interrupted download
    cannot leave a half-written model that fails confusingly later.
    """
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    log(f"Downloading the hand landmark model (~7.8 MB) to {path} ...")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as out:
            while chunk := response.read(256 * 1024):
                out.write(chunk)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    size = part.stat().st_size
    if size < 1_000_000:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"model download was only {size} bytes; expected ~7.8 MB")
    part.replace(path)
    log("Done.")
    return path


# ---------------------------------------------------------------------------
# Hand tracking, over whichever MediaPipe API is installed
# ---------------------------------------------------------------------------
class TasksTracker:
    """MediaPipe Tasks API (`mediapipe.tasks`), i.e. MediaPipe 1.x."""

    name = "mediapipe.tasks HandLandmarker"

    def __init__(self, mp, model_path: Path) -> None:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )
        )
        self._last_ts = -1

    def landmarks(self, rgb):
        # VIDEO mode requires strictly increasing timestamps, and a camera can
        # deliver two frames inside the same millisecond.
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts)
        return result.hand_landmarks[0] if result.hand_landmarks else None

    def close(self) -> None:
        self._landmarker.close()


class LegacyTracker:
    """MediaPipe 0.10.x `mediapipe.solutions.hands`, which bundles its model."""

    name = "mediapipe.solutions.hands"

    def __init__(self, mp) -> None:
        self._hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            # The lite landmark model. On a laptop CPU it is the difference
            # between comfortably real time and visibly lagging.
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

    def landmarks(self, rgb):
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        return result.multi_hand_landmarks[0].landmark

    def close(self) -> None:
        self._hands.close()


def make_tracker(mp, model_path: Path | None = None,
                 log: Callable[[str], None] = print):
    """Build a tracker on whichever MediaPipe API this install provides."""
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        return LegacyTracker(mp)
    model = ensure_model(model_path or default_model_path(), log=log)
    return TasksTracker(mp, model)


# ---------------------------------------------------------------------------
# Preview drawing
# ---------------------------------------------------------------------------
def draw_hand(cv2, frame, lm) -> None:
    """Draw the skeleton over the frame. Landmarks are normalised, pixels are not."""
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80, 200, 80), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, (240, 240, 240), -1)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
class Calibrator:
    """Records pinch/reach measurements for your own hand.

    The fist-versus-pinch threshold is the one number in this script that is a
    judgement call: the defaults come from photographs of other people's hands.
    Rather than guess for yours, hold each pose and press a key.
    """

    def __init__(self) -> None:
        self.samples: dict[str, list[tuple[float, float]]] = {"pinch": [], "fist": []}

    def record(self, pose: str, lm: Sequence) -> int:
        self.samples[pose].append((pinch_ratio(lm), index_reach(lm)))
        return len(self.samples[pose])

    def report(self) -> str:
        pinch, fist = self.samples["pinch"], self.samples["fist"]
        if not pinch or not fist:
            return ("Not enough samples. Hold a pinch and press p a few times,\n"
                    "then hold a fist and press f a few times.")
        pinch_reach = [r for _, r in pinch]
        fist_reach = [r for _, r in fist]
        pinch_d = [d for d, _ in pinch]
        lines = [
            f"pinch ({len(pinch)} samples): reach {min(pinch_reach):.2f}-{max(pinch_reach):.2f}"
            f"  thumb-index {min(pinch_d):.2f}-{max(pinch_d):.2f}",
            f"fist  ({len(fist)} samples): reach {min(fist_reach):.2f}-{max(fist_reach):.2f}",
        ]
        gap_lo, gap_hi = max(fist_reach), min(pinch_reach)
        if gap_lo >= gap_hi:
            lines += [
                "",
                "Your two poses overlap on reach, so no threshold separates them.",
                "Pinch with your hand a little further from your body, so the",
                "pinch happens in front of the palm rather than against it.",
            ]
        else:
            suggested = (gap_lo + gap_hi) / 2
            lines += [
                "",
                f"Suggested:  --reach-threshold {suggested:.2f}"
                f"  (clear of {gap_lo:.2f} and {gap_hi:.2f})",
                f"            --pinch-threshold {max(0.10, max(pinch_d) + 0.03):.2f}",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Camera loop
# ---------------------------------------------------------------------------
def quiet_opencv(cv2_module) -> None:
    """Silence OpenCV's native camera chatter.

    A camera that will not open makes the C++ layer print a dozen lines about
    V4L2, FFMPEG and UVC before our own one-line explanation ever appears. The
    advice that actually fixes it is in this script, not in that wall of text.
    """
    try:
        cv2_module.utils.logging.setLogLevel(
            cv2_module.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass    # older or trimmed OpenCV builds; the noise is not worth failing over


def camera_backend(cv2_module) -> int:
    """DirectShow on Windows, where the default backend can take ~10s to open."""
    if WINDOWS:
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
    p.add_argument("--pinch-threshold", type=float, default=PINCH_THRESHOLD,
                   help="thumb-index distance, in hand spans, that counts as a pinch")
    p.add_argument("--reach-threshold", type=float, default=REACH_THRESHOLD,
                   help="wrist-to-index-tip distance, in hand spans, below which a "
                        "pinch is treated as a fist")
    p.add_argument("--no-drag", action="store_true", help="disable the pinch-drag gesture")
    p.add_argument("--calibrate", action="store_true",
                   help="measure your own hand instead of scrolling: hold a pinch and "
                        "press p, hold a fist and press f, then quit")
    p.add_argument("--model", type=Path, default=None,
                   help="path to hand_landmarker.task (downloaded to a cache dir if absent)")
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
    except ImportError as exc:
        print(f"Missing dependency: {exc.name}\n"
              f"Install with: pip install -r requirements.txt", file=sys.stderr)
        return 1

    if args.dry_run or args.calibrate:
        def scroll_fn(clicks: int) -> None:
            print(f"scroll {clicks:+d}")
    else:
        try:
            import pyautogui
        except ImportError as exc:
            print(f"Missing dependency: {exc.name}", file=sys.stderr)
            return 1
        scroll_fn = pyautogui.scroll

    try:
        tracker = make_tracker(mp, args.model)
    except Exception as exc:
        print(f"Could not start hand tracking: {type(exc).__name__}: {exc}\n"
              f"Run python3 check_setup.py for a diagnosis.", file=sys.stderr)
        return 1

    quiet_opencv(cv2)
    cap = cv2.VideoCapture(args.camera, camera_backend(cv2))
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}. On macOS, grant Camera access to\n"
              f"your terminal in System Settings > Privacy & Security > Camera.",
              file=sys.stderr)
        tracker.close()
        return 1
    # 640x480 is plenty for hand landmarks and keeps the per-frame cost low.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    gestures = GestureFilter(hold=args.hold)
    scroller = Scroller(
        scroll_fn,
        step=args.step,
        interval=args.interval,
        drag_gain=args.drag_gain,
        invert_drag=args.invert_drag,
    )
    calibrator = Calibrator() if args.calibrate else None

    prev_palm_y: float | None = None
    fps = 0.0
    last_t = time.monotonic()

    print(f"Tracking with {tracker.name}.")
    if calibrator:
        print("Calibration mode. Nothing will scroll.")
        print("  hold a PINCH and press p (a few times), hold a FIST and press f,")
        print("  then press q to see the suggested thresholds.")
    else:
        print("Keep the mouse pointer over the document you want to scroll.")
        print("  one finger = up    two fingers = down    pinch and move = drag")
    print("  q or Esc in the preview window, or Ctrl-C here, to quit.")

    try:
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
            lm = tracker.landmarks(rgb)

            raw = IDLE
            if lm is not None:
                raw = classify(
                    lm,
                    pinch_threshold=args.pinch_threshold,
                    reach_threshold=args.reach_threshold,
                    allow_drag=not args.no_drag,
                )
            gesture = gestures.update(raw)

            if calibrator is None:
                if gesture == DRAG and lm is not None:
                    palm_y = lm[MIDDLE_MCP].y
                    span = hand_span(lm)
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

            key = -1
            if not args.no_preview:
                if lm is not None:
                    draw_hand(cv2, frame, lm)
                colour = (0, 220, 0) if gesture != IDLE else (160, 160, 160)
                label = gesture.upper()
                if calibrator is not None and lm is not None:
                    label += (f"   pinch {pinch_ratio(lm):.2f}"
                              f"   reach {index_reach(lm):.2f}")
                cv2.putText(frame, label, (12, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
                cv2.putText(frame, f"{fps:4.1f} fps", (12, 466),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("gesture scroll", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if calibrator is not None and lm is not None and key != -1:
                if key == ord("p"):
                    print(f"pinch sample {calibrator.record('pinch', lm)}: "
                          f"reach {index_reach(lm):.2f}")
                elif key == ord("f"):
                    print(f"fist sample {calibrator.record('fist', lm)}: "
                          f"reach {index_reach(lm):.2f}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        tracker.close()
        if not args.no_preview:
            cv2.destroyAllWindows()
        if calibrator is not None:
            print("\n" + calibrator.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

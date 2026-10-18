#!/usr/bin/env python3
"""Run the gesture classifier over real photographs of hands, and over degraded
versions of them.

self_test.py checks the geometry against synthetic landmarks, which is fast and
needs nothing installed -- but synthetic hands only ever confirm the assumptions
that built them. This test runs the actual MediaPipe landmarker over MediaPipe's
own published sample photographs, which is how the fist-reads-as-a-pinch bug was
found: every synthetic test passed while a real fist still triggered a drag.

Each photograph is then put through the things a laptop webcam does to an image
-- mirroring, a hand further away, a dim room, motion blur, compression, sensor
noise -- because a gesture recogniser that only works in good light on a clean
image is not much use for reading a book in the evening.

Needs mediapipe, opencv and about 8 MB of downloads the first time. Images and
model are cached in your cache directory, never in the repo.

    python3 image_test.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gesture_scroll import (  # noqa: E402
    DOWN, IDLE, UP,
    classify, default_model_path, ensure_model, index_reach, pinch_ratio,
)

IMAGE_BASE = "https://storage.googleapis.com/mediapipe-assets/"

# MediaPipe's own test photographs, and what this script should make of them.
# `pointing_up_rotated` is a hand pointing sideways: with the finger horizontal
# rather than up, it is correctly not a command, and it pins the edge of the
# envelope rather than being an awkward case to leave out.
EXPECTED = {
    "pointing_up.jpg": UP,
    "victory.jpg": DOWN,
    "fist.jpg": IDLE,
    "thumb_up.jpg": IDLE,
    "pointing_up_rotated.jpg": IDLE,
    "left_hands.jpg": IDLE,          # open palms
    "right_hands.jpg": IDLE,         # open palms
}

# Rotating the frame rotates the hand, so a pose that is defined by being
# upright can legitimately change meaning. Every pose here is rotation-stable
# over the swept range; the sideways one is tested separately below.
ROTATION_STABLE = [n for n in EXPECTED if n != "pointing_up_rotated.jpg"]

CHECKS = 0


def fetch(name: str, into: Path) -> Path:
    path = into / name
    if path.exists() and path.stat().st_size > 1000:
        return path
    into.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(IMAGE_BASE + name, timeout=60) as r, part.open("wb") as f:
        f.write(r.read())
    part.replace(path)
    return path


def build_augmentations(cv2, np):
    """The things a webcam and a room do to an image, as named transforms."""

    def scale(f):
        def go(img):
            h, w = img.shape[:2]
            return cv2.resize(img, (max(32, int(w * f)), max(32, int(h * f))),
                              interpolation=cv2.INTER_AREA)
        return go

    def rotate(deg):
        def go(img):
            h, w = img.shape[:2]
            m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
            return cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))
        return go

    def brightness(g):
        return lambda img: np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)

    def blur(k):
        return lambda img: cv2.GaussianBlur(img, (k, k), 0)

    def jpeg(q):
        def go(img):
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return go

    def noise(sigma, seed=0):
        def go(img):
            rng = np.random.default_rng(seed)
            return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape),
                           0, 255).astype(np.uint8)
        return go

    robust = [("identity", lambda i: i),
              # The live script mirrors the frame, so this is the path you use.
              ("mirror", lambda i: cv2.flip(i, 1))]
    robust += [(f"scale {f}", scale(f)) for f in (0.75, 0.5, 0.35, 0.25, 0.15)]
    robust += [(f"brightness {g}", brightness(g)) for g in (0.3, 0.5, 0.7, 1.5)]
    robust += [(f"blur {k}", blur(k)) for k in (3, 5, 9, 15)]
    robust += [(f"jpeg q{q}", jpeg(q)) for q in (40, 20)]
    robust += [("sensor noise 10", noise(10))]
    robust += [("mirror+far+dim", lambda i: brightness(0.6)(scale(0.5)(cv2.flip(i, 1))))]

    # Deliberately past the point of usefulness: what matters here is not that
    # it still works, but that it fails by seeing nothing rather than by
    # inventing a gesture. A wrong gesture scrolls the page; no gesture does not.
    degraded = [("jpeg q10", jpeg(10)), ("sensor noise 25", noise(25))]

    return robust, degraded, rotate


def main() -> int:
    try:
        import cv2
        import numpy as np
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        print(f"needs mediapipe and opencv ({exc.name} missing): "
              f"pip install -r requirements.txt")
        return 1

    cache = default_model_path().parent
    model = ensure_model(default_model_path())
    images = {name: cv2.imread(str(fetch(name, cache / "images")))
              for name in EXPECTED}
    missing = [n for n, img in images.items() if img is None]
    if missing:
        print(f"could not read: {missing}")
        return 1

    robust, degraded, rotate = build_augmentations(cv2, np)

    # IMAGE mode rather than the VIDEO mode the live script uses: these are
    # stills, and video mode would carry tracking state between unrelated photos.
    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1))

    def gesture_of(img) -> str | None:
        """None means no hand was detected at all."""
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            return None
        return classify(result.hand_landmarks[0])

    failures: list[str] = []

    def record(label: str, ok: bool) -> None:
        global CHECKS
        CHECKS += 1
        if not ok:
            failures.append(label)

    def expect_correct(label: str, img, want: str) -> None:
        """The gesture must be detected and right."""
        record(label, gesture_of(img) == want)

    def expect_not_wrong(label: str, img, want: str) -> None:
        """Weaker, and the one that actually matters for safety.

        Losing the hand is harmless -- nothing scrolls. Inventing a gesture is
        not. Conditions past the point of reliable detection are held to this.
        """
        record(label, gesture_of(img) in (None, want))

    # Multi-hand photographs at steep angles are where the detector gives up, so
    # the stricter rotation assertion applies to the single-hand ones.
    SINGLE_HAND = ["pointing_up.jpg", "victory.jpg", "fist.jpg", "thumb_up.jpg"]

    with landmarker:
        print("baseline measurements")
        for name, want in EXPECTED.items():
            rgb = np.ascontiguousarray(cv2.cvtColor(images[name], cv2.COLOR_BGR2RGB))
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            lm = res.hand_landmarks[0] if res.hand_landmarks else None
            got = classify(lm) if lm is not None else None
            record(f"baseline {name}", got == want)
            detail = (f"thumb-index={pinch_ratio(lm):.2f} reach={index_reach(lm):.2f}"
                      if lm is not None else "no hand detected")
            print(f"  {'ok ' if got == want else 'FAIL'} {name:24} {str(got):5} {detail}")

        print(f"\nrobustness: {len(EXPECTED)} photographs x {len(robust)} conditions")
        for aug_name, fn in robust:
            before = len(failures)
            for name, want in EXPECTED.items():
                expect_correct(f"{name} under {aug_name}", fn(images[name]), want)
            print(f"  {'ok  ' if len(failures) == before else 'FAIL'} {aug_name}")

        print(f"\nrotation: never a wrong command, and the single-hand photographs stay found")
        for deg in range(-50, 51, 10):
            before = len(failures)
            for name in ROTATION_STABLE:
                # Nothing may be misread at any angle ...
                expect_not_wrong(f"{name} rotated {deg} misread",
                                 rotate(deg)(images[name]), EXPECTED[name])
            for name in SINGLE_HAND:
                # ... and a clean single hand must still be found and read.
                expect_correct(f"{name} rotated {deg} lost",
                               rotate(deg)(images[name]), EXPECTED[name])
            print(f"  {'ok  ' if len(failures) == before else 'FAIL'} {deg:+d} degrees")

        print("\nthe edge of the envelope: a hand pointing sideways")
        sideways = images["pointing_up_rotated.jpg"]
        for deg in (-20, -10, -5, 0, 5):
            expect_correct(f"sideways rotated {deg} is not a command",
                           rotate(deg)(sideways), IDLE)
        print("  ok   idle while the finger is level or below")
        for deg in (15, 25, 35):
            expect_correct(f"sideways rotated {deg} reads as up",
                           rotate(deg)(sideways), UP)
        print("  ok   up once the finger is tilted above horizontal")

        print("\ndegradation must lose the hand, not invent a gesture")
        for aug_name, fn in degraded:
            before = len(failures)
            for name, want in EXPECTED.items():
                expect_not_wrong(f"{name} under {aug_name} invented a gesture",
                                 fn(images[name]), want)
            print(f"  {'ok  ' if len(failures) == before else 'FAIL'} {aug_name}")

    print(f"\n{CHECKS - len(failures)}/{CHECKS} checks passed "
          f"over {len(EXPECTED)} real photographs")
    for f in failures[:10]:
        print(f"  FAIL {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the gesture classifier over real photographs of hands.

self_test.py checks the geometry against synthetic landmarks, which is fast and
needs nothing installed -- but synthetic hands only ever confirm the assumptions
that built them. This test runs the actual MediaPipe landmarker over MediaPipe's
own published sample photographs, which is how the fist-reads-as-a-pinch bug was
found: every synthetic test passed while a real fist still triggered a drag.

Needs mediapipe, opencv and about 8 MB of downloads the first time. Images and
model are cached next to the model in your cache directory, never in the repo.

    python3 image_test.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gesture_scroll import (  # noqa: E402
    DOWN, IDLE, UP,
    classify, default_model_path, ensure_model, fingers_up, index_reach, pinch_ratio,
)

IMAGE_BASE = "https://storage.googleapis.com/mediapipe-assets/"

# MediaPipe's own test photographs, and what this script should make of them.
# There is no pinch image published, which is why the reach threshold is set
# clear of the fist rather than midway to a measured pinch -- and why
# gesture_scroll.py has a --calibrate mode for measuring your own hand.
EXPECTED = {
    "pointing_up.jpg": UP,
    "victory.jpg": DOWN,
    "fist.jpg": IDLE,
    "thumb_up.jpg": IDLE,
}


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


def main() -> int:
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        print(f"needs mediapipe and opencv ({exc.name} missing): "
              f"pip install -r requirements.txt")
        return 1

    cache = default_model_path().parent
    model = ensure_model(default_model_path())

    # IMAGE mode rather than the VIDEO mode the live script uses: these are
    # stills, and video mode would carry tracking state between unrelated photos.
    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1))

    failed = 0
    with landmarker:
        for name, want in EXPECTED.items():
            image_path = fetch(name, cache / "images")
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                print(f"FAIL {name}: could not be read")
                failed += 1
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not result.hand_landmarks:
                print(f"FAIL {name}: no hand detected")
                failed += 1
                continue
            lm = result.hand_landmarks[0]
            got = classify(lm)
            mark = "ok  " if got == want else "FAIL"
            failed += got != want
            print(f"{mark} {name:18} want={want:5} got={got:5} "
                  f"fingers={''.join('1' if f else '0' for f in fingers_up(lm))} "
                  f"thumb-index={pinch_ratio(lm):.2f} reach={index_reach(lm):.2f}")

    print(f"\n{len(EXPECTED) - failed}/{len(EXPECTED)} photographs classified as expected")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

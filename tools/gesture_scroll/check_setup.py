#!/usr/bin/env python3
"""Check whether this machine can run gesture_scroll.py, and say what to fix.

Run this first:

    python3 check_setup.py

Each check prints OK, WARN or FAIL with the exact command or setting that fixes
it. The checks are ordered so the first FAIL is the one worth fixing -- a camera
cannot be tested before OpenCV is installed.

The last check is interactive: it sends a real scroll event and asks whether the
page moved. That is the only honest way to test it, because on macOS a missing
Accessibility grant makes scroll events vanish silently rather than fail.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path

OK, WARN, FAIL = "OK", "WARN", "FAIL"

MAC = platform.system() == "Darwin"
WINDOWS = platform.system() == "Windows"

# Which Python versions MediaPipe can actually be installed on, read off the
# wheels it publishes:
#
#   MediaPipe 1.x  ships  py3-none-macosx_11_0_arm64, win_amd64, manylinux.
#                  "py3-none" means any Python 3 -- but there is NO Intel-Mac
#                  wheel at all.
#   MediaPipe 0.10 ships  cp39 to cp312 only, including Intel Mac.
#
# So on Apple Silicon any Python 3.9+ works, while on an Intel Mac pip has to
# fall back to 0.10.x and the interpreter must be 3.12 or older. Outside the
# supported range pip just says no matching distribution exists, which is a
# confusing way to learn you need a different interpreter.
PY_MIN = (3, 9)
PY_MAX_KNOWN = (3, 13)
PY_MAX_LEGACY = (3, 12)

# platform.machine() reports the *interpreter's* architecture, so a Rosetta
# Python on an Apple Silicon Mac correctly reads as x86_64 -- and does need the
# Intel wheel that MediaPipe 1.x does not publish.
MAC_INTEL = MAC and platform.machine() != "arm64"


class Report:
    """Collects results so the summary can point at the first thing to fix."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, name: str, status: str, detail: str = "", fix: str = "") -> str:
        self.rows.append((name, status, detail, fix))
        return status

    def show(self) -> int:
        width = max(len(name) for name, *_ in self.rows)
        print()
        for name, status, detail, _ in self.rows:
            print(f"  {status:<4}  {name:<{width}}  {detail}")

        warnings = [(n, f) for n, s, _, f in self.rows if s == WARN and f]
        # Several checks can share one fix -- three missing packages are one pip
        # command -- so group by fix text rather than listing it three times.
        problems: dict[str, list[str]] = {}
        for name, status, _, fix in self.rows:
            if status == FAIL and fix:
                problems.setdefault(fix, []).append(name)

        # Warnings print even when something failed: the Python version warning
        # in particular is usually the reason the pip install below will fail.
        if warnings:
            print("\nWorth knowing:\n")
            for name, fix in warnings:
                print(f"  - {name}")
                for line in fix.splitlines():
                    print(f"    {line}")
                print()

        if problems:
            print("Fix these, in order:\n")
            for i, (fix, names) in enumerate(problems.items(), 1):
                print(f"  {i}. {', '.join(names)}")
                for line in fix.splitlines():
                    print(f"     {line}")
                print()
            return 1

        print("\nReady. Put the mouse pointer over a PDF or web page, then run:\n")
        print("    python3 gesture_scroll.py\n")
        return 0


def _newer_python_fix(target: tuple[int, int]) -> str:
    major, minor = target
    if MAC:
        # brew --prefix rather than a hardcoded path: Homebrew lives in
        # /opt/homebrew on Apple Silicon and /usr/local on Intel.
        return (f"brew install python@{major}.{minor}\n"
                f"deactivate 2>/dev/null; rm -rf .venv\n"
                f"\"$(brew --prefix)/opt/python@{major}.{minor}/bin/python{major}.{minor}\" "
                f"-m venv .venv\n"
                f"source .venv/bin/activate && pip install -r requirements.txt")
    if WINDOWS:
        return (f"winget install Python.Python.{major}.{minor}\n"
                f"rmdir /s /q .venv\n"
                f"py -{major}.{minor} -m venv .venv\n"
                f".venv\\Scripts\\activate && pip install -r requirements.txt")
    return (f"sudo apt install python{major}.{minor}-venv\n"
            f"rm -rf .venv && python{major}.{minor} -m venv .venv\n"
            f"source .venv/bin/activate && pip install -r requirements.txt")


def check_python(r: Report) -> None:
    v = sys.version_info
    detail = f"{v.major}.{v.minor}.{v.micro}  ({sys.executable})"

    # On an Intel Mac the ceiling is lower, because the only installable
    # MediaPipe there is 0.10.x and its newest wheel is for CPython 3.12.
    ceiling = PY_MAX_LEGACY if MAC_INTEL else PY_MAX_KNOWN

    if v[:2] < PY_MIN:
        r.add("Python version", FAIL, detail,
              f"MediaPipe needs Python {PY_MIN[0]}.{PY_MIN[1]} or newer.\n"
              + _newer_python_fix(ceiling))
    elif v[:2] > ceiling:
        status = FAIL if MAC_INTEL else WARN
        why = ("MediaPipe publishes no Intel-Mac wheel for 1.x, so pip must install\n"
               f"0.10.x here -- and that has no wheel for Python {v.major}.{v.minor}.\n"
               f"Rebuild the virtualenv on Python {ceiling[0]}.{ceiling[1]}:\n"
               if MAC_INTEL else
               "This is newer than the Python versions MediaPipe is known to ship\n"
               "wheels for. If 'pip install -r requirements.txt' cannot find\n"
               f"mediapipe, rebuild the virtualenv on Python {ceiling[0]}.{ceiling[1]}:\n")
        r.add("Python version", status, detail, why + _newer_python_fix(ceiling))
    else:
        r.add("Python version", OK, detail)


def check_architecture(r: Report) -> None:
    """Only interesting on a Mac, where it decides which MediaPipe you get."""
    if not MAC:
        return
    if MAC_INTEL:
        r.add("Mac architecture", WARN, f"{platform.machine()} (Intel, or a Rosetta Python)",
              "MediaPipe 1.x has no Intel-Mac wheel, so pip installs 0.10.x instead.\n"
              "That is fine -- this script supports the older solutions API too, and\n"
              "skips the model download because 0.10.x bundles its model. It does mean\n"
              "the interpreter has to be Python 3.12 or older.")
    else:
        r.add("Mac architecture", OK, "arm64 (Apple Silicon)")


def check_virtualenv(r: Report) -> None:
    # sys.prefix differs from base_prefix only inside a venv.
    active = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if active:
        r.add("Virtualenv", OK, Path(sys.prefix).name)
    else:
        r.add("Virtualenv", WARN, "not active",
              "Installing into the system Python works but can collide with other\n"
              "projects. A virtualenv keeps this self-contained:\n"
              + ("  python3 -m venv .venv && source .venv/bin/activate"
                 if not WINDOWS else
                 "  py -m venv .venv ; .venv\\Scripts\\activate"))


def check_packages(r: Report) -> dict[str, object]:
    """Import each dependency and record its version."""
    found: dict[str, object] = {}
    for pip_name, import_name in (
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("pyautogui", "pyautogui"),
    ):
        try:
            mod = importlib.import_module(import_name)
        except ImportError:
            r.add(pip_name, FAIL, "not installed",
                  "pip install -r requirements.txt")
        except Exception as exc:
            # A package can import-fail for reasons other than absence.
            # pyautogui talks to X11 at import time, so on a headless or Wayland
            # session it raises here and reinstalling it will not help.
            if import_name == "pyautogui" and not (MAC or WINDOWS):
                fix = ("pyautogui needs an X11 display. Under Wayland it cannot inject\n"
                       "input at all; log into an X11/Xorg session instead. Gesture\n"
                       "recognition still works under --dry-run either way.")
            else:
                fix = "Reinstall it:\n  pip install --force-reinstall " + pip_name
            r.add(pip_name, FAIL, f"import failed: {exc}", fix)
        else:
            found[import_name] = mod
            r.add(pip_name, OK, str(getattr(mod, "__version__", "installed")))
    return found


def check_camera(r: Report, cv2, index: int) -> None:
    from gesture_scroll import camera_backend, quiet_opencv

    quiet_opencv(cv2)
    cap = cv2.VideoCapture(index, camera_backend(cv2))
    try:
        if not cap.isOpened():
            r.add(f"Camera {index}", FAIL, "will not open",
                  ("Grant Camera access to the app you launch from (Terminal,\n"
                   "iTerm, VS Code -- not 'Python') in System Settings >\n"
                   "Privacy & Security > Camera, then restart that app."
                   if MAC else
                   "Close any other app using the webcam (Zoom, Teams, Photo Booth).\n"
                   "If you have more than one camera, try: python3 check_setup.py --camera 1"))
            return
        ok, frame = cap.read()
        if not ok or frame is None:
            r.add(f"Camera {index}", FAIL, "opens but returns no frames",
                  "Another application is probably holding the camera. Close it and retry.")
            return
        h, w = frame.shape[:2]
        r.add(f"Camera {index}", OK, f"{w}x{h}")
    finally:
        cap.release()


def check_hand_tracking(r: Report, mp) -> None:
    """Build the real tracker and run one blank frame through it.

    This is the check that matters most, because it covers the two things most
    likely to be wrong on a fresh machine: which MediaPipe API is installed, and
    whether its native library can load.
    """
    from gesture_scroll import default_model_path, make_tracker

    legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
    if legacy:
        r.add("MediaPipe API", OK, "mediapipe.solutions.hands (0.10.x, model bundled)")
    else:
        model = default_model_path()
        cached = model.exists() and model.stat().st_size > 1_000_000
        r.add("MediaPipe API", OK,
              f"mediapipe.tasks ({'model cached' if cached else 'model will download, ~7.8 MB'})")

    try:
        import numpy as np
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        tracker = make_tracker(mp, log=lambda msg: None)
        try:
            tracker.landmarks(blank)
        finally:
            tracker.close()
    except OSError as exc:
        # MediaPipe 1.x loads a native library that links against the system
        # graphics stack. macOS and Windows ship it; a minimal Linux does not.
        missing = str(exc)
        if not (MAC or WINDOWS) and (".so" in missing):
            fix = ("MediaPipe needs the system graphics libraries:\n"
                   "  sudo apt install libegl1 libgles2        # Debian/Ubuntu\n"
                   "  sudo dnf install mesa-libEGL mesa-libGLES  # Fedora")
        else:
            fix = "Reinstall MediaPipe:\n  pip install --force-reinstall mediapipe"
        r.add("Hand tracking", FAIL, missing, fix)
    except Exception as exc:
        r.add("Hand tracking", FAIL, f"{type(exc).__name__}: {exc}",
              "Reinstall MediaPipe:\n  pip install --force-reinstall mediapipe")
    else:
        r.add("Hand tracking", OK, "model loads and runs")


def check_accessibility(r: Report) -> None:
    """macOS only: without this grant, scroll events are silently discarded."""
    if not MAC:
        return
    fix = ("System Settings > Privacy & Security > Accessibility: add the app you\n"
           "launch from (Terminal, iTerm, VS Code -- not 'Python') and switch it on,\n"
           "then quit that app completely and reopen it. Without this the script\n"
           "runs and recognises gestures but nothing ever scrolls.")
    try:
        from ApplicationServices import AXIsProcessTrusted  # type: ignore
    except Exception:
        r.add("Accessibility", WARN, "cannot be checked automatically", fix)
        return
    if AXIsProcessTrusted():
        r.add("Accessibility", OK, "granted")
    else:
        r.add("Accessibility", FAIL, "not granted", fix)


def check_scroll(r: Report, pyautogui, interactive: bool) -> None:
    if not interactive:
        r.add("Scroll event", WARN, "skipped",
              "Re-run without --no-interactive to test that scrolling actually works.")
        return

    print("\nScroll test")
    print("  Put the mouse pointer over something scrollable -- a long web page or")
    print("  a PDF -- and leave it there. Do not click. Press Enter when ready,")
    print("  or type s to skip.")
    if input("  > ").strip().lower().startswith("s"):
        r.add("Scroll event", WARN, "skipped by you")
        return

    for _ in range(6):
        pyautogui.scroll(-3)
    answer = input("  Did the page move? [y/N] ").strip().lower()
    if answer.startswith("y"):
        r.add("Scroll event", OK, "the page moved")
    else:
        r.add("Scroll event", FAIL, "nothing moved",
              ("The usual cause on macOS is the Accessibility grant above. The other\n"
               "is that the pointer was not over the scrollable window -- scroll\n"
               "events go where the pointer is, not where the keyboard focus is."
               if MAC else
               "Check that the pointer was over the scrollable window; scroll events go\n"
               "where the pointer is, not where the keyboard focus is.\n"
               "On Windows, a window running as administrator only accepts input from\n"
               "a program that is also running as administrator.\n"
               "On Linux this does not work under Wayland, only X11."))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--camera", type=int, default=0, help="camera index to test")
    p.add_argument("--no-interactive", action="store_true",
                   help="skip the scroll test, which needs an answer from you")
    args = p.parse_args(argv)

    print(f"gesture_scroll setup check on {platform.system()} "
          f"{platform.release()} ({platform.machine()})")

    r = Report()
    check_python(r)
    check_architecture(r)
    check_virtualenv(r)
    mods = check_packages(r)

    if "cv2" in mods:
        check_camera(r, mods["cv2"], args.camera)
    if "mediapipe" in mods:
        check_hand_tracking(r, mods["mediapipe"])
    check_accessibility(r)
    if "pyautogui" in mods:
        check_scroll(r, mods["pyautogui"], not args.no_interactive)

    return r.show()


if __name__ == "__main__":
    raise SystemExit(main())

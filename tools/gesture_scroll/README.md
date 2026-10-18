# Hand-gesture scroller

Scroll a PDF, an ebook or a web page with your hand instead of the mouse, using
nothing but the laptop webcam.

| Gesture | Action |
|---|---|
| ☝️ index finger up | scroll up, continuously while held |
| ✌️ index + middle up | scroll down, continuously while held |
| 🤏 pinch thumb to index, then move the hand | drag the page with your hand |
| ✋ open palm · ✊ fist · no hand | nothing |

```bash
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 gesture_scroll.py --dry-run                   # prints events, scrolls nothing
python3 gesture_scroll.py                             # for real
```

Start with `--dry-run`. It opens the camera and prints `scroll +2` / `scroll -2`
so you can see the gestures being recognised before granting the permissions
that let it move anything. Quit with `q` or `Esc` in the preview window, or
`Ctrl-C` in the terminal.

**Keep the mouse pointer over the document you want to scroll.** The script
emits the same scroll-wheel events a real mouse does, and the operating system
delivers those to whatever is under the pointer. Nothing needs to be clicked,
and the pointer does not need to move again afterwards.

## What it is built on

[MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
is a free Google library that finds a hand in a camera frame and returns 21
*landmarks* for it — the wrist plus four points along each finger, from knuckle
to fingertip. Each landmark comes back as an x/y position normalised to the
image, plus a rough depth that this script ignores. It is two small neural
networks chained together: the first draws a box around the palm, the second
places the 21 points inside that box. Palms are easier to detect than fingers
because they barely change shape. It runs in real time on a laptop CPU with no
GPU.

The loop, about 30 times a second: OpenCV grabs a frame, MediaPipe returns the
landmarks, ~60 lines of geometry turn them into a gesture, and
[pyautogui](https://pyautogui.readthedocs.io/) sends a scroll event.

## The three decisions that make it usable

Recognising a pose is the easy half. These are the parts that took thought:

**A finger is "up" when its tip is above its middle joint** — with a dead band.
Image coordinates grow downward, so "above" means a *smaller* y. Comparing the
two directly works, but a finger resting near the boundary flips state every
frame and the page stutters, so a tip has to clear its joint by a margin
(default 0.02 of the frame) before it counts.

**Everything is measured in hand spans, not pixels.** Thresholds are divided by
the wrist-to-middle-knuckle distance, so leaning back from the camera does not
change which gestures fire.

**A fist is not a pinch, even though it looks like one.** Distance between thumb
tip and index tip is the obvious pinch test, and it is wrong: curl your hand
into a fist and your thumb lands right next to your index fingertip, so the page
lurches every time you relax. The fix is a second condition — a real pinch
happens out in front of the palm, so the index tip must also be *reaching away*
from the wrist, which in a fist it never is.

And the one that matters most in practice: **a gesture must hold for 5
consecutive frames** before it does anything (`--hold`). Without that, every
frame your hand passes through on the way to a coffee mug fires a scroll.

## Tuning

Defaults are starting points — pyautogui's scroll "click" is not the same size
on every platform, so expect to adjust once.

| Flag | Default | What it does |
|---|---|---|
| `--step` | 3 on macOS, 1 on Windows, 2 on Linux | scroll clicks per emission for the finger gestures |
| `--interval` | `0.06` | minimum seconds between those emissions — raise it to slow scrolling down |
| `--drag-gain` | `12` | clicks per hand-span of pinch-drag movement |
| `--invert-drag` | off | push the hand down to scroll *down*, like a scrollbar, instead of dragging the page |
| `--hold` | `5` | frames a gesture must persist; raise it if you get accidental scrolls |
| `--pinch-threshold` | `0.22` | thumb-index distance, in hand spans, that counts as pinched |
| `--no-drag` | off | finger gestures only |
| `--no-preview` | off | no camera window; quit with `Ctrl-C` |
| `--camera` | `0` | camera index, if you have more than one |

Scrolling too fast is the usual first complaint: lower `--step` or raise
`--interval`.

## Permissions

**macOS.** Two separate grants, both in System Settings → Privacy & Security,
and both for the app you launch from (Terminal, iTerm, VS Code — not Python):

- **Camera** — without it the script exits with "could not open camera".
- **Accessibility** — without it the script runs, the preview works, gestures
  are recognised, and *nothing scrolls*. There is no error message. If gestures
  show up in the preview but the page never moves, this is why.

**Windows.** No permission prompts. One catch: a program cannot send input to a
window running with higher privileges, so if you want to scroll something opened
as administrator, launch the script as administrator too.

**Linux.** Works under X11. Under Wayland, pyautogui cannot inject input;
`--dry-run` will still show the gestures being recognised correctly.

## Troubleshooting

| Symptom | Cause |
|---|---|
| gestures recognised, nothing scrolls | macOS Accessibility not granted, or the pointer is not over the document |
| scrolls the wrong window | the pointer is somewhere else — the pointer decides, not keyboard focus |
| random scrolls while you move around | raise `--hold` |
| pinch fires when you relax your hand | lower `--pinch-threshold` toward `0.15` |
| low fps, laggy | close other camera apps; the script already uses the lite model at 640×480 |
| `pip` cannot find mediapipe | your Python is newer than MediaPipe's wheels — use 3.12 or older |

## Tests

The geometry is the part that is easy to get backwards, and none of it needs a
camera. Synthetic landmarks stand in for a hand:

```bash
python3 self_test.py        # no pytest required
```

14 checks, including that a fist is not read as a pinch and that the same pose
at half the apparent size still classifies the same.

## Where to take it next

- **Head tilt** — `mediapipe` Face Mesh gives you a nose and eye landmarks; tilt
  past a threshold to scroll. Less tiring than holding a hand up to read a long
  chapter.
- **Gaze zones** — Face Mesh also returns iris landmarks, which is the closest
  you can get on a laptop to what Apple Vision Pro does with Look to Scroll.
  Coarse zones (look at the bottom third → scroll down) work; true pointwise
  gaze does not, because the webcam is too far away and too low-resolution for
  the precision that infrared cameras millimetres from your eye give a headset.
- **Horizontal drag** too, for wide PDFs, from the x axis that is already there.

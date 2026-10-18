# Hand-gesture scroller

Scroll a PDF, an ebook or a web page with your hand instead of the mouse, using
nothing but the laptop webcam.

| Gesture | Action |
|---|---|
| ☝️ index finger up | scroll up, continuously while held |
| ✌️ index + middle up | scroll down, continuously while held |
| 🤏 pinch thumb to index, then move the hand | drag the page with your hand |
| ✋ open palm · ✊ fist · no hand | nothing |

---

## Getting it onto your laptop

It lives in this repository, so there is nothing to copy by hand. Pick the
block for your machine and paste it into a terminal. The clone is about 40 MB.

### macOS

```bash
git clone https://github.com/Pranjal101Shrivastava/Projects.git
cd Projects/tools/gesture_scroll
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 check_setup.py
```

Which Mac you have decides which MediaPipe you get, and on one of them the
Python version matters:

| | MediaPipe | Python | Model |
|---|---|---|---|
| **Apple Silicon** (M1–M4) | 1.x, via a `py3-none-macosx_11_0_arm64` wheel | any 3.9+ | downloaded once, ~7.8 MB |
| **Intel Mac** | 0.10.x — there is no Intel wheel for 1.x at all | **3.9–3.12 only** | bundled, no download |

Both paths work; the script detects which API is installed. But on an Intel Mac
with Python 3.13 or newer, `pip install` fails outright with "no matching
distribution", because neither MediaPipe generation publishes a wheel for that
combination. `check_setup.py` detects this and prints the Homebrew commands to
rebuild the virtualenv on 3.12.

### Windows (PowerShell)

```powershell
git clone https://github.com/Pranjal101Shrivastava/Projects.git
cd Projects\tools\gesture_scroll
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
py check_setup.py
```

### Linux

```bash
sudo apt install libegl1 libgles2          # MediaPipe's native library needs these
git clone https://github.com/Pranjal101Shrivastava/Projects.git
cd Projects/tools/gesture_scroll
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 check_setup.py
```

Already have the repo cloned? Then just:

```bash
git fetch origin && git checkout claude/awesome-hawking-5rl6qn
cd tools/gesture_scroll
```

### Then run it

`check_setup.py` is the first thing to run, every time something seems wrong. It
checks the Python version, each dependency, the camera, the hand model and
(on macOS) the Accessibility permission, and prints the exact command or setting
that fixes whatever failed. It ends by sending a real scroll event and asking
whether the page moved, which is the only honest way to test that part.

Once it says *Ready*:

```bash
python3 gesture_scroll.py --dry-run    # prints events, scrolls nothing
python3 gesture_scroll.py              # for real
```

Start with `--dry-run`. It opens the camera and prints `scroll +2` / `scroll -2`
so you can see your gestures being recognised before granting the permission
that lets it move anything. Quit with `q` or `Esc` in the preview window, or
`Ctrl-C` in the terminal.

**Keep the mouse pointer over the document you want to scroll.** The script
emits the same scroll-wheel events a real mouse does, and the operating system
delivers those to whatever is under the pointer. Nothing needs to be clicked,
and the pointer does not need to move again afterwards.

On first run you will see a one-off `Downloading the hand landmark model
(~7.8 MB)`. It goes to your cache directory (`~/Library/Caches/gesture_scroll`
on macOS, `%LOCALAPPDATA%\gesture_scroll` on Windows), not into the repository.

---

## What it is built on

[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
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

MediaPipe has two generations of Python API and this script supports both,
picking whichever `pip` gave you:

- **`mediapipe.tasks`** — MediaPipe 1.x. Needs the model file downloaded once.
  This is what Apple Silicon, Windows and Linux get.
- **`mediapipe.solutions.hands`** — 0.10.x, with the model bundled. Removed in
  1.0, so this path runs on older installs and on Intel Macs, which have no
  MediaPipe 1.x wheel.

---

## The three decisions that make it usable

Recognising a pose is the easy half. These are the parts that took thought:

**A finger is "up" when its tip is above its middle joint** — with a dead band.
Image coordinates grow downward, so "above" means a *smaller* y. Comparing the
two directly works, but a finger resting near the boundary flips state every
frame and the page stutters, so a tip has to clear its joint by a margin before
it counts.

**Everything is measured in hand spans, not pixels** — including that margin,
which is the point. Thresholds are divided by the wrist-to-middle-knuckle
distance, so leaning back from the camera does not change which gestures fire.
The margin was originally a fixed distance in frame units, and a scale sweep
found the consequence: sit far enough back and no finger can clear a fixed dead
band, so every gesture silently goes idle.

**A fist is not a pinch, even though it measures like one.** Distance between
thumb tip and index tip is the obvious pinch test, and it is not enough: curl
your hand into a fist and the thumb folds down right beside the curled index
fingertip. Measured on a real photograph, that distance is 0.20 hand spans —
well inside any sensible pinch threshold. So a pinch also has to happen *out in
front of the palm*:

| pose | thumb–index | wrist→index tip |
|---|---|---|
| fist | 0.20 | **0.94** |
| thumb up | 0.97 | 0.80 |
| index pointing | 1.06 | **1.92** |
| two fingers | 1.05 | 1.84 |

A curled index finger sits around 0.94 hand spans from the wrist and an extended
one around 1.9, which leaves a wide gap to put a threshold in. The default is
1.35.

Those numbers are measured, not guessed — see **Tests** below. An earlier
version of this script used 0.75, which passed every synthetic test and still
fired a drag every time a real hand relaxed into a fist.

And the decision that matters most in daily use: **a gesture must hold for 5
consecutive frames** before it does anything (`--hold`). Without that, every
frame your hand passes through on the way to a coffee mug fires a scroll.

---

## Calibrating to your own hand

The fist-versus-pinch threshold is the one number here that depends on your
hand, and the default came from photographs of other people's. There is no
published pinch photograph to measure, so if drag misfires — or will not
trigger — measure yourself:

```bash
python3 gesture_scroll.py --calibrate
```

Nothing scrolls in this mode. The preview shows live `pinch` and `reach`
numbers. Hold a pinch and press `p` a few times, hold a fist and press `f` a few
times, then `q`. It prints the two ranges and a suggested threshold placed
between them:

```
pinch (5 samples): reach 1.38-1.49  thumb-index 0.05-0.11
fist  (4 samples): reach 0.91-0.97
Suggested:  --reach-threshold 1.17  (clear of 0.97 and 1.38)
```

Then run with that value. If the two ranges overlap it says so rather than
inventing a threshold.

---

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
| `--pinch-threshold` | `0.22` | thumb–index distance, in hand spans, that counts as pinched |
| `--reach-threshold` | `1.35` | wrist→index-tip distance below which a pinch is treated as a fist |
| `--no-drag` | off | finger gestures only |
| `--calibrate` | off | measure your own hand instead of scrolling |
| `--no-preview` | off | no camera window; quit with `Ctrl-C` |
| `--camera` | `0` | camera index, if you have more than one |

Scrolling too fast is the usual first complaint: lower `--step` or raise
`--interval`.

---

## Permissions

**macOS.** Two separate grants, both in System Settings → Privacy & Security,
and both for the app you launch from (Terminal, iTerm, VS Code — not Python):

- **Camera** — without it the script exits with "could not open camera".
- **Accessibility** — without it the script runs, the preview works, gestures
  are recognised, and *nothing scrolls*. There is no error message. `check_setup.py`
  detects this one directly and tells you.

After granting Accessibility, quit the terminal app completely and reopen it;
the grant is read at launch.

**Windows.** No permission prompts. One catch: a program cannot send input to a
window running with higher privileges, so to scroll something opened as
administrator, launch the script as administrator too.

**Linux.** Works under X11. Under Wayland, pyautogui cannot inject input at all;
`--dry-run` still shows gestures being recognised correctly.

---

## Troubleshooting

Run `python3 check_setup.py` first — it diagnoses most of this directly.

| Symptom | Cause |
|---|---|
| gestures recognised, nothing scrolls | macOS Accessibility not granted, or the pointer is not over the document |
| scrolls the wrong window | the pointer is somewhere else — the pointer decides, not keyboard focus |
| random scrolls while you move around | raise `--hold` |
| drag fires when you relax your hand | `--calibrate`, then lower `--reach-threshold` |
| drag never fires | `--calibrate`, then raise `--reach-threshold` |
| `module 'mediapipe' has no attribute 'solutions'` | MediaPipe 1.x with an old copy of this script — `git pull` |
| `libEGL.so.1: cannot open shared object file` | Linux only: `sudo apt install libegl1 libgles2` |
| stops detecting in a dim room | darkness is fine, but a noisy sensor is not — add a light rather than turning up exposure |
| low fps, laggy | close other camera apps; the script already runs at 640×480 |
| `pip` cannot find mediapipe, Intel Mac | no MediaPipe 1.x Intel wheel exists; 0.10.x needs Python 3.9–3.12. Run `check_setup.py` for the Homebrew commands |
| `pip` cannot find mediapipe, elsewhere | your Python is newer than MediaPipe's wheels — rebuild the venv on an older one |

---

## What it survives

Measured, not assumed — every photograph put through what a webcam and a room
actually do to an image:

| Condition | Result |
|---|---|
| mirrored (the live script mirrors) | fine |
| hand far away — down to 15% of frame size | fine |
| dim room — down to 30% brightness | fine |
| motion blur — up to a 15px kernel | fine |
| JPEG compression down to quality 20 | fine |
| tilted hand, ±50° | fine; a clean single hand reads correctly to ±70° |
| heavy sensor noise | **hand is not detected** — nothing scrolls |
| hand pointing sideways | idle; tilt the finger ~15° above horizontal and it reads as up |

The important one is the second-to-last. Past the point where it can see
reliably, it loses the hand rather than inventing a gesture — a missed detection
does nothing, a wrong gesture scrolls your page. That property is asserted, not
hoped for.

## Tests

Two layers, because the first one alone was not enough.

```bash
python3 self_test.py     # 1,063 checks, no camera, no pytest needed
python3 image_test.py    # 272 checks over 7 real photographs, needs mediapipe
```

`self_test.py` runs the geometry against synthetic landmarks. Beyond the named
cases, it sweeps the properties that must hold across a range rather than at one
point: scale from 5% to 300%, position across the frame, rotation through ±80°,
seeded landmark jitter, a 15×15 grid of the pinch decision, and every one of the
243 possible five-frame gesture sequences through the hold filter.

`image_test.py` runs the real landmarker over MediaPipe's published photographs
and over degraded versions of each. This is the layer that earns its place: every
synthetic test passed while a real fist still triggered a drag, because the
synthetic fist had been built from the same wrong assumption as the threshold it
was testing. Both bugs found so far — that one, and the absolute-margin scale
bug — came from sweeps rather than from named cases.

---

## Where to take it next

- **Head tilt** — MediaPipe's Face Landmarker gives you nose and eye points;
  tilt past a threshold to scroll. Less tiring than holding a hand up for a
  whole chapter.
- **Gaze zones** — the same model returns iris landmarks, which is the closest a
  laptop gets to what Apple Vision Pro does with Look to Scroll. Coarse zones
  (look at the bottom third → scroll down) work; true pointwise gaze does not,
  because the webcam is too far away and too low-resolution for the precision
  that infrared cameras millimetres from your eye give a headset.
- **Horizontal drag** for wide PDFs, from the x axis that is already there.

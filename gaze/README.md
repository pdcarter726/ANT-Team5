# Eye-tracking cursor

Moves the mouse cursor with your eyes using a normal webcam, built on
[EyeGestures](https://github.com/NativeSensors/EyeGestures) v4 (GPLv3) and MediaPipe Face Landmarker.

Calibration is a Python port of the EyeGestures v4 engine's calibration (verified to advance on the exact
same frames as the original): same eye features, same linear model, and a dot only moves on once the live
estimate has settled on it. Porting it lets us use 35 points, ignore blinks, and never recalibrate on our
own (the stock engine adds points by itself whenever the head moves).

## Setup (do this with internet, BEFORE the ANT station)
1. Install **Python 3.13** (eyeGestures 4.x only ships for 3.13; on 3.12 pip quietly installs the old 3.2.4, which has a different API).
2. From the repo root:
   ```
   py -3.13 -m venv .venv
   .venv\Scripts\python -m pip install -r requirements-gaze.txt
   .venv\Scripts\python -m gaze.setup_check
   ```
   `setup_check` downloads the face model into `gaze/models/` (EyeGestures would otherwise try to download it at startup and crash without internet) and checks that the webcam sees your face.

## Run
```
.venv\Scripts\python -m gaze.app
```
(Plain `python` uses the system Python, which does not have the packages.)
1. A fullscreen calibration screen appears. Follow the dot through 35 points with your eyes, head still. A point only advances once the live estimate (grey dot) has settled on it, so take your time.
2. After calibration the cursor follows your gaze. A small preview window shows status and FPS.
   Keys (click the preview window first):
   - `c` recenter: look at the center dot (about 2 s). Uses the median of many frames, so single noisy frames don't matter. Fixes drift after a small head move.
   - `a` add head position: 9 more points at your current head position, added to the model (not a reset).
   - `r` full recalibration (starts over), `p` pause/resume, `q`/`Esc` quit.
3. Calibration happens only at start, on a key, or when the EEG app asks. Never on its own.

## Gaze model (picked per calibration)
After each calibration (and each `a`), several models are fitted to your calibration data and each one is
scored on calibration dots it did not train on (leave one dot out). The best one is used. The terminal
prints the scoreboard and the winner's held-out error in pixels, which is the honest accuracy number to
compare setups with (lighting, camera height, sitting position). Candidates:
- engine linear: the EyeGestures v4 model on its own eye features
- eye+pose: iris position inside each eye (in eye widths) and eyelid opening, plus head yaw/pitch/roll and
  position from MediaPipe, ridge regression, linear or with squared terms for the edges
- engine+eye+pose: both feature sets together, ridge regression

## Blinks
During a blink the eyelid and iris landmarks jump, which throws the gaze estimate. Blinks are detected from
MediaPipe's eyeBlink scores and those frames are ignored everywhere: calibration never learns from them,
recenter skips them, and the cursor holds still through them (plus 3 frames after). A "blink" longer than
0.6 s is treated as looking down or squinting, so those frames are used normally.

## Edges
Regression on noisy webcam features pulls predictions toward the screen center, so the raw cursor never
quite reaches the edges. For the chosen model, the app measures that pull on the held-out dots and
stretches the output per axis to undo it (simulated edge error about 40 px -> 7 px).

## Head movement
The model maps eye landmarks to the screen. The stock engine also feeds it head position and face size.
If you calibrate with your head still, those head numbers barely change, the regression gives them
enormous weights, and afterwards even a millimetre of head movement throws the cursor far off (and
"returning" the head is never exact enough to bring it back). So the model leaves the head features
out until calibration data shows real head movement, and uses them once it does.

Fixes, in order of effort:
1. **Recenter (`c`)** after a head shift. Takes a few seconds. Note: the correction stays until the next
   recenter, so if you move back to your old position, recenter again.
2. **Add head positions (`a`)**: after the full calibration, shift your head slightly (a little left, right,
   up, closer) and press `a` at each spot. 3 to 4 extra positions is a good target. Once the head moved
   enough across these, the model starts using the head features to compensate.
3. **Keep the head stable physically**: sit back against a high chair back or headrest, camera at eye level,
   laptop at arm's length, light from the front.
4. **Head gestures (EEG)**: the EEG app should call `control.freeze()` during a head turn/tilt so the cursor
   holds still, and the wearer returns the head to the same rest position afterwards.

Simulated test (head shifted after calibrating): stock engine model about 940 px off, this version about
30 px off; after `a` at the new spot about 25 px at both positions.

## Hooking up the EEG / ML app
The gaze app listens on `udp://127.0.0.1:50555`. From Python:
```python
from gaze import control
control.freeze(0.8)   # call the moment a gesture starts: holds the cursor still so
                      # eyebrow raises, blinks and head turns don't drag it away
control.pause(); control.resume(); control.toggle()
control.recalibrate()   # full 25-point calibration
control.add_pose()      # 9 extra points at the current head position
control.recenter()      # one center dot, fixes drift
control.status()      # {'face', 'x', 'y', 'calibrating', 'state'}
```
Any language works: send the plain-text message (`freeze 0.8`, `pause`, ...) as a UDP packet.
Clicks and keys stay in the EEG app (pyautogui clicks at the current cursor position).

## Tips for accuracy
- Light from the front, no bright window behind you, camera at eye level, laptop at arm's length.
- Calibrate on the demo laptop, in the demo room, sitting how you'll sit for the demo.
- Head movement shifts the estimate. Recalibrate (`r`) if the cursor drifts.
- Webcam gaze is coarse: design the demo around big targets.

## Files
- `app.py` main program (calibration screen, cursor loop, preview)
- `tracker.py` webcam, blink filter, calibration (EyeGestures port), gaze model, save/load
- `filters.py` spike rejection + One Euro smoothing
- `cursor.py` cursor movement, freeze/pause, dwell click
- `control.py` UDP channel for the EEG app
- `setup_check.py` one-time setup and diagnostics

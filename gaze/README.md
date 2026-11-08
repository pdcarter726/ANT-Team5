# Eye-tracking cursor

Moves the mouse cursor with your eyes using a normal webcam. The method comes from
[EyeGestures](https://github.com/NativeSensors/EyeGestures) v4 (GPLv3); face landmarks come from MediaPipe Face Landmarker.

How it works, in one paragraph: MediaPipe finds 478 face landmarks per frame (in VIDEO mode, so the face
is tracked between frames and landmarks are smoothed: about 55-60% less jitter than EyeGestures'
per-frame mode in a test with the real model, with no measurable lag). Calibration is a Python port of the
EyeGestures v4 engine's calibration, checked to advance on the exact same frames as the original: a dot
only moves on once the live estimate has settled on it. After calibration, several gaze models are fitted
and the one that predicts best on data it did not train on is used.

## Setup
From the repo root, in the same `.venv` as the controller:
```
.venv\Scripts\python -m pip install -r requirements-gaze.txt
.venv\Scripts\python -m gaze.setup_check
```
`setup_check` downloads the face model into `gaze/models/` (needed offline later), and reports the camera
resolution, the frames per second actually processed, and whether your face is found.

## Run
On its own: `.venv\Scripts\python -m gaze.app`. With the controller: `controller.bat --gaze-profile NAME`
(the controller starts it and talks to it, see the end of this file).

1. **Calibration** (about 1.5 minutes, no keys needed): 50 dots in 5 blocks of 10. Before each block the
   screen tells you where to put your head (normal, a little left, a little right, chin slightly lower,
   chin slightly higher) with a 4-second countdown; hold that position and follow the dots. A dot only
   advances once the live estimate (grey dot) has settled on it. Because the calibration contains head
   movement, the model learns to compensate for it, and every model is tested at a head position it did
   not train on. `s` skips a dot that will not settle (optional).
2. **Accuracy check**: press `v` (or Ctrl+Alt+V), then sit still and follow the 5 dots. It prints the
   typical and worst error in pixels, whether the cursor is shifted overall (drift, usually the head moved),
   and whether it falls short of the edges (e.g. "only reaches 86% of the way to the edges horizontally").
   It does not change the model.
   **Fix**: press `f` (or Ctrl+Alt+F) to apply a correction from that check (per-axis shift and scale).
   Then `v` again to confirm.
3. The cursor follows your gaze. Blinks are ignored (the cursor holds still through them).

### Profiles (save a calibration and reuse it)
```
.venv\Scripts\python -m gaze.app --profile henry          # loads "henry" if it exists (no calibration),
                                                           # otherwise calibrates and saves it as "henry"
.venv\Scripts\python -m gaze.app --save-profile henry     # copy the last calibration into "henry", then exit
.venv\Scripts\python -m gaze.app --list-profiles          # name, date, screen, model, score, last check
```
- While running with `--profile`, everything that changes the calibration (add head position, undo, recenter,
  the `f` correction, a full recalibration) is saved to that profile.
- One profile per person: a model only fits the person it was calibrated on.
- A profile is tied to the screen size, and works best in the same seat and camera position. After loading,
  press `c` (recenter) or `v` then `f` if you sit differently than when it was made.
- Profiles live in `gaze/profiles/` and are committed with the repo, so anyone on the team can pull and run
  `--profile <name>`. The face model (`gaze/models/face_landmarker.task`) is committed too, so a fresh clone
  works offline after `pip install`.
- A profile only loads on a screen with the same resolution it was made on (the model predicts screen pixels).

### Keys
Anywhere (Windows): **Ctrl+Alt+P** pause/resume, **Ctrl+Alt+C** recenter, **Ctrl+Alt+V** accuracy check,
**Ctrl+Alt+F** apply check correction, **Ctrl+Alt+Q** quit.

In the small preview window (click it first):
| key | what |
|---|---|
| `v` | accuracy check |
| `f` | apply a correction from the last accuracy check |
| `c` | recenter: look at the center dot (about 2 s). Fixes drift after a small head move. Stays until the next recenter, so recenter again if you move back. |
| `a` | add head position: 9 more dots at your current head position, added to the model |
| `u` | undo the last added head position |
| `r` | full recalibration (starts over) |
| `p` | pause/resume |
| `q` / `Esc` | quit |

Calibration only ever happens at start, on a key, or when the EEG app asks. Never on its own.
Each calibration is saved; `--load` reuses it (same seat, same screen), handy after a restart.

### Options
| flag | default | meaning |
|---|---|---|
| `--camera N` | 0 | webcam index |
| `--load` | off | reuse the last saved calibration |
| `--profile NAME` | none | use a calibration profile (load or create) |
| `--save-profile NAME` | | copy the last calibration into a profile and exit |
| `--list-profiles` | | show saved profiles and exit |
| `--min-cutoff` | 0.4 | lower = steadier cursor, more lag |
| `--beta` | 0.003 | higher = less lag on big eye jumps |
| `--deadzone` | 10 | pixels of jitter ignored |
| `--dwell-click S` | 0 (off) | click by holding gaze still for S seconds (accessibility fallback) |
| `--image-mode` | off | detect the face from scratch every frame (EyeGestures' original way) |

## How accurate to expect
Webcam eye trackers typically reach about 2 to 5 degrees of eye angle. On a 2560x1600 laptop screen at arm's
length, 1 degree is roughly 70-80 px, so about 150-350 px of error is normal for a webcam and around 120 px
is good. Big targets (a few hundred pixels) work best; the EEG gestures do the clicking.

## What makes it accurate (and what to do when it is not)
In order of impact:
1. **Light and camera**: light from the front, no bright window behind you, camera at eye level,
   laptop at arm's length. `setup_check` shows frames per second; under about 20, add light (webcams lower
   their frame rate in dim light) and close other camera apps.
2. **Head stability**: sit back against a high chair back. Small head moves are the main source of error.
3. **Add head positions** (`a`): after calibrating, shift your head slightly (a little left, right, up,
   closer) and press `a` at each spot; 2 to 4 positions. Undo with `u` if the check gets worse.
4. **Recenter** (`c`) after any head shift.
5. **Check** (`v`) after any of the above to see whether it helped. If the check says the cursor is shifted
   overall, `f` applies a correction from it.
6. **Press the keys without moving**: use the Ctrl+Alt hotkeys so you don't lean toward the keyboard or
   reach for the mouse between calibrating and checking.

## Details
- **Camera**: requests MJPG at 30 fps, 1280x720. On Windows many webcams only reach 30 fps at 720p in MJPG.
- **Calibration layout**: each block of 10 dots covers the whole screen (corners, edges, middle), and
  together the 50 dots fill a fine grid. Each block counts as its own head position for the model test.
- **Calibration data**: frames from the first 0.4 s after a dot appears are not stored (the eyes are still
  moving to it); the dot timing itself is unchanged.
- **Gaze models**: candidates are the EyeGestures linear model on its eye features; ridge regression on
  iris position inside each eye (in eye widths) and eyelid opening, linear or with squared terms for the
  edges; and both feature sets together. Each with and without head pose/position, which is only allowed
  once the calibration data actually contains head movement. Each is scored on held-out dots, and, once
  head positions were added, on a held-out head position. The terminal prints the scoreboard.
- **Edges**: regression on noisy features pulls predictions toward the center; the chosen model's output is
  stretched per axis to undo that, measured on held-out dots.
- **Blinks**: two detectors, either counts: MediaPipe's eyeBlink score (above 0.5, or well above your own
  normal level; in video mode the scores are smoothed and often stay low during real blinks) and the eyelid
  gap dropping below 65% of its recent normal value. Blink frames are ignored in calibration, recenter, the
  check and tracking. A "blink" longer than 0.6 s counts as looking down or squinting and is used.

## Diagnostics (terminal + log file)
Everything is logged with a timestamp and a category, to the terminal and to `gaze/logs/gaze_<date>_<time>.log`
(send that file when something looks off). `--log-every N` sets the status-line interval (default 2 s, 0 = off).
- `START` Python/package versions, screen size and Windows scaling (DPI), all options
- `CAM` camera backend, requested vs delivered size, format (MJPG), reported fps, exposure; read failures
- `FACE` face model and mode; face lost / found again
- `CAL` every calibration dot: target, seconds and frames to settle, blink frames, no-face frames,
  frames kept vs dropped (eyes still moving), estimate error at settle; skips; totals
- `FIT` every model's held-out score, the chosen model, number of features, whether head features are used,
  edge stretch, how much the head moved during calibration, error at the edges vs inside, worst dots
- `HB` status line every 2 s: mode, cursor state, camera fps vs processed fps, face detection time,
  face found %, blink %, blink score (median/max) and smallest eyelid gap (% of normal), dropped camera frames, gaze and cursor position, gaze spread, head angles and
  distance, head drift since calibration, brightness (0-255), face width in pixels, cursor moves
- `CHECK` every check dot (target, measured, dx/dy, error, spread), re-checks, the result, drift direction,
  head position during the check vs calibration, the f correction applied
- `RECENTER` frames used, estimate, offset applied
- `CMD` every key, Ctrl+Alt hotkey and command from the EEG app; `MODE` mode changes; `CURSOR` pause/resume,
  dwell clicks; `SAVE` saves and loads; `ERROR` errors with the full traceback (a bad frame is logged and
  skipped instead of crashing the app)

Reading it: low `cam_fps` or `light` = add front light; high `detect_ms` = close other apps; large
`head_drift` = you moved since calibrating (recenter or `f`); `face_%` under 100 = face partly out of view
or bad light; high `gaze_spread_last30_px` while looking at one spot = noisy tracking.

## Control channel (used by the controller)
The gaze app listens on `udp://127.0.0.1:50555`. From Python:
```python
from gaze import control
control.freeze(0.8)     # call the moment a gesture starts: holds the cursor still so
                        # eyebrow raises, blinks and head turns don't drag it away
control.pause(); control.resume(); control.toggle()
control.recalibrate()   # full 50-dot calibration
control.add_pose()      # 9 extra points at the current head position
control.recenter()      # one center dot, fixes drift
control.undo_pose()     # drop the last added head position
control.validate()      # 5-dot accuracy check
control.fix_from_check()  # apply a correction from the last check
control.status()        # {'face', 'x', 'y', 'calibrating', 'state'}
```
Any language works: send the plain-text message (`freeze 0.8`, `pause`, ...) as a UDP packet.
Clicks and keys stay in the EEG app (pyautogui clicks at the current cursor position).

## Files
- `app.py` main program (calibration screen, cursor loop, preview, hotkeys)
- `tracker.py` camera, MediaPipe face landmarks, blink filter, calibration (EyeGestures port), gaze models, accuracy check, save/load
- `filters.py` spike rejection + One Euro smoothing
- `cursor.py` cursor movement, freeze/pause, dwell click
- `control.py` UDP channel for the EEG app
- `setup_check.py` one-time setup and diagnostics
- `diag.py` logging (terminal + `logs/`)

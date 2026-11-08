# Universal Computer Controller

NC State Neurotech Fall 2026 Hackathon, ANT Neuro track, Team 5.

Hands-free computer control from an ANT Neuro eego 24-channel EEG amplifier. Facial, eye and head gestures picked
up in the EEG become clicks, cursor movement and shortcuts, so someone who cannot use a mouse or keyboard can still
operate a normal computer. Each session starts with a short calibration that fits the model to the person wearing
the cap. A webcam eye cursor and an SSVEP input are optional extras.

## Controls (default mapping)

| Gesture | Action |
|---|---|
| Head turn left / right / up | move the cursor (a short turn aims, holding it speeds up) |
| Left wink | move the cursor down |
| Jaw clench | left click; keep clenching to hold the button (drag) |
| Double blink | right click |
| Right wink | Enter |
| Eyebrow raise | open / close the quick bar (scroll, copy, paste, undo, switch app, Enter, Escape, pause) |

Every gesture can be remapped in the dashboard (Mapping tab). The optional webcam eye cursor moves the pointer
with your gaze, and the optional SSVEP squares add four more inputs.

## How it works

```
eego amp --LSL--> dashboard --> team model (src/) --> gesture detector --> mouse / keyboard actions
                     |              ^
                     |              '-- profile fitted per session ("Start session", ~10 min of cued gestures)
                     '-- or: simulated EEG (recorded trials injected live) when there is no headset
```

1. **Data collection** (`src/collect.py`): a cue app records each gesture plus "confounders" (talking, chewing,
   typing, idle, natural blinking) that must never trigger anything.
2. **Training** (`src/train.py`): causal filtering, 0.5 s windows every 0.125 s, features, and a threshold baseline,
   logistic regression or random forest, whichever scores best on held-out data. False triggers are penalised.
3. **Live control** (`controller/`): the dashboard streams the EEG through the same pipeline, fires gestures with
   tuned thresholds and persistence, and performs the mapped actions.

## Quick start (Windows, Python 3.13 or newer)

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-controller.txt
controller.bat
```

| Goal | Command |
|---|---|
| Live session: connect to the amp, click **Start session** to calibrate | `controller.bat` |
| Load a saved profile and skip calibration | `controller.bat --model team:session` |
| No headset: inject recorded EEG from the side panel, through the model | `controller_sim.bat` |
| Rehearsal with synthetic EEG and test buttons (no model) | `controller.bat --model sim` |
| Add the webcam eye cursor | `controller.bat --gaze-profile nico` (setup in `gaze/README.md`) |
| All options | `controller.bat --help` |

**The controller performs real clicks and key presses**, in simulation too. Ctrl+Alt+G pauses and resumes it from
any window, Ctrl+Alt+E stops the eye cursor and Ctrl+Alt+S turns SSVEP off.

**Epilepsy warning:** the optional SSVEP input flashes squares at 8-15 Hz. Do not use it if you or anyone in your
family has epilepsy or seizures, or if you are sensitive to flashing lights. It never starts without accepting a
warning.

## Repository

| Folder | What | Docs |
|---|---|---|
| `controller/` | the dashboard app: EEG input, session calibration, detection, actions, quick bar, simulator, SSVEP | [controller/README.md](controller/README.md) |
| `src/` | the EEG model: cue app, signal check, preprocessing, features, training, real-time engine | [src/README.md](src/README.md) |
| `gaze/` | webcam eye-tracking cursor (EyeGestures v4 method, MediaPipe) | [gaze/README.md](gaze/README.md) |
| `data/` | the recordings from the ANT station | [data/README.md](data/README.md) |
| `config/` | channels, sessions and classes (`config.yaml`), gesture mapping for `src/realtime.py` (`mapping.yaml`) | |
| `tests/` | `.venv\Scripts\python -m pytest -q` | |

`profiles/` (trained models) and `logs/` are created locally and are not committed.

# Universal Computer Controller (app)

Turns EEG gestures into computer actions. No hands needed.

## Default controls
| gesture | action |
|---|---|
| Head turn up / down / left / right | move the cursor (a short turn aims, holding it speeds up) |
| Eyebrow raise | open / close the quick bar |
| Jaw clench | left click; keep clenching to hold the button down (drag) |
| Double blink | right click |
| Left wink, right wink | not assigned (free to map) |

The webcam eye cursor (`gaze/`) is optional: add `--gaze-profile NAME`. With it on, the eyes move the cursor
and head turns fine-tune it (the eye cursor holds still while the head moves it).

## Install (once, with internet)
```
.venv\Scripts\python -m pip install -r requirements-controller.txt
```

## Run
```
.venv\Scripts\python -m controller --model ml/model.py:predict
```
| option | what it does |
|---|---|
| `--model file.py:function` | the gesture model (see below). Without it: rehearsal mode, gestures come from the test buttons |
| `--eeg sim` | simulated EEG (default when there is no model); otherwise the live LSL stream |
| `--stream-name NAME` | pick the LSL stream by name (default: the first stream of type EEG) |
| `--gaze-profile NAME` | also run the eye cursor with this profile (`python -m gaze.app --list-profiles`) |
| `--gaze` / `--gaze-running` | also run the eye cursor and calibrate first / use one that is already running |
| `--mapping game` | gesture mapping (`controller/mappings/*.json`), also switchable in the app |
| `--threshold 0.6 --k 3 --n 4 --refractory 0.8` | detection settings, also adjustable in the app |

**Ctrl+Alt+G** (any window) pauses / resumes gesture actions. The eye cursor has its own Ctrl+Alt keys
(P pause, C recenter, V accuracy check, F apply correction, Q quit).

## Plugging in the gesture model (ML side)
Write a function; the controller calls it every `STEP_S` seconds with the latest `WINDOW_S` seconds of EEG:
```python
WINDOW_S = 1.0     # optional, default 1.0
STEP_S = 0.1       # optional, default 0.1

def predict(window, fs=None, channels=None):
    # window: numpy array (channels, samples), raw units from the stream
    # fs, channels: only passed if your function has those parameters
    return {"rest": 0.1, "jaw_clench": 0.83, ...}   # or ("jaw_clench", 0.83) or just "jaw_clench"
```
`controller/example_model.py` is a working template (it only detects jaw clenches with a simple threshold).
A crashing model is logged and treated as "rest"; it never stops the app.

Gesture names the default mapping knows: `head_turn_up, head_turn_down, head_turn_left, head_turn_right,
eyebrow_raise, jaw_clench, double_blink, left_wink, right_wink` (and `rest`). Other names work too: map them
in the Mapping tab.

## How a gesture becomes an action
1. The model's probabilities go to the detector: a gesture **fires** when it is the top label with confidence
   at least 0.6 in 3 of the last 4 windows, and not within 0.8 s of firing before. It stays **held** while it
   keeps winning (for hold actions such as game keys).
2. The mapping turns it into an action. With the eye cursor on, it is frozen for 0.8 s (face gestures nudge
   the gaze estimate) and clicks land where it was 0.5 s before the gesture.

Action strings (Mapping tab or `mappings/*.json`):
- `move:up|down|left|right` cursor moves while held; speed ramps from `cursor_speed` to `cursor_max_speed`
  (px/s) over `cursor_accel_s` seconds (set in the mapping file)
- `mouse:left|right` short = click, held past `hold_s` = button held down until the gesture ends (drag)
- `click:left|right|double`, `key:space`, `hotkey:ctrl+c`, `scroll:5` / `scroll:-5`, `hold:w` (key held while
  the gesture is), `type:text`, `menu`, `eye:recenter|check|fix|toggle`, `controller:pause`, `none`

## Quick bar
The quick bar gesture (eyebrow raise) opens a row of big tiles centered 25% up from the bottom of the screen:
Scroll up, Scroll down, Copy, Paste, Undo, Switch app, Enter, Escape, Pause controller. Move the cursor onto
a tile and clench to run it. Eyebrow raise again, Esc, or 15 s closes it. It never takes keyboard focus, so
Copy, Enter and so on go to the app you were using. Tiles are defined per mapping (`"menu"` in the JSON).

## Mappings
- `default.json`: the no-hands controls above
- `game.json`: head turns hold W/A/S/D, jaw clench is the left mouse button, double blink right-clicks,
  DirectInput keys (most games need them)
Edit in the Mapping tab (changes apply immediately), then Save or Save as new mapping.

## The window
- **Live**: the EEG as a scrolling chart, with every detected gesture stamped in red where it happened;
  the model's confidence per gesture with the firing threshold; the last action and a log.
- **Mapping and test**: the mapping table, buttons that try each gesture (in rehearsal mode they run the whole
  pipeline; with a real model they run the mapped action; tick "Hold for 2 s" to try cursor movement or a
  drag), detection settings, eye cursor buttons.
- **Log**: everything the app logs; also saved to `controller/logs/`.
- A small overlay at the top of the screen shows what was just detected and done (clicks pass through it).

# Controller (dashboard app)

Turns EEG gestures into computer actions. Install once with
`.venv\Scripts\python -m pip install -r requirements-controller.txt`, then run `controller.bat`
(same as `.venv\Scripts\python -m controller`; extra flags pass through).

## Starting a session

The dashboard opens on the start screen. Nothing is detected or sent to the OS until a model is loaded.

1. **EEG source**: by default the dashboard connects to the amp's LSL stream. Start with `--no-eeg` to choose
   **LSL stream** or **Simulated** in the header instead.
2. **Start session**: pick a duration (2 to 30 minutes, default 10). The team's cue app (`src/collect.py`,
   signal check first) records the cued gestures into `data/s5_<time>/`.
3. The model is fitted to this person (`src/train.py`: adapts `profiles/default.joblib` if it exists, otherwise
   trains from this session) and saved as `profiles/session.joblib`.
4. The live view opens and listening starts, with the profile's tuned thresholds and persistence.

**Use last calibration** reloads the previous `profiles/session.joblib`. **Cancel calibration** also closes the
cue app. **Recalibrate** (header, live EEG) re-records only the gestures you tick and refits the model; the
previous model is kept as `profiles/session_previous.*`.

## Options

| Flag | What it does |
|---|---|
| `--model team` (default) | the team model, calibrated per session from the start screen |
| `--model team:NAME` | load `profiles/NAME.joblib` and skip the start screen |
| `--model sim` | rehearsal: synthetic EEG, gestures come from test buttons |
| `--model file.py:function` | any other model (see "Plugging in a model") |
| `--eeg lsl / sim / inject` | choose the EEG source up front instead of in the header |
| `--no-eeg` | start without a source and choose it in the dashboard |
| `--stream-name NAME`, `--stream-type TYPE` | pick the LSL stream |
| `--eeg-file FILE [--eeg-start S]` | replay a recording as if it were live |
| `--sim-data SPEC ...`, `--no-forge` | simulator sources and exact replay (see below) |
| `--mapping NAME` | gesture mapping from `mappings/` (also switchable in the app) |
| `--gaze-profile NAME`, `--gaze`, `--gaze-running` | add the webcam eye cursor (load a profile / calibrate first / use one already running) |
| `--threshold`, `--k`, `--n`, `--refractory` | override the detection settings (also adjustable in the app) |
| `--no-overlay` | hide the on-screen event overlay |

Hotkeys (any window): **Ctrl+Alt+G** pause / resume gesture actions, **Ctrl+Alt+E** stop the eye cursor,
**Ctrl+Alt+S** turn SSVEP off.

## Simulated EEG (no headset)

`controller_sim.bat` (`--eeg inject`), or **Simulated** in the header, streams recorded EEG into the dashboard as
if it were the live stream. Recordings are pooled from every session in `data/` with cue markers (calibrations
and training sessions; a folder without its own EEG is paired with the `data/*.cnt` that was recording at the
time), or chosen with `--sim-data FOLDER[@EEGFILE] ...`.

- The background is the wearer's recorded rest.
- The **Simulate EEG** side panel injects a recorded trial of any gesture or confounder when you press it;
  holding the button keeps a held gesture going (clench to drag, head turn to keep moving).
- **Forge new variants** (default on, `--no-forge` to turn off) gives each injection its own per-channel gain and
  another rest trial's activity, so the model never sees the exact recorded samples.
- Injected samples go through the model exactly like live EEG. **Start session** trains on the recordings
  instead of running the cue app. Training and injecting from the same recordings flatters the accuracy.

## SSVEP (optional)

**Epilepsy warning:** four corner squares flash at 8, 10, 12 and 15 Hz. Do not use it if you or anyone in your
family has epilepsy or seizures, or if you are sensitive to flashing lights. The warning is shown before
calibration and every time SSVEP is switched on.

Looking at a square makes the occipital channels oscillate at its frequency; the controller scores each square
(CCA at f and 2f) and fires the one you look at, alongside the gesture model.
1. **Calibrate** (about 3 minutes, once per session): **SSVEP calibration (optional)** on the start screen, or
   `.venv\Scripts\python.exe -m controller.ssvep.calibrate` (`--synthetic` to rehearse). It is saved to
   `profiles\ssvep.json` only if it passes its accuracy and false-trigger checks.
2. **Enable** with the **SSVEP** header button (live headset only). Ctrl+Alt+S turns it off.
3. The squares appear in the mapping table as "SSVEP top left" and so on. Recalibrate if the screen's refresh
   rate changes.

## Mappings and actions

`mappings/default.json` is the no-hands layout (see the main README); `mappings/game.json` is a game layout
with DirectInput keys. Edit in the **Mapping and test** tab (changes apply at once), then Save or Save as new.

Action strings: `move:up|down|left|right` (cursor moves while held, speeding up from `cursor_speed` to
`cursor_max_speed` over `cursor_accel_s`), `mouse:left|right` (short = click, held past `hold_s` = drag),
`click:left|right|double`, `key:enter`, `hotkey:ctrl+c`, `scroll:5`, `hold:w` (key held while the gesture is),
`type:text`, `menu`, `eye:recenter|check|fix|toggle`, `controller:pause`, `none`.

**Quick bar** (`menu`): a row of large tiles 25% up from the bottom of the screen (scroll up/down, copy, paste,
undo, switch app, Enter, Escape, pause). Point at a tile and clench to run it; the same gesture, Esc or 15 s
closes it. It never takes keyboard focus, so keys go to the app you were using. Tiles are set per mapping.

With the eye cursor on, it holds still for 0.8 s at each gesture and clicks land where it was 0.5 s before.

## The window

- **Live**: scrolling EEG with each detected gesture marked, confidence per gesture against its threshold,
  the last action and recent events.
- **Mapping and test**: the mapping table, buttons that try each gesture ("Hold for 2 s" for movement or drag),
  detection settings, eye cursor controls.
- **Log**: everything the app logs, also saved to `controller/logs/`.
- A click-through overlay at the top of the screen shows what was just detected and done.

## Plugging in a model

Any function works with `--model file.py:function`; it is called every `STEP_S` seconds with the latest
`WINDOW_S` seconds of EEG:
```python
WINDOW_S = 1.0     # optional, default 1.0
STEP_S = 0.1       # optional, default 0.1

def predict(window, fs=None, channels=None):
    # window: numpy array (channels, samples), raw stream units
    return {"rest": 0.1, "jaw_clench": 0.83}   # or ("jaw_clench", 0.83) or "jaw_clench"
```
`example_model.py` is a template. A crashing model is logged and treated as rest. The team model is wired in
through `ant_model.py`, which streams every sample into `src/detector.py` and maps its class names
(`head_left` -> `head_turn_left`, `confounder` -> `rest`).

## Files

| File | What |
|---|---|
| `app.py` | entry point, options, wiring |
| `eeg_source.py` | LSL, replay and synthetic sources, channel detection |
| `ant_model.py` | adapter for the team model |
| `session_calibrate.py` | the Start session pipeline (cue app, then training) |
| `sim_inject.py` | simulated EEG from recordings |
| `engine.py`, `detector.py`, `actions.py` | live loop, firing logic, OS actions |
| `gaze_link.py` | starts and talks to the eye cursor (`gaze/`) |
| `ssvep/` | SSVEP calibration, detection and flicker |
| `ui/` | dashboard, start screen, simulator panel, quick bar, overlay, theme |
| `mappings/` | gesture mappings |

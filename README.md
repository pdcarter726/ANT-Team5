# ANT-Team5
Neurotech at NC State Fall Hackathon, ANT Neuro Track, Team 5

Hands-free computer control from an ANT Neuro eego EEG amplifier (facial / eye / head gestures
for clicks, shortcuts and cursor control), with a webcam head-tracking cursor as backup.

## Setup (Windows 11, Python 3.14)
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install pylsl pyxdf mne antio scikit-learn scipy numpy pyautogui pynput opencv-python mediapipe pyyaml pytest
```
All of these install on Python 3.14 (mediapipe 1.2.2 included). The cue UI uses Tkinter (pygame does not build on 3.14).
All commands below are run from the repo root and use `.venv\Scripts\python.exe` (shortened to `python`).

## Config
`config/config.yaml` holds the channel list, sampling rate, channel groups, class catalogue and per-session
settings. Channel names are PLACEHOLDERS until the amp is seen: `collect.py` and `signal_check.py` print the
real stream info (name, channels, rate, labels, unit) on startup, then copy those into the config.
Set `eeg_stream.units_to_uv` if the stream is not already in microvolts.

## Stage 1 - synthetic source and collection

| Script | Command |
|---|---|
| Synthetic 24-ch EEG (follows CueMarkers, injects the cued gesture) | `python src/synthetic_eeg.py` |
| Synthetic, cycling all gestures (for signal-check testing) | `python src/synthetic_eeg.py --demo` |
| Live signal check only | `python src/signal_check.py` |
| Session 1 duration estimate + schedule preview | `python src/collect.py --session 1 --dry-run` |
| Collect session 1 | `python src/collect.py --session 1 --subject S01 --cap "24ch cap, gel" --notes "..."` |
| Cut trials if over budget | `python src/collect.py --session 1 --trials 20` |
| UI only (no EEG stream) | `python src/collect.py --session 1 --no-eeg` |
| Tests | `python -m pytest -q` |

Collect flow: signal-check screen (Enter to continue) -> gate screen (start LabRecorder with the EEG stream
AND `CueMarkers` ticked, then Enter) -> countdown -> trials -> DONE. Keys: SPACE pause (immediate in the rest
interval, otherwise after the current trial; also continues a break), ESC twice abort, F11 fullscreen.
Signal-check keys: `+`/`-` scale, `c` common-average toggle, left/right pick the band-power channel.

Output folder `data/s<N>_<timestamp>/`: `markers.csv` (wall-clock + LSL time for every event),
`eeg_backup.npz` (raw samples + timestamps; partial `eeg_part_*.npz` are flushed every 60 s so a crash loses
at most a minute), `metadata.json` (session id, notes, cap comments, config used, trial list, stream info).
Save LabRecorder's `.xdf` into the same folder. Marker events: `cue_on`, `cue_off`, `session_start/end`,
`pause/resume`, `break_start/end`; LSL markers are JSON strings (`event, session, trial, block, label, cue_s`).

Launchers (extra flags pass through): `collect_trials25.bat` = full session 1, 25 trials per class (about 24 min,
25 min cue-only); `collect_trials10.bat` = short session 1, 10 trials per class (about 10 min, 10.3 min cue-only).
Example: `collect_trials10.bat --cue-only --subject S01`. Any other count: `--trials N`.

### Using an eego `.cnt` recording
`antio` (needed by MNE to read ANT `.cnt`) is in the setup line above. The loader accepts `.cnt` (read in microvolts via
MNE; the companion `.evt` holds only impedance checks and is ignored), `.xdf` and `.npz`. Keep the cue-only session
folder (it has `markers.csv`) and the EEG file anywhere; point at both:
```
.venv\Scripts\python.exe src/load_data.py data\s1_<timestamp> --eeg data\Session1Recording.cnt
.venv\Scripts\python.exe src/feasibility.py data\s1_<timestamp> --eeg data\Session1Recording.cnt
```
(or copy the `.cnt` into the session folder and drop `--eeg`). The `.cnt` stores its own start time; if it agrees
with the sync-clench estimate to within 0.5 s the clock-based start is used (more precise), otherwise the sync result.
Channel names are matched case-insensitively (the amp reports `t7`, `cz`, `oz` in lower case).

### Cue-only mode (EEG is recorded on a separate device)

```
.venv\Scripts\python.exe src/collect.py --session 1 --cue-only --subject S01 --notes "..."
```
This PC only shows the cues: no EEG stream, no LSL needed. Steps: (1) start recording on the other device,
(2) press ENTER here, (3) clench the jaw hard on each of 4 white flashes at the start (after the 5 s countdown) and
again at the end, (4) follow the cues, (5) stop the other recording and copy its file (`.xdf`, or `eeg_backup.npz`
saved in this project's format) into the printed `data\s1_<timestamp>\` folder.
Every event is logged in `markers.csv` with wall-clock time and `t_rel` (seconds since ENTER). The loader finds the
sync clenches in the EEG (30-100 Hz power) to work out where the session started in the other device's clock, and
corrects linear clock drift using the end sync. Accuracy is about +/-0.15 s (reaction time to the flashes), inside
the 0.25 s skip after each cue used by the windowing. `python src/load_data.py <folder>` prints the offset, drift
and a confidence score (warns if low). If the sync fails or you know the offset, give it explicitly:
`python src/load_data.py <folder> <offset_seconds>` or `python src/feasibility.py <folder> --marker-offset <s>`
(EEG time = t_rel + offset). Add `--lsl-markers` to also broadcast `CueMarkers` on LSL if the other device runs
LabRecorder on the same network (then the XDF has markers and no alignment is needed).
The session is about 0.7 min longer than a normal run (two sync blocks of 16.5 s plus a 4 s pause after the first; flashes last 1 s, 3-4 s apart, with a live "next flash in N" countdown). Fake data for testing:
`python src/make_synthetic_session.py --session 1 --out data\synth_cue --cue-only --xdf --drift-ppm 800`.

## Stage 2 - feasibility analysis (run between sessions)

| Script | Command |
|---|---|
| Inspect a recording (channel names, rate, class counts, marker alignment notes) | `python src/load_data.py data\s1_<timestamp>` (folder, `.xdf` or `.npz`) |
| Feasibility analysis (LR + random forest, ~15 s) | `python src/feasibility.py data\s1_<timestamp>` |
| Faster, LR only | `python src/feasibility.py data\s1_<timestamp> --quick` |
| Tune windowing / thresholds | `--win 0.5 --step 0.25 --offset 0.25 --folds 5 --keep-recall 0.7 --keep-precision 0.7` |
| Make a full fake session offline (npz + csv + json, optional xdf), no LSL needed | `python src/make_synthetic_session.py --session 1 --trials 25 --out data\synth_s1 --xdf --difficulty normal` (`easy`, `normal`, `hard`) |

The loader prefers an `.xdf` in the session folder (LabRecorder output) and falls back to `eeg_backup.npz` +
`markers.csv`, then to unmerged `eeg_part_*.npz` after a crash. If both an XDF marker stream and `markers.csv`
exist they are compared and the worst timestamp difference is printed (warns above 50 ms).

`feasibility.py` writes `<session>\analysis\` with `feasibility_report.txt`, `results.json` and `keep_list.yaml`.
Copy the `keep: [...]` line into `config\config.yaml` under `sessions: 2:`. It reports per-class precision/recall
(trial level, plus window-level recall), the confusion matrix, confused pairs, the rest false-trigger rate, and a
head-vs-eye direction section including whether each axis is usable for cursor velocity. Cross-validation is
time-ordered: contiguous blocks of trials, all windows of a trial stay in the same fold.
Preprocessing (`src/preprocess.py`) is causal (CAR, 60 Hz notch, 1-40 / 0.1-10 / 30-100 Hz branches) so the
real-time loop sees the same features as the analysis. The config `left`/`right` channel lists must be in
matching order (Fp1<->Fp2, F7<->F8, ...) because asymmetry features pair them by position.

Testing with the synthetic stream: terminal 1 `python src/synthetic_eeg.py`, terminal 2
`python src/collect.py --session 1 --trials 2 --rest-s 0.8 --auto-start --skip-check`.

## Stage 3 - session 2 (keep list + confounders)

| Step | Command |
|---|---|
| Duration estimate and schedule | `.venv\Scripts\python.exe src/collect.py --session 2 --cue-only --dry-run` |
| Collect session 2 | `.venv\Scripts\python.exe src/collect.py --session 2 --cue-only --subject S01 --notes "..."` (drop `--cue-only` if the EEG stream is on this PC) |
| Analyse (same script, confounders are detected from the config) | `.venv\Scripts\python.exe src/feasibility.py data\s2_<timestamp> --eeg <file>.cnt` |
| Fake session 2 for testing | `.venv\Scripts\python.exe src/make_synthetic_session.py --session 2 --out data\synth_s2 --cue-only` |

`config\config.yaml` -> `sessions: 2:` holds the `keep:` list (taken from the session-1 analysis; `head_left` was added by
hand, `look_*` dropped) and the confounders (talking, chewing, natural_blinking, typing, idle, 6 s each, 10 trials each,
spread evenly across the blocks). Defaults: 30 trials per class (50 would take 41 min with 10 classes; the budget is
30 min), 28.3 min in cue-only mode. Change `trials_per_class` or `--trials N`; the app warns when over budget.
Do the confounders naturally: talk about anything, chew gum, just look at the screen and blink, type, idle.
`feasibility.py` trains the confounders as a reject class, leaves them out of the ranking and keep list, and prints a
"False triggers on confounders" table: percent of windows predicted as a gesture, and stretches of 3+ consecutive
identical predictions per minute (what a persistence rule would still let through).

## Stage 4 - training and real-time control

All commands use `.venv\Scripts\python.exe` (shortened to `python`). Profiles are saved to `profiles\` (git-ignored), logs to `logs\`.

| Step | Command |
|---|---|
| Train (compares threshold baseline, logistic regression, random forest; saves the best) | `python src/train.py --data data\s2_<timestamp>@data\Session2Recording.cnt --name default` |
| Train with extra sessions (training only) | `python src/train.py --data S2FOLDER@S2.cnt --extra S1FOLDER@S1.cnt --name default` |
| Faster, one model | `python src/train.py --data ... --models rf` |
| Replay a recording through the real-time engine (dry run, scored against the cues) | `python src/realtime.py --profile default --replay data\s1_<timestamp> --eeg data\Session1Recording.cnt --fast --no-ui` |
| Same, paced like real time with the overlay | `python src/realtime.py --profile default --replay data\s1_<timestamp> --eeg data\Session1Recording.cnt --speed 1` |
| Replay only the last 20% | add `--tail 0.2` |
| Live EEG, log only (nothing sent to the OS) | `realtime_dry.bat` or `python src/realtime.py --profile default --dry-run` |
| Live EEG, REAL mouse/keyboard | `realtime_live.bat` or `python src/realtime.py --profile default` |
| 2-minute calibration (live EEG stream), then use the new profile | `calibrate.bat` then `realtime_live.bat --profile calibrated` |
| Calibrate from an existing recording | `python src/calibrate.py --from-folder data\s3_<timestamp> --base default --name calibrated` |
| Fake EEG stream to exercise the live path | `python src/synthetic_eeg.py --demo` and, in another terminal, `realtime_dry.bat --no-ui --max-seconds 20` |

`@` separates the session folder from its EEG file (the folder holds `markers.csv`; the `.cnt` can live anywhere).

**Training split.** The main session is split in time order, never shuffled: first 60% of trials train, next 20% validation
(thresholds, persistence and per-class thresholds are tuned here by replaying the validation segment through the real event
logic), last 20% TEST (reported, never tuned on). Models are compared on false triggers per minute as well as recall. The
objective is `recall - 0.5 * false_per_min - 0.5 * wrong_class_per_trial`, so false triggers are penalised hard. The gaps
between trials (head returning to centre etc.), confounder cues and look_* cues are trained as a `confounder` reject class.
The chosen model is then refit on all data. Outputs: `profiles\<name>.joblib` (model, thresholds, cursor estimators, channel
list), `.json` summary, `_train.npz` (features, for calibration), `_report.txt`.

**Event logic** (`src/detector.py`, identical offline and live): 0.5 s windows every 0.125 s; a gesture fires when it is the
arg-max, above its probability threshold, for N consecutive windows (persistence); a held gesture fires once (needs 2
windows without it to re-arm), with a per-class refractory period and a 0.5 s global lockout. Preprocessing is causal and shared
with Stage 2.

**Mapping** (`config\mapping.yaml`, edit and restart): gesture -> event -> OS action (click, key, hotkey, scroll, pause_toggle,
cursor), per-event overrides (`min_prob`, `hold`, `refractory_s`), cursor settings (speed, dead zone 0.3, smoothing, 300 ms
click freeze) and safety settings. A typo is reported with the offending name before anything touches the OS.
Defaults: left wink = left click, right wink = right click, double blink = double click, eyebrows = Enter,
jaw clench (held) = pause/lock toggle, head left/right/up/down = cursor. Tab, space, alt-tab, copy/paste, scroll etc. are defined and can be
assigned to any gesture.

**Safety.** Global kill hotkey Ctrl+Alt+Q (pynput, works from any window), global pause Ctrl+Alt+P, pause/lock by jaw clench
(while paused only that gesture is listened to), event-rate limiter (more than 40 events/min pauses control), 6 s warm-up,
refusal to run if stream channels/rate differ from the profile. The overlay (top-right, click-through, never takes focus) shows
ACTIVE (green) / PAUSED (red) / DRY RUN (blue), the last event and the live class probabilities. Replays are dry runs
unless you pass `--live-actions`. Windows DPI awareness is enabled before any cursor coordinates are used.

**Calibration** (`calibrate.py`): runs `collect.py --session 3` (3 trials per gesture + short talking and natural blinking, about 2.1 min,
live EEG so no sync clenches), then refits the base profile with the new windows up-weighted (x2). Thresholds are kept from the base profile.
Needs the EEG as an LSL stream on this PC.

## Model test: `python modeltest`

One command: about 5 minutes of calibration on the live EEG stream, then it listens and writes every detected gesture.
```
python modeltest                         # works with the plain `python` on PATH (it relaunches itself with .venv)
python modeltest --skip-calibration      # listen right away with the trained base profile
python modeltest --synthetic --collect-args --trials 1   # pipeline check with a fake EEG stream (not a model check)
```
Needs a trained base profile (`profiles\default.joblib`, see Stage 4) and the EEG as an LSL stream on this PC.
Steps: find the stream (clear error if none) -> signal check -> calibration cues (config `sessions: 4`, 4.9 min: 6 trials per gesture,
2 of each confounder: talking, typing, chewing, idle, natural blinking) -> refit into `profiles\modeltest.joblib` (base untouched) ->
listen. Listening is a dry run: nothing is sent to the mouse or keyboard. Every detected gesture (including head directions and
unmapped gestures) is printed and appended to `logs\modeltest_<time>_events.txt`, the overlay shows the live class probabilities,
and the full CSV log goes to `logs\rt_<time>.csv`. Stop with Ctrl+Alt+Q or Ctrl+C; a per-gesture count is printed at the end.
Options: `--base NAME`, `--name NAME`, `--weight 2.0` (calibration weight), `--no-ui`, `--max-seconds N`, `--events-file PATH`.

**Quick mode:** `python modeltest --quick` runs a 1 minute calibration (config `sessions: 6`: 2 trials per gesture, no confounders and no `rest` cue,
the gaps between trials supply the rest data; weight 4 instead of 2 because there is less data) and then listens exactly as above.
`calibrate.py --session 6` does the same without the listening step.

## Headsets and channel names

The current cap (config `channels`) is `Fp1 FP2 F9 F7 F3 Fz F4 F8 F10 T9 T7 C3 C4 T8 T10 P9 P7 P3 Pz P4 P8 P10 O1 O2`. The amp's LSL
stream has 25 channels with no labels and the 25th is all zeros: the first 24 get those names and the extra one is ignored
(`src/channels.py`). Models are trained on a fixed set of 22 canonical channels (`model.channels`); everything else (cz, oz,
P9, P10, the zero channel) is dropped before preprocessing, in training, replay and live use alike. The first cap's
Session1/2 `.cnt` files (M1, M2, cz, oz) still work: `model.aliases` treats M1/M2 as T9/T10, an ASSUMPTION (both sit near the ears).
After any change to `model:` or `channel_groups:` retrain (`train.py`), because feature vectors change. A stream whose channels do
not match the profile is refused with the two name lists printed. The live loop shows "NO EEG DATA for N s" when the stream stalls.

**"No LSL stream found":** run `.venv\Scripts\python.exe src\lsl_scan.py` (add `--watch` to see streams appear/disappear). It lists every
LSL stream this PC can see, any type or host; `python modeltest` prints the same list when it fails. The amp's stream has come from a
different PC than the one running this code, so the usual causes are network (same subnet, no guest Wi-Fi) and Windows Firewall on both machines.

## Channel auto-detection (dashboard and scripts)

The dashboard (`python -m controller ...`) no longer needs the channel list to be set by hand. On connecting to the LSL stream it
reads about 2 s of data and works out the layout (`controller/eeg_source.py`, `src/channels.py`):
- stream **with labels** (any headset): the labels are used; older names map through `model.aliases` (M1/M2 -> T9/T10);
- stream **without labels** (the amp): the first channels are named from `channels:` in `config/config.yaml`; extra trailing channels
  (the amp's all-zero 25th) are reported as "not running" and ignored;
- channels that are **not running** (all zero, flat, NaN) are detected at start-up and again every tick: the EEG status chip shows
  e.g. `live, 500 Hz, 24 ch` and turns red with `FLAT: T7` if an electrode loses contact;
- the model picks the channels it was trained on (the profile's own `ch_names`), in its own order. If the stream lacks some, or one of them is
  not running, starting the model fails with the names listed. If the headset really lacks channels, train for what it has:
  `python src/train.py --data S2FOLDER@S2.cnt --channels Fp1,Fp2,F9,F7,F3,Fz,F4,F8,F10,T7,C3,C4,T8,P7,P3,Pz,P4,P8,O1,O2 --name subset`
  (any subset of the 22 canonical channels; the profile then selects exactly those).
`python src/realtime.py` does the same detection before it starts listening. Everything is logged in the controller log lines tagged `EEG`.

## Running the controller (dashboard)

Install once: `.venv\Scripts\python.exe -m pip install -r requirements-controller.txt` (PySide6, pyqtgraph, ...; works on Python 3.14).

`controller.bat` is the same as `.venv\Scripts\python.exe -m controller`. Plain `python -m controller` also works: it relaunches itself with `.venv`. Add any flag after it.

| Goal | Command |
|---|---|
| Real session: live EEG from the amp, calibrate with "Start session" | `controller.bat` |
| Same, skip the start screen and load a saved profile | `controller.bat --model team:session` |
| Choose the EEG source (LSL or Simulated) in the dashboard | `controller.bat --no-eeg` |
| Simulator as a headset replacement (side panel makes the EEG, the trained model reads it) | `controller.bat --eeg inject` |
| Simulator with a specific profile | `controller.bat --eeg inject --model team:default` |
| Simulator, replay recorded trials exactly (no variation) | `controller.bat --eeg inject --no-forge` |
| Simulator from chosen recordings | `controller.bat --eeg inject --sim-data data\s2_20261107_162913@data\Carter1_Peter_2026-11-07_16-33-59.cnt` |
| Rehearsal with synthetic EEG and test buttons (no model) | `controller.bat --model sim` |
| Replay a recording as if live | `controller.bat --model team:default --eeg-file data\Session2Recording.cnt` |
| Pick an LSL stream by name | `controller.bat --stream-name eego` |
| Game controls (WASD) | `controller.bat --mapping game` |
| Add the webcam eye cursor | `controller.bat --gaze-profile nico` |
| Hide the on-screen event overlay | `controller.bat --no-overlay` |

All options: `controller.bat --help`. More in `controller\README.md`.

WARNING: the controller performs REAL clicks and key presses, replay and simulation included. Pause or resume with Ctrl+Alt+G; Ctrl+Alt+E stops the eye cursor; Ctrl+Alt+S switches SSVEP off.

### SSVEP (optional, flashing squares as a second input)

**EPILEPSY WARNING.** SSVEP flashes four squares at 8-15 Hz. Do not use it if you or anyone in your family has epilepsy or seizures, or if you are sensitive to flashing lights. Stop at once if you feel unwell. The warning is shown before calibration and again every time SSVEP is switched on; nothing flashes unless it is accepted.

How it works: four squares sit in the screen corners (8, 10, 12 and 15 Hz). Looking at one makes the occipital electrodes (O1, O2, Pz, P3, P4, P7, P8) oscillate at its frequency; the controller scores each square (CCA against sines at f and 2f) and fires the one you look at. It reads the same LSL stream as the gesture detector and runs beside it; the gesture model is untouched.

1. **Calibrate (once per session, about 3 minutes):** on the start screen click **SSVEP calibration (optional)...**, or run `.venv\Scripts\python.exe -m controller.ssvep.calibrate`. Accept the warning, then look at the red-ringed square for 8 s per trial, or at the screen centre on "centre" trials (these measure your no-choice level, which keeps false triggers down). Esc stops. The result is saved to `profiles\ssvep.json` only if it passes its checks (right square fires in at least 75% of trials, every square at least 50%, at most 1 accidental event per minute while looking away). A failed calibration keeps the old one and says why. Rehearsal without a headset: add `--synthetic`. Options: `--blocks 3` (5 trials per block), `--stream-name`.
2. **Enable:** the **SSVEP** button in the dashboard header is available once calibrated and with the live headset (not simulated or replayed EEG). It shows the warning, then the four squares appear in the corners. Click it again to turn off, or press **Ctrl+Alt+S** from any window.
3. **Actions:** the squares appear in the mapping table as "SSVEP top left" etc. (default mapping: scroll up, scroll down, back, Enter; game mapping: hold W, D, A, S). Change them like any other gesture. `hold:` and `move:` actions stay on while you keep looking at the square. The mapping name of each square is `ssvep_top_left`, `ssvep_top_right`, `ssvep_bottom_left`, `ssvep_bottom_right`.

The screen's refresh rate is stored with the calibration; if it changes (another monitor), recalibrate. Use a 60 Hz or faster screen, a well-lit room and take breaks.

### Recalibrating specific gestures

After a full calibration, improve only the gestures that are not detected well, without redoing everything (about 2-3 minutes):

- Dashboard: click **Recalibrate** (top bar, live EEG only), tick the gestures, pick the repetitions (default 8), **Start recalibration**. Gesture actions are paused while the cues run and restored afterwards. "Forget the old recordings of these gestures" drops the earlier recordings of the ticked gestures; leave it off to keep them (the new ones count more either way). Also available on the start screen.
- Command line (same effect; `session` is the profile the dashboard uses):
  `.venv\Scripts\python.exe src\calibrate.py --base session --name session --only jaw_clench,double_blink`
  Extra options: `--trials 10`, `--class-weight 4`, `--replace`.
- The re-recorded gestures' new windows are weighted x4 in the refit, the other gestures keep what they had, and the thresholds are kept. From the dashboard, the previous model is saved as `profiles\session_previous.*` (copy over `session.*` to undo).
- Cue-only (e.g. to record now and train later): `.venv\Scripts\python.exe src\collect.py --session 7 --classes jaw_clench,double_blink`.

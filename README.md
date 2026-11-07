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

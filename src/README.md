# EEG model (src/)

Data collection, analysis, training and the real-time engine for the gesture model. The dashboard
(`controller/`) uses this code for its session calibration and live detection; the scripts here also run on
their own. Run everything from the repo root with `.venv\Scripts\python.exe` (shortened to `python` below).

Setup: `python -m pip install -r requirements-controller.txt` (includes pylsl, pyxdf, mne, antio, scikit-learn,
pyyaml). `pytest` for the tests. The cue app uses Tkinter.

## Config

`config/config.yaml` holds the stream settings, channel names and groups, the class catalogue (display text and
hints for every gesture and confounder) and the settings of each recording session. Set
`eeg_stream.units_to_uv` if the stream is not in microvolts.

| Session | Purpose | Length |
|---|---|---|
| 1 | first recording: 14 classes incl. look_* | 25 trials per class, about 24 min |
| 2 | keep list from session 1 + confounders (talking, chewing, natural blinking, typing, idle) | about 28 min |
| 3 | short calibration (`calibrate.py`) | about 2 min |
| 4 | `modeltest` calibration | about 5 min |
| 5 | dashboard "Start session" calibration (trial count set by the chosen duration) | 2 to 30 min, default 10 |
| 6 | `modeltest --quick` | about 1 min |
| 7 | re-record chosen gestures only (`--classes`) | 2-3 min |

## 1. Collect

| Task | Command |
|---|---|
| Duration estimate and schedule preview | `python src/collect.py --session 1 --dry-run` |
| Collect (EEG stream on this PC) | `python src/collect.py --session 1 --subject S01 --cap "..." --notes "..."` |
| Fewer trials | `--trials 10` (or `collect_trials10.bat` / `collect_trials25.bat`) |
| EEG recorded on another device | `--cue-only` (see below) |
| UI only, no stream | `--no-eeg` |
| Live signal check only | `python src/signal_check.py` |
| Synthetic 24-ch stream that follows the cues | `python src/synthetic_eeg.py` (`--demo` cycles all gestures) |
| Fake session on disk, no LSL | `python src/make_synthetic_session.py --session 1 --out data\synth_s1 --xdf` |

Flow: signal check (Enter) -> gate (start LabRecorder if you use it, Enter) -> countdown -> trials -> done.
Keys: Space pauses (or continues a break), Esc twice aborts, F11 fullscreen. Signal check: `+`/`-` scale,
`c` common average, left/right pick the band-power channel.

Output `data/s<N>_<timestamp>/`: `markers.csv` (wall-clock and LSL time of every event), `eeg_backup.npz`
(raw samples; partial files are flushed every 60 s), `metadata.json` (notes, config, trial list, stream info).
LSL markers (`CueMarkers`) are JSON: event, session, trial, block, label, cue_s.

**Cue-only mode** (`--cue-only`): this PC only shows the cues. Start the recording on the other device, press
Enter, clench hard on each of the 4 white flashes at the start and at the end, follow the cues, then copy the
EEG file into the session folder. The loader finds the clenches in the EEG to align the clocks and corrects
drift (about +/-0.15 s). Override with `python src/load_data.py <folder> <offset_s>` if needed.

**eego `.cnt` files**: point at the session folder and the EEG file with `FOLDER@FILE` (training) or `--eeg FILE`
(load/feasibility). The `.cnt` start time is used when it agrees with the sync estimate.

## 2. Analyse

| Task | Command |
|---|---|
| Inspect a recording (channels, rate, class counts, alignment) | `python src/load_data.py data\s1_<ts> [--eeg FILE]` |
| Feasibility (LR + random forest per class) | `python src/feasibility.py data\s1_<ts> [--eeg FILE] [--quick]` |

Writes `<session>\analysis\` (report, `results.json`, `keep_list.yaml`): per-class precision/recall, confusion,
rest and confounder false triggers, and head vs eye direction. Cross-validation keeps trials in time-ordered
blocks. Preprocessing (`preprocess.py`) is causal (common average, 60 Hz notch, 1-40 / 0.1-10 / 30-100 Hz bands),
so offline and live features match. `left`/`right` channel groups must be listed in matching order.

## 3. Train

```
python src/train.py --data data\s2_<ts>@data\Session2Recording.cnt --name default
python src/train.py --data S2@S2.cnt --extra S1@S1.cnt --name default     # extra sessions: training only
```
- Split in time order: first 60% of trials train, next 20% tune thresholds and persistence, last 20% test.
- Objective: `recall - 0.5 * false_per_min - 0.5 * wrong_class_per_trial`.
- Gaps between trials, confounders and look_* cues form a `confounder` class that never fires.
- Output `profiles\<name>.joblib` (model, thresholds, cursor estimators, channels) plus `.json`, `_train.npz`,
  `_report.txt`. `--models rf` trains one kind only; `--channels A,B,...` trains on a channel subset.

**Event logic** (`detector.py`, same offline and live): a gesture fires when it is the top class, above its
threshold, for N consecutive windows; a held gesture fires once; per-class refractory plus a 0.5 s global lockout.

## 4. Calibrate and run without the dashboard

| Task | Command |
|---|---|
| Replay a recording through the engine (scored, dry run) | `python src/realtime.py --profile default --replay data\s1_<ts> --eeg FILE --fast --no-ui` |
| Live, log only | `realtime_dry.bat` |
| Live, real mouse and keyboard | `realtime_live.bat` |
| 2-minute calibration, then use it | `calibrate.bat`, then `realtime_live.bat --profile calibrated` |
| Re-record chosen gestures | `python src/calibrate.py --base session --name session --only jaw_clench,double_blink` (`--trials`, `--class-weight`, `--replace`) |
| Calibrate, then list every detected gesture | `python modeltest` (`--quick`, `--skip-calibration`, `--synthetic`) |

`src/realtime.py` uses `config/mapping.yaml` (the dashboard has its own mappings). Safety: Ctrl+Alt+Q kills,
Ctrl+Alt+P pauses, more than 40 events per minute pauses, a stream that does not match the profile is refused.

## Channels and headsets

The amp's LSL stream has 25 unlabelled channels (the 25th is all zeros). `channels.py` names the first 24 from
`config.yaml` (`Fp1 FP2 F9 F7 F3 Fz F4 F8 F10 T9 T7 C3 C4 T8 T10 P9 P7 P3 Pz P4 P8 P10 O1 O2`), uses labels when a
stream has them, and flags channels that are flat or not running. Models use 22 canonical channels
(`model.channels`); the first cap's M1/M2 are treated as T9/T10 (`model.aliases`, an assumption). Retrain after
changing `model:` or `channel_groups:`.

**"No LSL stream found"**: `python src\lsl_scan.py --watch` lists every stream this PC can see. Usual causes are
the network (same subnet, no guest Wi-Fi) and Windows Firewall on either machine.

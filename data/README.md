# EEG recordings

Recorded 2026-11-07 at the ANT Neuro station with the eego 24 amplifier (EE-225) and a waveguard cap.

| File | Length | Samples |
|---|---|---|
| `Carter1_Peter_2026-11-07_15-26-28.cnt` | 10.6 min | 317,097 |
| `Carter1_Peter_2026-11-07_16-33-59.cnt` | 56.8 min | 1,704,856 |

Each `.cnt` has a matching `.evt` that only holds the impedance checks. The gesture labels live in the cue
app's session folders (`s<N>_<timestamp>/markers.csv`), which pair with a `.cnt` as `FOLDER@FILE`
(see `src/README.md`). Session folders recorded through the dashboard (`s5_*`) contain their own EEG
(`eeg_backup.npz`). Session folders and `profiles/` are not committed; only this README and the `.cnt`/`.evt`
files are.

## Format
- ANT EEProbe `.cnt`, 24 channels, **500 Hz**, microvolts, reference CPz
- Channel order: `Fp1 Fp2 F9 F7 F3 Fz F4 F8 F10 M1 t7 c3 c4 t8 M2 cz p7 p3 pz p4 p8 oz o1 o2`
  (some labels are lower case; match names without case). This is the first cap's layout; see
  `src/README.md` for how it maps to the current one.
- Raw and unfiltered: large DC offsets and drift, and **strong 60 Hz mains noise** (plus some at 120 Hz).
  High-pass and notch-filter before use; the project's preprocessing does both.

## Reading in Python
```python
import mne                                   # needs: pip install mne antio
raw = mne.io.read_raw_ant("data/Carter1_Peter_2026-11-07_15-26-28.cnt", preload=True)
raw.notch_filter([60, 120])
raw.filter(1.0, 100.0)
data = raw.get_data() * 1e6                  # (24, samples), microvolts
```

## Using the recordings in the app
- Replay one as if it were live: `controller.bat --model team:session --eeg-file data\Carter1_Peter_2026-11-07_16-33-59.cnt`
  (`--eeg-start 120` starts two minutes in).
- Simulated EEG from the labelled sessions: `controller_sim.bat` (see `controller/README.md`).

The file headers include the subject name as entered in the eego software.

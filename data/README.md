# EEG recordings

Recorded 2026-11-07 at the ANT Neuro station with the eego 24 amplifier (EE-225) and waveguard cap.

| file | length | samples |
|---|---|---|
| `Carter1_Peter_2026-11-07_15-26-28.cnt` | 10.6 min | 317,097 |
| `Carter1_Peter_2026-11-07_16-33-59.cnt` | 56.8 min | 1,704,856 |

Each `.cnt` has a matching `.evt` (event file). Both `.evt` files only contain the impedance checks; there
are no gesture markers in the recordings, so gesture labels (which gesture, when) have to come from our own
notes or timing.

## Format
- ANT EEProbe `.cnt` (RIFF64), 24 channels, **500 Hz**, values in microvolts
- **Reference: CPz** (not Cz; Cz is a recorded channel here)
- Channel order: `Fp1 Fp2 F9 F7 F3 Fz F4 F8 F10 M1 t7 c3 c4 t8 M2 cz p7 p3 pz p4 p8 oz o1 o2`
  (some labels are lower case, as the eego software wrote them; match names without case)
- Raw and unfiltered (0-250 Hz): large DC offsets and drift, so high-pass before use
- **Strong 60 Hz mains noise** (about 2000x the neighbouring frequencies) and some at 120 Hz: notch-filter
  60 and 120 Hz (or low-pass under 55 Hz) before training, or the noise will dominate the features

## Reading it in Python
```
.venv\Scripts\python -m pip install mne antio
```
```python
import mne
raw = mne.io.read_raw_ant("data/Carter1_Peter_2026-11-07_15-26-28.cnt", preload=True)
raw.notch_filter([60, 120])
raw.filter(1.0, 100.0)
data = raw.get_data() * 1e6          # (24, samples), microvolts
print(raw.ch_names, raw.info["sfreq"])
```

## Replaying a recording through the app
The controller can play a recording as if it were the live stream, to test the gesture model on real data:
```
.venv\Scripts\python -m controller --model ml/model.py:predict --eeg-file data/Carter1_Peter_2026-11-07_16-33-59.cnt
```
(`--eeg-start 120` starts two minutes in.) The app sends the model raw microvolts, like the live stream.

Note: the file headers include the subject name as entered in the eego software.

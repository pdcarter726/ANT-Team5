import csv
import json
import shutil

import numpy as np
import pytest

from load_data import load_recording
from make_synthetic_session import make_session
from windowing import trial_windows


@pytest.fixture(scope="module")
def session_dir(tmp_path_factory, cfg):
    d = tmp_path_factory.mktemp("synth")
    make_session(cfg, 1, d, trials=3, seed=5, write_xdf_file=True)
    return d


def test_load_npz_and_csv_fallback(session_dir, tmp_path):
    d = tmp_path / "npzonly"
    d.mkdir()
    for f in ("eeg_backup.npz", "markers.csv", "metadata.json"):
        shutil.copy(session_dir / f, d / f)
    r = load_recording(d, verbose=False)
    assert r.source.startswith("NPZ")
    assert r.data.shape[1] == 24 and r.ch_names[0] == "Fp1" and r.fs == 500
    assert r.class_counts()["jaw_clench"] == 3 and sum(r.class_counts().values()) == 42
    assert any("markers.csv" in n for n in r.notes)


def test_load_xdf_primary_matches_npz(session_dir):
    x = load_recording(session_dir, verbose=False)
    n = load_recording(session_dir, prefer="npz", verbose=False)
    assert x.source.startswith("XDF") and n.source.startswith("NPZ")
    assert x.data.shape == n.data.shape
    assert np.allclose(x.data, n.data, atol=1e-4)
    assert x.ch_names == n.ch_names
    assert abs(x.ts[0] - n.ts[0]) < 1e-3
    assert any("max time difference 0.0 ms" in s for s in x.notes)    # XDF markers == CSV markers


def test_xdf_csv_mismatch_is_flagged(session_dir, tmp_path):
    d = tmp_path / "shifted"
    shutil.copytree(session_dir, d)
    rows = list(csv.DictReader(open(d / "markers.csv")))
    for r in rows:
        r["lsl_time"] = f"{float(r['lsl_time']) + 0.2:.6f}"
    with open(d / "markers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    r = load_recording(d, verbose=False)
    assert any("WARNING" in n and "50 ms" in n for n in r.notes)


def test_recover_from_parts(session_dir, tmp_path):
    d = tmp_path / "crashed"
    d.mkdir()
    z = np.load(session_dir / "eeg_backup.npz")
    half = len(z["timestamps"]) // 2
    np.savez(d / "eeg_part_0001.npz", data=z["data"][:half], timestamps=z["timestamps"][:half])
    np.savez(d / "eeg_part_0002.npz", data=z["data"][half:], timestamps=z["timestamps"][half:])
    shutil.copy(session_dir / "markers.csv", d / "markers.csv")
    shutil.copy(session_dir / "metadata.json", d / "metadata.json")
    r = load_recording(d, verbose=False)
    assert r.data.shape[0] == len(z["timestamps"]) and "parts" in r.source


def test_missing_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_recording(tmp_path, verbose=False)


def test_marker_alignment_spike_lands_in_window():
    """A spike placed exactly 0.6 s after the cue must fall inside the first window of that trial."""
    fs = 500
    ts = 100.0 + np.arange(20 * fs) / fs
    x = np.zeros(len(ts))
    t_cue = 105.0
    x[np.searchsorted(ts, t_cue + 0.6)] = 1.0
    ev = [dict(event="cue_on", t=t_cue, label="a", trial=0, block=0, cue_s=1.5)]
    wins, skipped = trial_windows(ts, ev, fs, win_s=0.5, step_s=0.25, offset_s=0.25)
    assert skipped == 0 and len(wins) == 4
    hit = [w for w in wins if x[w.i0:w.i1].any()]
    assert [round(w.t_start - t_cue, 2) for w in hit] == [0.25, 0.5]   # windows [.25,.75) and [.5,1.0)
    assert all(w.i1 - w.i0 == 250 for w in wins)
    assert all(abs(ts[w.i0] - w.t_start) < 1 / fs for w in wins)


def test_windows_skip_trials_with_gaps_or_outside():
    fs = 500
    ts = np.arange(10 * fs) / fs
    ts[2000:] += 1.0                        # 1 s dropout inside trial 1's cue
    ev = [dict(event="cue_on", t=1.0, label="a", trial=0, block=0, cue_s=1.5),
          dict(event="cue_on", t=3.6, label="b", trial=1, block=0, cue_s=1.5),
          dict(event="cue_on", t=50.0, label="c", trial=2, block=0, cue_s=1.5)]
    wins, skipped = trial_windows(ts, ev, fs)
    assert skipped == 2 and {w.trial for w in wins} == {0}


def test_windows_of_a_trial_share_trial_id_and_are_contiguous_in_time():
    fs = 250
    ts = np.arange(60 * fs) / fs
    ev = [dict(event="cue_on", t=5.0 + 4 * i, label="a", trial=i, block=0, cue_s=1.5) for i in range(8)]
    wins, _ = trial_windows(ts, ev, fs)
    ids = [w.trial for w in wins]
    assert ids == sorted(ids)               # a trial's windows are consecutive -> never interleaved

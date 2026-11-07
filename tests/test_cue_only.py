import csv
import json
import shutil

import numpy as np
import pytest

from common import session_cfg
from load_data import load_recording
from make_synthetic_session import make_session
from protocol import SYNC_READY_S, SYNC_TOTAL_S, build_trials, estimate_duration


def _errors(rec, folder, drift_ppm, t_start=37.3):
    rows = [x for x in csv.DictReader(open(folder / "markers.csv")) if x["event"] == "cue_on"]
    cues = [e for e in rec.events if e["event"] == "cue_on"]
    d = 1 + drift_ppm * 1e-6
    return np.array([e["t"] - (rec.ts[0] + (float(x["t_rel"]) + t_start) * d) for e, x in zip(cues, rows)])


@pytest.mark.parametrize("drift", [0.0, 800.0])
def test_sync_alignment_recovers_offset_and_drift(tmp_path, cfg, drift):
    make_session(cfg, 1, tmp_path, trials=3, seed=4, cue_only=True, drift_ppm=drift, write_xdf_file=True)
    (tmp_path / "eeg_backup.npz").unlink()            # EEG arrives as an XDF from the other device
    rec = load_recording(tmp_path, verbose=False)
    assert rec.alignment and rec.alignment["conf_start"] > 10
    assert len(rec.class_counts()) == 14
    err = _errors(rec, tmp_path, drift)
    assert np.abs(err).max() < 0.2                    # well inside the 0.25 s window offset
    rel = {r["label"]: float(r["t_rel"]) for r in csv.DictReader(open(tmp_path / "markers.csv"))
           if r["event"] == "sync_pulse"}[("sync_end")] - 0.0
    expected = drift * 1e-6 * rel                      # clock drift accumulated between the two syncs
    assert abs(rec.alignment["drift_s"] - expected) < 0.15


def test_manual_offset_skips_search(tmp_path, cfg):
    make_session(cfg, 1, tmp_path, trials=2, seed=1, cue_only=True)
    true = json.load(open(tmp_path / "metadata.json"))["true_offset_s"]
    rec = load_recording(tmp_path, marker_offset=true, verbose=False)
    assert rec.alignment["method"] == "manual offset"
    first = next(e for e in rec.events if e["event"] == "session_start")
    assert abs(first["t"] - (rec.ts[0] + 37.3)) < 1e-3


def test_missing_sync_pulses_fails_loudly(tmp_path, cfg):
    make_session(cfg, 1, tmp_path, trials=2, seed=1, cue_only=True)
    rows = [r for r in csv.DictReader(open(tmp_path / "markers.csv")) if r["event"] != "sync_pulse"]
    with open(tmp_path / "markers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="sync_pulse"):
        load_recording(tmp_path, verbose=False)


def test_estimate_includes_sync_for_cue_only(cfg):
    s = session_cfg(cfg, 1)
    tr = build_trials(cfg, 1, seed=0)
    a, b = estimate_duration(tr, s), estimate_duration(tr, s, cue_only=True)
    assert abs((b["total_s"] - a["total_s"]) - (2 * SYNC_TOTAL_S + SYNC_READY_S)) < 1e-6
    assert b["total_min"] < 26


def test_markers_csv_has_t_rel_monotonic(tmp_path, cfg):
    make_session(cfg, 1, tmp_path, trials=2, seed=1, cue_only=True)
    t = [float(r["t_rel"]) for r in csv.DictReader(open(tmp_path / "markers.csv"))]
    assert t[0] == 0.0 and all(b >= a for a, b in zip(t, t[1:]))

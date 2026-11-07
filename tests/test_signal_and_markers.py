import csv
import json
import time

import numpy as np
import pylsl
import pytest

from signal_metrics import band_powers, channel_stats
from synthetic_eeg import SignalModel


def test_flat_saturated_noisy_flags():
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.standard_normal((n, 4)) * 10
    x[:, 1] = 5.0                         # flat (constant offset)
    x[:, 2] = np.clip(x[:, 2] * 400, -3000, 3000)   # clipped / saturated
    x[:, 3] *= 30                         # noisy (~300 uV rms)
    st = channel_stats(x)
    assert [s["flag"] for s in st] == ["", "FLAT", "SAT", "NOISY"]


def test_band_power_alpha_peak():
    fs = 500
    t = np.arange(4 * fs) / fs
    x = 20 * np.sin(2 * np.pi * 10 * t) + np.random.default_rng(0).standard_normal(len(t))
    bp = band_powers(x, fs)
    assert bp["rel"]["alpha 8-13"] > 0.8
    assert abs(bp["alpha_peak"] - 10) < 0.6


def test_synthetic_gestures_have_expected_signatures():
    m = SignalModel(500, seed=0, natural_blinks=False)
    names = m.names
    i = {n.lower(): k for k, n in enumerate(names)}
    ptp = lambda g: np.ptp(g - g.mean(0), axis=0)
    lw, rw = ptp(m.make("left_wink", 1.5)), ptp(m.make("right_wink", 1.5))
    assert lw[i["fp1"]] > 2 * lw[i["fp2"]] and rw[i["fp2"]] > 2 * rw[i["fp1"]]
    jc = ptp(m.make("jaw_clench", 1.5))
    assert jc[i["t7"]] > 3 * jc[i["cz"]]
    ll, lr = m.make("look_left", 1.5), m.make("look_right", 1.5)
    assert np.sign(ll[100:600, i["f7"]].mean()) != np.sign(lr[100:600, i["f7"]].mean())


def test_synthetic_stream_generation_is_continuous():
    m = SignalModel(500, seed=1, natural_blinks=False)
    m.schedule("jaw_clench", 400, 1.5)
    a = np.concatenate([m.generate(137) for _ in range(20)])
    m2 = SignalModel(500, seed=1, natural_blinks=False)
    m2.schedule("jaw_clench", 400, 1.5)
    assert a.shape == (137 * 20, 24) and np.isfinite(a).all()
    assert m.n == 137 * 20                      # sample counter matches what was generated
    assert m.events == [] or m.events[0][0] + len(m.events[0][1]) > m.n


def test_marker_sender_csv_matches_lsl(tmp_path):
    from lsl_utils import MarkerSender
    from protocol import Trial
    ms = MarkerSender(tmp_path / "m.csv", 7, name="TestMarkers")
    inlet = pylsl.StreamInlet(pylsl.resolve_byprop("name", "TestMarkers", timeout=5)[0])
    inlet.pull_chunk(timeout=0.2)
    t = Trial(0, 0, "left_wink", "LEFT WINK", "", 1.5, 2.0)
    sent = [ms.send("cue_on", t), ms.send("cue_off", t)]
    time.sleep(0.3)
    got, ts = inlet.pull_chunk(timeout=1.0)
    ms.close()
    rows = list(csv.DictReader(open(tmp_path / "m.csv")))
    assert [r["event"] for r in rows] == ["cue_on", "cue_off"]
    payload = json.loads(got[0][0])
    assert payload["label"] == "left_wink" and payload["session"] == 7 and payload["trial"] == 0
    for r, s, lt in zip(rows, sent, ts):
        assert abs(float(r["lsl_time"]) - s) < 1e-5
        assert abs(lt - s) < 0.05                  # LSL timestamp matches CSV lsl_time
        assert abs(float(r["wall_unix"]) - time.time()) < 30

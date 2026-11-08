"""SSVEP: decoding, calibration checks, live logic, the epilepsy-warning gates and the flicker geometry."""
import os
import time

import numpy as np
import pytest

from controller.ssvep import core
from controller.ssvep.core import (DEFAULT_FREQS, SsvepLogic, SsvepProfile, calibrate_profile, synth_eeg, window_r)

FS = 500.0


def trials(rng, n_per=3, gaze_s=6.0, snr=0.6, rest=3, targets=(0, 1, 2, 3)):
    segs, labels = [], []
    for _ in range(n_per):
        for t in targets:
            segs.append(synth_eeg(int(gaze_s * FS), FS, 7, t, snr=snr, rng=rng))
            labels.append(t)
    for _ in range(rest):
        segs.append(synth_eeg(int(gaze_s * FS), FS, 7, None, rng=rng))
        labels.append(-1)
    return segs, labels


def test_cca_scores_the_attended_square_highest():
    rng = np.random.default_rng(1)
    for t in range(4):
        r = window_r(synth_eeg(int(3 * FS), FS, 7, t, snr=0.6, rng=rng), FS)
        assert int(np.argmax(r)) == t


def test_reference_sines_do_not_overlap_between_squares():
    harm = [{round(h * f, 3) for h in range(1, core.HARMONICS + 1)} for f in DEFAULT_FREQS]
    assert sum(len(h) for h in harm) == len(set().union(*harm))


def test_calibration_passes_with_a_real_response_and_fails_without_one():
    rng = np.random.default_rng(2)
    segs, labels = trials(rng, snr=0.6)
    prof, report = calibrate_profile(segs, labels, FS, channels=core.DEFAULT_CHANNELS, win_options=(2.0, 3.0))
    assert prof.usable and prof.hit_rate >= 0.75 and prof.false_per_min <= 1.0, report
    segs, labels = trials(rng, snr=0.0)                          # nobody is responding to the squares
    bad, report = calibrate_profile(segs, labels, FS, channels=core.DEFAULT_CHANNELS, win_options=(2.0,))
    assert not bad.usable and bad.reasons


def test_calibration_without_rest_trials_is_not_usable():
    rng = np.random.default_rng(3)
    segs, labels = trials(rng, snr=0.8, rest=0)
    prof, _ = calibrate_profile(segs, labels, FS, channels=core.DEFAULT_CHANNELS, win_options=(2.0,))
    assert not prof.usable and any("rest" in r for r in prof.reasons)


def test_looking_away_does_not_trigger_after_calibration():
    rng = np.random.default_rng(4)
    segs, labels = trials(rng, snr=0.6)
    prof, _ = calibrate_profile(segs, labels, FS, channels=core.DEFAULT_CHANNELS, win_options=(2.0, 3.0))
    logic, fired = prof.logic(), 0
    for _ in range(20):                                           # 20 fresh windows of nothing in particular
        x = synth_eeg(int(prof.win_s * FS), FS, 7, None, rng=rng)
        fired += sum(1 for k, _ in logic.step(prof.adjusted(window_r(x, FS))) if k == "fire")
    assert fired == 0


def test_logic_persistence_release_and_margin():
    lg = SsvepLogic(thr=0.2, margin=0.05, persist=2, release=2)
    hi = [0.0, 0.5, 0.0, 0.0]
    assert lg.step(hi) == []                                     # first accepted step: not yet
    assert lg.step(hi) == [("fire", 1)]
    assert lg.step(hi) == []                                     # still held, no repeat
    assert lg.step([0.0] * 4) == []                              # one step away: still held
    assert lg.step([0.0] * 4) == [("release", 1)]
    assert lg.step([0.5, 0.48, 0, 0]) == [] and lg.cand is None  # too close to call (margin)
    assert lg.step([0.1, 0.0, 0, 0]) == []                       # below threshold
    assert lg.reset() == []


def test_profile_roundtrip_and_load_if_usable(tmp_path):
    p = SsvepProfile(list(DEFAULT_FREQS), FS, core.DEFAULT_CHANNELS, 3.0, usable=True, hit_rate=0.9)
    p.save(tmp_path / "s.json")
    assert core.load_if_usable(tmp_path / "s.json").hit_rate == 0.9
    p.usable = False
    p.save(tmp_path / "s.json")
    assert core.load_if_usable(tmp_path / "s.json") is None and core.load_if_usable(tmp_path / "none.json") is None


class FakeActions:
    def __init__(self):
        self.calls = []

    def handle(self, kind, gesture, prob=0.0):
        self.calls.append((kind, gesture))
        return "ok"


def test_runner_fires_and_releases_the_mapped_gesture():
    from controller.eeg_source import _Buffer
    from controller.ssvep.runner import SsvepRunner
    rng = np.random.default_rng(5)
    segs, labels = trials(rng, snr=0.6)
    prof, _ = calibrate_profile(segs, labels, FS, channels=core.DEFAULT_CHANNELS, win_options=(2.0,))

    class Src:
        fs, channels = FS, core.DEFAULT_CHANNELS
        live = None
        buf = _Buffer(7, FS)

    src = Src()
    act = FakeActions()
    r = SsvepRunner(src, prof, act)
    t = 0.0
    for target in (2, None):                                       # look at bottom-left, then away
        for _ in range(8):
            src.buf.push(synth_eeg(int(0.25 * FS), FS, 7, target, snr=0.6, rng=rng, t0=t))
            t += 0.25
            r.step_once()
    assert ("fire", "ssvep_bottom_left") in act.calls and ("release", "ssvep_bottom_left") in act.calls
    r.stop()


def test_calibration_program_stops_at_the_warning(monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from controller.ssvep import calibrate, warning
    created = []
    monkeypatch.setattr(warning, "confirm", lambda parent, purpose: purpose == "x")        # declined
    monkeypatch.setattr(calibrate, "SyntheticSource", lambda *a, **k: created.append(1))
    assert calibrate.main(["--synthetic"]) == 3 and not created                              # nothing was started


def test_calibration_end_to_end_with_synthetic_eeg(monkeypatch, tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from controller.ssvep import calibrate, warning
    asked = []
    monkeypatch.setattr(warning, "confirm", lambda parent, purpose: asked.append(purpose) or True)
    monkeypatch.setattr(calibrate, "PROFILE_PATH", tmp_path / "ssvep.json")
    code = calibrate.main(["--synthetic", "--blocks", "1", "--cue-s", "0.3", "--gaze-s", "4.0", "--drop-s", "0.5",
                           "--countdown-s", "0.3", "--out", str(tmp_path / "raw"), "--seed", "1"])
    assert asked == ["calibration"] and code == 0
    p = core.load_if_usable(tmp_path / "ssvep.json")
    assert p is not None and p.n_trials == 4 and (tmp_path / "raw" / "ssvep_calibration.npz").exists()


def test_flicker_geometry_and_luminance():
    from controller.ssvep.flicker import luminance, square_rects
    r = square_rects(1920, 1080)
    assert len(r) == 4 and r[0].left() < r[1].left() and r[0].top() < r[2].top()
    assert all(0 <= x.left() and x.right() < 1920 and 0 <= x.top() and x.bottom() < 1080 for x in r)
    vals = [luminance(10.0, k / 1000) for k in range(100)]                  # exactly one 10 Hz cycle
    assert min(vals) < 0.01 and max(vals) > 0.99


def test_manager_refuses_without_calibration_or_live_headset(tmp_path):
    pytest.importorskip("PySide6")
    from controller.ssvep.manager import SsvepManager
    m = SsvepManager({"source": None, "actions": None, "mapping": {}}, lambda *a: None, profile_path=tmp_path / "none.json")
    with pytest.raises(RuntimeError, match="not calibrated"):
        m.enable()
    SsvepProfile(list(DEFAULT_FREQS), FS, core.DEFAULT_CHANNELS, 3.0, usable=True).save(tmp_path / "ok.json")

    class Sim:
        name = "simulated"
        connected = True

    m = SsvepManager({"source": Sim(), "actions": None, "mapping": {}}, lambda *a: None, profile_path=tmp_path / "ok.json")
    with pytest.raises(RuntimeError, match="live headset"):
        m.enable()


class _Lbl:
    def __init__(self):
        self.t = ""

    def setText(self, t):
        self.t = t


def _fake_dashboard(mgr):
    from controller.ui.dashboard import Dashboard

    class Fake:
        ctx = {"ssvep": mgr}
        big, big_sub = _Lbl(), _Lbl()
        stopped = 0
        toggle = Dashboard.toggle_ssvep

        def stop_ssvep(self):
            self.stopped += 1

        def refresh_ssvep_button(self):
            pass

    return Fake()


class _Mgr:
    def __init__(self, calibrated=True, enabled=False):
        self.calibrated, self.enabled, self.enable_calls = calibrated, enabled, 0

    def calibration(self):
        return object() if self.calibrated else None

    def enable(self):
        self.enable_calls += 1
        self.enabled = True


def test_enabling_ssvep_in_the_controller_needs_the_warning_every_time(monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from controller.ssvep import warning
    answers = iter([False, True, True])
    asked = []
    monkeypatch.setattr(warning, "confirm", lambda parent, purpose: asked.append(purpose) or next(answers))
    m = _Mgr()
    d = _fake_dashboard(m)
    d.toggle()                                       # declined: nothing starts
    assert m.enable_calls == 0 and asked == ["enable"]
    d.toggle()                                       # accepted
    assert m.enable_calls == 1
    m.enabled = False
    d.toggle()                                       # off and on again: asked AGAIN
    assert m.enable_calls == 2 and asked == ["enable"] * 3


def test_uncalibrated_ssvep_is_never_started_and_never_shows_the_warning(monkeypatch):
    pytest.importorskip("PySide6")
    from controller.ssvep import warning
    monkeypatch.setattr(warning, "confirm", lambda *a: pytest.fail("no warning without a calibration"))
    m = _Mgr(calibrated=False)
    d = _fake_dashboard(m)
    d.toggle()
    assert m.enable_calls == 0 and "not calibrated" in d.big.t


def test_switching_ssvep_off_needs_no_warning(monkeypatch):
    pytest.importorskip("PySide6")
    from controller.ssvep import warning
    monkeypatch.setattr(warning, "confirm", lambda *a: pytest.fail("turning off must be instant"))
    m = _Mgr(enabled=True)
    d = _fake_dashboard(m)
    d.toggle()
    assert d.stopped == 1


def test_session_page_starts_the_ssvep_calibration_program(monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QProcess
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from controller.ui import session as S
    started = {}
    monkeypatch.setattr(QProcess, "start", lambda self, prog, args: started.update(args=args))
    page = S.SessionPage()
    page.stream_name = "eego"
    assert page.ssvep_btn.isVisibleTo(page)
    page.start_ssvep()
    assert started["args"][:3] == ["-u", "-m", "controller.ssvep.calibrate"] and started["args"][-2:] == ["--stream-name", "eego"]
    assert page.ssvep_btn.text().startswith("Stop")                  # the same button now stops it
    page.ssvep_proc = None

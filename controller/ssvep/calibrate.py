"""SSVEP calibration, run as its own program (the dashboard starts it from the start screen):

    python -m controller.ssvep.calibrate [--stream-name NAME] [--blocks 3]
    python -m controller.ssvep.calibrate --synthetic        # rehearsal with fake EEG, no headset

1. Shows the photosensitive-epilepsy warning. Nothing flashes unless it is accepted.
2. Reads the SAME LSL EEG stream as the gesture detector (a second reader; the detector is unaffected).
3. Full-screen: four squares flicker in the corners. Each trial the red ring marks one square (look at it for 8 s);
   some trials say "look at the centre" (these measure what the signal looks like when you are NOT choosing, which
   is what keeps false triggers down). 3 blocks of 5 trials is about 3 minutes. Esc stops at any time.
4. Fits profiles/ssvep.json (window length, per-square rest level, threshold). If the check fails the old
   calibration (if any) is kept and the reasons are printed.

Progress goes to stdout as "@@STATUS <phase> <message>" lines (phase: record, train, done, error), like
controller/session_calibrate.py. Exit code 0 = a usable calibration was saved.
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import warning
from .core import (DEFAULT_CHANNELS, DEFAULT_FREQS, PROFILE_PATH, POSITIONS, WIN_OPTIONS, calibrate_profile, synth_eeg,
                   synth_mix)

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
PRETTY = {"top_left": "TOP LEFT", "top_right": "TOP RIGHT", "bottom_left": "BOTTOM LEFT", "bottom_right": "BOTTOM RIGHT"}


def status(phase, msg):
    print(f"@@STATUS {phase} {msg}", flush=True)


class SyntheticSource:
    """Fake EEG for rehearsal and tests: noise, plus the target square's frequency once set_target(i) is called."""

    def __init__(self, fs=500.0, channels=None, freqs=DEFAULT_FREQS, snr=0.6, seed=0):
        from controller.eeg_source import _Buffer
        self.fs = fs
        self.channels = list(channels or DEFAULT_CHANNELS)
        self.name = "synthetic SSVEP"
        self.freqs, self.snr = freqs, snr
        self.target = None
        self.live = None
        self.buf = _Buffer(len(self.channels), fs)
        self.rng = np.random.default_rng(seed)
        self.mix = synth_mix(len(self.channels), self.rng)
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def set_target(self, i):
        self.target = i

    def _loop(self):
        t0, sent = time.time(), 0
        while self.running:
            due = int((time.time() - t0) * self.fs) - sent
            if due > 0:
                self.buf.push(synth_eeg(due, self.fs, len(self.channels), self.target, self.freqs, self.snr,
                                        self.rng, t0=sent / self.fs, mix=self.mix))
                sent += due
            time.sleep(0.02)

    def window(self, seconds):
        return self.buf.window(seconds)

    @property
    def connected(self):
        return self.running

    def close(self):
        self.running = False


def build_plan(blocks, cue_s, gaze_s, rng):
    """[(target 0..3 or -1 for the centre)] - every block holds each square once plus one centre trial, shuffled."""
    plan, prev = [], None
    for _ in range(blocks):
        b = [0, 1, 2, 3, -1]
        rng.shuffle(b)
        if b[0] == prev:
            b[0], b[-1] = b[-1], b[0]
        plan += b
        prev = b[-1]
    return plan


class Calibration:
    def __init__(self, source, args, idx, on_done):
        from PySide6.QtCore import QTimer
        from .flicker import FlickerOverlay
        self.src, self.args, self.idx, self.on_done = source, args, idx, on_done
        self.fs = float(source.fs)
        self.win = FlickerOverlay(DEFAULT_FREQS, mode="calibration", labels=["", "", "", ""])
        self.win.aborted.connect(self.abort)
        self.cursor = source.buf.count
        self.chunks, self.total = [], 0
        self.plan = build_plan(args.blocks, args.cue_s, args.gaze_s, np.random.default_rng(args.seed))
        self.trials = []                     # dicts: target, start, end (sample indices into the recording)
        self.k = -1
        self.phase = "idle"
        self.last_new = time.time()
        self.finished = False
        self.poll = QTimer(self.win, timeout=self.pull, interval=25)
        self.QTimer = QTimer

    # ------------------------------------------------------------ data
    def pull(self):
        x, self.cursor, lost = self.src.buf.since(self.cursor)
        if lost:
            self.fail("the EEG reader fell behind and lost samples")
            return
        if x is not None and x.shape[1]:
            self.chunks.append(x[self.idx].T.astype(np.float64))
            self.total += x.shape[1]
            self.last_new = time.time()
        elif self.phase == "gaze" and time.time() - self.last_new > 3.0:
            self.fail("no EEG data arrived for 3 seconds (is the LSL stream still running?)")

    # ------------------------------------------------------------ sequence
    def start(self):
        self.win.set_state(flicker=False, highlight=None, message="Get ready",
                           sub="Four squares will flash. Press Esc at any time to stop.")
        self.win.start()
        self.poll.start()
        status("record", f"{len(self.plan)} trials, about {len(self.plan) * (self.args.cue_s + self.args.gaze_s) / 60:.1f} minutes")
        self.QTimer.singleShot(int(self.args.countdown_s * 1000), self.next_trial)

    def next_trial(self):
        if self.finished:
            return
        self.k += 1
        if self.k >= len(self.plan):
            return self.finish()
        t = self.plan[self.k]
        self.cur = t
        self.phase = "cue"
        self.win.set_state(flicker=False, highlight=t if t >= 0 else None,
                           message=("Look at the red ring" if t >= 0 else "Look at the centre of the screen"),
                           sub=(f"{PRETTY[POSITIONS[t]]} square" if t >= 0 else "Do not look at any square") +
                               f"      trial {self.k + 1} of {len(self.plan)}")
        status("record", f"trial {self.k + 1}/{len(self.plan)}: " + (PRETTY[POSITIONS[t]] if t >= 0 else "centre"))
        self.QTimer.singleShot(int(self.args.cue_s * 1000), self.begin_gaze)

    def begin_gaze(self):
        if self.finished:
            return
        self.pull()
        self.phase = "gaze"
        self.src_target(self.cur)
        self.win.set_state(flicker=True, message="" if self.cur >= 0 else "+", sub="")
        self.start_n = self.total
        self.QTimer.singleShot(int(self.args.gaze_s * 1000), self.end_gaze)

    def end_gaze(self):
        if self.finished:
            return
        self.pull()
        self.trials.append(dict(target=self.cur, start=self.start_n, end=self.total))
        self.phase = "idle"
        self.src_target(None)
        self.win.set_state(flicker=False, highlight=None, message="", sub="")
        self.next_trial()

    def src_target(self, t):
        if hasattr(self.src, "set_target"):
            self.src.set_target(None if t is None or t < 0 else t)

    def abort(self):
        self.fail("stopped with Esc")

    def fail(self, msg):
        if self.finished:
            return
        self.finished = True
        self.poll.stop()
        self.win.stop()
        self.on_done(None, msg)

    def finish(self):
        self.pull()
        self.finished = True
        self.poll.stop()
        self.win.stop()
        X = np.concatenate(self.chunks) if self.chunks else np.zeros((0, len(self.idx)))
        self.on_done((X, self.trials), "")


def train_and_save(X, trials, fs, channels, args):
    drop = int(round(args.drop_s * fs))
    segs, labels = [], []
    for t in trials:
        seg = X[t["start"] + drop:t["end"]]
        if len(seg) >= int(min(WIN_OPTIONS) * fs):
            segs.append(seg)
            labels.append(t["target"])
    from PySide6.QtGui import QGuiApplication
    refresh = float(QGuiApplication.primaryScreen().refreshRate() or 0.0)
    wins = tuple(w for w in WIN_OPTIONS if w <= args.gaze_s - args.drop_s) or (min(WIN_OPTIONS),)
    prof, report = calibrate_profile(segs, labels, fs, DEFAULT_FREQS, channels, wins, refresh_hz=refresh, subject=args.subject)
    return prof, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stream-name")
    ap.add_argument("--stream-type", default="EEG")
    ap.add_argument("--synthetic", action="store_true", help="fake EEG that responds to the cues (rehearsal, no headset)")
    ap.add_argument("--blocks", type=int, default=3, help="blocks of 5 trials (4 squares + centre); 3 = about 3 minutes")
    ap.add_argument("--cue-s", type=float, default=2.0)
    ap.add_argument("--gaze-s", type=float, default=8.0)
    ap.add_argument("--drop-s", type=float, default=1.0, help="seconds at the start of each gaze that are not used")
    ap.add_argument("--countdown-s", type=float, default=5.0)
    ap.add_argument("--subject", default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", help="folder for the raw recording (default data/ssvep_<time>)")
    args = ap.parse_args(argv)
    if args.seed is None:
        args.seed = int(time.time()) % 100000
    for p in (str(REPO), str(SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    if not warning.confirm(None, "calibration"):
        status("error", "SSVEP calibration cancelled at the epilepsy warning; nothing was shown")
        return 3

    # ---- EEG source
    try:
        if args.synthetic:
            source = SyntheticSource()
        else:
            from controller.eeg_source import LSLSource
            source = LSLSource(args.stream_name, args.stream_type)
    except Exception as e:
        status("error", f"no EEG stream: {e}")
        return 4
    from channels import ChannelMap
    from common import load_config
    try:
        cm = ChannelMap(list(source.channels), load_config(), live=getattr(source, "live", None), wanted=DEFAULT_CHANNELS)
    except ValueError as e:
        status("error", f"EEG stream lacks the occipital channels SSVEP needs: {e}")
        return 5
    if cm.problems:
        status("error", "these electrodes are not running (flat/zero): " + ", ".join(cm.problems) +
               ". SSVEP needs the occipital/parietal electrodes; check their contact.")
        return 5

    result = {}

    def done(data, msg):
        result["data"], result["msg"] = data, msg
        app.quit()

    cal = Calibration(source, args, cm.idx, done)
    cal.start()
    app.exec()
    source.close()
    if result.get("data") is None:
        status("error", f"calibration not finished: {result.get('msg')}; the saved calibration is unchanged")
        return 2
    X, trials = result["data"]
    status("train", "fitting the SSVEP decoder")
    try:
        prof, report = train_and_save(X, trials, float(source.fs), list(cm.names), args)
    except Exception as e:
        status("error", f"could not fit: {type(e).__name__}: {e}")
        return 6
    print(report, flush=True)
    out = Path(args.out) if args.out else REPO / "data" / f"ssvep_{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "ssvep_calibration.npz", eeg=X.astype(np.float32), fs=float(source.fs),
                            channels=np.array(cm.names), freqs=np.array(DEFAULT_FREQS),
                            targets=np.array([t["target"] for t in trials]),
                            starts=np.array([t["start"] for t in trials]), ends=np.array([t["end"] for t in trials]))
        (out / "report.txt").write_text(report, encoding="utf-8")
    except Exception as e:
        print(f"(could not save the raw recording: {e})", flush=True)
    if prof.usable:
        prof.save(PROFILE_PATH)
        status("done", prof.summary())
        return 0
    prof.save(PROFILE_PATH.with_name("ssvep_last_failed.json"))
    status("error", "calibration did not pass its checks, the previous calibration is kept: " + "; ".join(prof.reasons))
    return 2


if __name__ == "__main__":
    sys.exit(main())

"""Live SSVEP: flicker overlay + a thread that scores the newest EEG window every 0.25 s and turns accepted squares
into the same fire / release actions the gesture detector uses (mapping gestures ssvep_top_left ... ssvep_bottom_right).

Runs beside the gesture detector (it only READS the shared EEG buffer), and is enabled from the dashboard after the
epilepsy warning. It needs a calibration: profiles/ssvep.json with usable = true."""
import sys
import threading
import time
from pathlib import Path

import numpy as np

from ..log import log
from .core import DEFAULT_CHANNELS, GESTURES, PROFILE_PATH, SsvepProfile, load_if_usable, window_r

SRC = Path(__file__).resolve().parents[2] / "src"


def _channel_indices(source, profile):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from channels import ChannelMap
    from common import load_config
    cm = ChannelMap(list(source.channels), load_config(), live=getattr(source, "live", None), wanted=list(profile.channels))
    if cm.problems:
        raise RuntimeError("these electrodes are not running (flat/zero): " + ", ".join(cm.problems))
    return list(cm.idx)


class SsvepRunner(threading.Thread):
    def __init__(self, source, profile: SsvepProfile, actions, on_event=None, on_scores=None):
        super().__init__(daemon=True)
        if abs(float(source.fs) - profile.fs) > 1e-6:
            raise RuntimeError(f"EEG is {source.fs} Hz but SSVEP was calibrated at {profile.fs} Hz")
        self.src, self.p, self.actions = source, profile, actions
        self.on_event = on_event or (lambda *a: None)
        self.on_scores = on_scores or (lambda *a: None)
        self.idx = _channel_indices(source, profile)
        self.logic = profile.logic()
        self.running = True
        self.active = None
        self.errors = 0

    def step_once(self) -> bool:
        """Score the newest window and handle any events. False if there is not enough EEG yet."""
        w = self.src.buf.window(self.p.win_s)
        if w is None:
            return False
        x = w[self.idx].T.astype(np.float64)
        a = self.p.adjusted(window_r(x, self.p.fs, self.p.freqs, self.p.harmonics))
        self.on_scores(a)
        for kind, i in self.logic.step(a):
            self.emit(kind, i, float(a[i]))
        return True

    def emit(self, kind, i, score):
        name = GESTURES[i]
        desc = self.actions.handle(kind, name, score)
        self.active = i if kind == "fire" else None
        log("SSVEP", f"{kind} {name}", score=score, action=desc or "-")
        self.on_event(kind, name, score, desc or "")

    def run(self):
        nxt = time.time()
        while self.running:
            now = time.time()
            if now < nxt:
                time.sleep(min(0.02, nxt - now))
                continue
            nxt = max(nxt + self.p.step_s, now)
            try:
                self.step_once()
            except Exception as e:
                self.errors += 1
                if self.errors in (1, 10, 100):
                    log("ERROR", f"SSVEP step failed ({type(e).__name__}: {e})", total=self.errors)

    def stop(self):
        self.running = False
        for kind, i in self.logic.reset():          # let go of any held key / button
            self.emit(kind, i, 0.0)

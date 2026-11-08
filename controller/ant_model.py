"""Adapter for the team's trained EEG model (src/: Profile + WindowScorer, profiles/<name>.joblib).

The model is streaming: its filters keep state, so it must see every sample exactly once and in order.
Instead of the usual "latest window" call, the engine calls poll() each step; poll() pushes only the samples
that arrived since the last poll into src/detector.WindowScorer and returns one probability dict per model
window (0, 1 or more per poll). Class names are translated to the controller's names:
  head_left/right/up/down -> head_turn_left/right/up/down, confounder -> rest (never fires).
"""
import sys
import time
from pathlib import Path

import numpy as np

from .log import log

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
PROFILE_DIR = REPO / "profiles"
SESSION_PROFILE = "session"          # written by the dashboard's calibration (controller/session_calibrate.py)

NAME_MAP = {"head_left": "head_turn_left", "head_right": "head_turn_right",
            "head_up": "head_turn_up", "head_down": "head_turn_down", "confounder": "rest"}


def _src_on_path():
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def profile_path(name):
    return PROFILE_DIR / f"{name}.joblib"


class AntModel:
    streaming = True

    def __init__(self, profile=SESSION_PROFILE):
        _src_on_path()
        from detector import Profile, make_inference_ready
        from common import load_config
        path = profile_path(profile)
        if not path.exists():
            raise FileNotFoundError(f"no trained profile {path}")
        self.profile = Profile.load(path)
        make_inference_ready(self.profile.model)
        self.profile_name = profile
        self.name = f"team model ({profile}, {self.profile.kind})"
        self.step_s = float(self.profile.step_s)
        self.window_s = float(self.profile.win_s)
        self.units_to_uv = float(load_config()["eeg_stream"].get("units_to_uv", 1.0))
        self.classes = [NAME_MAP.get(c, c) for c in self.profile.classes]
        self.source = None
        self.errors = 0
        log("MODEL", "team model loaded", profile=profile, kind=self.profile.kind, classes=",".join(self.classes),
            fs=self.profile.fs, win_s=self.window_s, step_s=self.step_s, params=self.params_text())

    def params_text(self):
        p = self.profile.params
        return f"thr {p.default_thr}, persist {p.default_persist}, per-class thr {p.thr}"

    def attach(self, source):
        """Bind to the EEG source. Checks rate and channels, then starts from the newest sample."""
        from detector import WindowScorer
        if abs(float(source.fs) - float(self.profile.fs)) > 1e-6:
            raise RuntimeError(f"EEG is {source.fs} Hz but the model was trained at {self.profile.fs} Hz")
        low = [c.lower() for c in source.channels]
        missing = [c for c in self.profile.ch_names if c.lower() not in low]
        if missing:
            raise RuntimeError("EEG stream is missing channels the model needs: " + ", ".join(missing))
        self.order = np.array([low.index(c.lower()) for c in self.profile.ch_names])
        # FileSource already gives microvolts; the live LSL stream is scaled like src/realtime.py does
        self.scale = self.units_to_uv if type(source).__name__ == "LSLSource" else 1.0
        self.source = source
        self.scorer = WindowScorer(self.profile)
        self.cursor_count = source.buf.count          # only samples from now on
        self.last = {c: 0.0 for c in self.classes}
        log("MODEL", "model attached to EEG", eeg=source.name, scale_to_uv=self.scale,
            reordered=bool((self.order != np.arange(len(self.order))).any()))

    def poll(self):
        """New samples -> list of {gesture: probability} (one per completed model window)."""
        x, self.cursor_count, lost = self.source.buf.since(self.cursor_count)
        if lost:
            log("EEG", "model fell behind; samples were skipped", lost=lost)
        if x is None:
            return []
        try:
            res = self.scorer.push(x[self.order].T.astype(np.float64) * self.scale)    # samples x channels
        except Exception as e:
            self.errors += 1
            if self.errors in (1, 10, 100) or self.errors % 1000 == 0:
                log("ERROR", f"model failed ({type(e).__name__}: {e})", total_failures=self.errors)
            return []
        out = []
        for r in res:
            d = {}
            for c, p in zip(self.classes, r.probs):
                d[c] = max(d.get(c, 0.0), float(p))      # confounder shows as rest; max keeps the model's arg-max
            out.append(d)
        if out:
            self.last = out[-1]
        return out

    def __call__(self, window, fs, channels):            # not used by the engine for streaming models
        return self.last


class Waiting:
    """Stand-in until the session is calibrated: the engine is not started while this is the model."""
    streaming = True
    name = "not calibrated"
    window_s, step_s = 0.5, 0.125

    def poll(self):
        return []

    def __call__(self, window, fs, channels):
        return {"rest": 1.0}


def available_profiles():
    return sorted(p.stem for p in PROFILE_DIR.glob("*.joblib")) if PROFILE_DIR.exists() else []

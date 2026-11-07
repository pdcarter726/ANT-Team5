"""How the controller calls the gesture model (the ML side plugs in here).

Contract - write a Python function, e.g. in ml/model.py:

    WINDOW_S = 1.0     # optional: seconds of EEG per call (default 1.0)
    STEP_S = 0.1       # optional: seconds between calls (default 0.1)

    def predict(window, fs=None, channels=None):
        # window:   numpy array, shape (channels, samples), latest WINDOW_S seconds, raw units from the stream
        # fs:       sampling rate in Hz            (only passed if your function has this parameter)
        # channels: list of channel names          (only passed if your function has this parameter)
        # return ONE of:
        #   "jaw_clench"                                   just a label ("rest" = nothing)
        #   ("jaw_clench", 0.83)                           label and confidence 0-1
        #   {"rest": 0.1, "jaw_clench": 0.83, ...}         probability per gesture (best: shows on the dashboard)
        ...

Run with:  python -m controller --model ml/model.py:predict      (or a module path: ml.model:predict)

Gesture names the default mapping knows: rest, head_turn_up, head_turn_down, head_turn_left, head_turn_right,
eyebrow_raise, jaw_clench, double_blink, left_wink, right_wink. Any other name works too; it just needs a
line in the mapping.
"""
import importlib
import importlib.util
import inspect
import sys
import threading
import time
from pathlib import Path

from .log import log

GESTURES = ["rest", "head_turn_up", "head_turn_down", "head_turn_left", "head_turn_right",
            "eyebrow_raise", "jaw_clench", "double_blink", "left_wink", "right_wink"]


def normalize(out):
    """Any supported return value -> {label: probability}."""
    if out is None:
        return {"rest": 1.0}
    if isinstance(out, str):
        return {out: 1.0}
    if isinstance(out, dict):
        return {str(k): float(v) for k, v in out.items()}
    if isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[0], str):
        return {out[0]: float(out[1])}
    raise TypeError(f"model returned {type(out).__name__}; expected str, (str, float) or dict")


class LoadedModel:
    """Wraps the teammate's predict function: passes only the arguments it accepts, times it, never crashes the app."""

    def __init__(self, spec):
        target, _, fn_name = spec.rpartition(":")
        if not target:
            raise ValueError("use --model path/to/file.py:function or package.module:function")
        if target.endswith(".py") or "/" in target or "\\" in target:
            path = Path(target).resolve()
            sys.path.insert(0, str(path.parent))          # let the model import its own helper files
            mod_spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(mod_spec)
            mod_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(target)
        self.fn = getattr(module, fn_name)
        self.name = f"{Path(target).name}:{fn_name}"
        self.window_s = float(getattr(module, "WINDOW_S", 1.0))
        self.step_s = float(getattr(module, "STEP_S", 0.1))
        params = inspect.signature(self.fn).parameters
        self.wants_fs = "fs" in params
        self.wants_channels = "channels" in params
        self.errors = 0
        log("MODEL", "gesture model loaded", model=self.name, window_s=self.window_s, step_s=self.step_s,
            passes_fs=self.wants_fs, passes_channels=self.wants_channels)

    def __call__(self, window, fs, channels):
        kw = {}
        if self.wants_fs:
            kw["fs"] = fs
        if self.wants_channels:
            kw["channels"] = channels
        try:
            return normalize(self.fn(window, **kw))
        except Exception as e:
            self.errors += 1
            if self.errors in (1, 10, 100) or self.errors % 1000 == 0:
                log("ERROR", f"model call failed ({type(e).__name__}: {e})", total_failures=self.errors)
            return {"rest": 1.0}


class SimModel:
    """No EEG model: gestures come from the dashboard's test buttons (for rehearsing the demo and checking
    the mapping). A pressed button reads as that gesture for `seconds`, ramping up like a real detection."""
    name = "simulated (test buttons)"
    window_s, step_s = 1.0, 0.1

    def __init__(self):
        self.lock = threading.Lock()
        self.active = None

    def trigger(self, gesture, seconds=0.6):
        """seconds = how long the simulated gesture is held (the test buttons use 0.6 s, or longer for holds)."""
        with self.lock:
            self.active = (gesture, time.time(), seconds)

    def __call__(self, window, fs, channels):
        with self.lock:
            a = self.active
        if a and time.time() - a[1] < a[2]:
            p = min(0.95, 0.5 + (time.time() - a[1]) * 2.0)
            return {a[0]: p, "rest": 1 - p}
        return {"rest": 0.97}

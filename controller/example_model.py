"""Example gesture model: a template for the ML side, and a working stand-in for testing.

It follows the contract in controller/model_api.py. It only detects a jaw clench (a burst of high-frequency
muscle power on the temporal channels) by comparing against the first seconds of the stream, so it is
NOT the real model; replace predict() with your trained one.

Run:  python -m controller --model controller/example_model.py:predict --gaze-profile nico
"""
import numpy as np

WINDOW_S = 0.5      # seconds of EEG per call
STEP_S = 0.1        # seconds between calls

_baseline = []


def predict(window, fs=None, channels=None):
    # window: (channels, samples) raw EEG; fs: Hz; channels: names
    low = [c.lower() for c in channels] if channels else []
    idx = [low.index(c) for c in ("t7", "t8", "f7", "f8") if c in low] or [0, 1]
    x = window[idx].astype(float)
    x = x - x.mean(axis=1, keepdims=True)
    d = np.diff(x, axis=1)                 # crude high-pass: muscle bursts dominate the derivative
    power = float(np.log(np.mean(d ** 2) + 1e-12))
    if len(_baseline) < 50:                # learn "relaxed" level from the first ~5 s
        _baseline.append(power)
        return {"rest": 1.0}
    mu, sd = np.mean(_baseline), np.std(_baseline) + 1e-6
    z = (power - mu) / sd
    p = float(1 / (1 + np.exp(-(z - 4))))  # ~0.5 at 4 standard deviations above relaxed
    return {"rest": 1 - p, "jaw_clench": p}

"""Causal preprocessing shared by offline analysis and the real-time loop.

Pipeline: common average reference -> 60 Hz notch -> three parallel causal band branches:
  eog  1-40 Hz    blinks / winks / eyebrows (amplitude + duration)
  slow 0.1-10 Hz  head and eye direction (slow drift / held offset)
  emg  30-100 Hz  jaw / neck / forehead muscle activity
Causal filters (sosfilt with carried state) are used on purpose so offline features
match what the real-time loop will see.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

BRANCHES = {"eog": (1.0, 40.0), "slow": (0.1, 10.0), "emg": (30.0, 100.0)}


def _bandpass_sos(lo, hi, fs, order=4):
    hi = min(hi, 0.45 * fs)
    return sps.butter(order, [lo, hi], btype="band", fs=fs, output="sos")


class StreamingPreprocessor:
    def __init__(self, fs: float, n_ch: int, line_hz: float = 60.0, car: bool = True):
        self.fs, self.n_ch, self.car = fs, n_ch, car
        b, a = sps.iirnotch(line_hz, Q=30.0, fs=fs) if line_hz and line_hz < fs / 2 else (None, None)
        self.notch = sps.tf2sos(b, a) if b is not None else None
        self.sos = {k: _bandpass_sos(lo, hi, fs, 2 if k == "slow" else 4) for k, (lo, hi) in BRANCHES.items()}
        self.zi: dict[str, np.ndarray] = {}
        self.zi_notch = None

    def _run(self, key, sos, x):
        if key not in self.zi:   # init state to the first sample (avoids a huge DC transient)
            zi = sps.sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]   # (sections, 2, ch) for axis=0
            self.zi[key] = zi
        y, self.zi[key] = sps.sosfilt(sos, x, axis=0, zi=self.zi[key])
        return y

    def process(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """x: (n, n_ch) in uV. Returns {branch: (n, n_ch)}; state carries across calls."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] == 0:
            return {k: np.zeros((0, self.n_ch)) for k in self.sos}
        if self.car:
            x = x - x.mean(axis=1, keepdims=True)
        if self.notch is not None:
            x = self._run("notch", self.notch, x)
        return {k: self._run(k, s, x) for k, s in self.sos.items()}


def preprocess_offline(data: np.ndarray, fs: float, line_hz: float = 60.0) -> dict[str, np.ndarray]:
    return StreamingPreprocessor(fs, data.shape[1], line_hz).process(data)

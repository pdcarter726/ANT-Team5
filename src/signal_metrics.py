"""Pure-numpy signal quality metrics for the live signal-check view."""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

BANDS = {"delta 1-4": (1, 4), "theta 4-8": (4, 8), "alpha 8-13": (8, 13),
         "beta 13-30": (13, 30), "EMG 30-100": (30, 100)}


def channel_stats(x: np.ndarray, flat_rms=0.3, saturated_uv=2000.0, noisy_rms=150.0) -> list[dict]:
    """x: (n_samples, n_channels) in uV. Returns per-channel dict(rms, ptp, flag).

    flag is one of '', 'FLAT', 'SAT', 'NOISY'. RMS is computed after removing the
    median (DC offset), so a large constant offset is not mistaken for noise.
    """
    out = []
    if x.shape[0] < 2:
        return [dict(rms=0.0, ptp=0.0, flag="NODATA") for _ in range(x.shape[1])]
    xc = x - np.median(x, axis=0, keepdims=True)
    rms = np.sqrt(np.mean(xc ** 2, axis=0))
    ptp = np.ptp(x, axis=0)
    for c in range(x.shape[1]):
        col = x[:, c]
        mx = np.max(np.abs(xc[:, c]))
        clipped = np.mean(np.isclose(np.abs(col), np.max(np.abs(col)), rtol=0, atol=1e-6)) > 0.02 \
            and np.max(np.abs(col)) > 0 and rms[c] > flat_rms
        flag = ""
        if rms[c] < flat_rms or ptp[c] == 0:
            flag = "FLAT"
        elif mx > saturated_uv or clipped:
            flag = "SAT"
        elif rms[c] > noisy_rms:
            flag = "NOISY"
        out.append(dict(rms=float(rms[c]), ptp=float(ptp[c]), flag=flag))
    return out


def band_powers(x: np.ndarray, fs: float, bands: dict | None = None) -> dict:
    """Relative band power (fractions summing to ~1 over the bands) for one channel,
    plus alpha peak frequency. x: 1-D uV."""
    bands = bands or BANDS
    if len(x) < int(fs):
        return dict(rel={k: 0.0 for k in bands}, abs={k: 0.0 for k in bands}, alpha_peak=0.0)
    f, p = sps.welch(x - np.mean(x), fs=fs, nperseg=min(len(x), int(2 * fs)))
    ab, df = {}, f[1] - f[0]
    for name, (lo, hi) in bands.items():
        hi = min(hi, fs / 2)
        m = (f >= lo) & (f < hi)
        ab[name] = float(np.sum(p[m]) * df)
    tot = sum(ab.values()) or 1.0
    am = (f >= 8) & (f <= 13)
    peak = float(f[am][np.argmax(p[am])]) if am.any() else 0.0
    return dict(rel={k: v / tot for k, v in ab.items()}, abs=ab, alpha_peak=peak)

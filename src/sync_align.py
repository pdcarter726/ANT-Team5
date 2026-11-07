"""Align cue-only marker times (seconds since session start) to an EEG recording made on another device.

During collect.py --cue-only the wearer clenches the jaw hard on each white flash, at the start and at
the end of the session. Those bursts show up as broadband 30-100 Hz power. We cross-correlate the EEG
muscle-power envelope with a box template built from the logged flash times to find where the session
started in the EEG's clock. The start and end patterns have the same timing, so candidate start positions
are scored together with where the end pattern must then fall (session length is known from the markers);
this also gives clock drift.

Accuracy is limited by the wearer's reaction time (about 0.3 s, modelled by REACTION_S), so expect
roughly +/-0.15 s. Pass an explicit offset if you know better.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from protocol import SYNC_FLASH_S

HZ = 20.0            # envelope resolution
REACTION_S = 0.3     # typical reaction time to a visual flash
BOX = (REACTION_S, REACTION_S + SYNC_FLASH_S)   # expected burst interval after flash onset
END_SEARCH_S = (-3.0, 4.5)                      # where the end pattern may fall relative to the prediction (drift)


def muscle_envelope(data: np.ndarray, ts: np.ndarray, fs: float, hz: float = HZ):
    """Log 30-100 Hz power summed over channels on a hz-spaced grid. Returns (env, bin_times)."""
    sos = sps.butter(4, [30, min(100, 0.45 * fs)], btype="band", fs=fs, output="sos")
    x = data.astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    y = sps.sosfiltfilt(sos, x, axis=0)
    nb = max(1, int(round(fs / hz)))
    n = len(y) // nb
    p = (y[:n * nb] ** 2).sum(axis=1).reshape(n, nb).mean(axis=1)
    env = np.log10(p + 1e-9)
    tb = np.asarray(ts[:n * nb], float).reshape(n, nb).mean(axis=1)   # real bin times (keeps clock drift)
    return env, tb


def _robust_z(a):
    med = np.median(a)
    mad = np.median(np.abs(a - med)) * 1.4826 + 1e-9
    return np.clip((a - med) / mad, -3, 10)


def _corr(env, tb, pulses_rel):
    """Correlation of the envelope with the pulse-train template. Returns (z-scored corr, lag times).
    lag time = EEG time of the first pulse."""
    dt = float(np.median(np.diff(tb)))
    pr = np.asarray(pulses_rel, float) - pulses_rel[0]
    n_t = int(np.ceil((pr[-1] + BOX[1]) / dt)) + 1
    tpl = np.zeros(n_t)
    for p in pr:
        tpl[int((p + BOX[0]) / dt):int((p + BOX[1]) / dt) + 1] = 1.0
    tpl -= tpl.mean()
    c = sps.correlate(_robust_z(env), tpl, mode="valid", method="fft")
    med = np.median(c)
    mad = np.median(np.abs(c - med)) * 1.4826 + 1e-9
    return (c - med) / mad, tb[:len(c)], dt


def estimate_alignment(data, ts, fs, events, min_conf=6.0):
    """events: dicts with event/label/t (t = seconds since session start).

    Returns dict(offset, drift_s, conf_start, conf_end, method, fn) where EEG time = fn(t).
    Raises ValueError if there are no sync pulses in the markers.
    """
    start = sorted(e["t"] for e in events if e["event"] == "sync_pulse" and e["label"] == "sync_start")
    end = sorted(e["t"] for e in events if e["event"] == "sync_pulse" and e["label"] == "sync_end")
    if len(start) < 3:
        raise ValueError("no sync_pulse events in markers; pass an explicit offset")
    env, tb = muscle_envelope(data, ts, fs)
    c_s, lags, dt = _corr(env, tb, start)
    have_end = len(end) >= 3
    if have_end:
        c_e, _, _ = _corr(env, tb, end)
        span = end[0] - start[0]
        peaks, _ = sps.find_peaks(c_s, distance=max(1, int(5 / dt)))
        cand = peaks[np.argsort(c_s[peaks])[::-1][:10]] if len(peaks) else np.array([int(np.argmax(c_s))])
        best, best_score, best_j = None, -np.inf, None
        for i in cand:
            m = (lags >= lags[i] + span + END_SEARCH_S[0]) & (lags <= lags[i] + span + END_SEARCH_S[1])
            m[len(c_e):] = False
            if m.any():
                js = np.flatnonzero(m)
                j = js[int(np.argmax(c_e[js]))]
                score, jj = c_s[i] + max(c_e[j], 0.0), j
            else:                       # end pattern would fall outside the recording: cannot be the start
                score, jj = c_s[i] - 50.0, None
            if score > best_score:
                best, best_score, best_j = i, score, jj
        i0, j0 = best, best_j
    else:
        i0, j0 = int(np.argmax(c_s)), None
    off_s = float(lags[i0] - start[0])
    conf_s = float(c_s[i0])
    out = dict(offset=off_s, conf_start=conf_s, conf_end=None, drift_s=0.0, method="start sync")
    out["fn"] = lambda t, o=off_s: np.asarray(t, float) + o
    if conf_s < min_conf:
        out["warning"] = f"start sync confidence {conf_s:.1f} < {min_conf}: check alignment manually"
    if have_end and j0 is not None:
        off_e, conf_e = float(lags[j0] - end[0]), float(c_e[j0])
        out["conf_end"] = conf_e
        if conf_e >= min_conf and conf_s >= min_conf:
            out["drift_s"] = off_e - off_s
            out["method"] = "start+end sync (linear drift correction)"
            out["fn"] = lambda t, a=off_s, b=off_e, s0=start[0], sp=span: \
                np.asarray(t, float) + a + (b - a) * (np.asarray(t, float) - s0) / sp
        elif conf_e < min_conf:
            out["warning"] = (out.get("warning", "") + f" end sync confidence {conf_e:.1f} low; "
                              "using start offset only").strip()
    return out

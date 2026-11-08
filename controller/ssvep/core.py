"""SSVEP decoding (no Qt, no LSL): which of four flickering squares is the wearer looking at?

Four squares, one per screen corner, each luminance-modulated at its own frequency. Looking at one makes the visual
cortex (occipital electrodes O1, O2, Pz, P3, P4, P7, P8) oscillate at that frequency and its harmonic. We score every
square with canonical correlation analysis (CCA) between the EEG window and sines at that square's frequency and 2x
frequency, subtract the per-square score seen while the wearer looked at nothing in particular (calibration "rest"),
and accept a square only when it clearly beats the rest-level noise AND the other squares.

Calibration (controller/ssvep/calibrate.py) fixes, per person: the channels' usability, the window length, the
per-square rest offsets (mu), the acceptance threshold and margin. Nothing here is trained in the usual sense: CCA
uses fixed reference sines, so a profile is just a handful of numbers (profiles/ssvep.json).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

REPO = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO / "profiles" / "ssvep.json"

POSITIONS = ("top_left", "top_right", "bottom_left", "bottom_right")
GESTURES = tuple(f"ssvep_{p}" for p in POSITIONS)             # names used in the mappings
# 8/10/12/15 Hz: all easy to show on a 60 Hz screen; with 2 harmonics (f, 2f) no reference sine is shared (16,20,24,30)
DEFAULT_FREQS = (8.0, 10.0, 12.0, 15.0)
HARMONICS = 2
BAND_HZ = (5.0, 45.0)
DEFAULT_CHANNELS = ["O1", "O2", "Pz", "P3", "P4", "P7", "P8"]
STEP_S = 0.25
WIN_OPTIONS = (2.0, 3.0, 4.0)
MIN_HIT_RATE = 0.75              # share of looked-at trials in which the right square fired
MIN_TARGET_HIT_RATE = 0.5        # ... and for the worst single square
MAX_FALSE_PER_MIN = 1.0          # accidental events while looking at nothing in particular


# ----------------------------------------------------------------------------- signal processing
@lru_cache(maxsize=8)
def _bandpass(fs: float):
    return butter(4, BAND_HZ, btype="bandpass", fs=fs, output="sos")


def preprocess(x: np.ndarray, fs: float) -> np.ndarray:
    """samples x channels -> mean removed (the amp has a large DC offset), zero-phase band-pass 5-45 Hz."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    return sosfiltfilt(_bandpass(float(fs)), x, axis=0)


@lru_cache(maxsize=32)
def _refs(freqs: tuple, fs: float, n: int, harmonics: int):
    t = np.arange(n) / fs
    out = []
    for f in freqs:
        cols = []
        for h in range(1, harmonics + 1):
            cols += [np.sin(2 * np.pi * h * f * t), np.cos(2 * np.pi * h * f * t)]
        Y = np.stack(cols, axis=1)
        Y = Y - Y.mean(axis=0, keepdims=True)
        out.append(np.linalg.qr(Y)[0])                      # orthonormal basis: reused for every window
    return out


def cca_r(X: np.ndarray, Qy: np.ndarray) -> float:
    """Largest canonical correlation between X (samples x channels) and the orthonormal reference basis Qy."""
    X = X - X.mean(axis=0, keepdims=True)
    Qx, _ = np.linalg.qr(X)
    return float(np.linalg.svd(Qx.T @ Qy, compute_uv=False)[0])


def window_r(x: np.ndarray, fs: float, freqs=DEFAULT_FREQS, harmonics: int = HARMONICS, filtered: bool = False):
    """One CCA correlation per square for a window (samples x channels)."""
    xf = x if filtered else preprocess(x, fs)
    refs = _refs(tuple(float(f) for f in freqs), float(fs), len(xf), harmonics)
    return np.array([cca_r(xf, Qy) for Qy in refs])


# ----------------------------------------------------------------------------- decisions
class SsvepLogic:
    """Adjusted scores (CCA r minus the square's rest level), one set per step -> fire / release events.

    A square fires after `persist` consecutive steps in which it was the accepted one, and is released after
    `release` consecutive steps in which it was not (so a "hold:w" action is held while the wearer keeps looking)."""

    def __init__(self, thr: float, margin: float, persist: int = 2, release: int = 2):
        self.thr, self.margin, self.persist, self.release = thr, margin, int(persist), int(release)
        self.cand = None
        self.streak = 0
        self.active = None
        self.off = 0

    def accepted(self, a) -> int | None:
        a = np.asarray(a, dtype=float)
        order = np.argsort(a)[::-1]
        top, second = a[order[0]], a[order[1]]
        return int(order[0]) if (top >= self.thr and top - second >= self.margin) else None

    def step(self, a) -> list[tuple[str, int]]:
        cand = self.accepted(a)
        if cand is not None and cand == self.cand:
            self.streak += 1
        else:
            self.cand, self.streak = cand, 1 if cand is not None else 0
        events = []
        if self.active is None:
            if cand is not None and self.streak >= self.persist:
                self.active, self.off = cand, 0
                events.append(("fire", cand))
        else:
            self.off = 0 if cand == self.active else self.off + 1
            if self.off >= self.release:
                events.append(("release", self.active))
                self.active, self.off = None, 0
        return events

    def reset(self) -> list[tuple[str, int]]:
        out = [("release", self.active)] if self.active is not None else []
        self.cand, self.streak, self.active, self.off = None, 0, None, 0
        return out


# ----------------------------------------------------------------------------- profile
@dataclass
class SsvepProfile:
    freqs: list
    fs: float
    channels: list
    win_s: float
    step_s: float = STEP_S
    harmonics: int = HARMONICS
    mu: list = field(default_factory=lambda: [0.0] * 4)
    thr: float = 0.1
    margin: float = 0.03
    persist: int = 2
    release: int = 2
    usable: bool = False
    reasons: list = field(default_factory=list)
    hit_rate: float = 0.0
    target_hit_rate: list = field(default_factory=list)
    window_accuracy: float = 0.0
    wrong_per_trial: float = 0.0
    false_per_min: float = 0.0
    n_trials: int = 0
    refresh_hz: float = 0.0
    created: str = ""
    subject: str = ""

    def logic(self) -> SsvepLogic:
        return SsvepLogic(self.thr, self.margin, self.persist, self.release)

    def adjusted(self, r) -> np.ndarray:
        return np.asarray(r, dtype=float) - np.asarray(self.mu, dtype=float)

    def save(self, path=PROFILE_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path=PROFILE_PATH):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def summary(self) -> str:
        return (f"{'USABLE' if self.usable else 'NOT usable'}: window {self.win_s:g} s, hit rate {self.hit_rate:.0%}, "
                f"false events {self.false_per_min:.1f}/min while looking away, "
                f"per square {', '.join(f'{h:.0%}' for h in self.target_hit_rate)}")


def load_if_usable(path=PROFILE_PATH):
    """The saved calibration if there is one and it passed its own checks, else None."""
    try:
        p = SsvepProfile.load(path)
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return p if p.usable else None


# ----------------------------------------------------------------------------- calibration
def window_starts(n: int, win_n: int, step_n: int) -> range:
    return range(0, max(0, n - win_n) + 1, step_n) if n >= win_n else range(0)


def segment_scores(seg: np.ndarray, fs: float, freqs, win_s: float, step_s: float = STEP_S, harmonics: int = HARMONICS):
    """Sliding-window CCA scores for one recorded segment (samples x channels): array (n_windows, 4)."""
    win_n, step_n = int(round(win_s * fs)), max(1, int(round(step_s * fs)))
    seg = preprocess(seg, fs)                      # filter once; windows are cut from the filtered segment
    return np.array([window_r(seg[s:s + win_n], fs, freqs, harmonics, filtered=True)
                     for s in window_starts(len(seg), win_n, step_n)])


def _simulate(logic: SsvepLogic, adjusted: np.ndarray) -> list[tuple[str, int]]:
    ev = []
    for a in adjusted:
        ev += logic.step(a)
    return ev


def calibrate_profile(segments, labels, fs: float, freqs=DEFAULT_FREQS, channels=None, win_options=WIN_OPTIONS,
                      step_s: float = STEP_S, refresh_hz: float = 0.0, subject: str = "") -> tuple[SsvepProfile, str]:
    """segments: list of arrays (samples x channels), the part of each trial that was gazing; labels: 0..3 = the
    square that was looked at, -1 = looked at the centre (rest). Returns (profile, report text)."""
    labels = [int(x) for x in labels]
    if not any(x >= 0 for x in labels):
        raise ValueError("no SSVEP trials to calibrate from")
    best = None
    lines = []
    for win in win_options:
        S = [segment_scores(s, fs, freqs, win, step_s) for s in segments]
        rest = [s for s, l in zip(S, labels) if l < 0 and len(s)]
        rest = np.concatenate(rest) if rest else np.zeros((0, 4))
        mu = rest.mean(axis=0) if len(rest) else np.zeros(4)
        top = (rest - mu).max(axis=1) if len(rest) else np.zeros(1)
        thr = max(float(top.max()) + 0.03, float(top.mean() + 3 * top.std()), 0.08)
        prof = SsvepProfile(list(map(float, freqs)), float(fs), list(channels or DEFAULT_CHANNELS), float(win), step_s,
                            HARMONICS, [float(m) for m in mu], thr, 0.03, 2, 2, refresh_hz=refresh_hz,
                            created=time.strftime("%Y-%m-%d %H:%M:%S"), subject=subject)
        # window accuracy: does the highest adjusted score belong to the square that was looked at?
        acc = [float((np.argmax(s - mu, axis=1) == l).mean()) for s, l in zip(S, labels) if l >= 0 and len(s)]
        prof.window_accuracy = float(np.mean(acc)) if acc else 0.0
        # events: replay every trial through the same logic the live runner uses
        hits, wrong, per = [], 0, {t: [] for t in range(4)}
        rest_events, rest_s = 0, 0.0
        for s, l in zip(S, labels):
            if not len(s):
                continue
            ev = _simulate(prof.logic(), s - mu)
            fires = [i for k, i in ev if k == "fire"]
            if l >= 0:
                ok = l in fires
                hits.append(ok)
                per[l].append(ok)
                wrong += sum(1 for i in fires if i != l)
            else:
                rest_events += len(fires)
                rest_s += (len(s) - 1) * step_s + win
        prof.n_trials = len(hits)
        prof.hit_rate = float(np.mean(hits)) if hits else 0.0
        prof.target_hit_rate = [float(np.mean(per[t])) if per[t] else 0.0 for t in range(4)]
        prof.wrong_per_trial = wrong / max(1, len(hits))
        prof.false_per_min = rest_events / (rest_s / 60.0) if rest_s > 0 else 0.0
        lines.append(f"  window {win:g} s: accuracy {prof.window_accuracy:.0%}, hit rate {prof.hit_rate:.0%}, "
                     f"false {prof.false_per_min:.1f}/min, threshold {thr:.3f}")
        score = prof.hit_rate - 0.5 * prof.wrong_per_trial - 0.1 * prof.false_per_min - 0.02 * win
        if best is None or score > best[0]:
            best = (score, prof)
    prof = best[1]
    why = []
    if prof.hit_rate < MIN_HIT_RATE:
        why.append(f"the right square fired in only {prof.hit_rate:.0%} of trials (need {MIN_HIT_RATE:.0%})")
    weak = [POSITIONS[t].replace("_", " ") for t, h in enumerate(prof.target_hit_rate) if h < MIN_TARGET_HIT_RATE]
    if weak:
        why.append("weak response for: " + ", ".join(weak))
    if prof.false_per_min > MAX_FALSE_PER_MIN:
        why.append(f"{prof.false_per_min:.1f} accidental events per minute while looking away (max {MAX_FALSE_PER_MIN:g})")
    if not any(l < 0 for l in labels):
        why.append("no rest trials, so the false-trigger level could not be measured")
    prof.reasons, prof.usable = why, not why
    report = ["SSVEP calibration", *lines, "Chosen: " + prof.summary()]
    report += ["Problems: " + "; ".join(why)] if why else []
    return prof, "\n".join(report)


# ----------------------------------------------------------------------------- synthetic EEG (rehearsal and tests)
def synth_eeg(n: int, fs: float, n_ch: int = 7, target: int | None = None, freqs=DEFAULT_FREQS, snr: float = 0.5,
              rng=None, t0: float = 0.0, mix=None) -> np.ndarray:
    """Samples x channels of fake occipital EEG: noise plus, if `target` is given, that square's frequency and 2x
    frequency (amplitude `snr` relative to the noise). `mix` = (gain, phase) per channel (from synth_mix) keeps the
    response coherent when a stream is generated chunk by chunk; t0 = time of the first sample."""
    rng = rng or np.random.default_rng()
    x = rng.standard_normal((n, n_ch)) * 10.0
    if target is not None:
        gain, ph = mix if mix is not None else synth_mix(n_ch, rng)
        t = (np.arange(n) / fs + t0)[:, None]
        f = freqs[target]
        x = x + 10.0 * snr * gain * (np.sin(2 * np.pi * f * t + ph) + 0.5 * np.sin(2 * np.pi * 2 * f * t + 2 * ph))
    return x + 1000.0                                                  # the amp's big DC offset


def synth_mix(n_ch: int, rng):
    """Per-channel gain and phase of a fake SSVEP response."""
    return rng.uniform(0.6, 1.4, n_ch)[None, :], rng.uniform(0, 2 * np.pi, n_ch)[None, :]

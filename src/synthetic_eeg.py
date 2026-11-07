"""Fake 24-channel LSL EEG stream with injected gesture-like artifacts.

Modes (combinable):
  default      : background EEG + occasional natural blinks; also FOLLOWS the CueMarkers
                 stream from collect.py and injects the matching gesture after each cue_on.
  --demo       : cycles through every gesture every few seconds (for the signal-check view).
  --no-follow  : ignore markers.

Signals are in microvolts. The model is crude on purpose: it only has to be separable
enough to exercise the whole pipeline (collection, loader, features, real-time loop).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from scipy import signal as sps

# Real eego layout seen in the first recording (names exactly as the amplifier reports them).
CHANNELS = ["Fp1", "Fp2", "F9", "F7", "F3", "Fz", "F4", "F8", "F10", "M1", "t7", "c3", "c4", "t8", "M2",
            "cz", "p7", "p3", "pz", "p4", "p8", "oz", "o1", "o2"]
# crude scalp coordinates: x = left(-)/right(+), y = back(-)/front(+). Looked up case-insensitively.
COORDS = {
    "fp1": (-.3, 1), "fp2": (.3, 1), "f9": (-.95, .75), "f10": (.95, .75), "f7": (-.8, .6), "f3": (-.4, .6),
    "fz": (0, .6), "f4": (.4, .6), "f8": (.8, .6), "fc1": (-.2, .35), "fc2": (.2, .35), "m1": (-1.0, -.35),
    "m2": (1.0, -.35), "t7": (-1, 0), "c3": (-.5, 0), "cz": (0, 0), "c4": (.5, 0), "t8": (1, 0),
    "cp1": (-.2, -.3), "cp2": (.2, -.3), "p7": (-.8, -.6), "p3": (-.4, -.6), "pz": (0, -.6), "p4": (.4, -.6),
    "p8": (.8, -.6), "o1": (-.3, -1), "o2": (.3, -1), "oz": (0, -1)}

GESTURES = ["left_wink", "right_wink", "double_blink", "jaw_clench", "eyebrow_raise",
            "head_left", "head_right", "head_up", "head_down",
            "look_left", "look_right", "look_up", "look_down",
            "talking", "chewing", "natural_blinking", "typing"]
REACTION_S = 0.25   # delay between cue onset and gesture onset


def _pink_state(rng, nch):
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    return b, a


class SignalModel:
    def __init__(self, fs=500, ch_names=None, seed=None, natural_blinks=True, line_hz=60.0,
                 noise_scale=1.0, gesture_scale=1.0):
        self.fs = fs
        self.noise_scale, self.gesture_scale = noise_scale, gesture_scale
        self.names = list(ch_names or CHANNELS)
        self.nch = len(self.names)
        self.rng = np.random.default_rng(seed)
        self.xy = np.array([COORDS.get(n.lower(), (0.0, 0.0)) for n in self.names], dtype=float)
        self.known = np.array([n.lower() in COORDS for n in self.names])
        self.b, self.a = _pink_state(self.rng, self.nch)
        self.zi = np.zeros((max(len(self.a), len(self.b)) - 1, self.nch))
        # calibrate pink filter gain to ~8 uV rms
        test = sps.lfilter(self.b, self.a, self.rng.standard_normal(40000))
        self.pink_gain = 8.0 / test[2000:].std()
        self.line_hz = line_hz
        self.n = 0                      # samples generated so far
        self.events: list[tuple[int, np.ndarray]] = []
        self.natural_blinks = natural_blinks
        self.next_blink = int(self.rng.uniform(2, 6) * fs)
        self.alpha_phase = self.rng.uniform(0, 2 * np.pi, self.nch)
        occ = np.exp(-np.linalg.norm(self.xy - np.array([0, -1.0]), axis=1) / 0.5)
        self.alpha_w = occ * 10.0       # up to 10 uV alpha at Oz
        self.hf_sos = sps.butter(4, [30, min(100, fs / 2 - 1)], btype="band", fs=fs, output="sos")

    # ---- spatial weights
    def w(self, x, y, lam):
        d = np.linalg.norm(self.xy - np.array([x, y]), axis=1)
        return np.exp(-d / lam) * self.known

    # ---- building blocks
    def _t(self, dur):
        return np.arange(int(dur * self.fs)) / self.fs

    def _bump(self, dur, t0, width):
        t = self._t(dur)
        return np.where((t >= t0) & (t < t0 + width), np.sin(np.pi * (t - t0) / width) ** 2, 0.0)

    def _step(self, dur, t_on, t_off, rise=0.12):
        t = self._t(dur)
        up = 1 / (1 + np.exp(-(t - t_on) / (rise / 4)))
        dn = 1 / (1 + np.exp(-(t - t_off) / (rise / 4)))
        return up - dn

    def _hf(self, dur, env):
        n = len(env)
        noise = sps.sosfilt(self.hf_sos, self.rng.standard_normal(n + 400))[400:]
        noise /= noise.std() + 1e-9
        return noise * env

    def make(self, label: str, dur: float) -> np.ndarray | None:
        n = int(dur * self.fs)
        dur = max(dur, 0.5)
        out = np.zeros((int(max(dur, 0.5) * self.fs), self.nch))
        fp = self.w(0, 1.2, 0.6)
        ampj = lambda lo, hi: self.rng.uniform(lo, hi)
        if label == "double_blink":
            sh = self._bump(dur, 0.0, 0.2) + self._bump(dur, 0.35, 0.2)
            out += np.outer(sh, fp) * ampj(100, 140)
        elif label == "natural_blink":
            out += np.outer(self._bump(dur, 0.0, 0.2), fp) * ampj(60, 110)
        elif label in ("left_wink", "right_wink"):
            sgn = -1 if label == "left_wink" else 1
            near = self.w(sgn * .3, 1.1, 0.35)
            far = self.w(-sgn * .3, 1.1, 0.35) * 0.1
            out += np.outer(self._bump(dur, 0.0, 0.3), near + far) * ampj(90, 130)
        elif label == "jaw_clench":
            env = np.clip(np.minimum(self._t(dur) / 0.1, 1), 0, 1) * (self._t(dur) < min(dur, 1.4))
            temporal = self.w(-1, 0, 0.5) + self.w(1, 0, 0.5)
            out += np.stack([self._hf(dur, env * 22) * temporal[c] for c in range(self.nch)], axis=1)
            out += np.outer(self._bump(dur, 0, 0.3) * 8, self.w(0, 0.8, 0.8))   # small movement
        elif label == "eyebrow_raise":
            env = self._bump(dur, 0.0, min(dur, 1.4))
            out += np.outer(env, self.w(0, 1.1, 0.5)) * ampj(25, 40)
            out += np.stack([self._hf(dur, env * 6) * self.w(0, 1.1, 0.5)[c] for c in range(self.nch)], axis=1)
        elif label in ("head_left", "head_right"):
            sgn = -1 if label == "head_left" else 1
            slow = self._step(dur, 0.2, 0.2 + 1.2, rise=0.3)
            env = self._bump(dur, 0.0, 0.5) + self._bump(dur, 1.0, 0.5)
            lat = self.xy[:, 0] * sgn * self.known
            out += np.outer(slow, lat) * ampj(35, 50)
            neck = self.w(sgn * 0.8, -0.6, 0.5) + self.w(-sgn * 0.8, -0.6, 0.5) * 0.4
            out += np.stack([self._hf(dur, env * 9) * neck[c] for c in range(self.nch)], axis=1)
        elif label in ("head_up", "head_down"):
            sgn = 1 if label == "head_up" else -1
            slow = self._step(dur, 0.2, 1.4, rise=0.3)
            env = self._bump(dur, 0.0, 0.5) + self._bump(dur, 1.0, 0.5)
            out += np.outer(slow, self.xy[:, 1] * sgn * self.known) * ampj(35, 50)
            neck = self.w(0, -1, 0.5) if sgn > 0 else self.w(0, 0.9, 0.4) * 0.6
            out += np.stack([self._hf(dur, env * 8) * neck[c] for c in range(self.nch)], axis=1)
        elif label in ("look_left", "look_right"):
            sgn = -1 if label == "look_left" else 1
            dip = (self.w(.8, .6, 0.35) - self.w(-.8, .6, 0.35)) * sgn      # F8 +, F7 - for right
            dip += (self.w(.3, 1, 0.4) - self.w(-.3, 1, 0.4)) * sgn * 0.5
            out += np.outer(self._step(dur, 0.15, 1.45, rise=0.1), dip) * ampj(35, 50)
        elif label in ("look_up", "look_down"):
            sgn = 1 if label == "look_up" else -1
            out += np.outer(self._step(dur, 0.15, 1.45, rise=0.1), self.w(0, 1.2, 0.6)) * sgn * ampj(35, 50)
        elif label == "talking":
            t = self._t(dur)
            env = (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t + self.rng.uniform(0, 6))) ** 2 * (np.sin(2 * np.pi * 0.3 * t) > -0.6)
            tw = self.w(-1, 0, 0.5) + self.w(1, 0, 0.5) + 0.5 * (self.w(-.8, .6, .4) + self.w(.8, .6, .4))
            out += np.stack([self._hf(dur, env * 9) * tw[c] for c in range(self.nch)], axis=1)
        elif label == "chewing":
            t = self._t(dur)
            env = np.clip(np.sin(2 * np.pi * 1.6 * t), 0, None) ** 2
            tw = self.w(-1, 0, 0.5) + self.w(1, 0, 0.5)
            out += np.stack([self._hf(dur, env * 16) * tw[c] for c in range(self.nch)], axis=1)
            out += np.outer(env * 4, self.w(0, 0.8, 0.8))
        elif label == "natural_blinking":
            k = max(1, int(dur / 2.2))
            for _ in range(k):
                s = int(self.rng.uniform(0, max(dur - 0.3, 0.1)) * self.fs)
                b = self.make("natural_blink", 0.3)
                m = min(len(b), len(out) - s)
                if m > 0:
                    out[s:s + m] += b[:m]
        elif label == "typing":
            t = self._t(dur)
            env = (self.rng.random(len(t) // 50 + 1) > 0.5).repeat(50)[:len(t)].astype(float)
            out += np.stack([self._hf(dur, env * 3) * self.w(0, 0, 1.2)[c] for c in range(self.nch)], axis=1)
            out += np.outer(self.rng.standard_normal(len(t)) * 0.3, np.ones(self.nch))
        elif label == "eyes_closed":
            t = self._t(dur)
            osc = np.sin(2 * np.pi * 10 * t)
            out += np.outer(osc * np.clip(t / 0.3, 0, 1), self.alpha_w * 3)
        else:
            return None
        return out

    def schedule(self, label: str, at_sample: int, dur: float) -> bool:
        g = self.make(label, dur)
        if g is None:
            return False
        self.events.append((int(at_sample), g * self.gesture_scale))
        return True

    # ---- streaming
    def generate(self, count: int) -> np.ndarray:
        fs = self.fs
        white = self.rng.standard_normal((count, self.nch))
        base, self.zi = sps.lfilter(self.b, self.a, white, axis=0, zi=self.zi)
        base *= self.pink_gain * self.noise_scale
        t = (self.n + np.arange(count)) / fs
        alpha_env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.17 * t)
        alpha = np.sin(2 * np.pi * 10.0 * t[:, None] + self.alpha_phase) * self.alpha_w * alpha_env[:, None]
        line = 1.5 * np.sin(2 * np.pi * self.line_hz * t)[:, None] * np.ones(self.nch)
        x = base + alpha + line
        if self.natural_blinks and self.n + count > self.next_blink:
            self.schedule("natural_blink", self.next_blink, 0.3)
            self.next_blink += int(self.rng.uniform(4, 9) * fs)
        keep = []
        for start, arr in self.events:
            a0, a1 = max(start, self.n), min(start + len(arr), self.n + count)
            if a1 > a0:
                x[a0 - self.n:a1 - self.n] += arr[a0 - start:a1 - start]
            if start + len(arr) > self.n + count:
                keep.append((start, arr))
        self.events = keep
        self.n += count
        return x.astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--name", default="SyntheticEEG")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--demo", action="store_true", help="cycle through all gestures")
    ap.add_argument("--demo-period", type=float, default=3.5)
    ap.add_argument("--no-follow", action="store_true", help="ignore CueMarkers")
    ap.add_argument("--no-natural-blinks", action="store_true")
    ap.add_argument("--marker-name", default="CueMarkers")
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = forever)")
    args = ap.parse_args(argv)

    import pylsl
    model = SignalModel(args.fs, seed=args.seed, natural_blinks=not args.no_natural_blinks)
    info = pylsl.StreamInfo(args.name, "EEG", model.nch, args.fs, "float32", f"synthetic-{args.name}")
    chs = info.desc().append_child("channels")
    for n in model.names:
        c = chs.append_child("channel")
        c.append_child_value("label", n)
        c.append_child_value("unit", "microvolts")
        c.append_child_value("type", "EEG")
    outlet = pylsl.StreamOutlet(info, chunk_size=0, max_buffered=60)
    print(f"[synthetic] streaming '{args.name}' {model.nch} ch @ {args.fs} Hz. Ctrl+C to stop.", flush=True)

    marker_inlet = None
    last_resolve = 0.0
    t0 = pylsl.local_clock()
    sent = 0
    demo_i, next_demo = 0, 2.0
    try:
        while True:
            now = pylsl.local_clock()
            if args.duration and now - t0 > args.duration:
                break
            if not args.no_follow:
                if marker_inlet is None and now - last_resolve > 2.0:
                    last_resolve = now
                    res = pylsl.resolve_byprop("name", args.marker_name, timeout=0.01)
                    if res:
                        marker_inlet = pylsl.StreamInlet(res[0])
                        print("[synthetic] following marker stream", flush=True)
                if marker_inlet is not None:
                    while True:
                        s, mts = marker_inlet.pull_sample(timeout=0.0)
                        if s is None:
                            break
                        try:
                            m = json.loads(s[0])
                        except Exception:
                            continue
                        if m.get("event") == "cue_on":
                            at = int((mts + REACTION_S - t0) * args.fs)
                            model.schedule(m["label"], at, float(m.get("cue_s", 1.5)) + 0.5)
            if args.demo and now - t0 > next_demo:
                lab = GESTURES[demo_i % len(GESTURES)] if demo_i % 5 else "eyes_closed"
                demo_i += 1
                next_demo += args.demo_period
                dur = 4.0 if lab == "eyes_closed" else 1.8
                model.schedule(lab, int((now - t0 + 0.1) * args.fs), dur)
                print(f"[synthetic] demo gesture: {lab}", flush=True)
            target = int((now - t0) * args.fs)
            n = target - sent
            if n > 0:
                x = model.generate(n)
                outlet.push_chunk(x.tolist(), t0 + (sent + n - 1) / args.fs)
                sent += n
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    print("[synthetic] stopped", flush=True)


if __name__ == "__main__":
    sys.exit(main())

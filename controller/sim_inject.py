"""Simulated EEG built from recorded sessions, streamed live into the controller (backup for when the LSL stream
is down). Use: python -m controller --eeg inject [--sim-data SPEC [SPEC ...]]

Sources (pooled): every recorded session with cue markers in data/: the per-session calibrations (s5_*) and the
model training sessions (s1_*, s2_*). A session folder whose EEG is a separate .cnt (training sessions recorded on
ANT's laptop) is paired automatically with the data/*.cnt that was recording when its cues started, or give it
as FOLDER@EEGFILE like src/train.py.

  background: the wearer's recorded "rest" trials, looped
  inject:     a recorded trial of the chosen class (gesture or confounder) is spliced in at the moment you press.
              Holding keeps the steady part looping (held clench / head turn); releasing plays the relaxation.
  forge:      (default on) each injection is a new variant rather than an exact replay: per-channel gain
              0.85-1.15 and the activity of another rest trial mixed in, so the model never sees the same
              samples twice.

Everything downstream is unchanged: the samples land in the same buffer the LSL stream fills, and the team
model reads them through its normal streaming path (filters, windows, features, classifier, detector, actions).
Each segment is offset to start exactly where the previous one ended, so the cuts add no steps.
"""
import csv
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .eeg_source import _Buffer
from .log import log

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PRE_S, POST_S = 0.3, 1.0         # clip = cue_on - PRE_S ... cue_off + POST_S
STEADY_FROM_S = 0.4              # hold loop = cue_on + STEADY_FROM_S ... cue_off
CHUNK_S = 0.02
GAIN = (0.85, 1.15)
MIX = 0.5                        # weight of the other rest trial mixed into a forged trial
CNT_NAME = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.cnt$", re.I)


def _session_start(folder):
    try:
        with open(folder / "markers.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                return float(r["wall_unix"])
    except Exception:
        return None


def _cnt_minutes(path):
    """Approximate length of an eego .cnt from its size (~1.15 MB per minute for 24 ch at 500 Hz; measured on
    our recordings). Reading the header would load the native library, which can abort on a bad file."""
    return path.stat().st_size / 1.15e6


def _pair_cnt(folder):
    """The data/*.cnt (named ..._YYYY-MM-DD_HH-MM-SS.cnt) that was recording when this session's first cue came."""
    t = _session_start(folder)
    if t is None:
        return None
    for c in sorted(DATA.glob("*.cnt")):
        m = CNT_NAME.search(c.name)
        if not m:
            continue
        start = datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S").timestamp()
        if start <= t:
            minutes = _cnt_minutes(c)
            if t <= start + minutes * 60 * 1.1:
                return c
    return None


def find_sessions():
    """Every finished session in data/ as FOLDER or FOLDER@EEGFILE specs, largest first."""
    out = []
    for d in sorted(p for p in DATA.glob("s*_*") if p.is_dir()):
        if not (d / "markers.csv").exists():
            continue
        npz = d / "eeg_backup.npz"
        if (npz.exists() and npz.stat().st_size > 100_000) or list(d.glob("*.xdf")):
            out.append((npz.stat().st_size if npz.exists() else 0, str(d)))
        elif not npz.exists():                       # EEG recorded elsewhere (training sessions): find its .cnt
            cnt = _pair_cnt(d)
            if cnt is not None:
                out.append((cnt.stat().st_size, f"{d}@{cnt}"))
    return [s for _, s in sorted(out, reverse=True)]


class InjectSource:
    def __init__(self, specs=None, seed=None, forge=True):
        import sys
        src = str(REPO / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from common import load_config
        from load_data import load_recording
        cfg = load_config()
        if isinstance(specs, (str, Path)):
            specs = [specs]
        specs = list(specs) if specs else find_sessions()
        if not specs:
            raise RuntimeError("no recorded session to simulate from (needs data/s*_*/ with markers.csv and its EEG: "
                               "eeg_backup.npz, an .xdf, or the session's .cnt in data/)")
        self.rng = np.random.default_rng(seed)
        self.forge = forge
        self.recs, self.specs, self.trials, self.rest = [], [], {}, []
        self.channels = None
        for spec in specs:
            folder, _, eeg = str(spec).partition("@")
            try:
                rec = load_recording(folder, units_to_uv=cfg["eeg_stream"].get("units_to_uv", 1.0),
                                     eeg_file=eeg or None, verbose=False)
                names, live = self._names(rec, cfg)
            except Exception as e:
                log("ERROR", f"skipping {Path(folder).name} for simulation ({type(e).__name__}: {e})")
                continue
            if not any(e.get("event") == "cue_on" and e.get("label") and float(e["t"]) >= rec.ts[0]
                       and float(e["t"]) <= rec.ts[-1] for e in rec.events):
                log("ERROR", f"skipping {Path(folder).name} for simulation: no cues inside its EEG")
                continue
            if self.channels is None:
                self.fs, self.channels, self.live = float(rec.fs), names, live
            elif float(rec.fs) != self.fs or [n.lower() for n in names] != [n.lower() for n in self.channels]:
                log("ERROR", f"skipping {Path(folder).name}: different rate or channel layout", fs=rec.fs)
                continue
            i = len(self.recs)
            self.recs.append(np.asarray(rec.data, dtype=np.float32))
            self.specs.append(str(spec))
            n_before = sum(len(v) for v in self.trials.values())
            self._cut(rec, i)
            log("EEG", "simulation source", session=Path(folder).name, eeg=Path(eeg).name if eeg else "in folder",
                trials=sum(len(v) for v in self.trials.values()) - n_before,
                minutes=round(len(rec.data) / rec.fs / 60, 1))
        if not self.recs:
            raise RuntimeError("none of the recorded sessions could be loaded for simulation")
        if not self.rest:
            raise RuntimeError("the recorded sessions have no rest trials to use as background")
        self.folder = Path(self.specs[0].partition("@")[0])
        self.name = f"simulated ({len(self.recs)} session{'s' if len(self.recs) > 1 else ''})"
        self.labels = sorted(self.trials)
        self.buf = _Buffer(len(self.channels), self.fs)
        self.lock = threading.Lock()
        self.queue, self.cur, self.pos = [], None, 0
        self.last = np.zeros(len(self.channels), dtype=np.float32)
        self.hold = None
        self.pending_tail = None
        self.in_steady = False
        self.injected = None
        self.injected_seq = 0
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        log("EEG", "simulated EEG ready", sessions=len(self.recs), fs=self.fs, channels=len(self.channels), forge=forge,
            classes=",".join(f"{k}:{len(v)}" for k, v in sorted(self.trials.items())), rest_trials=len(self.rest))

    def _names(self, rec, cfg):
        """Same channel naming as the live LSL stream (src/channels.py)."""
        from channels import detect_live, name_channels
        live = detect_live(np.asarray(rec.data[: int(5 * rec.fs)]))
        names, _ = name_channels(list(rec.ch_names), cfg, live)
        return names, live

    # ----------------------------------------------------------- library
    def _cut(self, rec, ri):
        ts, fs, n = np.asarray(rec.ts), float(rec.fs), len(rec.data)
        cue_times = sorted(float(e["t"]) for e in rec.events if e.get("event") == "cue_on")
        for e in rec.events:
            if e.get("event") != "cue_on" or not e.get("label"):
                continue
            cue = float(e.get("cue_s") or 1.5)
            i0 = int(np.searchsorted(ts, float(e["t"])))
            a, b = i0 - int(PRE_S * fs), i0 + int((cue + POST_S) * fs)
            if a < 0 or b > n:
                continue
            lab = e["label"]
            if lab == "rest":                        # the rest cue and the relaxed gap after it, up to the next cue
                nxt = [t for t in cue_times if t > float(e["t"]) + 0.1]
                end = int(np.searchsorted(ts, nxt[0])) if nxt else i0 + int(cue * fs)
                self.rest.append(dict(rec=ri, all=(i0, min(end, i0 + int(4.0 * fs), n))))
                continue
            s0, s1 = i0 + int(STEADY_FROM_S * fs), i0 + int(cue * fs)
            self.trials.setdefault(lab, []).append(dict(rec=ri, head=(a, s1), steady=(s0, s1), tail=(s1, b),
                                                        trial=e.get("trial")))

    def _seg(self, t, part):
        a, b = t[part]
        return self.recs[t["rec"]][a:b]

    def _forger(self):
        """A per-injection transform: per-channel gain plus another rest trial's activity mixed in."""
        if not self.forge:
            return lambda x: x
        g = self.rng.uniform(*GAIN, len(self.channels)).astype(np.float32)
        r = self._seg(self.rest[int(self.rng.integers(len(self.rest)))], "all").astype(np.float32)
        r = r - np.linspace(r[0], r[-1], len(r)).astype(np.float32)     # drop its drift, keep its activity
        off = int(self.rng.integers(len(r)))

        def f(x):
            x = x.astype(np.float32)
            base = x[:1]
            idx = (off + np.arange(len(x))) % len(r)
            return base + (x - base) * g + MIX * r[idx]
        return f

    # ----------------------------------------------------------- control
    def inject(self, label, hold=False):
        """Start a recorded (forged) `label` trial now. hold=True loops its steady part until release()."""
        trials = self.trials.get(label)
        if not trials:
            log("ERROR", f"no recorded '{label}' trials to simulate")
            return
        t = trials[int(self.rng.integers(len(trials)))]
        f = self._forger()
        with self.lock:
            self.queue = [f(self._seg(t, "head"))]
            self.cur, self.pos = None, 0          # cut the background now
            self.in_steady = False
            self.pending_tail = f(self._seg(t, "tail"))
            self.hold = (label, f(self._seg(t, "steady"))) if hold else None
            if not hold:
                self.queue.append(self.pending_tail)
                self.pending_tail = None
        session = Path(self.specs[t["rec"]].partition("@")[0]).name
        self.injected = (label, t["trial"], session)
        self.injected_seq += 1
        log("SIM", f"injected {'forged ' if self.forge else ''}{label}", session=session, trial=t["trial"], hold=hold)

    def release(self):
        with self.lock:
            if self.hold is None:
                return
            label = self.hold[0]
            self.hold = None
            if self.pending_tail is not None:
                self.queue.append(self.pending_tail)
                self.pending_tail = None
            if self.in_steady:
                self.cur = None                    # stop the held part now, go straight to the relaxation
        log("SIM", f"released {label}")

    # ----------------------------------------------------------- stream
    def _next_segment(self):
        self.in_steady = False
        if self.queue:
            return self.queue.pop(0)
        if self.hold is not None:
            self.in_steady = True
            return self.hold[1]
        return self._seg(self.rest[int(self.rng.integers(len(self.rest)))], "all")

    def _take(self, k):
        out = np.empty((k, len(self.channels)), dtype=np.float32)
        got = 0
        with self.lock:
            while got < k:
                if self.cur is None or self.pos >= len(self.cur):
                    seg = self._next_segment().astype(np.float32)
                    # continue from the last sample (no step), keeping the segment's own shape and drift
                    self.cur, self.pos = seg + (self.last - seg[0])[None, :], 0
                n = min(k - got, len(self.cur) - self.pos)
                out[got:got + n] = self.cur[self.pos:self.pos + n]
                self.pos += n
                got += n
                self.last = out[got - 1].copy()
        return out

    def _loop(self):
        t0, sent = time.time(), 0
        while self.running:
            due = int((time.time() - t0) * self.fs) - sent
            if due > 0:
                self.buf.push(self._take(due))
                sent += due
            time.sleep(CHUNK_S)

    def dead_names(self):
        return [] if self.live is None else [c for c, v in zip(self.channels, self.live) if not v]

    @property
    def n_running(self):
        return len(self.channels) - len(self.dead_names())

    def window(self, seconds):
        return self.buf.window(seconds)

    @property
    def connected(self):
        return self.running

    def close(self):
        self.running = False

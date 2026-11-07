"""LSL helpers: stream resolving/description, marker sender (LSL + CSV), EEG recorder."""
from __future__ import annotations

import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pylsl


# ---------------------------------------------------------------- stream info
def resolve_stream(stype: str, name: str | None = None, timeout: float = 5.0):
    """Return the first matching StreamInfo or None."""
    if name:
        res = pylsl.resolve_byprop("name", name, timeout=timeout)
    else:
        res = pylsl.resolve_byprop("type", stype, timeout=timeout)
    return res[0] if res else None


def channel_labels(info: pylsl.StreamInfo) -> list[str]:
    """Channel labels advertised in the stream description (empty list if absent)."""
    labels = []
    try:
        ch = info.desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
            labels.append(ch.child_value("label") or ch.child_value("name") or "")
            ch = ch.next_sibling()
    except Exception:
        return []
    return labels if any(labels) else []


def describe_stream(info: pylsl.StreamInfo) -> dict:
    units = ""
    try:
        units = info.desc().child("channels").child("channel").child_value("unit")
    except Exception:
        pass
    return dict(name=info.name(), type=info.type(), n_channels=info.channel_count(),
                srate=info.nominal_srate(), fmt=info.channel_format(),
                source_id=info.source_id(), hostname=info.hostname(),
                labels=channel_labels(info), unit=units)


def print_stream_info(d: dict, cfg: dict | None = None) -> list[str]:
    """Print stream details; return a list of warnings (config mismatches)."""
    warns = []
    print("=" * 60)
    print(f"LSL stream : {d['name']}  (type {d['type']}, host {d['hostname']})")
    print(f"Channels   : {d['n_channels']}   Sampling rate: {d['srate']} Hz   Unit: {d['unit'] or '?'}")
    print(f"Source id  : {d['source_id']}")
    if d["labels"]:
        print("Channel labels advertised by the stream:")
        print("  " + ", ".join(f"{i}:{l}" for i, l in enumerate(d["labels"])))
    else:
        print("Stream advertises NO channel labels - fill them into config.yaml 'channels'.")
    if cfg:
        if abs(d["srate"] - cfg.get("sampling_rate", d["srate"])) > 1e-6:
            warns.append(f"config sampling_rate={cfg['sampling_rate']} differs from stream {d['srate']} "
                         "(stream value is used)")
        names = d["labels"] or cfg.get("channels", [])
        if len(names) != d["n_channels"]:
            warns.append(f"channel name count {len(names)} != stream channels {d['n_channels']}; "
                         "using generic names ch0..chN")
        else:
            have = {n.lower() for n in names}
            missing = sorted({c for g in cfg.get("channel_groups", {}).values() for c in g if c.lower() not in have})
            if missing:
                warns.append(f"channel_groups reference channels not in the stream: {missing}")
    for w in warns:
        print("WARNING:", w)
    print("=" * 60)
    return warns


def effective_channel_names(d: dict, cfg: dict) -> list[str]:
    n = d["n_channels"]
    if len(d["labels"]) == n:
        return list(d["labels"])
    if len(cfg.get("channels", [])) == n:
        return list(cfg["channels"])
    return [f"ch{i}" for i in range(n)]


# ------------------------------------------------------------------- markers
CSV_FIELDS = ["wall_unix", "wall_iso", "lsl_time", "event", "session", "trial_id", "block",
              "label", "display", "cue_s", "t_rel"]   # t_rel = seconds since session_start (monotonic)


class MarkerSender:
    """Sends JSON markers on an LSL stream AND appends them to a CSV (flushed per row)."""

    def __init__(self, csv_path: str | Path, session: int, name="CueMarkers", stype="Markers",
                 use_lsl: bool = True):
        self.session = session
        self.t0 = time.perf_counter()
        self.csv_path = Path(csv_path)
        self._f = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=CSV_FIELDS)
        self._w.writeheader()
        self._f.flush()
        self.outlet = None
        if use_lsl:
            info = pylsl.StreamInfo(name, stype, 1, pylsl.IRREGULAR_RATE, "string",
                                    f"ant-collect-s{session}-{int(time.time())}")
            self.outlet = pylsl.StreamOutlet(info)

    def reset_clock(self):
        """Call at session start so t_rel is measured from there."""
        self.t0 = time.perf_counter()

    def send(self, event: str, trial=None, **extra) -> float:
        payload = dict(event=event, session=self.session,
                       trial=getattr(trial, "idx", -1), block=getattr(trial, "block", -1),
                       label=getattr(trial, "label", extra.pop("label", "")),
                       cue_s=getattr(trial, "cue_s", 0.0))
        payload.update(extra)
        lsl_t = pylsl.local_clock()
        wall = time.time()
        t_rel = time.perf_counter() - self.t0
        if self.outlet is not None:
            self.outlet.push_sample([json.dumps(payload)], lsl_t)
        self._w.writerow(dict(
            wall_unix=f"{wall:.6f}", wall_iso=datetime.fromtimestamp(wall).isoformat(timespec="milliseconds"),
            lsl_time=f"{lsl_t:.6f}", event=event, session=self.session,
            trial_id=payload["trial"], block=payload["block"], label=payload["label"],
            display=getattr(trial, "display", ""), cue_s=payload["cue_s"], t_rel=f"{t_rel:.6f}"))
        self._f.flush()
        return lsl_t

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


# ------------------------------------------------------------------ recorder
class EegRecorder(threading.Thread):
    """Background EEG puller. Keeps a rolling ring for live views and (when
    recording) stores every sample; flushes segments to disk so a crash loses <60 s."""

    def __init__(self, info: pylsl.StreamInfo, ring_seconds: float = 30.0,
                 units_to_uv: float = 1.0, segment_dir: str | Path | None = None,
                 segment_every_s: float = 60.0):
        super().__init__(daemon=True)
        self.inlet = pylsl.StreamInlet(info, max_buflen=360, processing_flags=pylsl.proc_clocksync)
        self.nch = info.channel_count()
        self.fs = info.nominal_srate()
        self.scale = units_to_uv
        self.ring_n = int(ring_seconds * self.fs)
        self.ring = np.zeros((self.ring_n, self.nch), dtype=np.float32)
        self.ring_ts = np.zeros(self.ring_n, dtype=np.float64)
        self.ring_count = 0  # total samples seen
        self.lock = threading.Lock()
        self.recording = False
        self._chunks: list[np.ndarray] = []
        self._tss: list[np.ndarray] = []
        self._unflushed = 0
        self.segment_dir = Path(segment_dir) if segment_dir else None
        self.segment_every_s = segment_every_s
        self._seg_i = 0
        self._last_flush = time.time()
        self._halt = threading.Event()
        self.last_sample_wall = 0.0
        self.error: str | None = None

    # -- thread
    def run(self):
        try:
            while not self._halt.is_set():
                x, ts = self.inlet.pull_chunk(timeout=0.1, max_samples=2048)
                if ts:
                    a = np.asarray(x, dtype=np.float32) * self.scale
                    t = np.asarray(ts, dtype=np.float64)
                    self._ingest(a, t)
                self._maybe_flush()
        except Exception as e:  # keep the app alive, surface the error in the UI
            self.error = repr(e)

    def _ingest(self, a, t):
        n = len(t)
        with self.lock:
            if n >= self.ring_n:
                a, t = a[-self.ring_n:], t[-self.ring_n:]
                n = self.ring_n
            self.ring = np.roll(self.ring, -n, axis=0)
            self.ring_ts = np.roll(self.ring_ts, -n)
            self.ring[-n:] = a
            self.ring_ts[-n:] = t
            self.ring_count += n
            if self.recording:
                self._chunks.append(a)
                self._tss.append(t)
                self._unflushed += n
            self.last_sample_wall = time.time()

    def _maybe_flush(self):
        if self.recording and self.segment_dir and time.time() - self._last_flush >= self.segment_every_s:
            self.flush_segment()

    # -- API
    def recent(self, seconds: float):
        """Return (data[n, ch], ts[n]) for the latest `seconds` seen (may be shorter)."""
        n = min(int(seconds * self.fs), self.ring_count, self.ring_n)
        with self.lock:
            if n <= 0:
                return np.zeros((0, self.nch), np.float32), np.zeros(0)
            return self.ring[-n:].copy(), self.ring_ts[-n:].copy()

    def start_recording(self):
        with self.lock:
            self._chunks, self._tss, self._unflushed = [], [], 0
            self.recording = True
        self._last_flush = time.time()

    def flush_segment(self):
        """Write samples gathered since the last flush to a part file."""
        with self.lock:
            if not self._chunks or self.segment_dir is None:
                return
            a = np.concatenate(self._chunks)
            t = np.concatenate(self._tss)
            self._chunks, self._tss, self._unflushed = [], [], 0
            self._seg_i += 1
            i = self._seg_i
        tmp = self.segment_dir / f"eeg_part_{i:04d}.tmp.npz"
        np.savez(tmp, data=a, timestamps=t)
        tmp.replace(self.segment_dir / f"eeg_part_{i:04d}.npz")
        self._last_flush = time.time()

    def stop(self):
        self._halt.set()
        self.join(timeout=2)

    def finalize(self, out_path: str | Path, ch_names: list[str], extra: dict | None = None) -> dict:
        """Stop recording, merge parts into one .npz, delete parts. Returns summary."""
        self.recording = False
        self.flush_segment()
        d = Path(self.segment_dir)
        data, ts = load_parts(d)
        np.savez(out_path, data=data, timestamps=ts, ch_names=np.array(ch_names),
                 fs=self.fs, units="uV", **(extra or {}))
        for p in d.glob("eeg_part_*.npz"):
            p.unlink()
        n = len(ts)
        meas = (n - 1) / (ts[-1] - ts[0]) if n > 1 and ts[-1] > ts[0] else 0.0
        return dict(n_samples=n, measured_srate=meas, duration_s=(ts[-1] - ts[0]) if n > 1 else 0.0)


def load_parts(folder: str | Path):
    parts = sorted(Path(folder).glob("eeg_part_*.npz"))
    if not parts:
        return np.zeros((0, 0), np.float32), np.zeros(0)
    ds, ts = [], []
    for p in parts:
        z = np.load(p)
        ds.append(z["data"])
        ts.append(z["timestamps"])
    return np.concatenate(ds), np.concatenate(ts)

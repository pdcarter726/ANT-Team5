"""EEG input. LSLSource reads the live eego stream over Lab Streaming Layer; FileSource replays a recording
(e.g. data/*.cnt) in real time as if it were live; SimSource makes synthetic EEG (testing without the
headset). All keep a rolling buffer and hand out the latest window as a (channels x samples) numpy array.
Channel names are matched without regard to case (the eego software writes e.g. "t7" and "Fp1")."""
import threading
import time
from pathlib import Path

import numpy as np

from .log import log

# Our cap's layout, as written in the recordings (data/*.cnt): reference CPz, 500 Hz
EEGO24_CHANNELS = ["Fp1", "Fp2", "F9", "F7", "F3", "Fz", "F4", "F8", "F10", "M1", "t7", "c3",
                   "c4", "t8", "M2", "cz", "p7", "p3", "pz", "p4", "p8", "oz", "o1", "o2"]


def find_channel(channels, name):
    """Index of a channel by name, ignoring case; None if absent."""
    low = [c.lower() for c in channels]
    return low.index(name.lower()) if name.lower() in low else None


class _Buffer:
    def __init__(self, n_ch, fs, seconds=10.0):
        self.fs = float(fs)
        self.n = int(seconds * fs)
        self.data = np.zeros((n_ch, self.n), dtype=np.float32)
        self.count = 0                  # total samples received
        self.lock = threading.Lock()
        self.last_time = 0.0

    def push(self, chunk):            # chunk: samples x channels
        chunk = np.asarray(chunk, dtype=np.float32).T
        k = chunk.shape[1]
        if k == 0:
            return
        with self.lock:
            if k >= self.n:
                self.data[:] = chunk[:, -self.n:]
            else:
                self.data = np.roll(self.data, -k, axis=1)
                self.data[:, -k:] = chunk
            self.count += k
            self.last_time = time.time()

    def since(self, count):
        """Samples received after total-sample index `count`: (channels x k array or None, new count, samples lost).
        Lost > 0 only if the caller fell more than the buffer length behind."""
        with self.lock:
            k = self.count - count
            if k <= 0:
                return None, self.count, 0
            lost = max(0, k - self.n)
            k = min(k, self.n)
            return self.data[:, -k:].copy(), self.count, lost

    def window(self, seconds):
        m = int(seconds * self.fs)
        with self.lock:
            if self.count < m:
                return None
            return self.data[:, -m:].copy()


class LSLSource:
    def __init__(self, name=None, stype="EEG", timeout=15.0):
        from pylsl import StreamInlet, resolve_byprop, resolve_streams
        prop, val = ("name", name) if name else ("type", stype)
        log("EEG", f"looking for LSL stream {prop}={val!r} ...")
        streams = resolve_byprop(prop, val, timeout=timeout)
        if not streams:
            found = [(s.name(), s.type(), s.channel_count()) for s in resolve_streams(wait_time=2.0)]
            raise RuntimeError(f"no LSL stream with {prop}={val!r}; streams on the network: {found}")
        self.inlet = StreamInlet(streams[0], max_buflen=10, recover=True)
        info = self.inlet.info()
        self.name, self.fs, n = info.name(), info.nominal_srate(), info.channel_count()
        labels, ch = [], info.desc().child("channels").child("channel")
        for i in range(n):
            labels.append(ch.child_value("label") or f"ch{i + 1}")
            ch = ch.next_sibling()
        self.channels = labels
        self.buf = _Buffer(n, self.fs)
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        log("EEG", "LSL stream connected", name=self.name, fs=self.fs, channels=n,
            labels=",".join(labels[:24]) + ("..." if n > 24 else ""))

    def _loop(self):
        while self.running:
            chunk, _ = self.inlet.pull_chunk(timeout=0.1, max_samples=4096)
            if chunk:
                self.buf.push(chunk)

    def window(self, seconds):
        return self.buf.window(seconds)

    @property
    def connected(self):
        return time.time() - self.buf.last_time < 1.0

    def close(self):
        self.running = False


class SimSource:
    """Synthetic EEG: background noise + alpha, plus a burst on plausible channels while a test gesture is
    active (only so the plot reacts during testing; the simulated model does not read it)."""
    BURSTS = {"eyebrow_raise": ["Fp1", "Fpz", "Fp2"], "jaw_clench": ["T7", "T8", "F7", "F8"],
              "head_turn_left": ["T8", "P8"], "head_turn_right": ["T7", "P7"],
              "head_turn_up": ["O1", "O2", "M1", "M2"], "head_turn_down": ["Fp1", "Fp2", "Fz"],
              "double_blink": ["Fp1", "Fp2"], "left_wink": ["Fp1", "F7"], "right_wink": ["Fp2", "F8"]}

    def __init__(self, fs=500.0):
        self.name, self.fs, self.channels = "simulated", fs, EEGO24_CHANNELS
        self.buf = _Buffer(len(self.channels), fs)
        self.active = None
        self.running = True
        self.rng = np.random.default_rng(0)
        threading.Thread(target=self._loop, daemon=True).start()
        log("EEG", "simulated EEG source", fs=fs, channels=len(self.channels))

    def _loop(self):
        t0, sent = time.time(), 0
        while self.running:
            due = int((time.time() - t0) * self.fs) - sent
            if due > 0:
                t = (sent + np.arange(due)) / self.fs
                x = self.rng.normal(0, 8, (due, len(self.channels))) + 10 * np.sin(2 * np.pi * 10 * t)[:, None]
                if self.active and self.active[1] > time.time():
                    for c in self.BURSTS.get(self.active[0], []):
                        i = find_channel(self.channels, c)
                        if i is not None:
                            x[:, i] += self.rng.normal(0, 60, due)
                self.buf.push(x)
                sent += due
            time.sleep(0.02)

    def trigger(self, gesture, seconds=0.6):
        self.active = (gesture, time.time() + seconds)

    def window(self, seconds):
        return self.buf.window(seconds)

    @property
    def connected(self):
        return True

    def close(self):
        self.running = False


class FileSource:
    """Replays a recording in real time, looping, as if it were the live stream (needs: pip install mne antio
    for ANT .cnt files; .edf/.vhdr/.fif work with mne alone). Data is in microvolts like the live stream."""

    def __init__(self, path, start_s=0.0):
        import mne
        p = str(path)
        if p.lower().endswith(".cnt"):
            raw = mne.io.read_raw_ant(p, preload=True, verbose="error")
        else:
            raw = mne.io.read_raw(p, preload=True, verbose="error")
        self.data = (raw.get_data() * 1e6).astype(np.float32)      # volts -> microvolts
        self.name = f"replay:{Path(p).name}"
        self.fs, self.channels = float(raw.info["sfreq"]), list(raw.ch_names)
        self.pos = int(start_s * self.fs)
        self.buf = _Buffer(len(self.channels), self.fs)
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        log("EEG", "replaying recording", file=Path(p).name, fs=self.fs, channels=len(self.channels),
            minutes=self.data.shape[1] / self.fs / 60, start_s=start_s)

    def _loop(self):
        t0, sent = time.time(), 0
        n = self.data.shape[1]
        while self.running:
            due = int((time.time() - t0) * self.fs) - sent
            if due > 0:
                idx = (self.pos + sent + np.arange(due)) % n
                self.buf.push(self.data[:, idx].T)
                sent += due
            time.sleep(0.02)

    @property
    def position_s(self):
        return (self.pos + self.buf.count) / self.fs

    def window(self, seconds):
        return self.buf.window(seconds)

    @property
    def connected(self):
        return True

    def close(self):
        self.running = False

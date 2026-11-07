"""The live loop: EEG window -> model -> detector -> actions, every model.step_s seconds (own thread)."""
import threading
import time
from collections import deque

import numpy as np

from .log import log


class Engine:
    def __init__(self, source, model, detector, actions):
        self.source, self.model, self.detector, self.actions = source, model, detector, actions
        self.on_probs = lambda probs: None              # set by the UI
        self.on_event = lambda kind, gesture, prob, desc: None
        self.running = False
        self.lat = deque(maxlen=50)
        self.calls = deque(maxlen=50)
        self.last_probs = {}

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        self.actions.release_all()

    @property
    def stats(self):
        now = time.time()
        rate = len([t for t in self.calls if now - t < 2.0]) / 2.0
        return {"model_ms": float(np.mean(self.lat)) if self.lat else 0.0, "calls_per_s": rate}

    def _loop(self):
        nxt = time.time()
        warned = False
        t_hb = time.time()
        while self.running:
            now = time.time()
            if now < nxt:
                time.sleep(min(0.01, nxt - now))
                continue
            nxt = max(nxt + self.model.step_s, now)
            self.actions.sample_cursor()
            win = self.source.window(self.model.window_s)
            if win is None:
                if not warned:
                    log("EEG", "waiting for enough EEG data for the first window", window_s=self.model.window_s)
                    warned = True
                continue
            t0 = time.perf_counter()
            probs = self.model(win, self.source.fs, self.source.channels)
            self.lat.append((time.perf_counter() - t0) * 1000)
            self.calls.append(time.time())
            self.last_probs = probs
            self.on_probs(probs)
            for kind, gesture, p in self.detector.update(probs):
                desc = self.actions.handle(kind, gesture, p)
                log("GESTURE", f"{kind} {gesture}", confidence=p, action=desc or "-")
                self.on_event(kind, gesture, p, desc)
            if time.time() - t_hb > 10:
                t_hb = time.time()
                s = self.stats
                top = max(probs.items(), key=lambda kv: kv[1]) if probs else ("-", 0)
                log("HB", "status", eeg=self.source.name, eeg_live=self.source.connected, model_ms=s["model_ms"],
                    calls_per_s=s["calls_per_s"], top=f"{top[0]}:{top[1]:.2f}", paused=self.actions.paused)

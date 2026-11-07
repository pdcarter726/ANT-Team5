"""Turns a stream of per-window probabilities into gesture events.

A gesture FIRES when it is the top label with probability >= threshold in at least K of the last N windows,
and it is not in its refractory period. While it stays on top it is HELD (for hold-type actions, e.g. holding
a game key); it is RELEASED after `release_windows` windows without it."""
import time
from collections import deque

from .log import log


class GestureDetector:
    def __init__(self, threshold=0.6, k=3, n=4, refractory_s=0.8, release_windows=3):
        self.threshold, self.k, self.n = threshold, k, n
        self.refractory_s, self.release_windows = refractory_s, release_windows
        self.history = deque(maxlen=n)
        self.last_fire = {}
        self.held = None
        self.miss = 0

    def update(self, probs, now=None):
        """probs: {label: p}. Returns a list of events: ("fire", label, p), ("release", label, 0)."""
        now = now or time.time()
        events = []
        label, p = max(probs.items(), key=lambda kv: kv[1]) if probs else ("rest", 1.0)
        top = label if (label != "rest" and p >= self.threshold) else "rest"
        self.history.append(top)
        if self.held:
            if top == self.held:
                self.miss = 0
            else:
                self.miss += 1
                if self.miss >= self.release_windows:
                    events.append(("release", self.held, 0.0))
                    self.held = None
        if top != "rest" and self.history.count(top) >= self.k and top != self.held \
                and now - self.last_fire.get(top, 0) >= self.refractory_s:
            if self.held:                                  # a different gesture took over
                events.append(("release", self.held, 0.0))
            self.last_fire[top] = now
            self.held, self.miss = top, 0
            events.append(("fire", top, p))
        return events

"""Moves the OS cursor from filtered gaze points, with freeze/pause and optional dwell click."""
import math
import time

from .diag import diag
from .filters import GazeSmoother


class CursorController:
    def __init__(self, screen_w, screen_h, mover=None, clicker=None,
                 min_cutoff=0.4, beta=0.003, deadzone=10, margin=6, dwell_click_s=0.0, dwell_radius=45):
        self.w, self.h = screen_w, screen_h
        self.mover, self.clicker = mover, clicker
        self.smooth = GazeSmoother(min_cutoff, beta)
        self.deadzone, self.margin = deadzone, margin
        self.paused = False
        self.frozen_until = 0.0
        self.pos = None
        self.dwell_s, self.dwell_r = dwell_click_s, dwell_radius
        self.dwell_anchor, self.dwell_t0, self.dwell_done = None, 0.0, False

    # ---- commands (from keys, UDP or the ML app) ----
    def freeze(self, seconds, now=None):
        self.frozen_until = max(self.frozen_until, (now or time.time()) + float(seconds))

    def pause(self):
        self.paused = True
        diag.log("CURSOR", "paused")

    def resume(self):
        self.paused = False
        self.smooth.reset()
        diag.log("CURSOR", "resumed")
    def toggle(self): self.resume() if self.paused else self.pause()

    @property
    def state(self):
        if self.paused:
            return "paused"
        if time.time() < self.frozen_until:
            return "frozen"
        return "tracking"

    def update(self, x, y, now=None):
        """Feed one raw gaze point. Returns the new cursor position or None if it did not move."""
        now = now or time.time()
        sx, sy = self.smooth(x, y, now)          # keep the filter warm even while frozen
        if self.paused or now < self.frozen_until:
            self.dwell_anchor = None
            return None
        sx = min(max(sx, self.margin), self.w - 1 - self.margin)   # stay off the exact corners
        sy = min(max(sy, self.margin), self.h - 1 - self.margin)
        if self.pos and math.dist(self.pos, (sx, sy)) < self.deadzone:
            self._dwell(now)
            return None
        self.pos = (sx, sy)
        if self.mover:
            self.mover(int(sx), int(sy))
        self._dwell(now)
        return self.pos

    def _dwell(self, now):
        if not self.dwell_s or self.pos is None:
            return
        if self.dwell_anchor is None or math.dist(self.dwell_anchor, self.pos) > self.dwell_r:
            self.dwell_anchor, self.dwell_t0, self.dwell_done = self.pos, now, False
        elif not self.dwell_done and now - self.dwell_t0 >= self.dwell_s:
            self.dwell_done = True
            diag.log("CURSOR", "dwell click", at=self.pos)
            if self.clicker:
                self.clicker()

"""Switches SSVEP on and off for the dashboard: overlay + runner, and what the dashboard needs to know about the
calibration (is there a usable one, when was it made)."""
import time

from ..log import log
from .core import GESTURES, PROFILE_PATH, load_if_usable
from .runner import SsvepRunner


class SsvepManager:
    def __init__(self, ctx, emit_event, profile_path=PROFILE_PATH):
        self.ctx, self.emit_event, self.path = ctx, emit_event, profile_path
        self.runner = None
        self.overlay = None
        self.profile = None
        self._mtime = None
        self.scores = None

    # ---- calibration status (cheap enough to call from a UI timer)
    def calibration(self):
        try:
            m = self.path.stat().st_mtime
        except OSError:
            self.profile, self._mtime = None, None
            return None
        if m != self._mtime:
            self._mtime = m
            self.profile = load_if_usable(self.path)
        return self.profile

    def describe(self) -> str:
        p = self.calibration()
        if p is None:
            return "SSVEP: not calibrated"
        return f"SSVEP calibrated {p.created[:16]}, hit rate {p.hit_rate:.0%}"

    @property
    def enabled(self) -> bool:
        return self.runner is not None

    def labels(self):
        from ..ui import theme
        g = self.ctx["mapping"].get("gestures", {})
        return [theme.describe(g.get(name, "none")) for name in GESTURES]

    def refresh_labels(self):
        if self.overlay is not None:
            self.overlay.set_labels(self.labels())

    # ---- on / off. The warning is the CALLER's job (dashboard), so this stays testable without a dialog.
    def enable(self):
        if self.enabled:
            return
        p = self.calibration()
        if p is None:
            raise RuntimeError("SSVEP is not calibrated yet. Use 'SSVEP calibration' on the start screen.")
        src = self.ctx["source"]
        if type(src).__name__ not in ("LSLSource", "SyntheticSource") or not getattr(src, "connected", False):
            raise RuntimeError("SSVEP needs the live headset (LSL stream); it does not work with simulated or replayed EEG.")
        from .flicker import FlickerOverlay
        runner = SsvepRunner(src, p, self.ctx["actions"], on_event=self.emit_event, on_scores=self._scores)
        overlay = FlickerOverlay(p.freqs, mode="live", labels=self.labels())
        if p.refresh_hz and abs(overlay.refresh_hz - p.refresh_hz) > 2.0:
            raise RuntimeError(f"The screen runs at {overlay.refresh_hz:.0f} Hz but SSVEP was calibrated at "
                               f"{p.refresh_hz:.0f} Hz. Calibrate again on this screen.")
        self.runner, self.overlay = runner, overlay
        overlay.start()
        runner.start()
        log("SSVEP", "enabled", freqs=p.freqs, win_s=p.win_s, thr=p.thr, refresh_hz=overlay.refresh_hz)

    def disable(self):
        if not self.enabled:
            return
        self.runner.stop()
        self.overlay.stop()
        self.overlay.deleteLater()
        self.runner = self.overlay = None
        log("SSVEP", "disabled")

    def _scores(self, a):
        self.scores = a

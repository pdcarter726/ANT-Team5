"""Starts the eye-tracking cursor (gaze/) as its own process and talks to it over its UDP control channel."""
import subprocess
import sys
import time
from pathlib import Path

from .log import log

REPO = Path(__file__).resolve().parents[1]


class GazeLink:
    def __init__(self, profile=None, start=True, extra_args=()):
        sys.path.insert(0, str(REPO))
        from gaze import control
        self.control = control
        self.proc = None
        if start:
            args = [sys.executable, "-m", "gaze.app", "--log-every", "10"]
            if profile:
                args += ["--profile", profile]
            args += list(extra_args)
            self.proc = subprocess.Popen(args, cwd=str(REPO))
            log("GAZE", "eye cursor started", profile=profile or "(calibrate)", pid=self.proc.pid)
        else:
            log("GAZE", "using an eye cursor that is already running (not starting one)")

    def status(self):
        return self.control.status()

    @property
    def running(self):
        return self.proc is None or self.proc.poll() is None

    def freeze(self, s): self.control.freeze(s)
    def pause(self): self.control.pause()
    def resume(self): self.control.resume()
    def toggle(self): self.control.toggle()
    def recenter(self): self.control.recenter()
    def validate(self): self.control.validate()
    def fix(self): self.control.fix_from_check()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(3)
            except Exception:
                self.proc.kill()
            log("GAZE", "eye cursor stopped")

"""Small always-on-top, click-through status overlay: mode (ACTIVE / PAUSED / DRY RUN), last event, live class bars.

It never takes focus and mouse clicks pass through it, so it cannot interfere with the thing being controlled.
Runs in the main thread (Tk); the detector engine runs in a worker thread and only writes to `State`.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class State:
    mode: str = "STARTING"             # ACTIVE | PAUSED | DRY RUN | CALIBRATING | STOPPED
    paused: bool = False
    dry_run: bool = False
    source: str = ""
    classes: list = field(default_factory=list)
    probs: np.ndarray | None = None
    top: str = ""
    last_event: str = ""
    last_event_wall: float = 0.0
    n_events: int = 0
    lag_s: float = 0.0
    message: str = ""
    stop: bool = False

    def label(self):
        if self.mode in ("STOPPED", "STARTING", "CALIBRATING"):
            return self.mode
        return "PAUSED" if self.paused else ("DRY RUN" if self.dry_run else "ACTIVE")


COLORS = {"ACTIVE": "#1b9e3e", "PAUSED": "#c62828", "DRY RUN": "#1565c0", "STOPPED": "#555555",
          "STARTING": "#8d6e00", "CALIBRATING": "#8d6e00"}


def _click_through(root):
    if sys.platform != "win32":
        return
    try:
        import ctypes
        u = ctypes.windll.user32
        hwnd = u.GetParent(root.winfo_id()) or root.winfo_id()
        GWL_EXSTYLE = -20
        style = u.GetWindowLongW(hwnd, GWL_EXSTYLE)
        # layered | transparent (click-through) | toolwindow (no taskbar) | noactivate (never takes focus)
        u.SetWindowLongW(hwnd, GWL_EXSTYLE, style | 0x80000 | 0x20 | 0x80 | 0x08000000)
    except Exception:
        pass


class Overlay:
    def __init__(self, state: State, corner="tr", show_bars=True):
        import tkinter as tk
        self.tk, self.st = tk, state
        self.root = tk.Tk()
        r = self.root
        r.overrideredirect(True)
        r.attributes("-topmost", True)
        try:
            r.attributes("-alpha", 0.88)
        except Exception:
            pass
        scale = max(1.0, r.winfo_fpixels("1i") / 96.0)
        self.s = scale
        W, H = int(250 * scale), int((92 + (110 if show_bars else 0)) * scale)
        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        x = sw - W - int(12 * scale) if "r" in corner else int(12 * scale)
        y = int(12 * scale) if "t" in corner else sh - H - int(60 * scale)
        r.geometry(f"{W}x{H}+{x}+{y}")
        self.cv = tk.Canvas(r, width=W, height=H, highlightthickness=0, bg="#202124")
        self.cv.pack()
        self.W, self.H, self.bars = W, H, show_bars
        r.update_idletasks()
        _click_through(r)
        self._tick()

    def _tick(self):
        st, cv, s = self.st, self.cv, self.s
        cv.delete("all")
        lab = st.label()
        cv.create_rectangle(0, 0, self.W, int(38 * s), fill=COLORS.get(lab, "#444"), outline="")
        cv.create_text(int(10 * s), int(19 * s), anchor="w", text=lab, fill="white",
                       font=("Segoe UI", int(15 * 0.75 * s) or 11, "bold"))
        cv.create_text(self.W - int(8 * s), int(19 * s), anchor="e", text=st.source, fill="white",
                       font=("Segoe UI", int(9 * 0.75 * s) or 8))
        recent = time.time() - st.last_event_wall < 2.0
        cv.create_text(int(10 * s), int(54 * s), anchor="w", text=("> " + st.last_event) if st.last_event else "no events yet",
                       fill="#ffe082" if recent else "#bbbbbb", font=("Segoe UI", int(11 * 0.75 * s) or 9, "bold" if recent else "normal"))
        cv.create_text(int(10 * s), int(76 * s), anchor="w",
                       text=f"{st.n_events} events   lag {st.lag_s * 1000:.0f} ms   kill: Ctrl+Alt+Q" if not st.message else st.message,
                       fill="#9aa0a6", font=("Segoe UI", int(8 * 0.75 * s) or 7))
        if self.bars and st.probs is not None and st.classes:
            order = np.argsort(-st.probs)[:6]
            y = int(94 * s)
            for i in order:
                cv.create_text(int(10 * s), y + int(8 * s), anchor="w", text=st.classes[i][:13], fill="white",
                               font=("Segoe UI", int(8 * 0.75 * s) or 7))
                x0, x1 = int(100 * s), self.W - int(10 * s)
                cv.create_rectangle(x0, y + int(2 * s), x1, y + int(14 * s), fill="#3c4043", outline="")
                cv.create_rectangle(x0, y + int(2 * s), x0 + int((x1 - x0) * float(min(st.probs[i], 1.0))), y + int(14 * s),
                                    fill="#8ab4f8" if st.classes[i] not in ("rest", "confounder") else "#5f6368", outline="")
                y += int(17 * s)
        if st.stop:
            self.root.destroy()
            return
        self.root.after(80, self._tick)

    def run(self):
        self.root.mainloop()

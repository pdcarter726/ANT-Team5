"""OS input layer: user-editable event mapping, action execution, cursor controller, global hotkeys.

Backends: PynputBackend (real mouse/keyboard), VirtualBackend (records actions, tracks a virtual cursor;
used for dry runs, replay and tests). The mapping file is validated before anything touches the OS.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from common import ROOT

DEFAULT_MAPPING = ROOT / "config" / "mapping.yaml"
ACTIONS = {"click", "key", "hotkey", "scroll", "pause_toggle", "cursor"}


# ------------------------------------------------------------------ mapping
@dataclass
class Mapping:
    gestures: dict
    events: dict
    cursor: dict
    safety: dict

    def event_for(self, gesture):
        name = self.gestures.get(gesture)
        return (name, self.events[name]) if name else (None, None)

    def overrides(self):
        """Per-gesture detector overrides from the mapped event: {gesture: {min_prob, hold, refractory_s}}."""
        out = {}
        for g, name in self.gestures.items():
            if name and self.events[name].get("action") != "cursor":
                o = {k: self.events[name][k] for k in ("min_prob", "hold", "refractory_s") if k in self.events[name]}
                if o:
                    out[g] = o
        return out


def load_mapping(path=None, known_gestures=None) -> Mapping:
    p = Path(path) if path else DEFAULT_MAPPING
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    m = Mapping(d.get("gestures") or {}, d.get("events") or {}, d.get("cursor") or {}, d.get("safety") or {})
    errs = []
    for g, name in m.gestures.items():
        if known_gestures is not None and g not in known_gestures:
            errs.append(f"gestures: '{g}' is not a gesture the profile knows (known: {sorted(known_gestures)})")
        if name and name not in m.events:
            errs.append(f"gestures: '{g}' -> '{name}' but there is no event '{name}' under events:")
    for n, e in m.events.items():
        a = e.get("action")
        if a not in ACTIONS:
            errs.append(f"events: '{n}' has action '{a}'; valid actions: {sorted(ACTIONS)}")
        if a == "click" and e.get("button", "left") not in ("left", "right", "middle"):
            errs.append(f"events: '{n}' button must be left/right/middle")
        if a == "key" and "key" not in e:
            errs.append(f"events: '{n}' needs key:")
        if a == "hotkey" and not e.get("keys"):
            errs.append(f"events: '{n}' needs keys: [..]")
        if a == "cursor" and (e.get("axis") not in ("x", "y") or e.get("dir") not in (-1, 1)):
            errs.append(f"events: '{n}' needs axis: x|y and dir: -1|1")
    if errs:
        raise ValueError(f"{p}:\n  " + "\n  ".join(errs))
    return m


# ------------------------------------------------------------------ backends
class VirtualBackend:
    """No OS access. Records every action and integrates a virtual cursor."""

    def __init__(self):
        self.log: list[tuple] = []
        self.x = self.y = 0.0

    def move(self, dx, dy): self.x += dx; self.y += dy; self.log.append(("move", dx, dy))
    def click(self, button="left", count=1): self.log.append(("click", button, count))
    def key(self, name): self.log.append(("key", name))
    def hotkey(self, keys): self.log.append(("hotkey", tuple(keys)))
    def scroll(self, dy): self.log.append(("scroll", dy))
    def actions(self): return [a for a in self.log if a[0] != "move"]
    def close(self): pass


class PynputBackend:
    """Real mouse and keyboard through pynput (physical pixels once the process is DPI aware)."""

    def __init__(self):
        from pynput import keyboard, mouse
        self._kb, self._ms = keyboard, mouse
        self.kb, self.ms = keyboard.Controller(), mouse.Controller()

    def move(self, dx, dy):
        x, y = self.ms.position
        self.ms.position = (int(round(x + dx)), int(round(y + dy)))

    def click(self, button="left", count=1):
        self.ms.click(getattr(self._ms.Button, button), count)

    def _key(self, name):
        k = getattr(self._kb.Key, name, None)
        return k if k is not None else name

    def key(self, name):
        k = self._key(name)
        self.kb.press(k)
        self.kb.release(k)

    def hotkey(self, keys):
        ks = [self._key(k) for k in keys]
        for k in ks:
            self.kb.press(k)
        for k in reversed(ks):
            self.kb.release(k)

    def scroll(self, dy): self.ms.scroll(0, dy)
    def close(self): pass


class ActionExecutor:
    def __init__(self, backend):
        self.b = backend

    def do(self, ev: dict):
        a = ev["action"]
        if a == "click":
            self.b.click(ev.get("button", "left"), int(ev.get("count", 1)))
        elif a == "key":
            self.b.key(ev["key"])
        elif a == "hotkey":
            self.b.hotkey(ev["keys"])
        elif a == "scroll":
            self.b.scroll(int(ev.get("dy", 0)))
        # pause_toggle / cursor are handled by the engine


# --------------------------------------------------------------- cursor
class CursorController:
    """Head-direction estimates -> smoothed, dead-zoned cursor velocity, with a post-click freeze.

    update() is called once per scoring window (8 Hz) with raw estimates; step() is called at ~100 Hz
    (or once per window in fast replay) and moves the backend. All times are passed in, so tests are deterministic.
    """

    def __init__(self, backend, cfg: dict, enabled_axes=("x", "y")):
        c = dict(max_speed_px_s=700, dead_zone=0.3, smoothing=0.5, accel_power=1.5, head_gate=0.5,
                 velocity_tau_s=0.12, freeze_ms=300)
        c.update(cfg or {})
        self.c, self.b, self.axes = c, backend, set(enabled_axes)
        self.est = np.zeros(2)       # smoothed estimate (x, y)
        self.target = np.zeros(2)    # target velocity, -1..1
        self.vel = np.zeros(2)
        self.frozen_until = -1e9
        self.enabled = True
        self._carry = np.zeros(2)
        self.last_dir = (0.0, 0.0)

    def freeze(self, now, ms=None):
        self.frozen_until = now + (self.c["freeze_ms"] if ms is None else ms) / 1000.0
        self.est[:] = 0
        self.target[:] = 0
        self.vel[:] = 0

    def update(self, h, v, p_head, now):
        if not self.enabled or now < self.frozen_until or p_head < self.c["head_gate"]:
            self.est *= 0.0
            self.target[:] = 0
            return
        a = float(self.c["smoothing"])
        raw = np.array([h if "x" in self.axes else 0.0, v if "y" in self.axes else 0.0])
        self.est = a * self.est + (1 - a) * raw
        dz = float(self.c["dead_zone"])
        mag = np.abs(self.est)
        u = np.where(mag > dz, (mag - dz) / (1 - dz), 0.0).clip(0, 1) ** float(self.c["accel_power"])
        self.target = np.sign(self.est) * u

    def step(self, now, dt):
        if not self.enabled or now < self.frozen_until:
            self.vel[:] = 0
            return 0.0, 0.0
        k = min(1.0, dt / max(float(self.c["velocity_tau_s"]), 1e-3))
        self.vel += (self.target - self.vel) * k
        d = self.vel * float(self.c["max_speed_px_s"]) * dt + self._carry
        move = np.round(d)
        self._carry = d - move
        if move.any():
            self.b.move(float(move[0]), float(move[1]))
        return float(move[0]), float(move[1])


# ---------------------------------------------------------------- hotkeys
def start_hotkeys(bindings: dict):
    """bindings: {'<ctrl>+<alt>+q': callable}. Returns the pynput listener (call .stop()) or None."""
    try:
        from pynput import keyboard
        h = keyboard.GlobalHotKeys(bindings)
        h.daemon = True
        h.start()
        return h
    except Exception as e:  # pragma: no cover
        print(f"WARNING: global hotkeys unavailable ({e!r}). Close the console window to stop.")
        return None

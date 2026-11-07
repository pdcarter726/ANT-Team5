"""Executes mapped actions.

Action strings (used in mappings/*.json and the dashboard):
  move:up | move:down | move:left | move:right
                       move the cursor while the gesture is held (starts slow for aiming, speeds up the longer
                       it is held; see cursor_* in the mapping). With the eye cursor on, it fine-tunes the
                       eye cursor's position (the eye cursor holds still while you move).
  mouse:left | mouse:right
                       a short gesture clicks; holding it past `hold_s` presses the button down until the
                       gesture ends (drag)
  click:left | click:right | click:double | click:middle      one click
  key:space            press a key (any pyautogui key name: enter, esc, tab, left, f5, a, ...)
  hotkey:ctrl+c        key combination
  scroll:5 | scroll:-5 scroll up / down
  hold:w               hold a key down while the gesture is held (games)
  type:hello           type text
  menu                 open / close the quick bar
  eye:pause | eye:resume | eye:toggle | eye:recenter | eye:check | eye:fix     control the eye cursor
  controller:pause     pause / resume all gesture actions (still works while paused)
  none                 do nothing
"""
import math
import threading
import time
from collections import deque

from .log import log

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


def parse(action):
    action = (action or "none").strip()
    kind, _, arg = action.partition(":")
    return kind.strip().lower(), arg.strip()


class CursorMover(threading.Thread):
    """Moves the cursor at 100 Hz while one or more directions are held. Speed ramps from `speed` to
    `max_speed` (px/s) over `accel_s` seconds of holding, so short turns aim and long turns travel."""

    def __init__(self, engine):
        super().__init__(daemon=True)
        self.e = engine
        self.active = {}            # direction -> start time
        self.frac = [0.0, 0.0]
        self.start()

    def press(self, d):
        self.active.setdefault(d, time.time())

    def release(self, d):
        self.active.pop(d, None)

    def run(self):
        last = time.time()
        while True:
            time.sleep(0.01)
            now = time.time()
            dt, last = now - last, now
            if not self.active:
                continue
            m = self.e.mapping
            v0, v1 = float(m.get("cursor_speed", 250)), float(m.get("cursor_max_speed", 1400))
            ramp = max(0.05, float(m.get("cursor_accel_s", 1.2)))
            vx = vy = 0.0
            for d, t0 in list(self.active.items()):
                k = min(1.0, (now - t0) / ramp)
                v = v0 + (v1 - v0) * k * k
                vx += DIRS[d][0] * v
                vy += DIRS[d][1] * v
            self.frac[0] += vx * dt
            self.frac[1] += vy * dt
            mx, my = int(self.frac[0]), int(self.frac[1])
            if mx or my:
                self.frac[0] -= mx
                self.frac[1] -= my
                try:
                    if self.e.gaze:
                        self.e.gaze.freeze(0.4)          # eye cursor holds still while the head fine-tunes
                    self.e.pg.moveRel(mx, my, _pause=False)
                except Exception:
                    pass


class ActionEngine:
    def __init__(self, mapping, gaze=None, on_menu=None):
        import pyautogui
        pyautogui.FAILSAFE = False      # the cursor legitimately reaches screen corners
        pyautogui.PAUSE = 0
        self.pg = pyautogui
        self.gaze = gaze
        self.on_menu = on_menu or (lambda: None)
        self.paused = False
        self.cursor_hist = deque(maxlen=200)      # (time, x, y)
        self.held_keys = set()
        self.mouse_held = {}                      # gesture -> {"button", "t0", "down"}
        self.lock = threading.Lock()
        self.set_mapping(mapping)
        self.mover = CursorMover(self)
        threading.Thread(target=self._mouse_hold_watch, daemon=True).start()

    def set_mapping(self, mapping):
        if hasattr(self, "mover"):
            self.release_all()
        self.mapping = mapping
        self.di = None
        if mapping.get("key_backend") == "directinput":
            try:
                import pydirectinput
                pydirectinput.FAILSAFE = False
                pydirectinput.PAUSE = 0
                self.di = pydirectinput
            except Exception as e:
                log("ACTION", f"pydirectinput not available ({e}); game keys fall back to pyautogui")
        log("ACTION", "mapping active", name=mapping.get("name"), gestures=len(mapping.get("gestures", {})),
            key_backend="directinput" if self.di else "pyautogui")

    # ---- cursor history for click lookback (eye cursor only) ----
    def sample_cursor(self):
        try:
            x, y = self.pg.position()
            self.cursor_hist.append((time.time(), x, y))
        except Exception:
            pass

    def _click_pos(self):
        """With the eye cursor, click where it was `click_lookback_s` ago: face gestures nudge the gaze
        estimate. With head-turn cursor movement the cursor is where you put it, so click right there."""
        if not self.gaze:
            return None
        t = time.time() - float(self.mapping.get("click_lookback_s", 0.5))
        best = None
        for (ts, x, y) in self.cursor_hist:
            if ts <= t:
                best = (x, y)
        return best

    # ---- events ----
    def handle(self, kind, gesture, prob=0.0):
        """kind: 'fire' or 'release'. Returns a short description of what was done (or None)."""
        action = self.mapping.get("gestures", {}).get(gesture, "none")
        a_kind, arg = parse(action)
        if kind == "release":
            if a_kind == "hold":
                self._key_up(arg)
                return f"released {arg}"
            if a_kind == "move":
                self.mover.release(arg)
                return "stopped moving"
            if a_kind == "mouse":
                return self._mouse_release(gesture)
            return None
        if self.paused and not (a_kind == "controller" and arg == "pause"):
            return f"(paused) {action}"
        if self.gaze and a_kind not in ("move",) and self.mapping.get("freeze_gaze_s", 0) > 0:
            self.gaze.freeze(self.mapping["freeze_gaze_s"])
        if a_kind == "move":
            if arg not in DIRS:
                return f"unknown direction {arg!r}"
            self.mover.press(arg)
            return f"moving {arg}"
        if a_kind == "mouse":
            self.mouse_held[gesture] = {"button": arg or "left", "t0": time.time(), "down": False}
            return f"{arg or 'left'} button (click, or hold to drag)"
        return self.run(action)

    def _mouse_hold_watch(self):
        """Turns a held mouse gesture into a button press once it has lasted `hold_s`."""
        while True:
            time.sleep(0.02)
            hold_s = float(self.mapping.get("hold_s", 0.5))
            for g, st in list(self.mouse_held.items()):
                if not st["down"] and time.time() - st["t0"] >= hold_s:
                    pos = self._click_pos()
                    try:
                        if pos:
                            self.pg.moveTo(*pos, _pause=False)
                        self.pg.mouseDown(button=st["button"], _pause=False)
                        st["down"] = True
                        log("ACTION", f"{st['button']} button held down (drag)", gesture=g)
                    except Exception as e:
                        log("ERROR", f"mouse down failed ({e})")

    def _mouse_release(self, gesture):
        st = self.mouse_held.pop(gesture, None)
        if not st:
            return None
        try:
            if st["down"]:
                self.pg.mouseUp(button=st["button"], _pause=False)
                return f"{st['button']} button released"
            pos = self._click_pos()
            if pos:
                self.pg.click(pos[0], pos[1], button=st["button"], _pause=False)
            else:
                self.pg.click(button=st["button"], _pause=False)
            return f"{st['button']} click"
        except Exception as e:
            log("ERROR", f"mouse action failed ({e})")
            return "FAILED mouse"

    def run(self, action):
        a_kind, arg = parse(action)
        try:
            if a_kind == "none":
                return "no action"
            if a_kind == "key":
                (self.di or self.pg).press(arg)
            elif a_kind == "hotkey":
                keys = [k.strip() for k in arg.split("+") if k.strip()]
                if self.di:
                    for k in keys: self.di.keyDown(k)
                    for k in reversed(keys): self.di.keyUp(k)
                else:
                    self.pg.hotkey(*keys)
            elif a_kind == "click":
                pos = self._click_pos()
                button = {"double": "left"}.get(arg, arg or "left")
                clicks = 2 if arg == "double" else 1
                if pos:
                    self.pg.click(pos[0], pos[1], clicks=clicks, button=button, _pause=False)
                else:
                    self.pg.click(clicks=clicks, button=button, _pause=False)
            elif a_kind == "scroll":
                self.pg.scroll(int(arg))
            elif a_kind == "hold":
                self._key_down(arg)
            elif a_kind == "type":
                self.pg.write(arg)
            elif a_kind == "menu":
                self.on_menu()
            elif a_kind == "eye":
                if not self.gaze:
                    return "eye cursor not running"
                {"pause": self.gaze.pause, "resume": self.gaze.resume, "toggle": self.gaze.toggle,
                 "recenter": self.gaze.recenter, "check": self.gaze.validate, "fix": self.gaze.fix}[arg]()
            elif a_kind == "controller" and arg == "pause":
                self.paused = not self.paused
                if self.paused:
                    self.release_all()
                log("ACTION", "controller paused" if self.paused else "controller resumed")
                return "controller paused" if self.paused else "controller resumed"
            else:
                return f"unknown action {action!r}"
        except Exception as e:
            log("ERROR", f"action {action!r} failed ({type(e).__name__}: {e})")
            return f"FAILED {action}"
        return action

    def _key_down(self, key):
        with self.lock:
            if key not in self.held_keys:
                (self.di or self.pg).keyDown(key)
                self.held_keys.add(key)

    def _key_up(self, key):
        with self.lock:
            if key in self.held_keys:
                (self.di or self.pg).keyUp(key)
                self.held_keys.discard(key)

    def release_all(self):
        """Let go of everything (pause, quit, mapping change): keys, mouse buttons, cursor movement."""
        for k in list(self.held_keys):
            self._key_up(k)
        for g in list(self.mouse_held):
            st = self.mouse_held.pop(g)
            if st["down"]:
                try:
                    self.pg.mouseUp(button=st["button"], _pause=False)
                except Exception:
                    pass
        for d in list(self.mover.active):
            self.mover.release(d)

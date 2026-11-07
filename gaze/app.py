"""Eye-tracking cursor (webcam + EyeGestures v4).

  python -m gaze.app                     # calibrate (50 dots, 5 head positions), then move the cursor with your eyes
  python -m gaze.app --load              # reuse the last saved calibration (same seat, same screen)
  python -m gaze.app --profile henry     # use profile "henry": loads it if it exists (no calibration),
                                         # otherwise calibrates and saves it; every update is saved to it
  python -m gaze.app --save-profile henry  # copy the last calibration into profile "henry", then exit
  python -m gaze.app --list-profiles
  python -m gaze.app --dwell-click 1.2   # also click by holding your gaze still for 1.2 s
Calibration only happens at start, on a key, or when the EEG app asks. Never on its own.
Hotkeys that work in any window (Windows): Ctrl+Alt+P pause/resume, Ctrl+Alt+C recenter,
  Ctrl+Alt+V accuracy check, Ctrl+Alt+F apply check correction, Ctrl+Alt+Q quit.
Keys in the preview window:
  v = accuracy check (5 dots, reports the real error in pixels and whether it is drift; does not change the model)
  f = apply a correction from the last accuracy check
  c = recenter (look at one dot, fixes drift after a small head move)
  a = add head position (9 more points at your current head position, model learns head movement)
Blinks are detected and ignored (the cursor holds still through them).
  u = undo the last added head position (if it made things worse)
  r = full recalibration     p = pause/resume     q / Esc = quit
Keys on the calibration screen: s = skip a dot that won't settle, Esc = quit
The EEG app controls it over UDP, see gaze/control.py (freeze during gestures!).
"""
import argparse
import sys
import time

import numpy as np

from .control import PORT, ControlServer
from .diag import diag
from .cursor import CursorController
from .tracker import CALIB_PATH, GazeTracker, list_profiles, profile_path

TEAL, WHITE, GREY, DARK = (197, 209, 79), (235, 235, 235), (120, 120, 120), (24, 20, 17)


MESSAGES = {
    "calibrate": "Look at the dot and keep your head still in this position",
    "addpose": "Adding this head position: look at the dot",
    "recenter": "Recenter: look at the center dot",
    "validate": "Accuracy check: look at the dot",
}
LABELS = {"calibrate": "Calibration point", "addpose": "Extra point", "recenter": "Steady frames",
          "validate": "Check dot"}


class GlobalHotkeys:
    """Ctrl+Alt+<key> hotkeys that work whichever window has focus (Windows only, no extra packages).
    Polled once per frame; each press fires once."""
    KEYS = {"P": "toggle", "C": "recenter", "V": "validate", "F": "fixcheck", "Q": "quit"}

    def __init__(self):
        self.user32 = None
        if sys.platform == "win32":
            import ctypes
            self.user32 = ctypes.windll.user32
        self.down = set()

    def poll(self):
        if self.user32 is None:
            return []
        gk = self.user32.GetAsyncKeyState
        held = lambda vk: bool(gk(vk) & 0x8000)
        if not (held(0x11) and held(0x12)):          # VK_CONTROL and VK_MENU (Alt)
            self.down.clear()
            return []
        fired = []
        for ch, action in self.KEYS.items():
            if held(ord(ch)):
                if ch not in self.down:
                    self.down.add(ch); fired.append(action)
            else:
                self.down.discard(ch)
        return fired


def draw_calibration(canvas, s, progress, t, extra=""):
    import cv2
    canvas[:] = DARK
    h, w = canvas.shape[:2]
    k, n = progress
    if s.note:                                   # instruction screen (e.g. move your head), no dot
        msg = s.note
        scale = 1.4 if w >= 1600 else 1.0
        size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)[0]
        cv2.putText(canvas, msg, ((w - size[0]) // 2, h // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, TEAL, 3, cv2.LINE_AA)
        cv2.putText(canvas, "the next dot appears when the countdown ends", ((w - size[0]) // 2, h // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{extra}      Esc = quit", (40, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2, cv2.LINE_AA)
        return
    cx, cy = int(s.target[0]), int(s.target[1])
    pulse = 34 + int(10 * np.sin(t * 6))
    cv2.circle(canvas, (cx, cy), pulse, TEAL, 3, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 10, WHITE, -1, cv2.LINE_AA)
    if s.face:
        cv2.circle(canvas, (int(s.x), int(s.y)), 8, GREY, -1, cv2.LINE_AA)   # live estimate
    if s.blink:
        cv2.putText(canvas, "blink", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2, cv2.LINE_AA)
    msg = MESSAGES.get(s.mode, "") if s.face else "No face found - face the camera, add light"
    cv2.putText(canvas, msg, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2, cv2.LINE_AA)
    hint = "s = skip this dot   Esc = quit" if s.mode in ("calibrate", "addpose") else "Esc = quit"
    cv2.putText(canvas, f"{LABELS.get(s.mode, '')} {min(k, n)} / {n}   {extra}      {hint}", (40, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2, cv2.LINE_AA)


def open_fullscreen(cv2, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--load", action="store_true", help="reuse the last saved calibration")
    ap.add_argument("--profile", help="calibration profile name: loaded if it exists, created otherwise")
    ap.add_argument("--save-profile", metavar="NAME", help="copy the last calibration into a profile and exit")
    ap.add_argument("--list-profiles", action="store_true", help="show saved profiles and exit")
    ap.add_argument("--min-cutoff", type=float, default=0.4, help="lower = steadier cursor, more lag")
    ap.add_argument("--beta", type=float, default=0.003, help="higher = less lag on big eye jumps")
    ap.add_argument("--deadzone", type=int, default=10, help="pixels of movement to ignore")
    ap.add_argument("--dwell-click", type=float, default=0.0, help="seconds of steady gaze that clicks (0 = off)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--log-every", type=float, default=2.0, help="seconds between status lines (0 = off)")
    ap.add_argument("--image-mode", action="store_true",
                    help="detect the face from scratch every frame (EyeGestures' way) instead of tracking it")
    a = ap.parse_args()

    if a.list_profiles:
        profs = list_profiles()
        if not profs:
            print("No profiles yet. Create one with --profile NAME or --save-profile NAME.")
        for name, info in profs:
            print(f"  {name:20s} " + "  ".join(f"{k}={v}" for k, v in info.items()))
        return
    if a.save_profile:
        if not CALIB_PATH.exists():
            print("No calibration to copy yet: run the app and calibrate first."); return
        import json
        dst = profile_path(a.save_profile)
        d = json.loads(CALIB_PATH.read_text())
        d["profile"] = dst.stem
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(d))
        print(f"Saved the last calibration ({d.get('saved_at', '?')}) as profile '{dst.stem}': {dst}")
        print(f"Use it with:  .venv\\Scripts\\python -m gaze.app --profile {dst.stem}")
        return

    import cv2
    import pyautogui
    pyautogui.FAILSAFE = False   # the gaze can legitimately reach screen corners
    pyautogui.PAUSE = 0
    sw, sh = pyautogui.size()
    diag.open_file()
    diag.hb_every = a.log_every if a.log_every > 0 else float("inf")
    diag.startup((sw, sh), a.camera, a)

    tracker = GazeTracker(sw, sh, a.camera, video_mode=not a.image_mode, start_calibration=False)
    hotkeys = GlobalHotkeys()
    diag.log("START", "global hotkeys", available=hotkeys.user32 is not None,
             keys="Ctrl+Alt+P pause, C recenter, V check, F fix, Q quit")

    def move(x, y):
        pyautogui.moveTo(x, y, _pause=False)
        diag.w_cursor_moves += 1
    ctl = CursorController(sw, sh, mover=move,
                           clicker=(lambda: pyautogui.click(_pause=False)) if a.dwell_click else None,
                           min_cutoff=a.min_cutoff, beta=a.beta, deadzone=a.deadzone, dwell_click_s=a.dwell_click)
    last = {"face": False, "x": 0, "y": 0, "calibrating": True}
    pending = {"cmd": None}     # mode change requested by a key or the EEG app

    def handle(cmd, arg):
        if cmd != "status":
            diag.log("CMD", f"from EEG app (UDP): {cmd}" + (f" {arg}" if arg else ""))
        if cmd == "freeze":
            ctl.freeze(float(arg or 0.8))
        elif cmd == "pause":
            ctl.pause()
        elif cmd == "resume":
            ctl.resume()
        elif cmd == "toggle":
            ctl.toggle()
        elif cmd == "recalibrate":
            pending["cmd"] = "calibrate"
        elif cmd == "addpose":
            pending["cmd"] = "addpose"
        elif cmd == "recenter":
            pending["cmd"] = "recenter"
        elif cmd == "undo":
            pending["cmd"] = "undo"
        elif cmd == "validate":
            pending["cmd"] = "validate"
        elif cmd == "fixcheck":
            pending["cmd"] = "fixcheck"
        elif cmd == "status":
            return {**last, "state": ctl.state}
        return None

    try:
        server = ControlServer(handle, a.port); server.start()
        diag.log("START", "control channel for the EEG app", address=f"udp://127.0.0.1:{a.port}")
    except OSError as e:
        server = None
        diag.log("ERROR", f"control channel unavailable ({e}); the EEG app cannot freeze the cursor")

    if a.profile:
        tracker.calib_path = profile_path(a.profile)
        tracker.profile = tracker.calib_path.stem
        if tracker.calib_path.exists():
            try:
                tracker.load_calibration()
                diag.log("SAVE", "profile loaded: if you sit differently than when it was made, press c (recenter) "
                                 "or v then f before relying on it")
            except Exception as e:
                diag.log("SAVE", f"could not load profile {tracker.profile!r} ({e}); calibrating, and the new "
                                 "calibration will be saved to it")
        else:
            diag.log("SAVE", f"profile {tracker.profile!r} does not exist yet: calibrating, then saving it",
                     path=str(tracker.calib_path))
    elif a.load and CALIB_PATH.exists():
        try:
            tracker.load_calibration()
        except Exception as e:
            diag.log("SAVE", f"could not load saved calibration ({e}); calibrating instead")

    if tracker.model is None:
        tracker.recalibrate()
    canvas = np.zeros((sh, sw, 3), np.uint8)
    calibrating_ui = tracker.calibrating
    if calibrating_ui:
        open_fullscreen(cv2, "calibration")
    frames, t_fps, fps = 0, time.time(), 0.0
    errors_in_row = 0
    try:
        while True:
            cmd, pending["cmd"] = pending["cmd"], None
            if cmd:
                diag.log("CMD", f"run: {cmd}", mode_before=tracker.mode)
                {"calibrate": tracker.recalibrate, "addpose": tracker.add_pose, "recenter": tracker.recenter,
                 "undo": tracker.undo_pose, "validate": tracker.validate, "skip": tracker.skip_point,
                 "fixcheck": tracker.apply_check_fix}[cmd]()
                if tracker.calibrating and not calibrating_ui:
                    calibrating_ui = True
                    open_fullscreen(cv2, "calibration")
            quit_now = False
            for action in hotkeys.poll():
                diag.log("CMD", f"hotkey Ctrl+Alt: {action}")
                if action == "quit":
                    quit_now = True
                elif action == "toggle":
                    ctl.toggle()
                else:
                    pending["cmd"] = action
            if quit_now:
                break
            t_loop = time.perf_counter()
            try:
                s = tracker.step()
                errors_in_row = 0
            except Exception as e:
                errors_in_row += 1
                diag.error(f"error while processing a frame ({e})", e)
                if errors_in_row >= 20:
                    diag.log("ERROR", "20 errors in a row, stopping")
                    break
                cv2.waitKey(1)
                continue
            if s is None:                     # no camera frame within 1 s: keep the windows responsive
                diag.log("CAM", "no camera frame for 1 s")
                cv2.waitKey(1)
                continue
            now = time.time()
            frames += 1
            if now - t_fps >= 1.0:
                fps, frames, t_fps = frames / (now - t_fps), 0, now
            last.update(face=s.face, calibrating=s.calibrating)
            if s.face:
                last.update(x=round(s.x), y=round(s.y))

            if calibrating_ui:
                if s.calibrating:
                    extra = ""
                    if tracker.mode == "calibrate" and getattr(tracker.session, "block_size", None):
                        nb = -(-len(tracker.session.points) // tracker.session.block_size)
                        extra = f"| head position {min(tracker.session.block + 1, nb)} / {nb}"
                    draw_calibration(canvas, s, tracker.progress, now, extra)
                    cv2.imshow("calibration", canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        diag.log("CMD", "key: quit (calibration screen)")
                        break
                    if key == ord("s"):
                        diag.log("CMD", "key: s (skip dot)")
                        pending["cmd"] = "skip"
                    diag.heartbeat(now, tracker.mode, "calibrating", (s.x, s.y) if s.face else None, None,
                                   extra=f"{tracker.mode} {tracker.progress[0]}/{tracker.progress[1]}")
                    continue
                calibrating_ui = False
                cv2.destroyWindow("calibration")
                cv2.waitKey(1)
                ctl.smooth.reset()
                diag.log("MODE", "eye cursor is live")

            if s.face and not s.blink:
                ctl.update(s.x, s.y, now)
            diag.w_loop_ms.append((time.perf_counter() - t_loop) * 1000)
            diag.heartbeat(now, tracker.mode, ctl.state, (s.x, s.y) if s.face else None, ctl.pos)

            if not a.no_preview:
                small = cv2.resize(s.frame, (320, 180))
                txt = ("blink" if s.blink else ctl.state) if s.face else "no face"
                cv2.putText(small, f"{txt}  {fps:.0f} fps", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEAL, 2, cv2.LINE_AA)
                if tracker.validation:
                    cv2.putText(small, f"check: typical {tracker.validation['median']:.0f} px, worst {tracker.validation['worst']:.0f} (f = fix)",
                                (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
                cv2.putText(small, "v check  f fix  c recenter  a pose  u undo", (8, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
                cv2.putText(small, "r recal  p pause  q quit", (8, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
                cv2.imshow("eye cursor", small)
                try:
                    cv2.setWindowProperty("eye cursor", cv2.WND_PROP_TOPMOST, 1)
                except cv2.error:
                    pass
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    diag.log("CMD", f"key: {chr(key) if 32 < key < 127 else key}")
                if key in (27, ord("q")):
                    break
                if key == ord("p"):
                    ctl.toggle()
                if key == ord("r"):
                    pending["cmd"] = "calibrate"
                if key == ord("a"):
                    pending["cmd"] = "addpose"
                if key == ord("c"):
                    pending["cmd"] = "recenter"
                if key == ord("u"):
                    pending["cmd"] = "undo"
                if key == ord("v"):
                    pending["cmd"] = "validate"
                if key == ord("f"):
                    pending["cmd"] = "fixcheck"
    except KeyboardInterrupt:
        diag.log("CMD", "Ctrl+C in terminal")
    except Exception as e:
        diag.error(f"app crashed ({e})", e)
        raise
    finally:
        diag.log("START", "shutting down", log_file=str(diag.path))
        if server:
            server.stop()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

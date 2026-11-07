"""Eye-tracking cursor (webcam + EyeGestures v4).

  python -m gaze.app                     # calibrate (35 points, EyeGestures method), then move the cursor with your eyes
  python -m gaze.app --load              # reuse the last saved calibration (same seat, same screen)
  python -m gaze.app --dwell-click 1.2   # also click by holding your gaze still for 1.2 s
Calibration only happens at start, on a key, or when the EEG app asks. Never on its own.
Keys (click the small preview window first):
  c = recenter (look at one dot, fixes drift after a small head move)
  a = add head position (9 more points at your current head position, model learns head movement)
Blinks are detected and ignored (the cursor holds still through them).
  r = full recalibration     p = pause/resume     q / Esc = quit
The EEG app controls it over UDP, see gaze/control.py (freeze during gestures!).
"""
import argparse
import time

import numpy as np

from .control import PORT, ControlServer
from .cursor import CursorController
from .tracker import CALIB_PATH, GazeTracker

TEAL, WHITE, GREY, DARK = (197, 209, 79), (235, 235, 235), (120, 120, 120), (24, 20, 17)


MESSAGES = {
    "calibrate": "Look at the dot and keep your head still",
    "addpose": "Adding this head position: look at the dot",
    "recenter": "Recenter: look at the center dot",
}
LABELS = {"calibrate": "Calibration point", "addpose": "Extra point", "recenter": "Steady frames"}


def draw_calibration(canvas, s, progress, t):
    import cv2
    canvas[:] = DARK
    h, w = canvas.shape[:2]
    k, n = progress
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
    cv2.putText(canvas, f"{LABELS.get(s.mode, '')} {min(k, n)} / {n}      Esc = quit", (40, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREY, 2, cv2.LINE_AA)


def open_fullscreen(cv2, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--load", action="store_true", help="reuse the last saved calibration")
    ap.add_argument("--min-cutoff", type=float, default=0.4, help="lower = steadier cursor, more lag")
    ap.add_argument("--beta", type=float, default=0.003, help="higher = less lag on big eye jumps")
    ap.add_argument("--deadzone", type=int, default=10, help="pixels of movement to ignore")
    ap.add_argument("--dwell-click", type=float, default=0.0, help="seconds of steady gaze that clicks (0 = off)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()

    import cv2
    import pyautogui
    pyautogui.FAILSAFE = False   # the gaze can legitimately reach screen corners
    pyautogui.PAUSE = 0
    sw, sh = pyautogui.size()
    print(f"Screen {sw}x{sh}. Starting camera {a.camera} ...")

    tracker = GazeTracker(sw, sh, a.camera)
    ctl = CursorController(sw, sh, mover=lambda x, y: pyautogui.moveTo(x, y, _pause=False),
                           clicker=(lambda: pyautogui.click(_pause=False)) if a.dwell_click else None,
                           min_cutoff=a.min_cutoff, beta=a.beta, deadzone=a.deadzone, dwell_click_s=a.dwell_click)
    last = {"face": False, "x": 0, "y": 0, "calibrating": True}
    pending = {"cmd": None}     # mode change requested by a key or the EEG app

    def handle(cmd, arg):
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
        elif cmd == "status":
            return {**last, "state": ctl.state}
        return None

    try:
        server = ControlServer(handle, a.port); server.start()
        print(f"Control channel on udp://127.0.0.1:{a.port}")
    except OSError as e:
        server = None
        print(f"WARNING: control channel unavailable ({e}); the EEG app cannot freeze the cursor")

    if a.load and CALIB_PATH.exists():
        try:
            tracker.load_calibration()
            print("Loaded saved calibration.")
        except Exception as e:
            print(f"Could not load calibration ({e}); calibrating.")

    canvas = np.zeros((sh, sw, 3), np.uint8)
    calibrating_ui = tracker.calibrating
    if calibrating_ui:
        open_fullscreen(cv2, "calibration")
    frames, t_fps, fps = 0, time.time(), 0.0
    try:
        while True:
            cmd, pending["cmd"] = pending["cmd"], None
            if cmd:
                {"calibrate": tracker.recalibrate, "addpose": tracker.add_pose, "recenter": tracker.recenter}[cmd]()
                if tracker.calibrating and not calibrating_ui:
                    calibrating_ui = True
                    open_fullscreen(cv2, "calibration")
            s = tracker.step()
            if s is None:
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
                    draw_calibration(canvas, s, tracker.progress, now)
                    cv2.imshow("calibration", canvas)
                    if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                        break
                    continue
                calibrating_ui = False
                cv2.destroyWindow("calibration")
                cv2.waitKey(1)
                ctl.smooth.reset()
                print("Calibration updated. Eye cursor is live.")

            if s.face and not s.blink:
                ctl.update(s.x, s.y, now)

            if not a.no_preview:
                small = cv2.resize(s.frame, (320, 180))
                txt = ("blink" if s.blink else ctl.state) if s.face else "no face"
                cv2.putText(small, f"{txt}  {fps:.0f} fps", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEAL, 2, cv2.LINE_AA)
                cv2.putText(small, "c recenter  a add pose  r recal", (8, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
                cv2.putText(small, "p pause  q quit", (8, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
                cv2.imshow("eye cursor", small)
                try:
                    cv2.setWindowProperty("eye cursor", cv2.WND_PROP_TOPMOST, 1)
                except cv2.error:
                    pass
                key = cv2.waitKey(1) & 0xFF
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
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.stop()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

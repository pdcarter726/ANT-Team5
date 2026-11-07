"""Diagnostics: timestamped log lines to the terminal and to gaze/logs/gaze_<date>_<time>.log.

Categories: START (environment), CAM (camera), FACE (detection), CAL (calibration), FIT (model fitting),
HB (heartbeat: a status line every few seconds), CHECK (accuracy check), RECENTER, CMD (keys, hotkeys,
UDP commands), MODE (mode changes), CURSOR, SAVE, ERROR.
"""
import os
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np

LOG_DIR = Path(__file__).parent / "logs"


class Diag:
    def __init__(self):
        self.lock = threading.Lock()
        self.file = None
        self.path = None
        self.hb_every = 2.0
        self.t_hb = time.time()
        self.reset_window()
        self.cam_frames = 0          # read by the camera thread (all time)
        self.cam_fail = 0
        self.cam_drops = 0           # frames replaced before the main loop took them
        self.face_was = None
        self.calib_head = None       # mean head pose of the calibration data (for drift)

    def open_file(self):
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.path = LOG_DIR / time.strftime("gaze_%Y%m%d_%H%M%S.log")
            self.file = open(self.path, "w", encoding="utf-8", buffering=1)
        except Exception as e:
            print(f"(could not open log file: {e})")

    def log(self, cat, msg, **fields):
        t = time.time()
        stamp = time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"
        extra = " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
        line = f"[{stamp}] [{cat}] {msg}" + (f" | {extra}" if extra else "")
        with self.lock:
            print(line, flush=True)
            if self.file:
                try:
                    self.file.write(line + "\n")
                except Exception:
                    pass

    def error(self, msg, exc=None):
        self.log("ERROR", msg)
        tb = traceback.format_exc() if exc else ""
        if tb and tb.strip() != "NoneType: None":
            for l in tb.rstrip().splitlines():
                self.log("ERROR", "  " + l)

    # ---------- heartbeat window ----------
    def reset_window(self):
        self.w_frames = 0
        self.w_face = 0
        self.w_blink = 0
        self.w_det_ms = []
        self.w_loop_ms = []
        self.w_raw = deque(maxlen=200)
        self.w_pose = []
        self.w_light = []
        self.w_face_px = []
        self.w_cursor_moves = 0
        self.w_blink_score = []
        self.w_ear_ratio = []
        self.w_cam0 = getattr(self, "cam_frames", 0)
        self.w_t0 = time.time()

    def frame(self, face, blink, det_ms, loop_ms=None, raw=None, extras=None, light=None, face_px=None):
        self.w_frames += 1
        self.w_face += bool(face)
        self.w_blink += bool(blink)
        self.w_det_ms.append(det_ms)
        if loop_ms is not None:
            self.w_loop_ms.append(loop_ms)
        if raw is not None:
            self.w_raw.append(raw)
        if extras is not None:
            self.w_pose.append(extras[8:14])
        if light is not None:
            self.w_light.append(light)
        if face_px is not None:
            self.w_face_px.append(face_px)
        if self.face_was is not None and bool(face) != self.face_was:
            self.log("FACE", "face found again" if face else "face LOST (cursor holds still)")
        self.face_was = bool(face)

    def heartbeat(self, now, mode, state, gaze, cursor, extra=""):
        if now - self.t_hb < self.hb_every:
            return
        dt = max(now - self.w_t0, 1e-6)
        n = max(self.w_frames, 1)
        f = {
            "mode": mode, "cursor_state": state,
            "cam_fps": (self.cam_frames - self.w_cam0) / dt, "proc_fps": self.w_frames / dt,
            "detect_ms": np.mean(self.w_det_ms) if self.w_det_ms else 0,
            "detect_max_ms": np.max(self.w_det_ms) if self.w_det_ms else 0,
            "face_%": 100 * self.w_face / n, "blink_%": 100 * self.w_blink / n,
            "cam_drops": self.cam_drops, "cam_fail": self.cam_fail,
        }
        if self.w_loop_ms:
            f["loop_ms"] = np.mean(self.w_loop_ms)
        if gaze is not None:
            f["gaze"] = (round(gaze[0]), round(gaze[1]))
        if cursor is not None:
            f["cursor"] = (round(cursor[0]), round(cursor[1]))
        if len(self.w_raw) >= 5:
            R = np.array(self.w_raw)
            f["gaze_spread_last30_px"] = float(np.hypot(*R[-30:].std(axis=0)))
        if self.w_pose:
            P = np.array(self.w_pose).mean(axis=0)
            f["head_yaw_pitch_roll_deg"] = tuple(np.round(np.degrees(P[:3]), 1))
            f["head_xyz_cm"] = tuple(np.round(P[3:], 1))
            f["dist_cm"] = round(abs(P[5]), 1)
            if self.calib_head is not None:
                d = P - self.calib_head
                f["head_drift_deg"] = tuple(np.round(np.degrees(d[:3]), 1))
                f["head_drift_cm"] = tuple(np.round(d[3:], 1))
        if self.w_light:
            f["light"] = np.mean(self.w_light)
        if self.w_face_px:
            f["face_width_px"] = np.mean(self.w_face_px)
        bs = [b for b in self.w_blink_score if b >= 0]
        if bs:
            f["blink_score_med"] = float(np.median(bs)); f["blink_score_max"] = float(np.max(bs))
        if self.w_ear_ratio:
            f["lid_gap_min_%"] = 100 * float(np.min(self.w_ear_ratio))
        f["cursor_moves"] = self.w_cursor_moves
        self.log("HB", extra or "status", **f)
        self.t_hb = now
        self.reset_window()

    def startup(self, screen, cam, args):
        import platform
        self.log("START", "gaze app starting", log_file=str(self.path) if self.path else "none")
        self.log("START", "system", python=sys.version.split()[0], os=platform.platform(), exe=sys.executable)
        try:
            import importlib.metadata as md
            vers = {p: md.version(p) for p in ("mediapipe", "opencv-python", "numpy", "pyautogui")}
            self.log("START", "packages", **vers)
        except Exception as e:
            self.log("START", f"package versions unavailable ({e})")
        info = {"screen_px": screen}
        if sys.platform == "win32":
            try:
                import ctypes
                info["system_dpi"] = ctypes.windll.user32.GetDpiForSystem()
                info["windows_scale_%"] = round(info["system_dpi"] / 96 * 100)
            except Exception:
                pass
        self.log("START", "display", **info)
        self.log("START", "options", **{k: v for k, v in vars(args).items()})


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:.1f}" if abs(v) >= 10 or v == 0 else f"{v:.3g}"
    if isinstance(v, (list, tuple)):
        return "(" + ",".join(_fmt(x) for x in v) + ")"
    if isinstance(v, np.ndarray):
        return _fmt(v.tolist())
    return str(v)


diag = Diag()

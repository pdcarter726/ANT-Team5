"""Webcam gaze tracking (method from EyeGestures v4).

Face landmarks come from MediaPipe Face Landmarker in VIDEO mode (it tracks the face between frames
and smooths the landmarks, unlike the per-frame IMAGE mode EyeGestures uses). Calibration is a Python
port of the EyeGestures v4 engine's calibration, checked to advance on the exact same frames as the
original: same eye features, same linear model, and a dot only moves on once the live estimate has
settled on it (within 10% of the screen width for 20+ frames). Porting it lets us use 50 points with head moves,
skip blink frames, keep only frames where the eyes had landed on the dot, and never recalibrate on
our own.
"""
import json
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .diag import diag

MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
CALIB_PATH = Path(__file__).parent / "models" / "gaze_calibration.json"   # last calibration (no profile)
PROFILE_DIR = Path(__file__).parent / "profiles"


def profile_path(name):
    safe = "".join(c for c in name.strip() if c.isalnum() or c in "-_ ").strip().replace(" ", "_")
    if not safe:
        raise ValueError("profile name must contain letters or digits")
    return PROFILE_DIR / f"{safe}.json"


def list_profiles():
    """[(name, info dict)] for every saved profile."""
    out = []
    for p in sorted(PROFILE_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            lc = (d.get("last_check") or {}).get("median")
            out.append((p.stem, {"saved": d.get("saved_at", "?"), "screen": d.get("screen"),
                                 "model": d.get("model", {}).get("type", "?"), "score_px": round(d.get("err") or 0),
                                 "last_check_px": round(lc) if lc else "none",
                                 "head_positions": len(set(d.get("data_s", [0])))}))
        except Exception as e:
            out.append((p.stem, {"error": str(e)}))
    return out

# Identical to the EyeGestures v4 engine (engine/src/eyegestures.rs)
LEFT = [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 144, 145, 153, 246, 468]
RIGHT = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 374, 466, 473]
TMP_MAX = 40          # engine keeps the last 40 frames of the current calibration point
SMOOTH_FRAMES = 20    # engine averages the last 20 predictions
SETTLE_FRAMES = 20    # engine: frames near the target before a point is accepted
SETTLE_FRAC = 0.10    # engine: "near" = within 10% of the screen width

# Full calibration: 50 dots in 5 blocks of 10. Each block covers the whole screen, and before each block
# the screen asks you to shift your head a little, so the model learns what head movement looks like
# and the "unseen head position" test can check it. (EyeGestures' own 25 points live on in ENGINE_POINTS.)
ENGINE_POINTS = [
    (0.25, 0.25), (0.5, 0.75), (1.0, 0.5), (0.75, 0.5), (0.0, 0.75),
    (0.5, 0.5), (1.0, 0.25), (0.75, 0.0), (0.25, 0.5), (0.5, 0.0),
    (0.0, 0.5), (1.0, 1.0), (0.75, 1.0), (0.25, 0.0), (1.0, 0.0),
    (0.0, 1.0), (0.25, 1.0), (0.75, 0.75), (0.5, 0.25), (0.0, 0.25),
    (1.0, 0.5), (0.75, 0.25), (0.5, 1.0), (0.25, 0.75), (0.0, 0.0),
]
CALIBRATION_BLOCKS = [
    [(0.0625, 0.0625), (0.25, 1.0), (1.0, 0.25), (0.75, 1.0), (0.875, 0.625), (0.5, 0.25), (0.5, 0.0), (0.125, 0.625), (0.5, 0.5), (0.5, 0.875)],
    [(0.75, 0.5), (0.0, 0.5), (0.5, 1.0), (1.0, 0.75), (0.75, 0.25), (0.625, 0.125), (0.5, 0.625), (0.125, 0.875), (0.25, 0.25), (0.0, 0.0)],
    [(0.125, 0.375), (0.25, 0.0), (1.0, 0.5), (1.0, 1.0), (0.875, 0.125), (0.5, 0.375), (0.25, 0.75), (0.625, 0.875), (0.375, 0.625), (0.625, 0.625)],
    [(0.25, 0.5), (0.0, 0.75), (0.5, 0.75), (1.0, 0.0), (0.625, 0.375), (0.5, 0.125), (0.875, 0.875), (0.875, 0.5), (0.125, 0.125), (0.375, 0.375)],
    [(0.375, 0.875), (0.375, 0.5), (0.625, 0.5), (0.375, 0.125), (0.75, 0.75), (0.875, 0.375), (0.0, 1.0), (0.125, 0.5), (0.75, 0.0), (0.0, 0.25)],
]
CALIBRATION_POINTS = [p for b in CALIBRATION_BLOCKS for p in b]          # 50
BLOCK_SIZE = 10
HEAD_PROMPTS = [
    "Sit in your normal position",
    "Move your head a little to the LEFT (a few cm), then hold still",
    "Move your head a little to the RIGHT (a few cm), then hold still",
    "Lower your chin slightly / lean in a little, then hold still",
    "Raise your chin slightly / sit up a little taller, then hold still",
]
HEAD_MOVE_S = 4.0     # pause before each block so you can read the prompt and move
ADD_POSE_POINTS = [(0.25, 0.25), (0.75, 0.75), (1.0, 0.5), (0.0, 0.5), (0.5, 0.5),
                   (0.75, 0.25), (0.25, 0.75), (0.5, 0.0), (0.5, 1.0)]   # 9

SACCADE_S = 0.4      # after a dot appears, frames this early are kept out of the stored data
                     # (the eyes are still travelling); the dot itself still advances exactly like the engine
VALIDATE_POINTS = [(0.5, 0.5), (0.15, 0.2), (0.85, 0.2), (0.85, 0.8), (0.15, 0.8)]   # easy center dot first
VALIDATE_SETTLE_S, VALIDATE_COLLECT_S = 1.5, 1.2   # first dot also waits for you to settle after pressing the key
VALIDATE_FIRST_EXTRA_S = 1.0

BLINK_SCORE = 0.5     # MediaPipe eyeBlink score (0-1) above which a frame always counts as a blink
BLINK_EAR_RATIO = 0.65  # eyelid gap below this fraction of its recent normal value = blink
BLINK_HOLD_FRAMES = 3 # also skip this many frames after a blink (lids still settling)
BLINK_MAX_S = 0.6     # "blinks" longer than this are not blinks (e.g. looking down): use the frames

RECENTER_WINDOW = 30          # recenter: median of this many frames ...
RECENTER_AGREE_PX = 25        # ... accepted once two windows in a row agree within this
RECENTER_MAX_FRAMES = 150     # ... or after this many frames (median of everything)


def model_present():
    """The face model must be downloaded once with internet (python -m gaze.setup_check);
    the ANT station network may have none."""
    return MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000


import contextlib
import sys


@contextlib.contextmanager
def _quiet_native_stderr():
    """MediaPipe's C++ code prints setup chatter (InitGoogle, XNNPACK, feedback tensors) straight to the
    process stderr. Point file descriptor 2 at the null device while it starts up."""
    try:
        sys.stderr.flush()
        saved = os.dup(2)
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 2)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(null)
        os.close(saved)


class FaceLandmarks:
    """MediaPipe Face Landmarker with blendshapes (blink scores) and the head transformation matrix.
    video=True uses VIDEO mode: the face is tracked from frame to frame and landmarks are smoothed."""

    def __init__(self, video=True):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp, self.video = mp, video
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.last_ts = -1
        with _quiet_native_stderr():
            self.lm = vision.FaceLandmarker.create_from_options(opts)
            self.find(np.zeros((360, 640, 3), np.uint8))   # warm-up: the first detection also prints chatter
        diag.log("FACE", "MediaPipe Face Landmarker ready", mode="VIDEO (tracking)" if video else "IMAGE (per frame)",
                 model=MODEL_PATH.name, model_bytes=MODEL_PATH.stat().st_size)

    def find(self, frame_bgr):
        import cv2
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        if self.video:
            ts = max(int(time.monotonic() * 1000), self.last_ts + 1)   # must strictly increase
            self.last_ts = ts
            res = self.lm.detect_for_video(img, ts)
        else:
            res = self.lm.detect(img)
        return res if res.face_landmarks else None

    def close(self):
        try:
            self.lm.close()
        except Exception:
            pass


@dataclass
class GazeSample:
    face: bool
    x: float = 0.0
    y: float = 0.0
    calibrating: bool = True
    mode: str = "calibrate"         # calibrate | addpose | recenter | run
    target: tuple = (0.0, 0.0)
    blink: bool = False
    note: str = ""                  # on-screen instruction (e.g. move your head); no dot while set
    frame: np.ndarray = None


class Camera:
    """Threaded webcam reader that always hands back the newest frame."""

    def __init__(self, index=0, width=1280, height=720, fps=30):
        import cv2
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}. Try --camera 1.")
        # On Windows (DirectShow) many webcams only reach 30 fps at 720p when MJPG is requested;
        # the default uncompressed format often gives 5-15 fps. Order: fps, format, then size.
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fcc >> 8 * i) & 0xFF) for i in range(4)) if fcc else "?"
        try:
            backend = self.cap.getBackendName()
        except Exception:
            backend = "?"
        diag.log("CAM", f"camera {index} open", backend=backend, requested=f"{width}x{height}@{fps}",
                 delivered=f"{self.size[0]}x{self.size[1]}", reported_fps=self.cap.get(cv2.CAP_PROP_FPS),
                 format=fourcc, auto_exposure=self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                 exposure=self.cap.get(cv2.CAP_PROP_EXPOSURE), brightness=self.cap.get(cv2.CAP_PROP_BRIGHTNESS))
        self.q = queue.Queue(maxsize=1)
        self.running = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                diag.cam_fail += 1
                if diag.cam_fail in (1, 10, 100) or diag.cam_fail % 1000 == 0:
                    diag.log("CAM", "camera read failed", total_failures=diag.cam_fail)
                time.sleep(0.01)        # camera hiccup: don't spin the CPU
                continue
            diag.cam_frames += 1
            if self.q.full():
                try:
                    self.q.get_nowait()
                    diag.cam_drops += 1    # processing slower than the camera: older frame skipped
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self, timeout=1.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self.running = False
        self.t.join(timeout=1)
        self.cap.release()


class Keypoints:
    """Exact port of the engine's feature vector (eye landmarks in the face box, scale, size, head offset)."""

    def __init__(self):
        self.start_w = self.start_h = 0.0
        self.head_start = None

    def __call__(self, lm):
        lm = np.asarray(lm, dtype=np.float64)
        off = lm.min(axis=0)
        size = lm.max(axis=0) - off
        if self.start_w * self.start_h == 0.0:
            self.start_w, self.start_h = size
        scale = size / np.array([self.start_w, self.start_h])
        if self.head_start is None:
            self.head_start = off * scale
        eyes = ((lm[LEFT + RIGHT] - off) / size) * scale
        return np.concatenate([eyes.reshape(-1), scale, size, off * scale - self.head_start])

    def state(self):
        return {"start_w": self.start_w, "start_h": self.start_h, "head_start": list(self.head_start)}

    def load(self, d):
        self.start_w, self.start_h = d["start_w"], d["start_h"]
        self.head_start = np.array(d["head_start"])


class BlinkFilter:
    """Flags blink frames. Two detectors, either one counts:
      1. MediaPipe's eyeBlinkLeft/Right score: above BLINK_SCORE, or well above your own normal level
         (in VIDEO mode the scores are smoothed and often stay under 0.5 during real blinks)
      2. the eyelid gap (eye aspect ratio) suddenly dropping well below its recent normal value
    Blink frames are ignored everywhere: calibration, recenter, the check and tracking.
    A "blink" longer than BLINK_MAX_S is looking down or squinting, so those frames are used."""

    def __init__(self):
        self.hold = 0
        self.closed_since = None
        self.scores = deque(maxlen=150)     # ~5 s of history for "your normal level"
        self.ears = deque(maxlen=150)
        self.last = (0.0, 1.0)              # (score, ear ratio) of the latest frame, for diagnostics

    @staticmethod
    def score(det, lm):
        try:
            cats = {c.category_name: c.score for c in det.face_blendshapes[0]}
            return (cats["eyeBlinkLeft"] + cats["eyeBlinkRight"]) / 2
        except Exception:
            return None

    @staticmethod
    def ear(lm):
        """Eye aspect ratio: lid gap / eye width, averaged over both eyes (~0.25-0.3 open, <0.1 closed)."""
        v = []
        for top, bot, a, b in ((159, 145, 33, 133), (386, 374, 362, 263)):
            v.append(np.linalg.norm(lm[top] - lm[bot]) / (np.linalg.norm(lm[a] - lm[b]) + 1e-9))
        return float(np.mean(v))

    def __call__(self, det, lm, now):
        sc = self.score(det, lm)
        e = self.ear(lm)
        base_sc = float(np.median(self.scores)) if len(self.scores) >= 15 else None
        base_ear = float(np.median(self.ears)) if len(self.ears) >= 15 else None
        ratio = e / base_ear if base_ear else 1.0
        closed = False
        if sc is not None:
            closed |= sc > BLINK_SCORE
            closed |= base_sc is not None and sc > 0.25 and sc - base_sc > 0.2
        closed |= base_ear is not None and ratio < BLINK_EAR_RATIO
        self.last = (sc if sc is not None else -1.0, ratio)
        diag.w_blink_score.append(self.last[0]); diag.w_ear_ratio.append(ratio)
        if not closed:                       # only open-eye frames define "normal"
            if sc is not None:
                self.scores.append(sc)
            self.ears.append(e)
        if closed:
            self.closed_since = self.closed_since or now
            if now - self.closed_since > BLINK_MAX_S:     # not a blink (looking down, squinting)
                return False
            self.hold = BLINK_HOLD_FRAMES
            return True
        self.closed_since = None
        if self.hold > 0:
            self.hold -= 1
            return True
        return False


N_ENGINE = 66   # engine feature columns; extra features follow (see extra_features)
N_FEATURES = N_ENGINE + 14

# Column layout of a stored sample: 0-59 engine eye landmarks, 60-63 face scale/size, 64-65 head offset,
# 66-73 iris/lid values per eye, 74-76 head yaw/pitch/roll, 77-79 head position (cm).
# Head columns, and how much they must vary in the calibration data before a model may use them.
HEAD_GROUPS = ((slice(60, 64), 0.02), (slice(64, 66), 0.01), (slice(74, 77), 0.05), (slice(77, 80), 1.0))
# physical scales: a change of this size counts as "one unit" for the ridge penalty
SCALES = np.array([0.01] * 60 + [0.02] * 4 + [0.01] * 2 + [0.05] * 8 + [0.05] * 3 + [1.0] * 3)
EYE_EXTRA = slice(66, 74)


def head_ok(X, use_head):
    """Boolean per column: may a model use it? Head columns only if use_head and they really varied."""
    ok = np.ones(N_FEATURES, dtype=bool)
    for sl, min_range in HEAD_GROUPS:
        cols = np.arange(N_FEATURES)[sl]
        ok[cols] = bool(use_head) and np.ptp(np.asarray(X)[:, cols], axis=0).max() >= min_range
    return ok


class LinearGaze:
    """The EyeGestures v4 model: ordinary linear regression with a bias on the engine features.
    Head columns are only used when allowed (see head_ok); with the head still during calibration they
    would get enormous weights and any later head shift would throw the cursor far off."""

    def __init__(self, use_head=True):
        self.use_head = use_head
        self.cols = self.W = None

    @property
    def name(self):
        return "engine linear" + (" + head" if self.use_head else "")

    def fit(self, X, Y):
        X = np.asarray(X)
        ok = head_ok(X, self.use_head)
        ok[N_ENGINE:] = False
        self.cols = np.flatnonzero(ok)
        A = np.hstack([X[:, self.cols], np.ones((len(X), 1))])
        AtA = A.T @ A + 1e-9 * np.eye(A.shape[1])
        try:
            self.W = np.linalg.solve(AtA, A.T @ np.asarray(Y, dtype=float))
        except np.linalg.LinAlgError:
            self.W = np.linalg.lstsq(A, np.asarray(Y, dtype=float), rcond=None)[0]
        return self

    def predict(self, x):
        return np.append(np.asarray(x)[self.cols], 1.0) @ self.W

    def to_dict(self):
        return {"type": "engine", "use_head": self.use_head, "cols": self.cols.tolist(), "W": self.W.tolist()}


# ---------- extra features: iris inside each eye + head pose ----------
EYES = ((33, 133, 468, 159, 145), (263, 362, 473, 386, 374))   # outer corner, inner corner, iris, top lid, bottom lid


def extra_features(lm, frame_shape, det):
    """Per eye: iris position along / across the eye axis and lid opening / lid position, all in eye
    widths (so they don't change when the head moves sideways); then head yaw, pitch, roll and position
    from MediaPipe's face transformation matrix."""
    h, w = frame_shape[:2]
    P = np.asarray(lm, dtype=np.float64) * np.array([w, h])
    out = []
    for outer, inner, iris, top, bot in EYES:
        a, b = P[outer], P[inner]
        c = (a + b) / 2
        width = np.linalg.norm(b - a) + 1e-9
        u = (b - a) / width
        v = np.array([-u[1], u[0]])
        d = P[iris] - c
        mid = (P[top] + P[bot]) / 2 - c
        out += [d @ u / width, d @ v / width, np.linalg.norm(P[top] - P[bot]) / width, mid @ v / width]
    try:
        M = np.asarray(det.facial_transformation_matrixes[0], dtype=np.float64).reshape(4, 4)
        R, t = M[:3, :3], M[:3, 3]
        out += [np.arctan2(R[0, 2], R[2, 2]), np.arcsin(np.clip(-R[1, 2], -1, 1)), np.arctan2(R[1, 0], R[1, 1])]
        out += list(t)
    except Exception:
        out += [0.0] * 6
    return np.array(out)


class RidgeGaze:
    """Ridge regression in physical units. kind="eye": iris/lid values (+ head pose if allowed);
    kind="combined": engine features too. quad adds squared/cross terms of the iris/lid values, since
    gaze to screen is not perfectly linear near the edges."""

    def __init__(self, lam=1.0, quad=False, kind="eye", use_head=True):
        self.lam, self.quad, self.kind, self.use_head = lam, quad, kind, use_head
        self.cols = self.mu = self.W = None

    @property
    def name(self):
        base = "engine+eye" if self.kind == "combined" else "eye"
        return f"{base} {'quadratic' if self.quad else 'linear'}{' + head' if self.use_head else ''} (ridge {self.lam:g})"

    def _phi(self, F):
        Z = (np.atleast_2d(F) - self.mu) / SCALES[self.cols]
        if self.quad:
            e = Z[:, self.eye_pos]
            iu = np.triu_indices(e.shape[1])
            Z = np.hstack([Z, (e[:, :, None] * e[:, None, :])[:, iu[0], iu[1]] / 4.0])
        return np.hstack([Z, np.ones((Z.shape[0], 1))])

    def fit(self, X, Y):
        X = np.asarray(X)
        ok = head_ok(X, self.use_head)
        if self.kind == "eye":
            ok[:N_ENGINE] = False
        self.cols = np.flatnonzero(ok)
        self.eye_pos = np.flatnonzero(np.isin(self.cols, np.arange(N_FEATURES)[EYE_EXTRA]))
        F = X[:, self.cols]
        self.mu = F.mean(axis=0)
        A = self._phi(F)
        R = self.lam * np.eye(A.shape[1]); R[-1, -1] = 0
        self.W = np.linalg.solve(A.T @ A + R, A.T @ np.asarray(Y, dtype=float))
        return self

    def predict(self, x):
        return (self._phi(np.asarray(x)[self.cols]) @ self.W)[0]

    def to_dict(self):
        return {"type": "ridge", "lam": self.lam, "quad": self.quad, "kind": self.kind, "use_head": self.use_head,
                "cols": self.cols.tolist(), "mu": self.mu.tolist(), "W": self.W.tolist()}


def model_from_dict(d):
    if d["type"] == "engine":
        m = LinearGaze(d["use_head"])
    else:
        m = RidgeGaze(d["lam"], d["quad"], d["kind"], d["use_head"])
        m.mu = np.array(d["mu"])
    m.cols, m.W = np.array(d["cols"]), np.array(d["W"])
    if d["type"] != "engine":
        m.eye_pos = np.flatnonzero(np.isin(m.cols, np.arange(N_FEATURES)[EYE_EXTRA]))
    return m


def _candidates():
    for use_head in (False, True):
        yield lambda h=use_head: LinearGaze(h)
        for lam in (1.0, 10.0):
            yield lambda l=lam, h=use_head: RidgeGaze(l, False, "eye", h)
        for lam in (3.0, 30.0):
            yield lambda l=lam, h=use_head: RidgeGaze(l, True, "eye", h)
        for lam in (0.3, 3.0):
            yield lambda l=lam, h=use_head: RidgeGaze(l, False, "combined", h)


def select_model(X, Y, G, S, log=print):
    """Try every candidate model and keep the one that predicts best on data it did NOT train on:
      - leave one dot out (accuracy on the screen), and
      - if head positions were added with 'a': leave one added head position out (does it still work
        at a head position it never saw?). A model that memorizes the seen head positions fails this.
    The score is the average of the two. Also fits the edge stretch (regression pulls predictions
    toward the center; screen = a * pred + b per axis, measured on held-out dots).
    Returns (model, stretch, score in px, report lines, detail)."""
    X, Y, G, S = np.asarray(X), np.asarray(Y, dtype=float), np.asarray(G), np.asarray(S)
    pts = np.unique(G)
    T = np.array([Y[G == g][0] for g in pts])
    pose_sessions = [s for s in np.unique(S) if s != S.min()]
    results = []
    for make in _candidates():
        try:
            P = []
            for g in pts:
                m = make().fit(X[G != g], Y[G != g])
                P.append(np.mean([m.predict(x) for x in X[G == g]], axis=0))
            P = np.array(P)
            stretch = []
            for i in range(2):
                a, b = np.polyfit(P[:, i], T[:, i], 1)
                if not 0.8 <= a <= 1.8:
                    a, b = 1.0, 0.0
                stretch.append([float(a), float(b)])
            st = lambda p: np.array([stretch[0][0] * p[0] + stretch[0][1], stretch[1][0] * p[1] + stretch[1][1]])
            dot_err = float(np.mean([np.linalg.norm(st(p) - t) for p, t in zip(P, T)]))
            pose_err = None
            if pose_sessions:
                errs = []
                for s_ in pose_sessions:
                    m = make().fit(X[S != s_], Y[S != s_])
                    for g in np.unique(G[S == s_]):
                        p = np.mean([m.predict(x) for x in X[G == g]], axis=0)
                        errs.append(np.linalg.norm(st(p) - Y[G == g][0]))
                pose_err = float(np.mean(errs))
            score = dot_err if pose_err is None else (dot_err + pose_err) / 2
            per_dot = [(tuple(t), float(np.linalg.norm(st(p) - t))) for p, t in zip(P, T)]
            results.append((score, dot_err, pose_err, make, stretch, per_dot))
        except Exception as e:                      # a candidate failing must never break calibration
            log(f"  candidate skipped: {e}")
    results.sort(key=lambda r: r[0])
    lines = []
    for score, de, pe, make, _, _ in results:
        extra = "" if pe is None else f", at an unseen head position {pe:5.0f} px"
        lines.append(f"  {make().name:40s} held-out dots {de:5.0f} px{extra}")
    score, de, pe, make, stretch, per_dot = results[0]
    return make().fit(X, Y), stretch, score, lines, {"per_dot": per_dot}


class CalibrationSession:
    """Python port of the EyeGestures v4 engine calibration (engine/src/eyegestures.rs + calibrator.rs):
    each frame the live estimate (mean of the last 20 predictions) is compared with the dot; the dot
    only moves on once the estimate has stayed within 10% of the screen width for 20+ frames. The model
    is refit every frame on the last 40 frames of the current dot plus everything kept from earlier dots."""

    def __init__(self, points, w, h, label="calibration", block_size=None, prompts=None):
        self.label = label
        self.block_size, self.prompts = block_size, prompts or []
        self.pause_until, self.note = None, ""
        self.stable_b = []                   # block (head position) of each stored sample
        self.points = [(fx * w, fy * h) for fx, fy in points]
        self.w = w
        self.idx = 0
        self.tmp = deque(maxlen=TMP_MAX)
        self.stable_x, self.stable_y, self.stable_g = [], [], []
        self.S_AtA = None
        self.S_Aty = None
        self.W = None
        self.buf = deque(maxlen=SMOOTH_FRAMES)
        self.frames_on_point = 0
        self.estimate = (w / 2, h / 2)
        self.point_t0 = None
        self.skipped = 0
        self.dot_frames = self.dot_blinks = self.dot_lost = 0
        self.t_start = time.time()

    @property
    def done(self):
        return self.idx >= len(self.points)

    @property
    def block(self):
        return self.idx // self.block_size if self.block_size else 0

    def paused(self, now):
        """At the start of every block: show the head-position prompt for HEAD_MOVE_S, ignore frames."""
        if self.block_size and not self.done and self.idx % self.block_size == 0 and self.dot_frames == 0 \
                and self.pause_until is None and not getattr(self, "_paused_for", None) == self.idx:
            self._paused_for = self.idx
            self.pause_until = now + HEAD_MOVE_S
            b = self.block
            self.note = self.prompts[b] if b < len(self.prompts) else "Move your head a little, then hold still"
            diag.log("CAL", f"head position {b + 1}/{-(-len(self.points) // self.block_size)}: {self.note}")
        if self.pause_until is not None:
            if now < self.pause_until:
                return True
            self.pause_until, self.note = None, ""
            self.buf.clear(); self.frames_on_point = 0; self.point_t0 = None
        return False

    @property
    def target(self):
        return self.points[min(self.idx, len(self.points) - 1)]

    def _refit(self):
        Xt = np.array([np.append(k, 1.0) for k, _, _ in self.tmp])
        Yt = np.array([t for _, t, _ in self.tmp], dtype=float)
        AtA, Aty = Xt.T @ Xt, Xt.T @ Yt
        if self.S_AtA is not None:
            AtA, Aty = AtA + self.S_AtA, Aty + self.S_Aty
        try:
            self.W = np.linalg.solve(AtA + 1e-9 * np.eye(AtA.shape[0]), Aty)
        except np.linalg.LinAlgError:
            pass

    def skip(self):
        """User asked to skip a dot that won't settle: drop its frames and move on."""
        t = self.target
        diag.log("CAL", f"{self.label} dot {self.idx + 1}/{len(self.points)} SKIPPED by user", target=(round(t[0]), round(t[1])),
                 frames=self.dot_frames, blink_frames=self.dot_blinks, no_face_frames=self.dot_lost,
                 estimate=(round(self.estimate[0]), round(self.estimate[1])))
        self.dot_frames = self.dot_blinks = self.dot_lost = 0
        self.tmp.clear(); self.buf.clear()
        self.frames_on_point = 0
        self.point_t0 = None
        self.idx += 1
        self.skipped += 1

    def step(self, k, full, now=None):
        """k: engine features (drive the dot exactly like the engine), full: everything we store."""
        now = time.time() if now is None else now
        if self.point_t0 is None:
            self.point_t0 = now
        self.dot_frames += 1
        target = self.target
        raw = np.append(k, 1.0) @ self.W if self.W is not None else np.zeros(2)
        self.buf.append(raw)
        est = np.mean(self.buf, axis=0)
        self.estimate = (float(est[0]), float(est[1]))
        self.tmp.append((k, target, (full, now - self.point_t0)))
        self._refit()
        near = np.hypot(est[0] - target[0], est[1] - target[1]) < SETTLE_FRAC * self.w
        if near and self.frames_on_point > SETTLE_FRAMES:
            X = np.array([np.append(kk, 1.0) for kk, _, _ in self.tmp])
            Y = np.array([t for _, t, _ in self.tmp], dtype=float)
            self.S_AtA = X.T @ X + (self.S_AtA if self.S_AtA is not None else 0)
            self.S_Aty = X.T @ Y + (self.S_Aty if self.S_Aty is not None else 0)
            # stored data: only frames after the eyes had time to land on the dot
            keep = [(ff, t) for _, t, (ff, age) in self.tmp if age >= SACCADE_S]
            if len(keep) < 10:                       # very fast settle: keep the latest frames anyway
                keep = [(ff, t) for _, t, (ff, _) in list(self.tmp)[-10:]]
            self.stable_x += [ff for ff, _ in keep]
            self.stable_y += [t for _, t in keep]
            self.stable_g += [self.idx] * len(keep)
            self.stable_b += [self.block] * len(keep)
            diag.log("CAL", f"{self.label} dot {self.idx + 1}/{len(self.points)} settled",
                     target=(round(target[0]), round(target[1])), secs=now - self.point_t0, frames=self.dot_frames,
                     blink_frames=self.dot_blinks, no_face_frames=self.dot_lost,
                     kept=len(keep), dropped_eyes_moving=len(self.tmp) - len(keep),
                     estimate_err_px=float(np.hypot(est[0] - target[0], est[1] - target[1])))
            self.dot_frames = self.dot_blinks = self.dot_lost = 0
            self.tmp.clear()
            self.buf.clear()
            self.frames_on_point = 0
            self.point_t0 = None
            self.idx += 1
            if self.done:
                diag.log("CAL", f"{self.label} finished", secs=time.time() - self.t_start,
                         dots=len(self.points) - self.skipped, skipped=self.skipped, samples=len(self.stable_x))
        elif near:
            self.frames_on_point += 1


class GazeTracker:
    """Modes:
      calibrate  full calibration (50 dots, 5 head positions), replaces the model
      addpose    9 more points at the current head position, ADDED to the existing data, then refit.
                 Teaches the model what head movement looks like.
      recenter   look at one center dot; shifts all predictions by the measured offset
      run        normal tracking
    Nothing switches mode on its own: only recalibrate(), add_pose() and recenter() do.
    """

    def __init__(self, screen_w, screen_h, camera=0, video_mode=True, start_calibration=True):
        if not model_present():
            raise SystemExit("Face model missing. Run once with internet:  python -m gaze.setup_check")
        self.w, self.h = int(screen_w), int(screen_h)
        self.faces = FaceLandmarks(video=video_mode)
        self.keypoints = Keypoints()
        self.blinks = BlinkFilter()
        self.cam = Camera(camera)
        self.validation = None
        self.calib_path = CALIB_PATH
        self.profile = None
        self.model = None
        self.err = 0.0
        self.data_x, self.data_y, self.data_g, self.data_s = [], [], [], []
        self.history = []
        self.stretch = [[1.0, 0.0], [1.0, 0.0]]
        self.offset = np.zeros(2)
        self.buf = deque(maxlen=SMOOTH_FRAMES)
        self.last_xy = (self.w / 2, self.h / 2)
        self.mode = "idle"
        if start_calibration:
            self.recalibrate()

    # ---------- mode switches (only ever called by the user / the EEG app) ----------
    def recalibrate(self):
        self.data_x, self.data_y, self.data_g, self.data_s = [], [], [], []
        self.history = []
        self.mode = "calibrate"
        self.session = CalibrationSession(CALIBRATION_POINTS, self.w, self.h, "calibration",
                                          block_size=BLOCK_SIZE, prompts=HEAD_PROMPTS)
        diag.log("MODE", "full calibration started", dots=len(CALIBRATION_POINTS), head_positions=len(HEAD_PROMPTS),
                 settle_frames=SETTLE_FRAMES,
                 settle_radius_px=SETTLE_FRAC * self.w, dropped_after_dot_appears_s=SACCADE_S)

    def add_pose(self):
        if self.model is None:
            return self.recalibrate()
        self.mode = "addpose"
        self.session = CalibrationSession(ADD_POSE_POINTS, self.w, self.h, "add-head-position")
        diag.log("MODE", "add head position started", dots=len(ADD_POSE_POINTS))

    def recenter(self):
        if self.model is None:
            return
        self.mode = "recenter"
        self.recenter_preds = []
        self.recenter_t0 = time.time()
        diag.log("MODE", "recenter started", old_offset=self.offset)

    def validate(self):
        """Accuracy check: 5 dots, measure where the gaze estimate actually lands (does not change the model)."""
        if self.model is None:
            return
        self.mode = "validate"
        self.val_order = list(range(len(VALIDATE_POINTS)))   # dot indices still to show
        self.val_idx, self.val_t0, self.val_samples = 0, None, []
        self.val_result = {}                                  # dot index -> (mean point, error)
        self.val_retried = False
        self.val_pose = []
        diag.log("MODE", "accuracy check started", dots=len(VALIDATE_POINTS),
                 settle_s=VALIDATE_SETTLE_S, collect_s=VALIDATE_COLLECT_S)

    def apply_check_fix(self):
        """Use the last accuracy check to correct the output: per axis, screen = a * current + b,
        fitted on the 5 check dots (bounded; offset only if the scale looks unreliable). Only when asked."""
        v = self.validation
        if not v or len(v.get("means", [])) < 4 or self.model is None:
            diag.log("CHECK", "no accuracy check to correct from yet: run v first")
            return False
        keep = [i for i in range(len(v["means"])) if i not in v.get("outliers", [])]
        P, T = np.array(v["means"])[keep], np.array(v["targets"])[keep]
        new_stretch, new_offset = [], []
        for i in range(2):
            a, b = np.polyfit(P[:, i], T[:, i], 1)
            if not 0.75 <= a <= 1.35:                      # scale not trustworthy: shift only
                a, b = 1.0, float(np.mean(T[:, i] - P[:, i]))
            # current output = stretch(raw) + offset; corrected = a * that + b
            sa, sb = self.stretch[i]
            new_stretch.append([float(a * sa), float(a * sb)])
            new_offset.append(float(a * self.offset[i] + b))
        self.stretch, self.offset = new_stretch, np.array(new_offset)
        self.buf.clear()
        self.validation = None
        self.save()
        diag.log("CHECK", "applied correction from the accuracy check (press v to check again)", stretch_x=self.stretch[0], stretch_y=self.stretch[1], offset=self.offset)
        return True

    def skip_point(self):
        """Skip a calibration dot that won't settle (only when the user asks)."""
        if self.mode in ("calibrate", "addpose"):
            self.session.skip()
            if self.session.done:
                diag.log("CAL", f"{self.session.label} finished after skip", skipped=self.session.skipped)
                self._finish_session()

    @property
    def calibrating(self):
        return self.mode != "run"

    @property
    def progress(self):
        """(current, total) for the on-screen counter."""
        if self.mode in ("calibrate", "addpose"):
            return min(self.session.idx + 1, len(self.session.points)), len(self.session.points)
        if self.mode == "recenter":
            return min(len(self.recenter_preds), RECENTER_WINDOW * 2), RECENTER_WINDOW * 2
        if self.mode == "validate":
            return min(self.val_idx + 1, len(self.val_order)), len(self.val_order)
        return 0, 0

    # ---------- model ----------
    def undo_pose(self):
        """Throw away the most recent 'add head position' and go back to the model before it."""
        if not self.history:
            return False
        n, self.model, self.stretch, self.err = self.history.pop()
        del self.data_x[n:], self.data_y[n:], self.data_g[n:], self.data_s[n:]
        self.offset = np.zeros(2)
        self.buf.clear()
        self.save()
        diag.log("FIT", "undid last head position", model=self.model.name, score_px=self.err, samples=len(self.data_x))
        return True

    def _finish_session(self):
        if self.mode == "addpose":
            self.history.append((len(self.data_x), self.model, self.stretch, self.err))
        if not self.session.stable_g:                 # every dot skipped: nothing learned
            diag.log("CAL", "no calibration data collected (all dots skipped)")
            self.mode = "run" if self.model is not None else "calibrate"
            if self.model is None:
                self.session = CalibrationSession(CALIBRATION_POINTS, self.w, self.h)
            return
        base = max(self.data_g, default=-1) + 1
        sid = max(self.data_s, default=-1) + 1
        self.data_x += self.session.stable_x
        self.data_y += self.session.stable_y
        self.data_g += [base + g for g in self.session.stable_g]
        self.data_s += [sid + b for b in (self.session.stable_b or [0] * len(self.session.stable_g))]
        X = np.asarray(self.data_x)
        diag.log("FIT", "fitting gaze models ...", samples=len(X), dots=len(set(self.data_g)),
                 head_positions=len(set(self.data_s)))
        t0 = time.time()
        self.model, self.stretch, self.err, lines, detail = select_model(self.data_x, self.data_y, self.data_g, self.data_s)
        diag.log("FIT", f"done in {time.time() - t0:.1f} s; models tried on your calibration (lower is better):")
        for l in lines:
            diag.log("FIT", l.strip())
        used_head = bool(np.isin(self.model.cols, np.r_[60:66, 74:80]).any())
        diag.log("FIT", f"USING {self.model.name}", score_px=self.err, features=len(self.model.cols),
                 uses_head_features=used_head, stretch_x=self.stretch[0], stretch_y=self.stretch[1])
        hp = X[:, 74:80]
        diag.log("FIT", "head movement inside the calibration data",
                 yaw_pitch_roll_range_deg=np.round(np.degrees(np.ptp(hp[:, :3], axis=0)), 1),
                 xyz_range_cm=np.round(np.ptp(hp[:, 3:], axis=0), 1))
        diag.calib_head = hp.mean(axis=0)
        # where on the screen the chosen model is weakest (held-out error per dot)
        per = sorted(detail["per_dot"], key=lambda r: -r[1])
        edge = [e for (pt, e) in per if min(pt[0], self.w - pt[0], pt[1], self.h - pt[1]) < 0.1 * self.w]
        inner = [e for (pt, e) in per if min(pt[0], self.w - pt[0], pt[1], self.h - pt[1]) >= 0.1 * self.w]
        diag.log("FIT", "held-out error by screen area", edge_dots_px=np.mean(edge) if edge else 0,
                 inner_dots_px=np.mean(inner) if inner else 0)
        diag.log("FIT", "worst dots (target -> held-out error px)",
                 worst=[f"({pt[0]:.0f},{pt[1]:.0f})->{e:.0f}" for pt, e in per[:5]])
        self.offset = np.zeros(2)
        self.buf.clear()
        self.mode = "run"
        self.save()

    def save(self):
        """Write the calibration in the background (it can be a few MB; writing it inline caused a hitch)."""
        last_check = None
        if self.validation:
            last_check = {k: self.validation[k] for k in ("median", "mean", "worst") if k in self.validation}
        path = self.calib_path
        snap = {"profile": self.profile, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "last_check": last_check,
                "model": self.model.to_dict(), "err": self.err, "stretch": self.stretch,
                "offset": self.offset.tolist(), "keypoints": self.keypoints.state(), "screen": [self.w, self.h],
                "data_x": np.asarray(self.data_x).round(6).tolist(),
                "data_y": np.asarray(self.data_y).tolist(), "data_g": [int(g) for g in self.data_g],
                "data_s": [int(v) for v in self.data_s]}

        def write():
            t0 = time.time()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(snap))
                tmp.replace(path)                    # never leave a half-written file behind
                diag.log("SAVE", f"calibration saved{' to profile ' + repr(self.profile) if self.profile else ''}",
                         path=f"{path.parent.name}/{path.name}", kb=path.stat().st_size // 1024,
                         secs=time.time() - t0)
            except Exception as e:
                diag.error(f"saving calibration failed: {e}", e)
        threading.Thread(target=write, daemon=True).start()

    def load_calibration(self, path=None):
        d = json.loads(Path(path or self.calib_path).read_text())
        if d["screen"] != [self.w, self.h]:
            raise ValueError("saved calibration is for a different screen size")
        if "model" not in d or "cols" not in d["model"]:
            raise ValueError("saved calibration is from an older version, please recalibrate")
        self.model = model_from_dict(d["model"])
        self.err = d.get("err", 0.0)
        self.stretch = d.get("stretch", [[1.0, 0.0], [1.0, 0.0]])
        self.offset = np.array(d.get("offset", [0.0, 0.0]))
        self.data_x = [np.array(x) for x in d.get("data_x", [])]
        self.data_y = [tuple(y) for y in d.get("data_y", [])]
        self.data_g = d.get("data_g", [])
        self.data_s = d.get("data_s", [0] * len(self.data_g))
        self.history = []
        self.keypoints.load(d["keypoints"])
        self.buf.clear()
        self.mode = "run"
        X = np.asarray(self.data_x)
        if len(X):
            diag.calib_head = X[:, 74:80].mean(axis=0)
        diag.log("SAVE", f"loaded {'profile ' + repr(self.profile) if self.profile else 'saved calibration'}",
                 saved_at=d.get("saved_at", "?"), last_check_px=(d.get("last_check") or {}).get("median"),
                 model=self.model.name, score_px=self.err,
                 samples=len(self.data_x), head_positions=len(set(self.data_s)), offset=self.offset)

    def _features(self, lm, frame, det):
        k = self.keypoints(lm)
        return k, np.concatenate([k, extra_features(lm, frame.shape, det)])

    def _predict(self, full):
        raw = self.model.predict(full)
        return np.array([self.stretch[0][0] * raw[0] + self.stretch[0][1],
                         self.stretch[1][0] * raw[1] + self.stretch[1][1]])

    # ---------- per frame ----------
    def step(self):
        frame = self.cam.read()
        if frame is None:
            return None
        now = time.time()
        t_det = time.perf_counter()
        det = self.faces.find(frame)
        det_ms = (time.perf_counter() - t_det) * 1000
        self._nframe = getattr(self, "_nframe", 0) + 1
        light = None
        if self._nframe % 10 == 0:                          # average brightness 0-255 (cheap, every 10th frame)
            light = float(frame[::8, ::8].mean())
        if det is None or not det.face_landmarks:
            diag.frame(False, False, det_ms, light=light)
            if self.mode in ("calibrate", "addpose"):
                self.session.dot_lost += 1
            return GazeSample(face=False, calibrating=self.calibrating, mode=self.mode, frame=frame)
        lm = np.array([(p.x, p.y) for p in det.face_landmarks[0]], dtype=np.float64)
        blink = self.blinks(det, lm, now)
        k, full = self._features(lm, frame, det)
        face_px = float(np.ptp(lm[:, 0]) * frame.shape[1])
        self._diag_args = dict(det_ms=det_ms, extras=full[N_ENGINE:], light=light, face_px=face_px)

        if self.mode in ("calibrate", "addpose"):
            sess = self.session
            diag.frame(True, blink, **self._diag_args)
            if sess.paused(now):
                left = sess.pause_until - now
                return GazeSample(face=True, x=sess.estimate[0], y=sess.estimate[1], calibrating=True, mode=self.mode,
                                  target=(self.w / 2, self.h / 2), note=f"{sess.note}   ({left:.0f})", frame=frame)
            if blink:
                sess.dot_blinks += 1
            else:                                           # blink frames never reach calibration
                sess.step(k, full, now)
                if sess.done:
                    self._finish_session()
            x, y = sess.estimate
            tx, ty = sess.target
            return GazeSample(face=True, x=x, y=y, calibrating=self.calibrating, mode=self.mode,
                              target=(min(max(tx, 0), self.w), min(max(ty, 0), self.h)), blink=blink, frame=frame)

        if blink:                                            # hold the last position through blinks
            diag.frame(True, True, **self._diag_args)
            x, y = self.last_xy
            target = ((self.w / 2, self.h / 2) if self.mode == "recenter" else
                      self._val_target() if self.mode == "validate" else (0.0, 0.0))
            return GazeSample(face=True, x=x, y=y, calibrating=self.calibrating, mode=self.mode,
                              target=target, blink=True, frame=frame)

        raw = self._predict(full)
        diag.frame(True, False, raw=raw.tolist(), **self._diag_args)
        if self.mode == "validate":
            self.val_pose.append(full[74:80])
        if self.mode == "recenter":
            # Median of recent frames: single noisy frames can't move it, so the grey dot stays put.
            c = np.array([self.w / 2, self.h / 2])
            self.recenter_preds.append(raw)
            P = np.array(self.recenter_preds)
            n = len(P)
            est = np.median(P[-RECENTER_WINDOW:], axis=0)
            done = False
            if n >= 2 * RECENTER_WINDOW:
                prev = np.median(P[-2 * RECENTER_WINDOW:-RECENTER_WINDOW], axis=0)
                done = np.linalg.norm(est - prev) < RECENTER_AGREE_PX
            if n >= RECENTER_MAX_FRAMES:
                est, done = np.median(P[RECENTER_WINDOW // 2:], axis=0), True
            if done:
                old = self.offset.copy()
                self.offset = c - est
                diag.log("RECENTER", "done", frames=n, secs=time.time() - self.recenter_t0,
                         estimate_before=(round(est[0]), round(est[1])), new_offset=self.offset, old_offset=old,
                         spread_px=float(np.hypot(*P[-RECENTER_WINDOW:].std(axis=0))))
                self.buf.clear()
                self.mode = "run"
                self.save()
            x, y = est + self.offset
            self.last_xy = (float(x), float(y))
            return GazeSample(face=True, x=float(x), y=float(y), calibrating=self.calibrating,
                              mode=self.mode, target=tuple(c), frame=frame)

        self.buf.append(raw + self.offset)
        x, y = np.mean(self.buf, axis=0)
        self.last_xy = (float(np.clip(x, 0, self.w)), float(np.clip(y, 0, self.h)))
        if self.mode == "validate":
            target = self._val_target()
            self._val_step(now)
            return GazeSample(face=True, x=self.last_xy[0], y=self.last_xy[1], calibrating=self.calibrating,
                              mode="validate" if self.mode == "validate" else "run", target=target, frame=frame)
        return GazeSample(face=True, x=self.last_xy[0], y=self.last_xy[1], calibrating=False,
                          mode="run", frame=frame)

    def _val_target(self):
        fx, fy = VALIDATE_POINTS[self.val_order[min(self.val_idx, len(self.val_order) - 1)]]
        return (fx * self.w, fy * self.h)

    def _val_step(self, now):
        if self.val_t0 is None:
            self.val_t0 = now
        settle = VALIDATE_SETTLE_S + (VALIDATE_FIRST_EXTRA_S if self.val_idx == 0 else 0)
        age = now - self.val_t0
        if settle <= age < settle + VALIDATE_COLLECT_S:
            self.val_samples.append(self.last_xy)
        elif age >= settle + VALIDATE_COLLECT_S:
            if self.val_samples:
                m = np.median(self.val_samples, axis=0)
                t = np.array(self._val_target())
                self.val_result[self.val_order[self.val_idx]] = (m.tolist(), float(np.hypot(*(m - t))))
                S_ = np.array(self.val_samples)
                diag.log("CHECK", f"dot {self.val_idx + 1}/{len(self.val_order)}", target=(round(t[0]), round(t[1])),
                         measured=(round(m[0]), round(m[1])), dx=m[0] - t[0], dy=m[1] - t[1],
                         error_px=float(np.hypot(*(m - t))), samples=len(S_),
                         spread_px=float(np.hypot(*S_.std(axis=0))))
            else:
                diag.log("CHECK", f"dot {self.val_idx + 1}: no samples (face lost or eyes closed)")
            self.val_idx += 1
            self.val_t0, self.val_samples = None, []
            if self.val_idx >= len(self.val_order):
                # a dot far worse than the rest usually means you were not looking at it yet: show it once more
                errs = {i: e for i, (_, e) in self.val_result.items()}
                med = np.median(list(errs.values())) if errs else 0
                redo = [i for i, e in errs.items() if e > max(2.5 * med, 250)]
                if redo and not self.val_retried:
                    diag.log("CHECK", f"re-checking {len(redo)} dot(s) that looked like a missed look", dots=redo)
                    self.val_retried = True
                    self.val_order += redo
                    return
                self._report_validation()
                self.mode = "run"

    def _report_validation(self):
        idx = sorted(self.val_result)
        if not idx:
            self.validation = None
            diag.log("CHECK", "no data (face not found or eyes closed)")
            return
        P = np.array([self.val_result[i][0] for i in idx])
        T = np.array([[VALIDATE_POINTS[i][0] * self.w, VALIDATE_POINTS[i][1] * self.h] for i in idx])
        errs = [self.val_result[i][1] for i in idx]
        D = P - T
        shift = np.median(D, axis=0)                            # robust: one bad dot can't drag it
        after_shift = float(np.median(np.linalg.norm(D - shift, axis=1)))
        med = float(np.median(errs))
        outliers = [k for k, e in enumerate(errs) if e > max(2.5 * med, 250)]
        self.validation = {"mean": float(np.mean(errs)), "median": med, "worst": float(np.max(errs)),
                           "per_point": errs, "shift": shift.tolist(), "after_shift": after_shift,
                           "means": P.tolist(), "targets": T.tolist(), "outliers": outliers}
        diag.log("CHECK", f"RESULT typical error {med:.0f} px, average {np.mean(errs):.0f}, worst {np.max(errs):.0f}",
                 per_dot_center_first=[round(e) for e in errs], shift_px=shift, error_without_shift_px=after_shift)
        if diag.calib_head is not None and self.val_pose:
            d = np.mean(self.val_pose, axis=0) - diag.calib_head
            diag.log("CHECK", "head position during the check vs calibration",
                     drift_yaw_pitch_roll_deg=np.round(np.degrees(d[:3]), 1), drift_xyz_cm=np.round(d[3:], 1))
        if outliers:
            diag.log("CHECK", f"{len(outliers)} dot(s) still far off after a second look; left out of the shift and the f correction")
        reach = []
        if len(P) >= 4:
            for i, ax in enumerate(("horizontally", "vertically")):
                a = np.polyfit(T[:, i], P[:, i], 1)[0]          # measured spread / target spread
                self.validation[f"reach_{'x' if i == 0 else 'y'}_%"] = float(100 * a)
                if abs(a - 1) > 0.06:
                    reach.append(f"{'only reaches' if a < 1 else 'overshoots to'} {100 * a:.0f}% of the way to the edges {ax}")
        if reach:
            diag.log("CHECK", "the cursor " + " and ".join(reach) + "; press f to correct it",
                     reach_x_pct=self.validation.get("reach_x_%"), reach_y_pct=self.validation.get("reach_y_%"))
        dirs = []
        drift = float(np.hypot(*shift))
        if drift > max(20.0, 0.75 * after_shift):          # the shift is a big part of the error
            if abs(shift[0]) > 0.3 * drift:
                dirs.append(f"{abs(shift[0]):.0f} px too far {'right' if shift[0] > 0 else 'left'}")
            if abs(shift[1]) > 0.3 * drift:
                dirs.append(f"{abs(shift[1]):.0f} px too {'low' if shift[1] > 0 else 'high'}")
        if dirs:
            diag.log("CHECK", f"overall the cursor sits {' and '.join(dirs)} (drift, usually from the head moving)")
            diag.log("CHECK", f"without that shift the typical error would be about {after_shift:.0f} px; press f to correct it")
        elif not reach:
            diag.log("CHECK", "no overall drift or scale error; the remaining error is general accuracy (light, head stability, recalibration)")

    def close(self):
        self.cam.close()
        self.faces.close()

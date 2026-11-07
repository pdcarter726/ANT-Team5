"""Webcam gaze tracking built on EyeGestures v4.

Face landmarks come from EyeGestures' FaceFinder (MediaPipe). Calibration is a Python port of the
EyeGestures v4 engine's calibration: same eye features, same linear model, and a point only advances
once the live estimate has settled on it (within 10% of the screen width for 20+ frames). Porting it
lets us use more points (35), skip blink frames, and never recalibrate on our own.
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

MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
CALIB_PATH = Path(__file__).parent / "models" / "gaze_calibration.json"

# Identical to the EyeGestures v4 engine (engine/src/eyegestures.rs)
LEFT = [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 144, 145, 153, 246, 468]
RIGHT = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373, 374, 466, 473]
TMP_MAX = 40          # engine keeps the last 40 frames of the current calibration point
SMOOTH_FRAMES = 20    # engine averages the last 20 predictions
SETTLE_FRAMES = 20    # engine: frames near the target before a point is accepted
SETTLE_FRAC = 0.10    # engine: "near" = within 10% of the screen width

# The engine's 25 points (same order), then 10 more filling the gaps between them.
ENGINE_POINTS = [
    (0.25, 0.25), (0.5, 0.75), (1.0, 0.5), (0.75, 0.5), (0.0, 0.75),
    (0.5, 0.5), (1.0, 0.25), (0.75, 0.0), (0.25, 0.5), (0.5, 0.0),
    (0.0, 0.5), (1.0, 1.0), (0.75, 1.0), (0.25, 0.0), (1.0, 0.0),
    (0.0, 1.0), (0.25, 1.0), (0.75, 0.75), (0.5, 0.25), (0.0, 0.25),
    (1.0, 0.5), (0.75, 0.25), (0.5, 1.0), (0.25, 0.75), (0.0, 0.0),
]
EXTRA_POINTS = [
    (0.125, 0.125), (0.875, 0.875), (0.875, 0.125), (0.125, 0.875), (0.375, 0.375),
    (0.625, 0.625), (0.625, 0.375), (0.375, 0.625), (0.125, 0.5), (0.875, 0.5),
]
CALIBRATION_POINTS = ENGINE_POINTS + EXTRA_POINTS            # 35
ADD_POSE_POINTS = [(0.25, 0.25), (0.75, 0.75), (1.0, 0.5), (0.0, 0.5), (0.5, 0.5),
                   (0.75, 0.25), (0.25, 0.75), (0.5, 0.0), (0.5, 1.0)]   # 9

BLINK_SCORE = 0.5     # MediaPipe eyeBlink score (0-1) above which a frame counts as a blink
BLINK_HOLD_FRAMES = 3 # also skip this many frames after a blink (lids still settling)
BLINK_MAX_S = 0.6     # "blinks" longer than this are not blinks (e.g. looking down): use the frames

RECENTER_WINDOW = 30          # recenter: median of this many frames ...
RECENTER_AGREE_PX = 25        # ... accepted once two windows in a row agree within this
RECENTER_MAX_FRAMES = 150     # ... or after this many frames (median of everything)


def use_local_model():
    """Point EyeGestures at our local copy of the face model so it never tries to download
    (the ANT station network may have no internet). Returns True if the model is present."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        os.environ["EYEGESTURES_FACE_LANDMARKER_MODEL"] = str(MODEL_PATH)
        return True
    return False


@dataclass
class GazeSample:
    face: bool
    x: float = 0.0
    y: float = 0.0
    calibrating: bool = True
    mode: str = "calibrate"         # calibrate | addpose | recenter | run
    target: tuple = (0.0, 0.0)
    blink: bool = False
    frame: np.ndarray = None


class Camera:
    """Threaded webcam reader that always hands back the newest frame."""

    def __init__(self, index=0, width=1280, height=720):
        import cv2
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}. Try --camera 1.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.q = queue.Queue(maxsize=1)
        self.running = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            if self.q.full():
                try:
                    self.q.get_nowait()
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
    """Flags blink frames from MediaPipe's eyeBlinkLeft/Right blendshape scores (eye aspect ratio
    as a fallback). Blink frames are ignored everywhere: calibration, recenter and tracking."""

    def __init__(self):
        self.hold = 0
        self.closed_since = None

    @staticmethod
    def score(det, lm):
        try:
            cats = {c.category_name: c.score for c in det.face_blendshapes[0]}
            return (cats["eyeBlinkLeft"] + cats["eyeBlinkRight"]) / 2
        except Exception:
            # eye aspect ratio: lid gap / eye width, ~0.25-0.3 open, <0.1 closed
            ear = []
            for top, bot, a, b in ((159, 145, 33, 133), (386, 374, 362, 263)):
                ear.append(np.linalg.norm(lm[top] - lm[bot]) / (np.linalg.norm(lm[a] - lm[b]) + 1e-9))
            return float(np.clip(1.0 - np.mean(ear) / 0.25, 0, 1))

    def __call__(self, det, lm, now):
        closed = self.score(det, lm) > BLINK_SCORE
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


N_ENGINE = 66   # engine feature columns; the rest of a stored sample are the extra features below


class LinearGaze:
    """Same model as the engine (ordinary linear regression with a bias, one per axis) on the engine
    features, with one guard: the last 6 engine features describe head position and size. If the head
    barely moved during calibration, those columns carry almost no information but get enormous
    weights, so any later head shift throws the cursor far off. Such columns are left out until
    calibration data (e.g. from "add head position") shows real head movement."""

    name = "engine linear"
    HEAD_GROUPS = ((slice(-6, -2), 0.02),   # scale_x, scale_y, width, height: needs > 2% size change
                   (slice(-2, None), 0.01))  # head offset x, y: needs > 1% of the image

    def __init__(self, W=None, mask=None):
        self.W = W
        self.mask = None if mask is None else np.asarray(mask, dtype=bool)

    def fit(self, X, Y, head_guard=True):
        X = np.asarray(X)[:, :N_ENGINE]
        self.mask = np.ones(X.shape[1], dtype=bool)
        if head_guard:
            idx = np.arange(X.shape[1])
            for sl, min_range in self.HEAD_GROUPS:
                cols = idx[sl]
                if np.ptp(X[:, cols], axis=0).max() < min_range:
                    self.mask[cols] = False
        A = np.hstack([X[:, self.mask], np.ones((len(X), 1))])
        AtA = A.T @ A + 1e-9 * np.eye(A.shape[1])
        try:
            self.W = np.linalg.solve(AtA, A.T @ np.asarray(Y, dtype=float))
        except np.linalg.LinAlgError:
            self.W = np.linalg.lstsq(A, np.asarray(Y, dtype=float), rcond=None)[0]
        return self

    def predict(self, x):
        x = np.asarray(x)[:N_ENGINE]
        x = x if self.mask is None else x[self.mask]
        return np.append(x, 1.0) @ self.W

    def to_dict(self):
        return {"type": "engine", "W": self.W.tolist(), "mask": self.mask.tolist()}


# ---------- extra features: iris inside each eye + head pose ----------
EYES = ((33, 133, 468, 159, 145), (263, 362, 473, 386, 374))   # outer corner, inner corner, iris, top lid, bottom lid
# physical scales: a change of this size counts as "one unit" for the ridge penalty
EXTRA_SCALES = np.array([0.05] * 8 + [0.05, 0.05, 0.05, 1.0, 1.0, 1.0])   # 8 eye values, yaw/pitch/roll (rad), x/y/z (cm)


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
    """Ridge regression on the extra features (optionally with squared/cross terms of the eye values,
    since gaze to screen is not perfectly linear, especially near the edges). Features are measured in
    physical units, so anything that never changed during calibration (e.g. head pose when the head was
    still) gets almost no weight instead of an enormous one."""

    ENGINE_SCALES = np.array([0.01] * 60 + [0.02] * 4 + [0.01] * 2)

    def __init__(self, lam=1.0, quad=False, with_engine=False):
        self.lam, self.quad, self.with_engine = lam, quad, with_engine
        self.mu = self.W = None

    @property
    def name(self):
        base = "engine+eye+pose" if self.with_engine else "eye+pose"
        return f"{base} {'quadratic' if self.quad else 'linear'} (ridge {self.lam:g})"

    def _cols(self, X):
        X = np.atleast_2d(X)
        return X if self.with_engine else X[:, N_ENGINE:]

    @property
    def scales(self):
        return np.concatenate([self.ENGINE_SCALES, EXTRA_SCALES]) if self.with_engine else EXTRA_SCALES

    def _phi(self, F):
        Z = (np.atleast_2d(F) - self.mu) / self.scales
        if self.quad:
            e = Z[:, -14:-6]
            iu = np.triu_indices(8)
            Z = np.hstack([Z, (e[:, :, None] * e[:, None, :])[:, iu[0], iu[1]] / 4.0])
        return np.hstack([Z, np.ones((Z.shape[0], 1))])

    def fit(self, X, Y):
        F = self._cols(np.asarray(X))
        self.mu = F.mean(axis=0)
        A = self._phi(F)
        R = self.lam * np.eye(A.shape[1]); R[-1, -1] = 0
        self.W = np.linalg.solve(A.T @ A + R, A.T @ np.asarray(Y, dtype=float))
        return self

    def predict(self, x):
        return (self._phi(self._cols(np.asarray(x))) @ self.W)[0]

    def to_dict(self):
        return {"type": "ridge", "lam": self.lam, "quad": self.quad, "with_engine": self.with_engine,
                "mu": self.mu.tolist(), "W": self.W.tolist()}


def model_from_dict(d):
    if d.get("type", "engine") == "engine":
        return LinearGaze(np.array(d["W"]), d.get("mask"))
    m = RidgeGaze(d["lam"], d["quad"], d.get("with_engine", False))
    m.mu, m.W = np.array(d["mu"]), np.array(d["W"])
    return m


CANDIDATES = [lambda: LinearGaze()] + \
    [lambda l=l: RidgeGaze(l, quad=False) for l in (0.3, 3.0, 30.0)] + \
    [lambda l=l: RidgeGaze(l, quad=True) for l in (1.0, 10.0, 100.0)] + \
    [lambda l=l: RidgeGaze(l, quad=False, with_engine=True) for l in (0.1, 1.0, 10.0)]


def select_model(X, Y, G, log=print):
    """Try every candidate model, score each on calibration points it did NOT train on (leave one point
    out), keep the best. Also fits the edge stretch for it: linear regression on noisy features pulls
    predictions toward the center, so a per-axis stretch (screen = a * pred + b) measured on the held-out
    predictions undoes that. Returns (model, stretch, held-out error in px, report lines)."""
    X, Y, G = np.asarray(X), np.asarray(Y, dtype=float), np.asarray(G)
    pts = np.unique(G)
    T = np.array([Y[G == g][0] for g in pts])
    results = []
    for make in CANDIDATES:
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
            Ps = np.stack([stretch[0][0] * P[:, 0] + stretch[0][1], stretch[1][0] * P[:, 1] + stretch[1][1]], 1)
            err = float(np.mean(np.linalg.norm(Ps - T, axis=1)))
            results.append((err, make, stretch))
        except Exception as e:                      # a candidate failing must never break calibration
            log(f"  candidate skipped: {e}")
    results.sort(key=lambda r: r[0])
    lines = [f"  {r[1]().name:34s} held-out error {r[0]:6.0f} px" for r in results]
    err, make, stretch = results[0]
    return make().fit(X, Y), stretch, err, lines


class CalibrationSession:
    """Python port of the EyeGestures v4 engine calibration (engine/src/eyegestures.rs + calibrator.rs):
    each frame the live estimate (mean of the last 20 predictions) is compared with the dot; the dot
    only moves on once the estimate has stayed within 10% of the screen width for 20+ frames. The model
    is refit every frame on the last 40 frames of the current dot plus everything kept from earlier dots."""

    def __init__(self, points, w, h):
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

    @property
    def done(self):
        return self.idx >= len(self.points)

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

    def step(self, k, full):
        """k: engine features (drive the dot exactly like the engine), full: everything we store."""
        target = self.target
        raw = np.append(k, 1.0) @ self.W if self.W is not None else np.zeros(2)
        self.buf.append(raw)
        est = np.mean(self.buf, axis=0)
        self.estimate = (float(est[0]), float(est[1]))
        self.tmp.append((k, target, full))
        self._refit()
        near = np.hypot(est[0] - target[0], est[1] - target[1]) < SETTLE_FRAC * self.w
        if near and self.frames_on_point > SETTLE_FRAMES:
            X = np.array([np.append(kk, 1.0) for kk, _, _ in self.tmp])
            Y = np.array([t for _, t, _ in self.tmp], dtype=float)
            self.S_AtA = X.T @ X + (self.S_AtA if self.S_AtA is not None else 0)
            self.S_Aty = X.T @ Y + (self.S_Aty if self.S_Aty is not None else 0)
            self.stable_x += [ff for _, _, ff in self.tmp]
            self.stable_y += [t for _, t, _ in self.tmp]
            self.stable_g += [self.idx] * len(self.tmp)
            self.tmp.clear()
            self.buf.clear()
            self.frames_on_point = 0
            self.idx += 1
        elif near:
            self.frames_on_point += 1


class GazeTracker:
    """Modes:
      calibrate  full calibration (35 points), replaces the model
      addpose    9 more points at the current head position, ADDED to the existing data, then refit.
                 Teaches the model what head movement looks like.
      recenter   look at one center dot; shifts all predictions by the measured offset
      run        normal tracking
    Nothing switches mode on its own: only recalibrate(), add_pose() and recenter() do.
    """

    def __init__(self, screen_w, screen_h, camera=0):
        if not use_local_model():
            raise SystemExit("Face model missing. Run once with internet:  python -m gaze.setup_check")
        from eyeGestures.face import FaceFinder
        self.w, self.h = int(screen_w), int(screen_h)
        self.finder = FaceFinder(str(MODEL_PATH))
        self.keypoints = Keypoints()
        self.blinks = BlinkFilter()
        self.cam = Camera(camera)
        self.model = None
        self.err = 0.0
        self.data_x, self.data_y, self.data_g = [], [], []
        self.stretch = [[1.0, 0.0], [1.0, 0.0]]
        self.offset = np.zeros(2)
        self.buf = deque(maxlen=SMOOTH_FRAMES)
        self.last_xy = (self.w / 2, self.h / 2)
        self.recalibrate()

    # ---------- mode switches (only ever called by the user / the EEG app) ----------
    def recalibrate(self):
        self.data_x, self.data_y, self.data_g = [], [], []
        self.mode = "calibrate"
        self.session = CalibrationSession(CALIBRATION_POINTS, self.w, self.h)

    def add_pose(self):
        if self.model is None:
            return self.recalibrate()
        self.mode = "addpose"
        self.session = CalibrationSession(ADD_POSE_POINTS, self.w, self.h)

    def recenter(self):
        if self.model is None:
            return
        self.mode = "recenter"
        self.recenter_preds = []

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
        return 0, 0

    # ---------- model ----------
    def _finish_session(self):
        base = max(self.data_g, default=-1) + 1
        self.data_x += self.session.stable_x
        self.data_y += self.session.stable_y
        self.data_g += [base + g for g in self.session.stable_g]
        self.model, self.stretch, self.err, lines = select_model(self.data_x, self.data_y, self.data_g)
        print("Gaze models tried on your calibration (lower is better):")
        print("\n".join(lines))
        print(f"Using: {self.model.name}, held-out error about {self.err:.0f} px")
        self.offset = np.zeros(2)
        self.buf.clear()
        self.mode = "run"
        self.save()

    def save(self):
        CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
        CALIB_PATH.write_text(json.dumps({
            "model": self.model.to_dict(), "err": self.err, "stretch": self.stretch,
            "offset": self.offset.tolist(), "keypoints": self.keypoints.state(), "screen": [self.w, self.h],
            "data_x": np.asarray(self.data_x).round(6).tolist(),
            "data_y": np.asarray(self.data_y).tolist(), "data_g": [int(g) for g in self.data_g]}))

    def load_calibration(self):
        d = json.loads(CALIB_PATH.read_text())
        if d["screen"] != [self.w, self.h]:
            raise ValueError("saved calibration is for a different screen size")
        if "model" not in d:
            raise ValueError("saved calibration is from an older version, please recalibrate")
        self.model = model_from_dict(d["model"])
        self.err = d.get("err", 0.0)
        self.stretch = d.get("stretch", [[1.0, 0.0], [1.0, 0.0]])
        self.offset = np.array(d.get("offset", [0.0, 0.0]))
        self.data_x = [np.array(x) for x in d.get("data_x", [])]
        self.data_y = [tuple(y) for y in d.get("data_y", [])]
        self.data_g = d.get("data_g", [])
        self.keypoints.load(d["keypoints"])
        self.buf.clear()
        self.mode = "run"

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
        det = self.finder.find(frame)
        if det is None or not det.face_landmarks:
            return GazeSample(face=False, calibrating=self.calibrating, mode=self.mode, frame=frame)
        lm = np.array([(p.x, p.y) for p in det.face_landmarks[0]], dtype=np.float64)
        blink = self.blinks(det, lm, now)

        if self.mode in ("calibrate", "addpose"):
            sess = self.session
            if not blink:                                   # blink frames never reach calibration
                k, full = self._features(lm, frame, det)
                sess.step(k, full)
                if sess.done:
                    self._finish_session()
            x, y = sess.estimate
            tx, ty = sess.target
            return GazeSample(face=True, x=x, y=y, calibrating=self.calibrating, mode=self.mode,
                              target=(min(max(tx, 0), self.w), min(max(ty, 0), self.h)), blink=blink, frame=frame)

        if blink:                                            # hold the last position through blinks
            x, y = self.last_xy
            target = (self.w / 2, self.h / 2) if self.mode == "recenter" else (0.0, 0.0)
            return GazeSample(face=True, x=x, y=y, calibrating=self.calibrating, mode=self.mode,
                              target=target, blink=True, frame=frame)

        raw = self._predict(self._features(lm, frame, det)[1])
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
                self.offset = c - est
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
        return GazeSample(face=True, x=self.last_xy[0], y=self.last_xy[1], calibrating=False,
                          mode="run", frame=frame)

    def close(self):
        self.cam.close()

"""Real-time detector core (shared by train.py, realtime.py and replay).

  WindowScorer  raw EEG chunks -> causal preprocessing -> sliding windows -> features -> class probabilities
                (+ continuous head-direction estimates for the cursor)
  EventLogic    probability stream -> discrete events with probability threshold, persistence and refractory
  ThresholdModel  single-feature-rule baseline (sklearn-like) for comparison with LR / RF
  Profile       everything needed at run time, saved with joblib

The offline path (train.py) and the live path use the same WindowScorer arithmetic, so replay results
predict live behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata

from features import FeatureExtractor
from preprocess import StreamingPreprocessor

NULL_CLASSES = ("rest", "confounder")
HEAD_CLASSES = ("head_left", "head_right", "head_up", "head_down")


# ------------------------------------------------------------ baseline model
class ThresholdModel:
    """One-vs-rest, one-feature-per-class threshold rules (the 'threshold baseline').

    For each gesture class the single most discriminative feature (ROC AUC, either direction) and the
    best balanced-accuracy threshold are chosen. predict_proba turns the rule margins into pseudo
    scores (0..1, rows need not sum to 1) with the same interface as the sklearn models: the null class
    gets 1 - the strongest rule score, so a gesture wins the arg-max once its rule score exceeds 0.5.
    """

    def __init__(self, width: float = 0.25):
        self.width = width

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.sigma_ = X.std(axis=0) + 1e-9
        ranks = rankdata(X, axis=0)
        self.rules_ = {}
        for c in self.classes_:
            if c in NULL_CLASSES:
                continue
            pos = y == c
            n1, n0 = pos.sum(), (~pos).sum()
            if n1 == 0 or n0 == 0:
                continue
            auc = (ranks[pos].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)
            f = int(np.argmax(np.abs(auc - 0.5)))
            sign = 1.0 if auc[f] >= 0.5 else -1.0
            v = X[:, f] * sign
            cands = np.quantile(v, np.linspace(0.02, 0.98, 97))
            tpr = np.array([(v[pos] > t).mean() for t in cands])
            tnr = np.array([(v[~pos] <= t).mean() for t in cands])
            thr = float(cands[np.argmax(tpr + tnr)])
            self.rules_[c] = (f, sign, thr)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        cols = []
        scores = {}
        for c, (f, sign, thr) in self.rules_.items():
            z = (X[:, f] * sign - thr) / (self.width * self.sigma_[f])
            scores[c] = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        null = 1.0 - (np.max(np.stack(list(scores.values())), axis=0) if scores else 0.0)
        for c in self.classes_:
            if c in scores:
                cols.append(scores[c])
            elif c == "rest":
                cols.append(null)
            else:
                cols.append(np.zeros(len(X)))
        return np.stack(cols, axis=1)      # NOT normalised: a rule score of 0.9 means 'rule fires', whatever the others say

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ----------------------------------------------------------------- profile
@dataclass
class LogicParams:
    default_thr: float = 0.7
    default_persist: int = 3           # consecutive windows
    default_refractory_s: float = 1.0
    global_refractory_s: float = 0.5
    rearm_windows: int = 2             # windows without the class before it may fire again
    thr: dict = field(default_factory=dict)
    persist: dict = field(default_factory=dict)
    refractory_s: dict = field(default_factory=dict)

    def thr_for(self, c): return self.thr.get(c, self.default_thr)
    def persist_for(self, c): return self.persist.get(c, self.default_persist)
    def refr_for(self, c): return self.refractory_s.get(c, self.default_refractory_s)

    def copy(self):
        return LogicParams(self.default_thr, self.default_persist, self.default_refractory_s,
                           self.global_refractory_s, self.rearm_windows,
                           dict(self.thr), dict(self.persist), dict(self.refractory_s))


@dataclass
class Profile:
    kind: str
    classes: list                       # model class order (probabilities follow it)
    model: object
    cursor_models: dict                 # {"h": pipeline, "v": pipeline} -> estimate in [-1, 1]
    ch_names: list
    fs: float
    win_s: float
    step_s: float
    groups: dict
    line_hz: float = 60.0
    params: LogicParams = field(default_factory=LogicParams)
    meta: dict = field(default_factory=dict)

    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "Profile":
        import joblib
        return joblib.load(path)

    @property
    def gestures(self):
        return [c for c in self.classes if c not in NULL_CLASSES]


def make_inference_ready(model):
    """RandomForest predict with n_jobs=-1 costs ~100 ms of thread overhead per single window."""
    try:
        for step in getattr(model, "steps", []):
            if hasattr(step[1], "n_jobs"):
                step[1].n_jobs = 1
    except Exception:
        pass
    return model


# ------------------------------------------------------------ window scoring
@dataclass
class WindowResult:
    end: int                # exclusive end sample index (counted from the first sample pushed)
    t: float                # time of the window's last sample
    probs: np.ndarray       # class probabilities in profile.classes order
    cursor: tuple           # (h, v) raw direction estimates (+right, +down), rest ~0
    feats: np.ndarray | None = None


class WindowScorer:
    """Streaming: push(chunk) -> list[WindowResult]. Chunk size does not matter."""

    def __init__(self, profile: Profile, t0: float = 0.0, step_s: float | None = None, keep_feats=False):
        self.p = profile
        fs = profile.fs
        self.win_n = int(round(profile.win_s * fs))
        self.step_n = int(round((step_s or profile.step_s) * fs))
        self.t0 = t0
        self.pre = StreamingPreprocessor(fs, len(profile.ch_names), profile.line_hz)
        self.fe = FeatureExtractor(profile.ch_names, profile.groups, fs)
        self.buf = {k: np.zeros((0, len(profile.ch_names))) for k in self.pre.sos}
        self.n_seen = 0
        self.next_end = self.win_n
        self.keep_feats = keep_feats

    def push(self, x) -> list[WindowResult]:
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] == 0:
            return []
        B = self.pre.process(x)
        for k in self.buf:
            self.buf[k] = np.concatenate([self.buf[k], B[k]])
        self.n_seen += len(x)
        buf_start = self.n_seen - len(self.buf["eog"])
        ends, feats = [], []
        while self.next_end <= self.n_seen:
            i1 = self.next_end - buf_start
            feats.append(self.fe.window(self.buf, i1 - self.win_n, i1))
            ends.append(self.next_end)
            self.next_end += self.step_n
        for k in self.buf:
            self.buf[k] = self.buf[k][-self.win_n:]
        if not ends:
            return []
        F = np.stack(feats)
        return score_features(self.p, F, ends, self.t0, self.keep_feats)


def score_features(profile: Profile, F, ends, t0=0.0, keep_feats=False) -> list[WindowResult]:
    P = profile.model.predict_proba(F)
    h = profile.cursor_models["h"].predict(F) if profile.cursor_models else np.zeros(len(F))
    v = profile.cursor_models["v"].predict(F) if profile.cursor_models else np.zeros(len(F))
    fs = profile.fs
    return [WindowResult(int(e), t0 + (e - 1) / fs, P[i], (float(h[i]), float(v[i])),
                         F[i] if keep_feats else None) for i, e in enumerate(ends)]


def offline_window_ends(n_samples: int, win_n: int, step_n: int) -> np.ndarray:
    return np.arange(win_n, n_samples + 1, step_n)


def offline_features(B: dict, fe: FeatureExtractor, ends, win_n: int) -> np.ndarray:
    return np.stack([fe.window(B, int(e) - win_n, int(e)) for e in ends]) if len(ends) else np.zeros((0, len(fe.feature_names)))


# --------------------------------------------------------------- event logic
@dataclass
class Fired:
    t: float
    cls: str
    p: float


class EventLogic:
    """Probability stream -> discrete events.

    A class fires when it is the arg-max, its probability is >= its threshold, and that holds for
    `persist` consecutive windows. After firing, the same class needs `rearm_windows` windows without it
    (so a held gesture fires once) and `refractory_s` seconds; any event also blocks all classes for
    `global_refractory_s`. rest / confounder never fire.
    """

    def __init__(self, classes, params: LogicParams):
        self.classes = list(classes)
        self.P = params
        self.reset()

    def reset(self):
        self.cand = None
        self.streak = 0
        self.last = {}
        self.last_any = -1e9
        self.armed = {c: True for c in self.classes}
        self.miss = {c: 0 for c in self.classes}

    def step(self, t: float, probs) -> list[Fired]:
        i = int(np.argmax(probs))
        c, p = self.classes[i], float(probs[i])
        cand = c if (c not in NULL_CLASSES and p >= self.P.thr_for(c)) else None
        if cand is not None and cand == self.cand:
            self.streak += 1
        else:
            self.streak = 1 if cand is not None else 0
        self.cand = cand
        for k in self.classes:
            if not self.armed[k]:
                self.miss[k] = 0 if k == cand else self.miss[k] + 1
                if self.miss[k] >= self.P.rearm_windows:
                    self.armed[k] = True
        out = []
        if (cand is not None and self.streak >= self.P.persist_for(cand) and self.armed[cand]
                and t - self.last.get(cand, -1e9) >= self.P.refr_for(cand)
                and t - self.last_any >= self.P.global_refractory_s):
            self.last[cand] = self.last_any = t
            self.armed[cand] = False
            self.miss[cand] = 0
            out.append(Fired(t, cand, p))
        return out

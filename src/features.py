"""Window feature extraction (shared by feasibility, training and the real-time loop)."""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis

EPS = 1e-6


class FeatureExtractor:
    def __init__(self, ch_names: list[str], groups: dict[str, list[str]], fs: float,
                 duration_thr_uv: float = 30.0):
        self.names, self.fs, self.thr = list(ch_names), fs, duration_thr_uv
        idx = {c.lower(): i for i, c in enumerate(self.names)}   # names matched case-insensitively
        self.groups = {g: [idx[c.lower()] for c in cs if c.lower() in idx] for g, cs in groups.items()
                       if g not in ("left", "right")}
        self.groups = {g: v for g, v in self.groups.items() if v}
        left = [c for c in groups.get("left", [])]
        right = [c for c in groups.get("right", [])]
        # NOTE: config 'left' and 'right' lists must be in matching order (Fp1<->Fp2, F7<->F8, ...)
        self.pairs = [(idx[l.lower()], idx[r.lower()]) for l, r in zip(left, right) if l.lower() in idx and r.lower() in idx]
        self.front = self.groups.get("frontal") or list(range(min(3, len(self.names))))
        self.feature_names = self._make_names()

    def _make_names(self):
        n = []
        for pre in ("eog_ptp", "slow_mean", "emg_logp"):
            n += [f"{pre}:{c}" for c in self.names]
        for g in self.groups:
            n += [f"slow_slope:{g}", f"emg_logp:grp_{g}", f"eog_rms:grp_{g}"]
        for l, r in self.pairs:
            a, b = self.names[l], self.names[r]
            n += [f"asym_slow:{a}-{b}", f"asym_eogptp:{a}-{b}", f"asym_emg:{a}-{b}"]
        n += ["front_rms", "front_kurt", "front_linelen", "front_dur_frac", "front_peak"]
        return n

    def window(self, B: dict[str, np.ndarray], i0: int, i1: int) -> np.ndarray:
        eog, slow, emg = B["eog"][i0:i1], B["slow"][i0:i1], B["emg"][i0:i1]
        f_ptp = np.ptp(eog, axis=0)
        f_slow = slow.mean(axis=0)
        f_emg = np.log10(np.mean(emg ** 2, axis=0) + EPS)
        feats = [f_ptp, f_slow, f_emg]
        t = np.arange(i1 - i0) / self.fs
        tc = t - t.mean()
        for g, ix in self.groups.items():
            gm = slow[:, ix].mean(axis=1)
            slope = float(np.dot(tc, gm - gm.mean()) / np.dot(tc, tc))
            feats.append(np.array([slope, f_emg[ix].mean(), np.sqrt(np.mean(eog[:, ix] ** 2))]))
        for l, r in self.pairs:
            feats.append(np.array([f_slow[l] - f_slow[r], f_ptp[l] - f_ptp[r], f_emg[l] - f_emg[r]]))
        fm = eog[:, self.front].mean(axis=1)
        feats.append(np.array([
            np.sqrt(np.mean(fm ** 2)), float(kurtosis(fm)) if fm.std() > 1e-9 else 0.0,
            np.sum(np.abs(np.diff(fm))), float(np.mean(np.abs(fm) > self.thr)), np.max(np.abs(fm))]))
        return np.concatenate(feats)

    def batch(self, B, wins) -> np.ndarray:
        return np.stack([self.window(B, w.i0, w.i1) for w in wins]) if wins else np.zeros((0, len(self.feature_names)))

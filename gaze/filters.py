"""Smoothing for noisy webcam gaze points."""
import math
from collections import deque


class OneEuro:
    """One Euro filter (Casiez et al. 2012): heavy smoothing when the gaze is still,
    little lag when it moves fast. min_cutoff lowers jitter, beta lowers lag."""

    def __init__(self, min_cutoff=0.4, beta=0.003, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = self.dx_prev = self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = self.dx_prev = self.t_prev = None

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, 0.0, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class GazeSmoother:
    """Median of the last 3 raw points (kills single-frame spikes, e.g. blinks),
    then a One Euro filter per axis."""

    def __init__(self, min_cutoff=0.4, beta=0.003):
        self.buf = deque(maxlen=3)
        self.fx, self.fy = OneEuro(min_cutoff, beta), OneEuro(min_cutoff, beta)

    def reset(self):
        self.buf.clear(); self.fx.reset(); self.fy.reset()

    def __call__(self, x, y, t):
        self.buf.append((x, y))
        xs = sorted(p[0] for p in self.buf); ys = sorted(p[1] for p in self.buf)
        mx, my = xs[len(xs) // 2], ys[len(ys) // 2]
        return self.fx(mx, t), self.fy(my, t)

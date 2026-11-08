"""Dashboard pieces: the scrolling EEG strip with gesture markers, and the gesture confidence meters."""
import time

import numpy as np
import pyqtgraph as pg
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from . import theme

PREFERRED = ["Fp1", "Fp2", "F7", "F8", "T7", "T8", "C3", "Cz", "C4", "M1", "M2", "O1", "O2"]


class EEGStrip(pg.PlotWidget):
    """Scrolling stacked traces on graph paper. Gesture events are marked where they happened."""

    def __init__(self, seconds=6.0):
        super().__init__(background=theme.PAPER_DEEP)
        self.seconds = seconds
        self.setMenuEnabled(False)
        self.setMouseEnabled(False, False)
        self.hideButtons()
        pi = self.getPlotItem()
        pi.showGrid(x=True, y=False, alpha=0.12)
        for ax in ("left", "bottom"):
            a = pi.getAxis(ax)
            a.setPen(pg.mkPen(theme.GRID))
            a.setTextPen(pg.mkPen(theme.MUTED))
            a.setStyle(tickFont=QFont(theme.BODY, 8))
        pi.getAxis("bottom").setLabel("seconds", color=theme.MUTED)
        self.setXRange(-seconds, 0, padding=0)
        self.curves, self.idx, self.names = [], [], []
        self.stamps = []        # (time, label) ; drawn items are rebuilt each frame
        self.items = []
        self._sos_fs = None

    def set_channels(self, channels):
        low = [c.lower() for c in channels]
        pick = [channels[low.index(c.lower())] for c in PREFERRED if c.lower() in low] or list(channels[:12])
        self.idx = [channels.index(c) for c in pick]
        self.names = pick
        pi = self.getPlotItem()
        for c in self.curves:
            pi.removeItem(c)
        self.curves = [pi.plot(pen=pg.mkPen(theme.TRACE, width=1.1)) for _ in pick]
        n = len(pick)
        self.setYRange(-0.8, n - 0.2, padding=0)
        ax = pi.getAxis("left")
        ax.setTicks([[(n - 1 - i, name) for i, name in enumerate(pick)]])

    def stamp(self, label):
        self.stamps.append((time.time(), label))
        self.stamps = self.stamps[-12:]

    def _display_filter(self, fs):
        """For the picture only (the model gets raw data): 1-45 Hz band and a 60 Hz notch. The raw eego
        signal drifts and carries strong 60 Hz mains noise, which would hide everything else."""
        if self._sos_fs != fs:
            band = butter(4, [1.0, min(45.0, fs / 2 - 1)], btype="band", fs=fs, output="sos")
            b, a = iirnotch(60.0, 30.0, fs=fs)
            self._sos = np.vstack([band, tf2sos(b, a)])
            self._sos_fs = fs
        return self._sos

    def update_data(self, window, fs):
        if window is None or not self.idx:
            return
        window = sosfiltfilt(self._display_filter(fs), window[self.idx].astype(float), axis=1)
        x = (np.arange(window.shape[1]) - window.shape[1]) / fs
        step = max(1, window.shape[1] // 1500)          # keep drawing cheap
        n = len(self.idx)
        for i, curve in enumerate(self.curves):
            y = window[i, ::step]
            s = np.percentile(np.abs(y), 95) or 1.0
            curve.setData(x[::step], np.clip(y / (s * 2.2), -0.48, 0.48) + (n - 1 - i))
        pi = self.getPlotItem()
        for it in self.items:
            pi.removeItem(it)
        self.items = []
        now = time.time()
        for k, (t, label) in enumerate(self.stamps):
            xpos = t - now
            if xpos < -self.seconds:
                continue
            line = pg.InfiniteLine(pos=xpos, angle=90, pen=pg.mkPen(theme.ACCENT, width=1.5))
            line.setZValue(20)
            right_side = xpos > -self.seconds * 0.3          # near the live edge: write the note to the left
            txt = pg.TextItem(label, color=theme.ACCENT, anchor=(1, 0) if right_side else (0, 0),
                              fill=pg.mkBrush(theme.PAPER_DEEP))
            txt.setFont(QFont(theme.DISPLAY, 10, QFont.DemiBold))
            txt.setZValue(21)
            txt.setPos(xpos, n - 0.25 - (k % 3) * 0.9)     # stagger so close stamps don't overlap
            for it in (line, txt):
                pi.addItem(it)
                self.items.append(it)


class ConfidenceMeters(QWidget):
    """One row per gesture: name, a thin track filled in ink to the model's confidence, the firing threshold
    as a thin tick. A row highlights when its gesture fires."""

    def __init__(self, gestures, threshold=0.6):
        super().__init__()
        self.gestures = [g for g in gestures if g != "rest"]
        self.threshold = threshold
        self.probs = {}
        self.flash = {}
        self.setFixedHeight(28 * len(self.gestures) + 10)

    def set_gestures(self, gestures):
        self.gestures = [g for g in gestures if g != "rest"]
        self.setFixedHeight(28 * len(self.gestures) + 10)
        self.update()

    def set_probs(self, probs):
        self.probs = probs
        for g in probs:
            if g != "rest" and g not in self.gestures:
                self.gestures.append(g)
                self.setFixedHeight(28 * len(self.gestures) + 10)
        self.update()

    def fired(self, gesture):
        self.flash[gesture] = time.time()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        label_w = 150
        track_x, track_w = label_w + 10, max(60, w - label_w - 70)
        row_h = 28
        now = time.time()
        f_body = QFont(theme.BODY, 10)
        f_num = QFont(theme.BODY, 9)
        for i, g in enumerate(self.gestures):
            y = 6 + i * row_h
            v = float(self.probs.get(g, 0.0))
            hot = now - self.flash.get(g, 0) < 0.8
            ink = QColor(theme.ACCENT if hot else theme.INK)
            p.setFont(f_body)
            p.setPen(ink)
            p.drawText(QRectF(0, y, label_w, row_h - 6), Qt.AlignVCenter | Qt.AlignLeft, theme.nice(g))
            cy = y + (row_h - 6) / 2
            p.setPen(QPen(QColor(theme.GRID), 5, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(track_x, cy, track_x + track_w, cy)
            if v > 0.01:
                p.setPen(QPen(QColor(theme.ACCENT) if v >= self.threshold or hot else QColor(theme.TRACE), 5, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(track_x, cy, track_x + track_w * min(v, 1.0), cy)
            tx = track_x + track_w * self.threshold
            p.setPen(QPen(QColor(theme.MUTED), 1))
            p.drawLine(tx, cy - 7, tx, cy + 7)
            p.setFont(f_num)
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(track_x + track_w + 8, y, 50, row_h - 6), Qt.AlignVCenter | Qt.AlignLeft, f"{v:.2f}")
        p.end()

"""The four flickering squares, one per screen corner (top_left, top_right, bottom_left, bottom_right).

live mode:        transparent, always-on-top and click-through: only the squares are visible, everything else on the
                  screen stays usable. Also never takes focus.
calibration mode: opaque dark full-screen window with a message in the centre; Esc stops (signal `aborted`).

Luminance follows a sine at each square's frequency, computed from the clock (not from a frame counter), so the
frequency is right whatever the screen's refresh rate and a dropped frame does not shift the rhythm.
"""
import math
import time

from PySide6.QtCore import QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .core import DEFAULT_FREQS, POSITIONS

LOW, HIGH = 18, 255                      # darkest / brightest grey of a square


def square_rects(width: int, height: int, size: int | None = None, margin: int = 24) -> list[QRect]:
    """Rectangles for top_left, top_right, bottom_left, bottom_right inside a width x height area."""
    size = size or max(110, int(min(width, height) * 0.15))
    r, b = width - margin - size, height - margin - size
    return [QRect(margin, margin, size, size), QRect(r, margin, size, size),
            QRect(margin, b, size, size), QRect(r, b, size, size)]


def luminance(freq: float, t: float) -> float:
    """0..1 brightness of a square at time t (seconds)."""
    return 0.5 + 0.5 * math.sin(2 * math.pi * freq * t)


class FlickerOverlay(QWidget):
    aborted = Signal()

    def __init__(self, freqs=DEFAULT_FREQS, mode: str = "live", labels=None, screen=None):
        flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        if mode == "live":
            flags |= Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus
        super().__init__(None, flags)
        self.mode = mode
        self.freqs = [float(f) for f in freqs]
        self.labels = list(labels or [""] * 4)
        self.flicker_on = True
        self.highlight = None                 # index of the square to look at (calibration cue)
        self.message = ""
        self.sub = ""
        self.t0 = time.perf_counter()
        if mode == "live":
            self.setAttribute(Qt.WA_TranslucentBackground)
            self.setAttribute(Qt.WA_ShowWithoutActivating)
        else:
            self.setStyleSheet("background: #0d0f12;")
            self.setFocusPolicy(Qt.StrongFocus)
        scr = screen or QGuiApplication.primaryScreen()
        self.screen_obj = scr
        self.setGeometry(scr.geometry())
        self.rects = square_rects(self.width(), self.height())
        self.refresh_hz = float(scr.refreshRate() or 60.0)
        self.timer = QTimer(self, timeout=self._frame)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.setInterval(max(1, int(round(1000.0 / self.refresh_hz))))

    # ---------------------------------------------------------------- control
    def start(self):
        self.t0 = time.perf_counter()
        if self.mode == "live":
            self.show()
        else:
            self.showFullScreen()
            self.activateWindow()
            self.setFocus()
        self.timer.start()

    def stop(self):
        self.timer.stop()
        self.hide()

    def set_labels(self, labels):
        self.labels = list(labels)
        self.update()

    def set_state(self, flicker=None, highlight="keep", message=None, sub=None):
        if flicker is not None:
            self.flicker_on = flicker
        if highlight != "keep":
            self.highlight = highlight
        if message is not None:
            self.message = message
        if sub is not None:
            self.sub = sub
        self.update()

    def _frame(self):
        if self.flicker_on:
            for r in self.rects:
                self.update(r.adjusted(-2, -2, 2, 2))
        elif self.mode != "live":
            pass

    def keyPressEvent(self, e):
        if self.mode != "live" and e.key() == Qt.Key_Escape:
            self.aborted.emit()
        else:
            super().keyPressEvent(e)

    # ---------------------------------------------------------------- drawing
    def brightness(self, i: int, now: float | None = None) -> float:
        t = (time.perf_counter() if now is None else now) - self.t0
        return luminance(self.freqs[i], t) if self.flicker_on else 0.0

    def paintEvent(self, _):
        p = QPainter(self)
        now = time.perf_counter()
        if self.mode != "live":
            p.fillRect(self.rect(), QColor("#0d0f12"))
        small = QFont("Segoe UI", 11)
        for i, r in enumerate(self.rects):
            g = int(LOW + (HIGH - LOW) * self.brightness(i, now))
            p.fillRect(r, QColor(g, g, g))
            p.setPen(QPen(QColor(120, 120, 120), 1))
            p.drawRect(r)
            if self.highlight == i:
                p.setPen(QPen(QColor("#CC0000"), 6))
                p.drawRect(r.adjusted(-10, -10, 10, 10))
        p.setFont(small)
        p.setPen(QColor(150, 150, 150))
        for i, r in enumerate(self.rects):
            if not self.labels[i]:
                continue
            left, top = i % 2 == 0, i < 2
            box = QRect(r.left() + (r.width() + 12 if left else -212), r.top() + (0 if top else r.height() - 40), 200, 40)
            p.drawText(box, (Qt.AlignLeft if left else Qt.AlignRight) | (Qt.AlignTop if top else Qt.AlignBottom),
                       self.labels[i])
        if self.mode != "live" and (self.message or self.sub):
            p.setPen(QColor(235, 235, 235))
            p.setFont(QFont("Segoe UI", 26, QFont.DemiBold))
            area = QRectF(self.width() * 0.2, self.height() * 0.35, self.width() * 0.6, 100)
            p.drawText(area, Qt.AlignCenter | Qt.TextWordWrap, self.message)
            p.setPen(QColor(150, 155, 165))
            p.setFont(QFont("Segoe UI", 14))
            p.drawText(QRectF(self.width() * 0.2, self.height() * 0.35 + 110, self.width() * 0.6, 120),
                       Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap, self.sub)
        p.end()

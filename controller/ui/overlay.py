"""A small always-on-top strip at the top of the screen: what was just detected and what it did.
It never takes clicks or focus (clicks pass through to whatever is underneath)."""
import time

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QWidget

from . import theme


class Overlay(QWidget):
    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                         | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.resize(520, 64)
        scr = QGuiApplication.primaryScreen().availableGeometry()
        self.move(scr.center().x() - self.width() // 2, scr.top() + 14)
        self.gesture, self.action, self.t = "", "Listening", 0.0
        self.paused = False
        self.timer = QTimer(self, timeout=self.update, interval=50)
        self.timer.start()

    def show_event(self, gesture, action):
        self.gesture, self.action, self.t = gesture, action, time.time()
        self.update()

    def set_paused(self, paused):
        self.paused = paused
        self.update()

    def paintEvent(self, _):
        age = time.time() - self.t
        fresh = age < 1.6
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(1, 1, self.width() - 2, self.height() - 2)
        bg = QColor(theme.PAPER)
        bg.setAlpha(240 if fresh else 170)
        p.setBrush(bg)
        p.setPen(QPen(QColor(theme.RED if (fresh or self.paused) else theme.GRID), 2))
        p.drawRoundedRect(r, 6, 6)
        # red margin rule, like an annotation in a chart margin
        p.setPen(QPen(QColor(theme.RED), 4))
        p.drawLine(16, 14, 16, self.height() - 14)
        if self.paused:
            main, sub = "Controller paused", "gestures are ignored until you resume"
        elif fresh:
            main, sub = self.action, f"from {theme.nice(self.gesture).lower()}"
        else:
            main, sub = "Listening", "make a gesture"
        p.setPen(QColor(theme.INK if fresh or self.paused else theme.MUTED))
        p.setFont(QFont(theme.DISPLAY, 16, QFont.DemiBold))
        p.drawText(QRectF(32, 6, self.width() - 44, 32), Qt.AlignVCenter | Qt.AlignLeft, main)
        p.setPen(QColor(theme.MUTED))
        p.setFont(QFont(theme.BODY, 9))
        p.drawText(QRectF(32, 36, self.width() - 44, 20), Qt.AlignVCenter | Qt.AlignLeft, sub)
        p.end()

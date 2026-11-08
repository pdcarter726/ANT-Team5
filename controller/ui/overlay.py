"""A small always-on-top card at the top of the screen: what was just detected and what it did.
It never takes clicks or focus (clicks pass through to whatever is underneath)."""
import time

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication, QIcon, QPainter, QPen
from PySide6.QtWidgets import QWidget
from pathlib import Path

from . import theme


class Overlay(QWidget):
    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
                         | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.gesture, self.action, self.t = "", "Listening", 0.0
        self.paused = False
        self._icon_dir = Path(__file__).parent / "icons"
        self._place(166, 36)
        self.timer = QTimer(self, timeout=self.update, interval=50)
        self.timer.start()

    def _place(self, width, height):
        if self.width() != width or self.height() != height:
            self.resize(width, height)
        scr = QGuiApplication.primaryScreen().availableGeometry()
        self.move(max(scr.left(), scr.left() + 14), scr.top() + 14)

    def show_event(self, gesture, action):
        self.gesture, self.action, self.t = gesture, action, time.time()
        self.update()

    def set_paused(self, paused):
        self.paused = paused
        self.update()

    def paintEvent(self, _):
        fresh = time.time() - self.t < 1.6
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self.paused:
            main = "Controller paused"
            icon_name = "pause.svg"
        else:
            main = self.action if fresh else "Listening"
            icon_name = "cpu-chip.svg" if fresh else "signal.svg"

        if fresh and self.gesture:
            main = f"{self.action} - {theme.nice(self.gesture)}"

        font = QFont(theme.BODY, 9, QFont.DemiBold)
        p.setFont(font)
        text_width = p.fontMetrics().horizontalAdvance(main)
        width = min(max(166, 38 + text_width + 12), 420)
        self._place(width, 36)
        bg = QColor(theme.PAPER_DEEP)
        p.setBrush(bg)
        p.setPen(QPen(QColor(theme.GRID), 1))
        p.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1), 6, 6)
        if icon_name:
            icon = QIcon(str(self._icon_dir / icon_name)).pixmap(18, 18)
            p.drawPixmap(12, (self.height() - 18) // 2, icon)
        p.setPen(QColor(theme.INK))
        p.setFont(font)
        p.drawText(QRectF(38, 0, self.width() - 48, self.height()),
                   Qt.AlignVCenter | Qt.AlignLeft, main)
        p.end()

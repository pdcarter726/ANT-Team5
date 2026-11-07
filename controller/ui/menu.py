"""Quick bar: a row of big tiles centered 25% up from the bottom of the screen.
Point at a tile (head turns or the eye cursor) and clench to run it. The quick-bar gesture again
(eyebrow raise), Esc, or 15 s without a choice closes it. The bar never takes keyboard focus, so
Copy, Enter and so on go to the app you were using."""
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import theme

BAR_QSS = f"""
QWidget#barRoot {{ background: {theme.PAPER}; border: 2px solid {theme.INK}; border-radius: 8px; }}
QLabel#barHint {{ font-size: 10pt; color: {theme.MUTED}; }}
QPushButton#tile {{ font-family: "{theme.DISPLAY}"; font-size: 16pt; background: {theme.PAPER_DEEP};
    border: 2px solid {theme.GRID}; border-radius: 6px; padding: 6px; }}
QPushButton#tile:hover {{ border: 3px solid {theme.RED}; color: {theme.RED}; background: {theme.PAPER}; }}
"""

CENTER_FROM_BOTTOM = 0.25      # bar centered 25% of the screen height above the bottom edge
TIMEOUT_S = 15


class QuickMenu(QWidget):
    chosen = Signal(str, str)        # label, action

    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(BAR_QSS)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.root = QWidget(objectName="barRoot")
        outer.addWidget(self.root)
        lay = QVBoxLayout(self.root)
        lay.setContentsMargins(16, 10, 16, 16)
        lay.setSpacing(8)
        self.hint = QLabel("Point at an action and clench to run it. Raise your eyebrows again to close.",
                           objectName="barHint")
        lay.addWidget(self.hint)
        self.row = QHBoxLayout()
        self.row.setSpacing(12)
        lay.addLayout(self.row)
        self.timer = QTimer(self, singleShot=True, timeout=self.hide)

    def set_items(self, items):
        while self.row.count():
            w = self.row.takeAt(0).widget()
            if w:
                w.deleteLater()
        scr = QGuiApplication.primaryScreen().availableGeometry()
        n = max(1, len(items))
        bar_w = int(scr.width() * 0.92)
        tile_w = max(120, int((bar_w - 32 - 12 * (n - 1)) / n))
        tile_h = max(110, int(scr.height() * 0.13))
        for it in items:
            label = it["label"]
            if len(label) > 9 and " " in label and tile_w < 220:   # two lines instead of clipping
                cut = label.rfind(" ", 0, len(label) // 2 + 3)
                cut = cut if cut > 0 else label.find(" ")
                label = label[:cut] + "\n" + label[cut + 1:]
            b = QPushButton(label, objectName="tile")
            b.setFixedSize(tile_w, tile_h)
            b.setFocusPolicy(Qt.NoFocus)
            b.clicked.connect(lambda _=False, it=it: self._pick(it))
            self.row.addWidget(b)
        self.adjustSize()

    def _pick(self, it):
        self.hide()
        self.chosen.emit(it["label"], it["action"])

    def toggle(self, timeout_s=TIMEOUT_S):
        if self.isVisible():
            self.hide()
            return
        scr = QGuiApplication.primaryScreen().availableGeometry()
        self.adjustSize()
        cy = scr.bottom() - int(scr.height() * CENTER_FROM_BOTTOM)
        self.move(scr.center().x() - self.width() // 2, cy - self.height() // 2)
        self.show()
        self.raise_()
        self.timer.start(int(timeout_s * 1000))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.hide()

"""Quick bar: a row of big tiles centered 25% up from the bottom of the screen.
Point at a tile (head turns or the eye cursor) and clench to run it. The quick-bar gesture again
(eyebrow raise), Esc, or 15 s without a choice closes it. The bar never takes keyboard focus, so
Copy, Enter and so on go to the app you were using."""
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal, QSize
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from . import theme

BAR_QSS = f"""
QWidget#barRoot {{ background: {theme.PAPER}; border: 1px solid {theme.GRID}; border-radius: 6px; }}
QToolButton#tile {{ font-size: 14pt; font-weight: 600; background: {theme.PAPER}; color: {theme.INK};
    border: 1px solid {theme.GRID}; border-radius: 6px; padding: 12px; }}
QToolButton#tile:hover {{ border-color: {theme.ACCENT}; background: {theme.ACCENT_SOFT}; color: {theme.ACCENT}; }}
QToolButton#tile:pressed {{ background: #E4E8EE; }}
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
        self.rows = []
        self._content_layout = lay
        self.timer = QTimer(self, singleShot=True, timeout=self.hide)
        self._icon_dir = Path(__file__).parent / "icons"

    def set_items(self, items):
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())
        scr = QGuiApplication.primaryScreen().availableGeometry()
        ranked = sorted(enumerate(items), key=lambda entry: (self._popularity(entry[1]), entry[0]))
        primary_count = min(4, len(ranked))
        tiers = [ranked[:primary_count]]
        if len(ranked) > primary_count:
            tiers.append(ranked[primary_count:])
        for tier_index, tier in enumerate(tiers):
            row = QHBoxLayout()
            row.setSpacing(12)
            row.setContentsMargins(0, 0, 0, 0)
            tile_h = max(92, int(scr.height() * (0.13 if tier_index == 0 else 0.105)))
            tile_w = min(240, max(150, int((scr.width() * 0.88 - 12 * (len(tier) - 1)) / max(1, len(tier)))))
            if tier_index > 0:
                tile_w = min(190, max(130, tile_w - 30))
            for _, it in tier:
                label = it["label"]
                if len(label) > 11 and " " in label and tile_w < 220:
                    words = label.split()
                    cut = len(words) // 2
                    label = " ".join(words[:cut]) + "\n" + " ".join(words[cut:])
                b = QToolButton(objectName="tile")
                b.setText(label)
                b.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                b.setFixedSize(tile_w, tile_h)
                b.setFocusPolicy(Qt.NoFocus)
                icon = self._menu_icon(it.get("action", ""))
                b.setIcon(QIcon(str(self._icon_dir / icon)))
                b.setIconSize(QSize(28 if tier_index == 0 else 24, 28 if tier_index == 0 else 24))
                b.clicked.connect(lambda _=False, item=it: self._pick(item))
                row.addWidget(b)
            self._content_layout.addLayout(row)
        self.adjustSize()

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            elif item.layout():
                QuickMenu._clear_layout(item.layout())

    @staticmethod
    def _popularity(item):
        action = item.get("action", "")
        label = item.get("label", "").lower()
        ranks = {
            "hotkey:ctrl+c": 0,
            "hotkey:ctrl+v": 1,
            "hotkey:ctrl+z": 2,
            "scroll:10": 3,
            "scroll:-10": 4,
            "hotkey:alt+tab": 5,
            "key:enter": 6,
            "key:esc": 7,
            "key:space": 8,
            "controller:pause": 20,
        }
        return ranks.get(action, 10 if label else 30)

    @staticmethod
    def _menu_icon(action):
        kind = action.partition(":")[0]
        if kind in {"move", "mouse", "click"}:
            return "cursor-arrow-rays.svg"
        if kind in {"key", "hotkey", "type", "hold"}:
            return "command-line.svg"
        if kind == "eye":
            return "eye.svg"
        if kind == "menu":
            return "bars-3.svg"
        return "hand-raised.svg"

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

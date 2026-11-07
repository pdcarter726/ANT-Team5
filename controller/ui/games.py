"""Games tab: the arcade plays inside the dashboard (Qt WebEngine), not a separate browser."""
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .. import arcade
from ..log import log


def _webengine_view():
    from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
    from PySide6.QtWebEngineWidgets import QWebEngineView

    class Page(QWebEnginePage):
        def createWindow(self, _type):
            return self          # keep links inside this view

        def javaScriptConsoleMessage(self, level, message, line, source):
            if "error" in str(level).lower() or "warning" in str(level).lower():
                log("ARCADE", f"js: {message}", line=line)

    view = QWebEngineView()
    page = Page(view)
    page.setBackgroundColor(QColor("#0b0d12"))
    view.setPage(page)
    s = view.settings()
    s.setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
    s.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
    s.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
    view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    return view


class GamesPane(QWidget):
    """Picker strip + live game view. `opened` is (display name, url)."""
    opened = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current = None
        self.view = None
        self.picks = {}

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 12, 0, 0)
        v.setSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(6)
        for game, name in (("", "Arcade"), *arcade.GAMES.items()):
            b = QPushButton(name, objectName="game")
            b.setCheckable(True)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, g=game or None: self.open_game(g))
            self.picks[game] = b
            row.addWidget(b)
        row.addStretch(1)
        self.reload_btn = QPushButton("Reload")
        self.reload_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.reload_btn.clicked.connect(self.reload)
        row.addWidget(self.reload_btn)
        v.addLayout(row)

        self.arcade_mapping = QCheckBox("Use Arcade mapping (eyebrow raise = Space, jaw clench = click)")
        self.arcade_mapping.setChecked(True)
        v.addWidget(self.arcade_mapping)

        try:
            self.view = _webengine_view()
        except Exception as e:
            log("ERROR", f"Qt WebEngine unavailable ({e}); games cannot play inside the tab")
            fail = QLabel(
                "The games need Qt WebEngine to play in this tab.\n"
                "Install it in the same venv:  python -m pip install PySide6-WebEngine\n"
                f"({type(e).__name__}: {e})",
                objectName="bigSub", wordWrap=True)
            v.addWidget(fail, 1)
            return

        self.view.loadFinished.connect(self._on_loaded)
        self.view.urlChanged.connect(self._sync_picker)
        v.addWidget(self.view, 1)
        self.open_game(None)

    def open_game(self, game=None):
        if self.view is None:
            return None
        url = arcade.url(game)
        self.current = game
        self._mark(game)
        self.view.setUrl(QUrl(url))
        self.focus_view()
        name = arcade.GAMES.get(game, "arcade menu")
        log("ARCADE", f"opened in dashboard: {name}", url=url)
        self.opened.emit(name, url)
        return url

    def reload(self):
        if self.view:
            self.view.reload()
            self.focus_view()

    def focus_view(self):
        if self.view:
            self.view.setFocus(Qt.FocusReason.OtherFocusReason)

    def _on_loaded(self, ok):
        self.focus_view()
        if not ok:
            log("ERROR", "arcade page failed to load", url=self.view.url().toString())

    def _sync_picker(self, qurl):
        parts = [p for p in qurl.path().split("/") if p]
        game = parts[0] if parts and parts[0] in arcade.GAMES else None
        self.current = game
        self._mark(game)

    def _mark(self, game):
        key = game or ""
        for g, b in self.picks.items():
            on = g == key
            b.setChecked(on)
            b.setProperty("current", "true" if on else "false")
            b.style().unpolish(b)
            b.style().polish(b)

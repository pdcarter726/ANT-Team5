"""The controller window: live EEG with gesture stamps, confidence meters, last action and log,
plus a Mapping & test tab and a Log tab."""
import json
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal, QObject, QSize
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QMainWindow, QPlainTextEdit, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                               QTabWidget, QVBoxLayout, QWidget, QFileDialog, QAbstractItemView, QListView, QSizePolicy)

from . import theme
from .widgets import ConfidenceMeters, EEGStrip

ACTION_PRESETS = ["move:up", "move:down", "move:left", "move:right", "mouse:left", "mouse:right", "click:left",
                  "click:right", "click:double", "menu", "key:space", "key:enter", "key:esc", "key:tab",
                  "scroll:5", "scroll:-5", "hotkey:alt+left", "hotkey:alt+right", "hotkey:ctrl+c", "hotkey:ctrl+v",
                  "hotkey:alt+tab", "hold:w", "hold:a", "hold:s", "hold:d", "eye:recenter", "eye:toggle",
                  "controller:pause", "none"]


def clean_combo(combo):
    view = QListView(combo)
    view.setFrameShape(QFrame.NoFrame)
    view.setUniformItemSizes(True)
    combo.setView(view)
    return combo


class Bridge(QObject):
    """Engine thread -> UI thread."""
    probs = Signal(dict)
    event = Signal(str, str, float, str)
    menu = Signal()
    logline = Signal(str)
    gaze_status = Signal(object)


class StatusItem(QWidget):
    def __init__(self, key, icon, compact=False):
        super().__init__()
        self.setObjectName("statusCard")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 5, 8, 5)
        lay.setSpacing(0)
        self.icon = QLabel()
        self.icon.setObjectName("statusIcon")
        self.icon.setPixmap(QIcon(str(icon)).pixmap(16, 16))
        self.icon.setFixedSize(18, 18)
        lay.addWidget(self.icon)
        if compact:
            lay.setContentsMargins(7, 7, 7, 7)
            self.setFixedSize(34, 34)
            self.setToolTip(key)
            return
        lay.addSpacing(6)
        text = QVBoxLayout()
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(0)
        text.addWidget(QLabel(key, objectName="statusKey"))
        self.v = QLabel("...", objectName="statusVal")
        self.v.setMinimumWidth(0)
        self.v.setTextFormat(Qt.PlainText)
        text.addWidget(self.v)
        lay.addLayout(text, 1)
        self.setFixedSize(118, 34)

    def set(self, text, color=None, listening=False):
        if not hasattr(self, "v"):
            self.setToolTip(text)
            return
        metrics = self.v.fontMetrics()
        self.v.setText(metrics.elidedText(text, Qt.ElideRight, 78))
        self.v.setToolTip(text)
        self.v.setStyleSheet(f"color: {color};" if color and not listening else "")
        self.set_listening(listening)

    def set_listening(self, listening):
        self.setProperty("listening", "true" if listening else "false")
        self.style().unpolish(self)
        self.style().polish(self)


class Dashboard(QMainWindow):
    test_gesture = Signal(str, float)      # gesture, seconds held
    eeg_selected = Signal(object)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx                      # app context: source, model, detector, actions, engine, gaze, mapping info
        self.setWindowTitle("Universal Computer Controller")
        screen = QGuiApplication.primaryScreen().availableGeometry()
        width = min(1320, int(screen.width() * 0.82))
        height = min(840, int(screen.height() * 0.78))
        width = min(width, screen.width() - 32)
        height = min(height, screen.height() - 32)
        self.setMinimumSize(min(width, 800), min(height, 600))
        self.resize(width, height)
        self.move(screen.center().x() - width // 2, screen.center().y() - height // 2)
        self.setStyleSheet(theme.QSS)
        root = QWidget(objectName="root")
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(22, 16, 22, 16)
        v.setSpacing(10)

        # ---- header: title, status, pause ----
        head = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(0)
        title_box.addWidget(QLabel("Universal Computer Controller", objectName="title"))
        head.addLayout(title_box)
        head.addSpacing(18)
        icons = Path(__file__).parent / "icons"
        self.s_eeg = StatusItem("EEG", icons / "signal.svg")
        self.s_model = StatusItem("Gesture model", icons / "cpu-chip.svg")
        self.s_eye = StatusItem("Eye cursor", icons / "eye.svg", compact=True)
        self.s_ctl = StatusItem("Controller", icons / "computer-desktop.svg")
        for s in (self.s_eeg, self.s_model, self.s_eye, self.s_ctl):
            head.addWidget(s)
        if ctx["source"].name == "not connected":
            head.addSpacing(8)
            self.source_switch = QFrame(objectName="sourceSwitch")
            switch_layout = QHBoxLayout(self.source_switch)
            switch_layout.setContentsMargins(1, 1, 1, 1)
            switch_layout.setSpacing(1)
            self.source_switch.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            self.source_switch.setFixedHeight(34)
            self.lsl_btn = QPushButton("LSL stream", objectName="sourceOption")
            self.lsl_btn.setCheckable(True)
            self.lsl_btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            self.lsl_btn.setFixedHeight(30)
            self.lsl_btn.setToolTip("Search for and connect to an LSL EEG stream")
            self.lsl_btn.clicked.connect(lambda checked=False: self._select_eeg_source("lsl"))
            switch_layout.addWidget(self.lsl_btn)
            self.sim_btn = QPushButton("Simulated", objectName="sourceOption")
            self.sim_btn.setCheckable(True)
            self.sim_btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            self.sim_btn.setFixedHeight(30)
            self.sim_btn.setToolTip("Use synthetic EEG to test the interface")
            self.sim_btn.clicked.connect(lambda checked=False: self._select_eeg_source("sim"))
            switch_layout.addWidget(self.sim_btn)
            self.source_group = QButtonGroup(self.source_switch)
            self.source_group.setExclusive(True)
            self.source_group.addButton(self.lsl_btn)
            self.source_group.addButton(self.sim_btn)
            self.lsl_btn.setChecked(True)
            head.addWidget(self.source_switch)
        head.addSpacing(8)
        head.addStretch(1)
        # eye cursor on/off, right in the header next to pause
        self.eye_btn = QPushButton(objectName="eye")
        self.eye_btn.setToolTip("Start or stop the webcam eye cursor. Ctrl+Alt+E turns it off from any window.")
        self.eye_btn.setIconSize(QSize(18, 18))
        self.eye_btn.clicked.connect(self.toggle_eye)
        head.addWidget(self.eye_btn)
        head.addSpacing(8)
        self._eye_label()
        self.pause_btn = QPushButton(objectName="icon")
        self.pause_btn.setToolTip("Stop all gesture actions (Ctrl+Alt+G from any window)")
        self.pause_btn.setIconSize(QSize(18, 18))
        self.pause_btn.clicked.connect(self.toggle_pause)
        head.addWidget(self.pause_btn)
        v.addLayout(head)

        from PySide6.QtWidgets import QStackedWidget
        from .session import SessionPage
        self.pages = QStackedWidget()
        self.page_row = QHBoxLayout()                  # pages + (in simulated mode) the injection panel
        self.page_row.addWidget(self.pages, 1)
        v.addLayout(self.page_row, 1)
        self.sim_panel = None
        if hasattr(ctx["source"], "inject"):
            self.show_sim_panel(ctx["source"])
        self.session = SessionPage()
        self.pages.addWidget(self.session)
        tabs = QTabWidget()
        self.pages.addWidget(tabs)
        self.pages.setCurrentWidget(tabs if ctx.get("listening", True) else self.session)
        tabs.addTab(self._live_tab(), "Live")
        tabs.addTab(self._mapping_tab(), "Mapping and test")
        tabs.addTab(self._log_tab(), "Log")

        for lbl in self.findChildren(QLabel):          # labels keep their height; tables and plots take the rest
            if not lbl.wordWrap():
                lbl.setSizePolicy(lbl.sizePolicy().horizontalPolicy(), QSizePolicy.Fixed)
        self.eeg.set_channels(ctx["source"].channels)
        self.timer = QTimer(self, timeout=self.tick, interval=50)
        self.timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, "_centered_once", False):
            return
        self._centered_once = True
        QTimer.singleShot(0, self._center_window)

    def _center_window(self):
        screen = self.screen().availableGeometry()
        frame = self.frameGeometry()
        frame.moveCenter(screen.center())
        self.move(frame.topLeft())

    # ---------------- tabs ----------------
    def _live_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 12, 0, 0)
        self.eeg = EEGStrip()
        self.eeg.setMinimumHeight(220)
        v.addWidget(self.eeg, 3)
        row = QHBoxLayout()
        row.setSpacing(36)
        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(QLabel("Gesture confidence", objectName="section"))
        self.meters = ConfidenceMeters(self._gesture_names(), self.ctx["detector"].threshold)
        self.meters.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        left.addWidget(self.meters)
        left.addStretch(1)
        row.addLayout(left, 5)
        right = QVBoxLayout()
        right.setSpacing(6)
        right.addWidget(QLabel("Last action", objectName="section"))
        last = QHBoxLayout()
        self.big = QLabel("None yet", objectName="lastAction")
        self.big_sub = QLabel("", objectName="bigSub")
        last.addWidget(self.big)
        last.addWidget(self.big_sub, 1, Qt.AlignBottom)
        right.addLayout(last)
        right.addSpacing(8)
        right.addWidget(QLabel("Recent events", objectName="section"))
        self.events = QTableWidget(0, 3)
        self.events.setHorizontalHeaderLabels(["Time", "Gesture", "Action"])
        self.events.verticalHeader().setVisible(False)
        self.events.setShowGrid(False)
        self.events.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.events.setSelectionMode(QAbstractItemView.NoSelection)
        self.events.setFocusPolicy(Qt.NoFocus)
        hh = self.events.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.Stretch)
        hh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        right.addWidget(self.events, 1)
        row.addLayout(right, 4)
        v.addLayout(row, 2)
        return w

    def _mapping_tab(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 14, 0, 0)
        h.setSpacing(36)
        # mapping table
        left = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(QLabel("Mapping", objectName="section"))
        self.preset = clean_combo(QComboBox())
        for p in sorted(self.ctx["mapping_dir"].glob("*.json")):
            self.preset.addItem(json.loads(p.read_text()).get("name", p.stem), str(p))
        self.preset.setCurrentIndex(max(0, self.preset.findData(str(self.ctx["mapping_path"]))))
        self.preset.currentIndexChanged.connect(self.load_preset)
        top.addWidget(self.preset, 1)
        left.addLayout(top)
        left.addSpacing(6)
        self.table = QTableWidget(0, 2, objectName="mapTable")
        self.table.setHorizontalHeaderLabels(["Gesture", "Action"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 180)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        left.addWidget(self.table, 1)
        btns = QHBoxLayout()
        b_save = QPushButton("Save mapping")
        b_save.clicked.connect(self.save_mapping)
        b_saveas = QPushButton("Save as new mapping...")
        b_saveas.clicked.connect(lambda: self.save_mapping(as_new=True))
        self.save_note = QLabel("", objectName="bigSub")
        btns.addWidget(b_save)
        btns.addWidget(b_saveas)
        btns.addWidget(self.save_note, 1)
        left.addLayout(btns)
        h.addLayout(left, 3)
        # test + settings
        right_w = QWidget()
        right = QVBoxLayout(right_w)
        right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(QLabel("Try a gesture", objectName="section"))
        grid = QGridLayout()
        grid.setSpacing(8)
        for i, g in enumerate(self._gesture_names()):
            b = QPushButton(theme.nice(g), objectName="test")
            b.clicked.connect(lambda _=False, g=g: self.test_gesture.emit(g, 2.0 if self.hold_box.isChecked() else 0.35))
            grid.addWidget(b, i // 2, i % 2)
        right.addLayout(grid)
        self.hold_box = QCheckBox("Hold for 2 s")
        right.addWidget(self.hold_box)
        right.addSpacing(16)
        right.addWidget(QLabel("Detection", objectName="section"))
        det = self.ctx["detector"]
        form = QGridLayout()
        self.sp_thr = QDoubleSpinBox(minimum=0.05, maximum=0.99, singleStep=0.05, value=det.threshold)
        self.sp_k = QSpinBox(minimum=1, maximum=10, value=det.k)
        self.sp_n = QSpinBox(minimum=1, maximum=10, value=det.n)
        self.sp_ref = QDoubleSpinBox(minimum=0.0, maximum=5.0, singleStep=0.1, value=det.refractory_s)
        rows = [("Confidence needed", self.sp_thr), ("Windows in agreement", self.sp_k),
                ("...out of the last", self.sp_n), ("Wait before repeating (s)", self.sp_ref)]
        for r, (lab, wdg) in enumerate(rows):
            form.addWidget(QLabel(lab), r, 0)
            form.addWidget(wdg, r, 1)
            wdg.valueChanged.connect(self.apply_detection)
        right.addLayout(form)
        right.addSpacing(16)
        right.addWidget(QLabel("Eye cursor", objectName="section"))
        eye = QHBoxLayout()
        for text, act in (("Recenter", "eye:recenter"), ("Accuracy check", "eye:check"),
                          ("Apply correction", "eye:fix"), ("Pause tracking", "eye:toggle")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, a=act: self.run_action(a))
            eye.addWidget(b)
        right.addLayout(eye)
        h.addWidget(right_w, 2, Qt.AlignTop)
        self.fill_table()
        return w

    def _log_tab(self):
        self.logview = QPlainTextEdit(readOnly=True)
        self.logview.setMaximumBlockCount(3000)
        if self.ctx.get("log_file"):
            self.logview.appendPlainText(f"Log file: {self.ctx['log_file']}")
        return self.logview

    # ---------------- mapping ----------------
    def _gesture_names(self):
        from ..model_api import GESTURES
        names = [g for g in GESTURES if g != "rest"]
        for g in self.ctx["mapping"].get("gestures", {}):
            if g not in names:
                names.append(g)
        return names

    def fill_table(self):
        gestures = self.ctx["mapping"].get("gestures", {})
        names = self._gesture_names()
        self.table.setRowCount(len(names))
        for r, g in enumerate(names):
            item = QTableWidgetItem(theme.nice(g))
            item.setFlags(Qt.ItemIsEnabled)
            self.table.setItem(r, 0, item)
            self.table.setItem(r, 1, QTableWidgetItem())       # gives the row its divider
            cur = gestures.get(g, "none")
            combo = clean_combo(QComboBox())
            codes = list(dict.fromkeys(ACTION_PRESETS + [cur]))
            for c in codes:
                combo.addItem(theme.describe(c), c)
            combo.setCurrentIndex(codes.index(cur))
            combo.setToolTip(cur)
            combo.currentIndexChanged.connect(lambda i, g=g, combo=combo: self.edit_action(g, combo))
            self.table.setCellWidget(r, 1, combo)
            self.table.setRowHeight(r, 42)

    def edit_action(self, gesture, combo):
        code = combo.currentData()
        self.ctx["mapping"].setdefault("gestures", {})[gesture] = code
        combo.setToolTip(code)
        self.save_note.setText("Unsaved changes")

    def load_preset(self, i):
        path = Path(self.preset.itemData(i))
        mapping = json.loads(path.read_text())
        self.ctx["mapping"].clear()
        self.ctx["mapping"].update(mapping)
        self.ctx["mapping_path"] = path
        self.ctx["actions"].set_mapping(self.ctx["mapping"])
        self.ctx["menu"].set_items(self.ctx["mapping"].get("menu", []))
        self.fill_table()
        self.save_note.setText("")

    def save_mapping(self, as_new=False):
        path = self.ctx["mapping_path"]
        if as_new:
            fn, _ = QFileDialog.getSaveFileName(self, "Save mapping as", str(self.ctx["mapping_dir"]), "Mapping (*.json)")
            if not fn:
                return
            path = Path(fn)
            self.ctx["mapping"]["name"] = path.stem.replace("_", " ").capitalize()
        path.write_text(json.dumps(self.ctx["mapping"], indent=2))
        self.ctx["mapping_path"] = path
        self.save_note.setText(f"Saved to {path.name}")

    def sync_detection(self):
        """Show the detector's current settings (e.g. the team model's tuned values) without re-applying them."""
        d = self.ctx["detector"]
        for sp, v in ((self.sp_thr, d.threshold), (self.sp_k, d.k), (self.sp_n, d.n), (self.sp_ref, d.refractory_s)):
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)
        self.meters.threshold = d.threshold

    def apply_detection(self):
        d = self.ctx["detector"]
        if abs(self.sp_thr.value() - d.threshold) > 1e-9:
            d.thr = {}                     # a hand-set confidence applies to every gesture
        d.threshold, d.k, d.n, d.refractory_s = (self.sp_thr.value(), self.sp_k.value(),
                                                 max(self.sp_n.value(), self.sp_k.value()), self.sp_ref.value())
        from collections import deque
        d.history = deque(d.history, maxlen=d.n)
        self.meters.threshold = d.threshold

    # ---------------- live updates ----------------
    def on_probs(self, probs):
        self.meters.set_probs(probs)

    def on_event(self, kind, gesture, prob, desc):
        if kind == "release":
            # mouse gestures act on release (click) or end a drag; show that in the log
            if desc and ("click" in desc or "released" in desc):
                self._log_row(gesture, desc[0].upper() + desc[1:])
            return
        words = theme.describe(self.ctx["mapping"].get("gestures", {}).get(gesture, "none"))
        failed = desc and desc.startswith("FAILED")
        self.big.setText(words if not failed else f"{words} failed")
        self.big_sub.setText(f"{theme.nice(gesture)}, {prob:.2f}")
        self.eeg.stamp(theme.nice(gesture))
        self.meters.fired(gesture)
        self._log_row(gesture, words)

    def _log_row(self, gesture, words):
        self.events.insertRow(0)
        for c, text in enumerate((time.strftime("%H:%M:%S"), theme.nice(gesture), words)):
            self.events.setItem(0, c, QTableWidgetItem(text))
        self.events.setRowHeight(0, 28)
        while self.events.rowCount() > 200:
            self.events.removeRow(self.events.rowCount() - 1)

    def on_log(self, line):
        self.logview.appendPlainText(line)

    def toggle_pause(self):
        a = self.ctx["actions"]
        a.paused = not a.paused
        self.refresh_pause()

    def refresh_pause(self):
        paused = self.ctx["actions"].paused
        icon_path = Path(__file__).parent / "icons" / ("play-light.svg" if paused else "pause.svg")
        self.pause_btn.setIcon(QIcon(str(icon_path)))
        self.pause_btn.setToolTip("Resume gestures (Ctrl+Alt+G)" if paused else "Pause gestures (Ctrl+Alt+G)")
        self.pause_btn.setProperty("paused", "true" if paused else "false")
        self.pause_btn.style().unpolish(self.pause_btn)
        self.pause_btn.style().polish(self.pause_btn)
        self.ctx["overlay"].set_paused(paused)

    def run_action(self, action):
        desc = self.ctx["actions"].run(action)
        self.big.setText(theme.describe(action))
        self.big_sub.setText("Dashboard")

    @staticmethod
    def _gaze_profiles():
        from pathlib import Path
        folder = Path(__file__).resolve().parents[2] / "gaze" / "profiles"
        return sorted(p.stem for p in folder.glob("*.json"))

    def _eye_label(self):
        on = bool(self.ctx.get("gaze"))
        icon = "eye.svg" if not on else "eye-light.svg"
        icon_path = Path(__file__).parent / "icons" / icon
        self.eye_btn.setIcon(QIcon(str(icon_path)))
        self.eye_btn.setToolTip("Stop eye cursor (Ctrl+Alt+E)" if on else "Start eye cursor")
        self.eye_btn.setProperty("on", "true" if on else "false")
        self.eye_btn.style().unpolish(self.eye_btn)
        self.eye_btn.style().polish(self.eye_btn)

    def eye_off(self):
        """Emergency stop for the eye cursor (Ctrl+Alt+E)."""
        if self.ctx.get("gaze"):
            self.toggle_eye()

    def toggle_eye(self):
        from ..gaze_link import GazeLink
        from ..log import log
        g = self.ctx.get("gaze")
        if g:
            if g.proc is None:
                g.pause()            # an eye cursor we did not start: pause it instead of killing it
            g.stop()
            self.ctx["gaze"] = None
            log("GAZE", "eye cursor turned off from the dashboard")
        else:
            prof = self.ctx.get("gaze_profile")
            g = GazeLink(prof)
            self.ctx["gaze"], self.ctx["gaze_profile"] = g, prof
        self.ctx["actions"].gaze = self.ctx["gaze"]
        self._eye_label()
        self.on_gaze_status(None)

    def on_gaze_status(self, st):
        if not self.ctx.get("gaze"):
            self.s_eye.set("off", theme.MUTED, listening=not self.ctx["actions"].paused)
        elif st is None:
            self.s_eye.set("starting / not responding", theme.RED, listening=False)
        elif st.get("calibrating"):
            self.s_eye.set("calibrating", theme.TRACE, listening=False)
        else:
            state = st.get("state", "?")
            self.s_eye.set(("no face" if not st.get("face") else state) +
                           (f", {self.ctx['gaze_profile']}" if self.ctx.get("gaze_profile") else ""),
                           theme.RED if not st.get("face") else (theme.GOOD if state == "tracking" else theme.TRACE),
                           listening=False)

    def show_live(self):
        self.pages.setCurrentIndex(1)

    def _select_eeg_source(self, kind):
        from ..eeg_source import LSLSource
        from ..log import log
        self.lsl_btn.setEnabled(False)
        self.sim_btn.setEnabled(False)
        self.lsl_btn.setChecked(kind == "lsl")
        self.sim_btn.setChecked(kind == "sim")
        if kind == "sim":                     # loading recordings takes seconds: do it off the UI thread
            import threading
            self.s_eeg.set("loading recordings...", theme.MUTED)
            box = {}
            threading.Thread(target=lambda: box.update(src=self._make_sim_source()), daemon=True).start()
            timer = QTimer(self, interval=100)

            def check():
                if "src" in box:
                    timer.stop()
                    self._use_source(box["src"])
            timer.timeout.connect(check)
            timer.start()
            return
        try:
            source = LSLSource(self.ctx.get("stream_name"), self.ctx.get("stream_type", "EEG"))
        except Exception as exc:
            log("ERROR", f"could not connect EEG source ({exc})")
            self.s_eeg.set("connection failed", theme.RED)
            self.lsl_btn.setEnabled(True)
            self.sim_btn.setEnabled(True)
            return
        self._use_source(source)

    @staticmethod
    def _make_sim_source():
        """Recorded EEG injection when available; synthetic EEG otherwise."""
        from ..eeg_source import SimSource
        from ..log import log
        try:
            from ..sim_inject import InjectSource
            return InjectSource()
        except Exception as exc:
            log("ERROR", f"recorded simulation unavailable ({exc}); using synthetic EEG")
            return SimSource()

    def _use_source(self, source):
        from ..log import log
        old = self.ctx["source"]
        if hasattr(old, "close"):
            old.close()
        self.ctx["source"] = source
        self.ctx["engine"].source = source
        self.eeg.set_channels(source.channels)
        self.lsl_btn.setVisible(False)
        self.sim_btn.setVisible(False)
        self.source_switch.setVisible(False)
        if hasattr(source, "inject"):
            self.show_sim_panel(source)
        self.eeg_selected.emit(source)
        log("START", "eeg source selected", source=source.name)

    def show_sim_panel(self, source):
        """Unlock the Simulate EEG side panel for an injecting source (controller/sim_inject.py)."""
        from .sim_panel import SimPanel
        if self.sim_panel is not None:
            self.sim_panel.setParent(None)
        self.sim_panel = SimPanel(source)
        self.page_row.addWidget(self.sim_panel)

    def tick(self):
        src, eng = self.ctx["source"], self.ctx["engine"]
        self.eeg.update_data(src.window(self.eeg.seconds), src.fs)
        live = src.connected
        kind = (" (simulated)" if src.name.startswith("simulated") else
                f" (replay {getattr(src, 'position_s', 0) / 60:.1f} min)" if src.name.startswith("replay:") else "")
        paused = self.ctx["actions"].paused
        listening = not paused
        self.s_eye.set_listening(listening)
        flat = getattr(src, "flat_channels", lambda: [])() if live else []
        n_run = getattr(src, "n_running", len(src.channels))
        chtxt = f"{n_run} ch" + (f", FLAT: {', '.join(flat[:3])}{'...' if len(flat) > 3 else ''}" if flat else "")
        self.s_eeg.set(f"{'live' if live else 'no data'}, {src.fs:.0f} Hz, {chtxt}" + kind,
                       (theme.GOOD if live and not flat else theme.RED), listening=listening)
        st = eng.stats
        self.s_model.set(f"{self.ctx['model'].name.split(':')[0][:28]}, {st['model_ms']:.0f} ms, {st['calls_per_s']:.0f}/s",
                         listening=listening)
        if not self.ctx.get("listening", True):
            self.s_ctl.set("not started", theme.MUTED, listening=False)
        else:
            self.s_ctl.set("paused" if paused else "listening", theme.RED if paused else theme.GOOD,
                           listening=listening)
        if self.pause_btn.property("paused") != ("true" if paused else "false"):
            self.refresh_pause()

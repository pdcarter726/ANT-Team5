"""Start screen shown before listening: runs the per-session calibration (controller/session_calibrate.py) in its
own process and reports when a profile is ready. Nothing is detected or sent to the OS until then."""
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QTimer, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..log import log

REPO = Path(__file__).resolve().parents[2]
SESSION_PROFILE = REPO / "profiles" / "session.joblib"
PHASES = {"record": "Calibrating: follow the cues in the calibration window",
          "train": "Fitting the model to this session",
          "done": "Calibration complete", "error": "Calibration stopped"}
CALIBRATION_TRIALS = {2: 3, 5: 6, 10: 14, 15: 21, 30: 42}


class SessionPage(QWidget):
    ready = Signal(str)            # profile name to load

    def __init__(self):
        super().__init__()
        self.proc = None
        self.t0 = 0.0
        outer = QVBoxLayout(self)
        outer.addStretch(2)
        box = QVBoxLayout()
        box.setSpacing(10)
        self.title = QLabel("Start session", objectName="sessionTitle")
        self.info = QLabel("Calibration takes about 10 minutes and is done once per session.", objectName="bigSub")
        duration_row = QHBoxLayout()
        duration_row.setSpacing(10)
        duration_row.addWidget(QLabel("Calibration duration"))
        self.duration = QComboBox()
        for minutes in CALIBRATION_TRIALS:
            self.duration.addItem(f"{minutes} minutes", minutes)
        self.duration.setCurrentIndex(2)
        self.duration.currentIndexChanged.connect(self._duration_changed)
        duration_row.addWidget(self.duration)
        duration_box = QWidget()
        duration_box.setLayout(duration_row)
        self.duration_box = duration_box
        box.addWidget(duration_box, 0, Qt.AlignHCenter)
        self.state = QLabel("", objectName="sessionState")
        self.detail = QLabel("", objectName="bigSub")
        self.detail.setWordWrap(True)
        self.detail.setAlignment(Qt.AlignHCenter)
        self.detail.setFixedWidth(640)
        for w in (self.title, self.info, self.state, self.detail):
            box.addWidget(w, 0, Qt.AlignHCenter)
        btns = QHBoxLayout()
        btns.setSpacing(10)
        self.start_btn = QPushButton("Start session", objectName="primary")
        self.start_btn.clicked.connect(self.start)
        self.last_btn = QPushButton("Use last calibration")
        self.last_btn.clicked.connect(lambda: self.ready.emit("session"))
        self.cancel_btn = QPushButton("Cancel calibration")
        self.cancel_btn.clicked.connect(self.cancel)
        btns.addStretch(1)
        for b in (self.start_btn, self.last_btn, self.cancel_btn):
            btns.addWidget(b)
        btns.addStretch(1)
        box.addSpacing(8)
        box.addLayout(btns)
        outer.addLayout(box)
        outer.addStretch(3)
        self.timer = QTimer(self, timeout=self._tick, interval=500)
        self.idle()

    # ---------------- states ----------------
    def _duration_changed(self):
        minutes = self.duration.currentData()
        self.info.setText(f"Calibration takes about {minutes} minutes and is done once per session.")

    def idle(self, error=""):
        running = False
        self.start_btn.setVisible(not running)
        self.start_btn.setText("Start session" if not error else "Try again")
        self.duration_box.setVisible(True)
        has_last = SESSION_PROFILE.exists()
        self.last_btn.setVisible(has_last)
        if has_last:
            when = time.strftime("%b %d, %H:%M", time.localtime(SESSION_PROFILE.stat().st_mtime))
            self.last_btn.setText(f"Use last calibration ({when})")
        self.cancel_btn.setVisible(False)
        self.state.setText(PHASES["error"] if error else "")
        self.state.setVisible(bool(error))
        self.detail.setText(error)
        self.detail.setVisible(bool(error))
        self.info.setVisible(True)

    def show_error(self, msg):
        """Called by the app when the profile could not be loaded or attached."""
        self.idle(msg)

    def start(self):
        if self.proc is not None:
            return
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(REPO))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self.last_status = ("record", "starting the cue app")
        self.last_line = ""
        self.t0 = time.time()
        minutes = self.duration.currentData()
        trials = CALIBRATION_TRIALS[minutes]
        log("SESSION", "calibration started")
        self.proc.start(sys.executable, ["-u", "-m", "controller.session_calibrate",
                                         "--trials", str(trials)])
        self.start_btn.setVisible(False)
        self.duration_box.setVisible(False)
        self.last_btn.setVisible(False)
        self.cancel_btn.setVisible(True)
        self.info.setVisible(False)
        self.state.setVisible(True)
        self.detail.setVisible(True)
        self.timer.start()
        self._tick()

    def cancel(self):
        if self.proc is None:
            return
        log("SESSION", "calibration cancelled")
        self.cancelled = True
        self.kill_tree()

    def kill_tree(self):
        """Stops the calibration and the cue app it started (QProcess.kill alone leaves the child on Windows)."""
        if self.proc is None:
            return
        pid = self.proc.processId()
        if sys.platform == "win32" and pid:
            subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            self.proc.kill()
        self.proc.waitForFinished(3000)

    # ---------------- process output ----------------
    def _read(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in data.splitlines():
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("@@STATUS "):
                phase, _, msg = line[9:].partition(" ")
                self.last_status = (phase, msg)
                log("SESSION", f"{phase}: {msg}")
            else:
                self.last_line = line[:200]
                log("CALIB", line[:300])

    def _tick(self):
        phase, msg = self.last_status
        el = int(time.time() - self.t0)
        self.state.setText(f"{PHASES.get(phase, phase)}  {el // 60:02d}:{el % 60:02d}")
        self.detail.setText(msg)

    def _finished(self, code, _status):
        self._read()
        self.timer.stop()
        self.proc = None
        phase, msg = self.last_status
        if getattr(self, "cancelled", False):
            self.cancelled = False
            self.idle("Calibration was cancelled; the model is unchanged.")
        elif code == 0 and phase == "done":
            log("SESSION", "calibration finished", seconds=int(time.time() - self.t0))
            self.state.setText(PHASES["done"])
            self.ready.emit("session")
        else:
            err = msg if phase == "error" else f"Calibration exited with code {code}; the model is unchanged."
            if self.last_line and phase != "train":
                err += f"\nLast message from the cue app: {self.last_line}"
            self.idle(err)

"""Start screen shown before listening: runs the per-session calibration (controller/session_calibrate.py) in its
own process and reports when a profile is ready. Nothing is detected or sent to the OS until then."""
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from ..log import log
from .widgets import DropdownCombo

REPO = Path(__file__).resolve().parents[2]
SESSION_PROFILE = REPO / "profiles" / "session.joblib"
PHASES = {"record": "Calibrating: follow the cues in the calibration window",
          "train": "Fitting the model to this session",
          "done": "Calibration complete", "error": "Calibration stopped"}
CALIBRATION_TRIALS = {2: 3, 5: 6, 10: 14, 15: 21, 30: 42}
RECAL_GESTURES = ["jaw_clench", "double_blink", "eyebrow_raise", "left_wink", "right_wink",
                  "head_left", "head_right", "head_up", "head_down"]
RECAL_NAMES = {"head_left": "Head turn left", "head_right": "Head turn right", "head_up": "Head turn up",
               "head_down": "Head turn down"}


def recal_label(g):
    return RECAL_NAMES.get(g, g.replace("_", " ").capitalize())


class SessionPage(QWidget):
    ready = Signal(str)            # profile name to load
    back = Signal()                # leave this page and return to the live view
    ssvep_changed = Signal()       # an SSVEP calibration just ended

    def __init__(self):
        super().__init__()
        self.proc = None
        self.sim_folder = None
        self.t0 = 0.0
        self.stream_name, self.stream_type = None, "EEG"
        self.ssvep_proc = None
        self.can_return = False        # the live view is running behind this page (set by the dashboard)
        self.recal_mode = False
        outer = QVBoxLayout(self)
        outer.addStretch(2)
        box = QVBoxLayout()
        box.setSpacing(10)
        self.title = QLabel("Start session", objectName="sessionTitle")
        self.info = QLabel("Calibration takes about 10 minutes and is done once per session.", objectName="bigSub")
        duration_row = QHBoxLayout()
        duration_row.setSpacing(10)
        duration_row.addWidget(QLabel("Calibration duration"))
        self.duration = DropdownCombo()
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
        # ---- targeted recalibration: pick gestures, record only those, improve them in the current model
        self.recal_btn = QPushButton("Recalibrate specific gestures...")
        self.recal_btn.setToolTip("Re-record only the gestures that are not detected well, about 2-3 minutes. "
                                  "The rest of the model is kept.")
        self.recal_btn.clicked.connect(lambda: self.show_recal(True))
        self.back_btn = QPushButton("Back to live view")
        self.back_btn.clicked.connect(self.back.emit)
        row2 = QHBoxLayout()
        row2.addStretch(1)
        row2.addWidget(self.recal_btn)
        row2.addWidget(self.back_btn)
        row2.addStretch(1)
        self.row2 = row2
        box.addLayout(row2)
        # ---- SSVEP (optional, separate from the gesture model): four flickering squares as another input
        self.ssvep_btn = QPushButton("SSVEP calibration (optional)...")
        self.ssvep_btn.setToolTip("Flashing squares: a photosensitive-epilepsy warning is shown first. About 3 minutes. "
                                  "Uses the same LSL stream; the gesture model is not changed.")
        self.ssvep_btn.clicked.connect(self.start_ssvep)
        self.ssvep_note = QLabel("", objectName="bigSub")
        self.ssvep_note.setWordWrap(True)
        self.ssvep_note.setAlignment(Qt.AlignHCenter)
        self.ssvep_note.setFixedWidth(640)
        row3 = QHBoxLayout()
        row3.addStretch(1)
        row3.addWidget(self.ssvep_btn)
        row3.addStretch(1)
        box.addLayout(row3)
        box.addWidget(self.ssvep_note, 0, Qt.AlignHCenter)
        self.recal_box = QWidget()
        rv = QVBoxLayout(self.recal_box)
        rv.setSpacing(8)
        rv.addWidget(QLabel("Re-record these gestures (the rest of the model is kept)", objectName="sessionState"),
                     0, Qt.AlignHCenter)
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        self.recal_checks = {}
        for i, g in enumerate(RECAL_GESTURES):
            c = QCheckBox(recal_label(g))
            c.toggled.connect(self._recal_changed)
            self.recal_checks[g] = c
            grid.addWidget(c, i // 3, i % 3)
        gw = QWidget()
        gw.setLayout(grid)
        rv.addWidget(gw, 0, Qt.AlignHCenter)
        trow = QHBoxLayout()
        trow.addStretch(1)
        trow.addWidget(QLabel("Repetitions per gesture"))
        self.recal_trials = QSpinBox(minimum=3, maximum=30, value=8)
        self.recal_trials.valueChanged.connect(self._recal_changed)
        trow.addWidget(self.recal_trials)
        self.recal_replace = QCheckBox("Forget the old recordings of these gestures")
        self.recal_replace.setToolTip("Use when the old recordings were bad. Off: the new ones count more, the old stay.")
        trow.addWidget(self.recal_replace)
        trow.addStretch(1)
        rv.addLayout(trow)
        self.recal_note = QLabel("", objectName="bigSub")
        rv.addWidget(self.recal_note, 0, Qt.AlignHCenter)
        rb = QHBoxLayout()
        rb.addStretch(1)
        self.recal_go = QPushButton("Start recalibration", objectName="primary")
        self.recal_go.clicked.connect(self.start)
        rb.addWidget(self.recal_go)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(lambda: self.show_recal(False))
        rb.addWidget(cancel)
        rb.addStretch(1)
        rv.addLayout(rb)
        self.recal_box.setVisible(False)
        box.addWidget(self.recal_box)
        outer.addLayout(box)
        outer.addStretch(3)
        self.timer = QTimer(self, timeout=self._tick, interval=500)
        self.idle()

    # ---------------- states ----------------
    def use_recording(self, specs):
        """Simulation: no cue app; the model is trained on the recorded sessions that are being injected."""
        self.sim_folder = list(specs)
        self.duration_box.setVisible(False)
        n = len(self.sim_folder)
        self.info.setText(f"Simulation: the model is trained on the {n} recorded session{'s' if n > 1 else ''}.")

    def _duration_changed(self):
        minutes = self.duration.currentData()
        self.info.setText(f"Calibration takes about {minutes} minutes and is done once per session.")

    def show_recal(self, on):
        if not on:
            self.idle()
            return
        self.recal_mode = on
        self.recal_box.setVisible(on)
        self.refresh_ssvep()
        for b in (self.start_btn, self.last_btn, self.recal_btn):
            b.setVisible(not on and (b is not self.last_btn or SESSION_PROFILE.exists())
                         and (b is not self.recal_btn or SESSION_PROFILE.exists()))
        self.duration_box.setVisible(not on and self.sim_folder is None)
        self.info.setVisible(not on)
        if on:
            self.state.setVisible(False)
            self.detail.setVisible(False)
            self._recal_changed()

    def _recal_changed(self, *_):
        n = [g for g, c in self.recal_checks.items() if c.isChecked()]
        # per trial: cue 1.5 s + rest 2 s; rest and two confounders are recorded too
        minutes = (len(n) + 1) * self.recal_trials.value() * 3.5 / 60 + 0.4
        self.recal_go.setEnabled(bool(n))
        self.recal_note.setText(f"About {minutes:.1f} minutes." if n else "Tick at least one gesture.")

    def refresh_ssvep(self):
        """SSVEP row: shown on the idle start screen; says whether there is a usable calibration."""
        from ..ssvep.core import PROFILE_PATH, SsvepProfile
        busy = self.proc is not None or self.recal_mode
        show = not busy and self.sim_folder is None
        self.ssvep_btn.setVisible(show)
        self.ssvep_btn.setText("Stop SSVEP calibration" if self.ssvep_proc is not None else
                               "SSVEP calibration (optional)...")
        if self.ssvep_proc is None and not getattr(self, "_ssvep_msg", ""):
            try:
                p = SsvepProfile.load(PROFILE_PATH)
                txt = (f"SSVEP calibrated {p.created[:16]}: {p.summary()}" if p.usable else
                       "SSVEP: the last calibration did not pass its checks. " + "; ".join(p.reasons))
            except (OSError, ValueError, TypeError, KeyError):
                txt = "SSVEP: not calibrated. It can be enabled from the controller once calibrated."
            self.ssvep_note.setText(txt)
        self.ssvep_note.setVisible(show and bool(self.ssvep_note.text()))

    def start_ssvep(self):
        if self.ssvep_proc is not None:                         # the button is "Stop" while it runs
            self.kill_tree(self.ssvep_proc)
            return
        if self.proc is not None:
            return
        p = QProcess(self)
        p.setWorkingDirectory(str(REPO))
        p.setProcessChannelMode(QProcess.MergedChannels)
        p.readyReadStandardOutput.connect(self._ssvep_read)
        p.finished.connect(self._ssvep_finished)
        self.ssvep_proc, self._ssvep_msg = p, "The epilepsy warning is opening in its own window..."
        self.ssvep_note.setText(self._ssvep_msg)
        args = ["-u", "-m", "controller.ssvep.calibrate"]
        if self.stream_name:
            args += ["--stream-name", self.stream_name]
        else:
            args += ["--stream-type", self.stream_type or "EEG"]
        log("SESSION", "SSVEP calibration started")
        p.start(sys.executable, args)
        self.start_btn.setEnabled(False)
        self.recal_btn.setEnabled(False)
        self.refresh_ssvep()

    def _ssvep_read(self):
        data = bytes(self.ssvep_proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in data.splitlines():
            line = line.strip()
            if line.startswith("@@STATUS "):
                phase, _, msg = line[9:].partition(" ")
                self._ssvep_msg = f"SSVEP calibration: {msg}"
                self.ssvep_note.setText(self._ssvep_msg)
                self.ssvep_note.setVisible(True)
                log("SESSION", f"ssvep {phase}: {msg}")
            elif line:
                log("CALIB", line[:300])

    def _ssvep_finished(self, code, _status):
        self._ssvep_read()
        self.ssvep_proc = None
        self.start_btn.setEnabled(True)
        self.recal_btn.setEnabled(True)
        if code != 0 and not self._ssvep_msg.startswith("SSVEP calibration: "):
            self._ssvep_msg = f"SSVEP calibration stopped (exit code {code})."
            self.ssvep_note.setText(self._ssvep_msg)
        elif code == 0:
            self._ssvep_msg = ""            # the next refresh reads the saved calibration
        log("SESSION", "SSVEP calibration finished", code=code)
        self.refresh_ssvep()
        self.ssvep_changed.emit()

    def idle(self, error=""):
        running = False
        self.recal_mode = False
        self.recal_box.setVisible(False)
        self.recal_btn.setVisible(SESSION_PROFILE.exists() and self.sim_folder is None)
        self.back_btn.setVisible(self.can_return)
        self.start_btn.setVisible(not running)
        self.start_btn.setText("Start session" if not error else "Try again")
        self.duration_box.setVisible(self.sim_folder is None)
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
        self.refresh_ssvep()

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
        if self.recal_mode:
            chosen = [g for g, c in self.recal_checks.items() if c.isChecked()]
            if not chosen:
                self.proc = None
                return
            args = ["--only", ",".join(chosen), "--trials", str(self.recal_trials.value())]
            if self.recal_replace.isChecked():
                args.append("--replace")
            log("SESSION", "recalibration started", gestures=",".join(chosen), trials=self.recal_trials.value())
        else:
            log("SESSION", "calibration started")
            args = (["--from-folder"] + list(self.sim_folder) if self.sim_folder else ["--trials", str(trials)])
        self.proc.start(sys.executable, ["-u", "-m", "controller.session_calibrate"] + args)
        self.start_btn.setVisible(False)
        self.duration_box.setVisible(False)
        self.last_btn.setVisible(False)
        self.recal_btn.setVisible(False)
        self.back_btn.setVisible(False)
        self.recal_box.setVisible(False)
        self.ssvep_btn.setVisible(False)
        self.ssvep_note.setVisible(False)
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

    def kill_tree(self, proc=None):
        """Stops the calibration and the cue app it started (QProcess.kill alone leaves the child on Windows).
        With no argument: everything this page started, also the SSVEP calibration, so no flashing window outlives the app."""
        if proc is None:
            for p in (self.proc, self.ssvep_proc):
                if p is not None:
                    self.kill_tree(p)
            return
        pid = proc.processId()
        if sys.platform == "win32" and pid:
            subprocess.call(["taskkill", "/PID", str(pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            proc.kill()
        proc.waitForFinished(3000)

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
        self.recal_mode = False
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

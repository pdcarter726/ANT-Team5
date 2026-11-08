"""Side panel for simulated EEG (controller/sim_inject.py): one button per recorded class. Pressing injects a real
recorded trial into the live stream; holding keeps the steady part going (held clench / head turn)."""
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QCheckBox, QFrame, QLabel, QPushButton, QVBoxLayout

from . import theme

NAME_MAP = {"head_left": "head_turn_left", "head_right": "head_turn_right",
            "head_up": "head_turn_up", "head_down": "head_turn_down"}
CONFOUNDERS = ("talking", "chewing", "natural_blinking", "typing", "idle")
DISCARDED = ("look_left", "look_right", "look_up", "look_down")     # dropped from the gesture set (confused with head turns)


class SimPanel(QFrame):
    def __init__(self, source):
        super().__init__(objectName="simPanel")
        self.src = source
        self.setFixedWidth(230)
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(6)
        v.addWidget(QLabel("Simulate EEG", objectName="sessionState"))
        n = len(source.specs)
        total = sum(len(t) for t in source.trials.values())
        src_lbl = QLabel(f"{total} recorded trials, {n} session{'s' if n > 1 else ''}", objectName="bigSub")
        src_lbl.setToolTip("\n".join(source.specs) + "\nPress = one gesture, hold = held gesture.")
        v.addWidget(src_lbl)
        forge = QCheckBox("Forge new variants")
        forge.setToolTip("Each injection gets its own per-channel gain and another rest trial's background, "
                         "so the model never sees the exact recorded samples")
        forge.setChecked(source.forge)
        forge.toggled.connect(lambda on: setattr(self.src, "forge", on))
        v.addWidget(forge)
        gestures = [l for l in source.labels if l not in CONFOUNDERS and l not in DISCARDED]
        others = [l for l in source.labels if l in CONFOUNDERS]
        for title, labels in (("Gestures", gestures), ("Other activity", others)):
            if not labels:
                continue
            v.addSpacing(6)
            v.addWidget(QLabel(title, objectName="section"))
            for lab in labels:
                b = QPushButton(theme.nice(NAME_MAP.get(lab, lab)), objectName="test")
                b.setToolTip(f"{len(source.trials[lab])} recorded trials")
                b.pressed.connect(lambda lab=lab: self.src.inject(lab, hold=True))
                b.released.connect(self.src.release)
                v.addWidget(b)
        v.addStretch(1)
        self.last = QLabel("", objectName="bigSub")
        self.last.setWordWrap(True)
        v.addWidget(self.last)
        QTimer(self, timeout=self._tick, interval=300).start()

    def _tick(self):
        inj = self.src.injected
        if inj:
            lab, trial, session = inj
            self.last.setText(f"Last: {theme.nice(NAME_MAP.get(lab, lab))}\n{session}, trial {trial}")

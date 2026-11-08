"""Targeted recalibration: cue only chosen gestures, and weight only those in the refit."""
import copy

import numpy as np
import pytest

from protocol import build_trials, restrict_classes


def test_restrict_classes_cues_only_chosen_gestures_plus_rest(cfg):
    c = copy.deepcopy(cfg)
    restrict_classes(c, 7, ["jaw_clench", "double_blink", "jaw_clench"])
    labels = {t.label for t in build_trials(c, 7, seed=1, trials_override=4)}
    assert {"rest", "jaw_clench", "double_blink"} <= labels
    assert not labels & {"head_left", "left_wink", "eyebrow_raise"}


def test_restrict_classes_rejects_unknown_and_empty(cfg):
    c = copy.deepcopy(cfg)
    with pytest.raises(KeyError):
        restrict_classes(c, 7, ["not_a_gesture"])
    with pytest.raises(ValueError):
        restrict_classes(c, 7, [" ", "rest"])


def test_collect_dry_run_with_classes(capsys):
    import collect
    assert collect.main(["--session", "7", "--classes", "head_left,jaw_clench", "--trials", "3", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "head_left" in out and "jaw_clench" in out and "left_wink" not in out


def test_adapt_weights_only_the_rerecorded_gestures(cfg, tmp_path, monkeypatch):
    """Fake dataset: the new session has jaw_clench and rest windows; only jaw_clench gets the class weight."""
    import train
    from detector import Profile

    seen = {}

    class DS:
        def train_arrays(self):
            y = np.array(["jaw_clench"] * 4 + ["rest"] * 6 + ["confounder"] * 2)
            return np.ones((12, 3)), y, np.zeros(12), np.zeros(12)

    class M:
        classes_ = np.array(["confounder", "jaw_clench", "rest"])

    base = Profile("stub", ["confounder", "jaw_clench", "rest"], M(), {}, ["a"], 500.0, 0.5, 0.25, {})
    old_y = np.array(["jaw_clench"] * 5 + ["rest"] * 5 + ["head_left"] * 5)
    np.savez(tmp_path / "base_train.npz", X=np.zeros((15, 3)), y=old_y, th=np.zeros(15), tv=np.zeros(15), w=np.ones(15))
    monkeypatch.setattr(train, "PROFILE_DIR", tmp_path)
    monkeypatch.setattr(train.Profile, "load", staticmethod(lambda p: base))
    monkeypatch.setattr(train, "load_dataset", lambda *a, **k: DS())
    monkeypatch.setattr(train, "fit_model", lambda kind, X, y, w: seen.update(y=y, w=w) or M())
    monkeypatch.setattr(train, "fit_cursor", lambda *a: None)
    monkeypatch.setattr(train, "make_inference_ready", lambda m: m)
    monkeypatch.setattr(train, "save_profile", lambda *a, **k: None)
    ns = dict(adapt="x", base="base", name="out", adapt_weight=2.0, config=None, adapt_classes=["jaw_clench"],
              class_weight=5.0, replace=False)
    import argparse
    train.adapt(argparse.Namespace(**ns))
    y, w = seen["y"], seen["w"]
    assert len(y) == 27 and w[:15].tolist() == [1.0] * 15                   # old windows untouched
    assert set(w[15:19]) == {5.0} and set(w[19:]) == {1.0}                  # new jaw windows heavy, rest/confounder normal
    train.adapt(argparse.Namespace(**{**ns, "replace": True}))
    y, w = seen["y"], seen["w"]
    assert len(y) == 22 and (y == "jaw_clench").sum() == 4                  # old jaw_clench windows dropped
    with pytest.raises(SystemExit):
        train.adapt(argparse.Namespace(**{**ns, "adapt_classes": ["nope"]}))


def test_session_page_builds_recalibration_command(monkeypatch):
    pytest.importorskip("PySide6")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QProcess
    app = QApplication.instance() or QApplication([])
    from controller.ui import session as S
    started = {}
    monkeypatch.setattr(QProcess, "start", lambda self, prog, args: started.update(args=args))
    page = S.SessionPage()
    page.can_return = True
    page.show_recal(True)
    assert not page.recal_go.isEnabled()                                    # nothing ticked yet
    page.recal_checks["jaw_clench"].setChecked(True)
    page.recal_checks["head_left"].setChecked(True)
    page.recal_trials.setValue(6)
    page.recal_replace.setChecked(True)
    assert page.recal_go.isEnabled()
    page.start()
    a = started["args"]
    assert a[a.index("--only") + 1] == "jaw_clench,head_left" and a[a.index("--trials") + 1] == "6" and "--replace" in a
    page.proc = None


def test_dropdown_combo_ignores_wheel_and_arrow_keys_until_opened():
    pytest.importorskip("PySide6")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QKeyEvent, QWheelEvent
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from controller.ui.widgets import DropdownCombo
    c = DropdownCombo()
    for t in ("a", "b", "c"):
        c.addItem(t)
    c.show()
    c.setCurrentIndex(1)
    w = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120), Qt.NoButton, Qt.NoModifier,
                    Qt.NoScrollPhase, False)
    app.sendEvent(c, w)
    assert c.currentIndex() == 1 and not w.isAccepted()                       # wheel passes through
    for key in (Qt.Key_Down, Qt.Key_Up, Qt.Key_End):
        app.sendEvent(c, QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))
    assert c.currentIndex() == 1
    c.setCurrentIndex(2)                                                      # choosing from the list still works
    assert c.currentText() == "c"

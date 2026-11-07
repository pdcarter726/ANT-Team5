"""Universal Computer Controller: EEG gestures -> actions. No hands needed.

Default controls: head turns move the cursor, eyebrow raise opens the quick bar, jaw clench clicks
(hold to drag), double blink right-clicks. The webcam eye cursor is optional.

  python -m controller --model ml/model.py:predict                      # the real thing (head-turn cursor)
  python -m controller --model ml/model.py:predict --gaze-profile nico  # plus the eye cursor (head turns fine-tune)
  python -m controller                                                  # rehearsal: simulated EEG, test buttons
  python -m controller --model ml/model.py:predict --stream-name eego  # pick an LSL stream by name
  python -m controller --mapping game                                   # game controls (WASD)
  python -m controller --model ml/model.py:predict --eeg-file data/X.cnt   # test the model on a recording

Ctrl+Alt+G (any window, Windows) pauses/resumes gesture actions. The eye cursor keeps its own keys
(Ctrl+Alt+P pause, C recenter, V check, F fix, Q quit).
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

from . import log as L
from .log import log

MAPPING_DIR = Path(__file__).parent / "mappings"


def resolve_mapping(name):
    p = Path(name)
    if p.suffix == ".json" and p.exists():
        return p
    p = MAPPING_DIR / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"mapping {name!r} not found; available: {', '.join(x.stem for x in MAPPING_DIR.glob('*.json'))}")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="sim", help="file.py:function or module:function; 'sim' = test buttons")
    ap.add_argument("--eeg", choices=["lsl", "sim"], help="EEG source (default: lsl with a real model, sim otherwise)")
    ap.add_argument("--eeg-file", help="replay a recording (e.g. data/xxx.cnt) as if it were live")
    ap.add_argument("--eeg-start", type=float, default=0.0, help="seconds into the recording to start the replay")
    ap.add_argument("--stream-name", help="LSL stream name (default: first stream of --stream-type)")
    ap.add_argument("--stream-type", default="EEG")
    ap.add_argument("--mapping", default="default", help="mapping name in controller/mappings or a .json path")
    ap.add_argument("--gaze-profile", help="also run the eye cursor with this profile (python -m gaze.app --list-profiles)")
    ap.add_argument("--gaze", action="store_true", help="also run the eye cursor, calibrating first")
    ap.add_argument("--gaze-running", action="store_true", help="an eye cursor is already running; use it")
    ap.add_argument("--threshold", type=float, default=0.6, help="confidence needed to count a window")
    ap.add_argument("--k", type=int, default=3, help="windows that must agree ...")
    ap.add_argument("--n", type=int, default=4, help="... out of the last n")
    ap.add_argument("--refractory", type=float, default=0.8, help="seconds before the same gesture can fire again")
    ap.add_argument("--no-overlay", action="store_true")
    a = ap.parse_args()

    log_file = L.open_file()
    log("START", "controller starting", log_file=log_file, python=sys.version.split()[0])
    log("START", "options", **vars(a))

    from .detector import GestureDetector
    from .eeg_source import FileSource, LSLSource, SimSource
    from .model_api import LoadedModel, SimModel

    model_is_sim = a.model == "sim"
    model = SimModel() if model_is_sim else LoadedModel(a.model)
    eeg = a.eeg or ("sim" if model_is_sim else "lsl")
    try:
        if a.eeg_file:
            source = FileSource(a.eeg_file, a.eeg_start)
        else:
            source = SimSource() if eeg == "sim" else LSLSource(a.stream_name, a.stream_type)
    except Exception as e:
        log("ERROR", f"EEG stream not available ({e})")
        raise SystemExit("No EEG stream. Start the eego LSL stream (or ANT's laptop on the same network), "
                         "or run with --eeg sim to rehearse.")
    detector = GestureDetector(a.threshold, a.k, a.n, a.refractory)

    gaze = None
    if a.gaze or a.gaze_profile or a.gaze_running:
        from .gaze_link import GazeLink
        gaze = GazeLink(a.gaze_profile, start=not a.gaze_running)
    else:
        log("START", "cursor: head turns (eye cursor off; add --gaze-profile NAME to use it)")

    mapping_path = resolve_mapping(a.mapping)
    mapping = json.loads(mapping_path.read_text())

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("Universal Computer Controller")

    from .actions import ActionEngine
    from .engine import Engine
    from .ui.dashboard import Bridge, Dashboard
    from .ui.menu import QuickMenu
    from .ui.overlay import Overlay

    bridge = Bridge()
    menu = QuickMenu()
    menu.set_items(mapping.get("menu", []))
    actions = ActionEngine(mapping, gaze=gaze, on_menu=bridge.menu.emit)
    engine = Engine(source, model, detector, actions)
    overlay = Overlay()
    ctx = {"source": source, "model": model, "detector": detector, "actions": actions, "engine": engine,
           "gaze": gaze, "gaze_profile": a.gaze_profile, "mapping": mapping, "mapping_path": mapping_path,
           "mapping_dir": MAPPING_DIR, "menu": menu, "overlay": overlay, "model_is_sim": model_is_sim,
           "log_file": log_file}
    dash = Dashboard(ctx)

    # engine thread -> UI thread
    engine.on_probs = bridge.probs.emit
    engine.on_event = bridge.event.emit
    L.listeners.append(bridge.logline.emit)
    bridge.probs.connect(dash.on_probs)
    bridge.event.connect(dash.on_event)
    bridge.logline.connect(dash.on_log)
    bridge.gaze_status.connect(dash.on_gaze_status)

    def on_event(kind, gesture, prob, desc):
        if kind == "fire":
            from .ui.theme import describe
            overlay.show_event(gesture, describe(mapping.get("gestures", {}).get(gesture, "none")))
    bridge.event.connect(on_event)
    bridge.menu.connect(lambda: menu.toggle())

    def on_menu_pick(label, action):
        # run after the menu has disappeared, so keys go to the app underneath
        def run():
            desc = actions.run(action)
            log("QUICKBAR", f"picked {label!r}", action=action, result=desc)
            from .ui.theme import describe
            overlay.show_event("quick bar", label)
            dash.big.setText(label)
            dash.big_sub.setText("from the quick bar")
        QTimer.singleShot(120, run)
    menu.chosen.connect(on_menu_pick)

    def on_test(g, seconds):
        log("TEST", f"test button: {g}", held_s=seconds)
        if model_is_sim:
            model.trigger(g, seconds)
            if hasattr(source, "trigger"):
                source.trigger(g, seconds)
        else:                                # real model: run the mapped action directly
            desc = actions.handle("fire", g, 1.0)
            bridge.event.emit("fire", g, 1.0, desc or "")

            def release():
                d = actions.handle("release", g)
                if d:
                    bridge.event.emit("release", g, 0.0, d)
            QTimer.singleShot(int(seconds * 1000), release)
    dash.test_gesture.connect(on_test)

    # eye cursor status, polled off the UI thread (the UDP call can wait up to 0.3 s)
    def poll_gaze():
        while True:
            st = gaze.status() if gaze else None
            bridge.gaze_status.emit(st)
            time.sleep(0.5)
    threading.Thread(target=poll_gaze, daemon=True).start()

    # Ctrl+Alt+G from any window toggles gesture actions (Windows)
    if sys.platform == "win32":
        import ctypes
        gk = ctypes.windll.user32.GetAsyncKeyState
        state = {"down": False}

        def poll_keys():
            down = bool(gk(0x11) & 0x8000) and bool(gk(0x12) & 0x8000) and bool(gk(ord("G")) & 0x8000)
            if down and not state["down"]:
                dash.toggle_pause()
                log("CMD", "Ctrl+Alt+G: gestures " + ("paused" if actions.paused else "resumed"))
            state["down"] = down
        QTimer(app, timeout=poll_keys, interval=40).start()

    dash.show()
    if not a.no_overlay:
        overlay.show()
    engine.start()
    log("START", "ready", eeg=source.name, model=model.name, mapping=mapping.get("name"),
        eye_cursor=("off" if not gaze else (a.gaze_profile or "calibrating")))
    code = app.exec()
    engine.stop()
    source.close()
    if gaze and not a.gaze_running:
        gaze.stop()
    log("START", "controller stopped")
    sys.exit(code)


if __name__ == "__main__":
    main()

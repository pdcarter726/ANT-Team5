"""Universal Computer Controller: EEG gestures -> actions. No hands needed.

Default controls: head turns move the cursor, eyebrow raise opens the quick bar, jaw clench clicks
(hold to drag), double blink right-clicks. The webcam eye cursor is optional.

  python -m controller                              # the real thing: live EEG, "Start session" calibrates (~10 min)
  python -m controller --gaze-profile nico          # plus the eye cursor (head turns fine-tune)
  python -m controller --model team:session         # skip the start screen, load profiles/session.joblib
  python -m controller --model sim                  # rehearsal: simulated EEG, test buttons
  python -m controller --eeg inject                 # no headset: recorded trials injected live, through the model
  python -m controller --stream-name eego           # pick an LSL stream by name
  python -m controller --mapping game               # game controls (WASD)
  python -m controller --model team:session --eeg-file data/X.cnt   # test a profile on a recording
  python -m controller --model ml/model.py:predict  # any other model function (see model_api.py)

Ctrl+Alt+G (any window, Windows) pauses/resumes gesture actions. Ctrl+Alt+E stops the eye cursor. The eye cursor keeps its own keys
(Ctrl+Alt+P pause, C recenter, V check, F fix, Q quit).
"""
import argparse
import json
from collections import deque
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
    ap.add_argument("--model", default="team",
                    help="team = the team model, calibrated per session from the start screen; team:NAME = load "
                         "profiles/NAME.joblib directly; sim = test buttons; or file.py:function / module:function")
    ap.add_argument("--eeg", choices=["lsl", "sim", "inject"],
                    help="EEG source: lsl (default with a real model), sim (rehearsal), inject (recorded trials "
                         "injected live from the side panel, through the model)")
    ap.add_argument("--no-eeg", action="store_true",
                    help="start without an EEG source; choose LSL or simulated EEG in the dashboard")
    ap.add_argument("--sim-data", nargs="+", help="recorded sessions for --eeg inject, FOLDER or FOLDER@EEGFILE "
                    "(default: every calibration and training session in data/)")
    ap.add_argument("--no-forge", action="store_true", help="--eeg inject: replay recorded trials exactly")
    ap.add_argument("--eeg-file", help="replay a recording (e.g. data/xxx.cnt) as if it were live")
    ap.add_argument("--eeg-start", type=float, default=0.0, help="seconds into the recording to start the replay")
    ap.add_argument("--stream-name", help="LSL stream name (default: first stream of --stream-type)")
    ap.add_argument("--stream-type", default="EEG")
    ap.add_argument("--mapping", default="default", help="mapping name in controller/mappings or a .json path")
    ap.add_argument("--gaze-profile", help="also run the eye cursor with this profile (python -m gaze.app --list-profiles)")
    ap.add_argument("--gaze", action="store_true", help="also run the eye cursor, calibrating first")
    ap.add_argument("--gaze-running", action="store_true", help="an eye cursor is already running; use it")
    # detection defaults: the team model's tuned values from its profile, otherwise 0.6 / 3 of 4 / 0.8 s
    ap.add_argument("--threshold", type=float, help="confidence needed to count a window")
    ap.add_argument("--k", type=int, help="windows that must agree ...")
    ap.add_argument("--n", type=int, help="... out of the last n")
    ap.add_argument("--refractory", type=float, help="seconds before the same gesture can fire again")
    ap.add_argument("--no-overlay", action="store_true")
    a = ap.parse_args()

    log_file = L.open_file()
    log("START", "controller starting", log_file=log_file, python=sys.version.split()[0])
    log("START", "options", **vars(a))

    from .detector import GestureDetector
    from .eeg_source import FileSource, LSLSource, NoSource, SimSource
    from .model_api import LoadedModel, SimModel

    from .ant_model import AntModel, Waiting
    model_is_sim = a.model == "sim"
    team = a.model == "team" or a.model.startswith("team:")
    team_profile = a.model.partition(":")[2] if a.model.startswith("team:") else None
    defer_eeg = a.no_eeg or (team and a.eeg is None and not a.eeg_file)
    if model_is_sim:
        model = SimModel()
    elif team:
        model = Waiting()               # replaced by AntModel once a session profile is ready
    else:
        model = LoadedModel(a.model)
    eeg = a.eeg or ("sim" if model_is_sim else "lsl")
    try:
        if a.eeg_file:
            source = FileSource(a.eeg_file, a.eeg_start)
        elif defer_eeg:
            source = NoSource()
        else:
            if eeg == "inject":
                from .sim_inject import InjectSource
                source = InjectSource(a.sim_data, forge=not a.no_forge)
            else:
                source = SimSource() if eeg == "sim" else LSLSource(a.stream_name, a.stream_type)
    except Exception as e:
        log("ERROR", f"EEG stream not available ({e})")
        raise SystemExit("No EEG stream. Start the eego LSL stream (or ANT's laptop on the same network), "
                         "or run with --eeg sim to rehearse.")
    detector = GestureDetector(a.threshold or 0.6, a.k or 3, a.n or 4,
                               a.refractory if a.refractory is not None else 0.8)

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
    app.setStyle("Fusion")          # same modern base on every Windows version; theme.QSS styles on top
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
           "stream_name": a.stream_name, "stream_type": a.stream_type, "log_file": log_file, "listening": not team}
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
            g = ctx.get("gaze")
            st = g.status() if g else None
            try:
                bridge.gaze_status.emit(st)
            except RuntimeError:                 # window already closed
                return
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
            esc = bool(gk(0x11) & 0x8000) and bool(gk(0x12) & 0x8000) and bool(gk(ord("E")) & 0x8000)
            if esc and not state.get("esc") and ctx.get("gaze"):
                dash.eye_off()
                log("CMD", "Ctrl+Alt+E: eye cursor stopped")
            state["esc"] = esc
        QTimer(app, timeout=poll_keys, interval=40).start()

    def start_listening(profile=None):
        """Load the team model for this session, hook it to the EEG, then start detecting."""
        selected_source = ctx["source"]
        if selected_source.name == "not connected":
            dash.session.show_error("Choose an LSL stream or simulated EEG above before starting.")
            return
        engine.source = selected_source
        if team:
            try:
                m = AntModel(profile)
                m.attach(selected_source)
            except Exception as e:
                log("ERROR", f"could not start the team model ({type(e).__name__}: {e})")
                dash.session.show_error(f"Could not load the model: {e}")
                return
            p = m.profile.params
            to_ours = dict(zip(m.profile.classes, m.classes))
            detector.threshold = a.threshold or p.default_thr
            detector.k = a.k or p.default_persist
            detector.n = a.n or max(detector.k, p.default_persist)
            detector.refractory_s = a.refractory if a.refractory is not None else p.default_refractory_s
            detector.thr = {} if a.threshold else {to_ours.get(c, c): v for c, v in p.thr.items()}
            detector.history = deque(maxlen=detector.n)
            engine.model = m
            ctx["model"] = m
            ctx["listening"] = True
            log("MODEL", "detection settings", threshold=detector.threshold, per_gesture=detector.thr,
                k=detector.k, n=detector.n, refractory_s=detector.refractory_s)
            dash.sync_detection()
        dash.show_live()
        if not a.no_overlay:
            overlay.show()
        engine.start()
        log("START", "listening", eeg=selected_source.name, model=ctx["model"].name, mapping=mapping.get("name"),
            eye_cursor=("off" if not ctx.get("gaze") else (ctx.get("gaze_profile") or "calibrating")))

    def start_simulated(source):
        simulated_model = SimModel()
        ctx["source"] = source
        ctx["model"] = simulated_model
        ctx["listening"] = True
        ctx["model_is_sim"] = True
        engine.source = source
        engine.model = simulated_model
        dash.sync_detection()
        dash.show_live()
        if not a.no_overlay:
            overlay.show()
        engine.start()
        log("START", "listening with simulated EEG", eeg=source.name, model=simulated_model.name)

    dash.session.ready.connect(start_listening)

    def start_injected(source):
        """Simulator = headset replacement: the side panel creates the EEG, the team model reads it and predicts.
        Profile: --model team:NAME, else default (trained on the recorded sessions), else the last session.
        Falls back to the button-driven fake model only if no profile can read the stream."""
        ctx["source"] = source
        engine.source = source
        if team:
            from .ant_model import available_profiles
            have = available_profiles()
            for name in dict.fromkeys([team_profile, "default", "session"]):
                if name and name in have:
                    log("START", "simulated EEG goes through the team model", profile=name)
                    start_listening(name)
                    if ctx.get("listening"):
                        return
            log("ERROR", "no usable trained profile for the simulated EEG; using the button-driven fake model "
                         "(train one: python src/train.py --data ... --name default)")
        start_simulated(source)

    def on_eeg_selected(selected):
        if hasattr(selected, "inject"):               # recorded EEG injected from the side panel, through the model
            start_injected(selected)
        elif selected.name == "simulated":            # synthetic EEG, simulated model (test buttons)
            start_simulated(selected)
        elif not team:
            start_listening()

    dash.eeg_selected.connect(on_eeg_selected)
    app.aboutToQuit.connect(dash.session.kill_tree)
    dash.show()
    if not team and not defer_eeg:
        start_listening()
    elif (model_is_sim or source.name == "simulated") and not defer_eeg:
        start_simulated(source)
    elif hasattr(source, "inject") and not defer_eeg:
        start_injected(source)
    elif team_profile and not defer_eeg:
        start_listening(team_profile)
    elif not defer_eeg:
        log("START", "waiting for the session calibration (Start session on the dashboard)", eeg=source.name)
    code = app.exec()
    engine.stop()
    ctx["source"].close()
    if ctx.get("gaze"):
        ctx["gaze"].stop()        # only stops a process this app started
    log("START", "controller stopped")
    sys.exit(code)


if __name__ == "__main__":
    main()

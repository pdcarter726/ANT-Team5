"""Stage 4 real-time control: EEG gestures -> mouse / keyboard actions.

  python src/realtime.py --profile default                       # live LSL EEG, real OS actions
  python src/realtime.py --profile default --dry-run             # live EEG, log only (nothing sent to the OS)
  python src/realtime.py --profile default --replay data/s1_X --eeg data/Session1Recording.cnt
                                                                 # replay a recording (dry run, scored against the cues)
  python src/realtime.py --replay ... --fast --no-ui             # replay as fast as possible, no window

Safety: global kill hotkey (default Ctrl+Alt+Q), global pause hotkey (Ctrl+Alt+P), pause/lock gesture
(jaw clench, hold), event-rate limiter. Mapping of gestures to actions: config/mapping.yaml.
Replays never touch the OS unless you pass --live-actions.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from common import ROOT, enable_dpi_awareness, load_config
from detector import HEAD_CLASSES, NULL_CLASSES, EventLogic, LogicParams, Profile, WindowScorer
from evaluation import format_score, score_events
from input_layer import (ActionExecutor, CursorController, PynputBackend, VirtualBackend, load_mapping,
                         start_hotkeys)
from overlay import Overlay, State

PROFILE_DIR = ROOT / "profiles"
LOG_DIR = ROOT / "logs"


def apply_mapping_overrides(params: LogicParams, mapping) -> LogicParams:
    p = params.copy()
    for g, o in mapping.overrides().items():
        if "min_prob" in o:
            p.thr[g] = float(o["min_prob"])
        if "hold" in o:
            p.persist[g] = int(o["hold"])
        if "refractory_s" in o:
            p.refractory_s[g] = float(o["refractory_s"])
    return p


def check_stream_matches(profile: Profile, names, fs):
    """Refuse to run on a stream whose channels differ from what the profile was trained on."""
    if abs(fs - profile.fs) > 1e-6:
        raise SystemExit(f"Stream rate {fs} Hz != profile {profile.fs} Hz. Retrain or fix the stream.")
    a, b = [n.lower() for n in names], [n.lower() for n in profile.ch_names]
    if a != b:
        raise SystemExit("Channel names/order differ from the profile.\n  stream : " + ", ".join(names) +
                         "\n  profile: " + ", ".join(profile.ch_names) +
                         "\nFix the stream labels or retrain; channel order is not guessed.")


class Engine:
    """Chunks in -> windows -> events -> actions. Time base = sample time, so replay is deterministic."""

    def __init__(self, profile: Profile, mapping, backend, state: State, executing: bool, fast=False,
                 log_path=None, log_probs=False, cursor_backend=None, events_file=None, report_all=False):
        self.p, self.map, self.st = profile, mapping, state
        self.fast = fast
        self.report_all = report_all
        self.events_f = open(events_file, 'w', encoding='utf-8', buffering=1) if events_file else None
        self.executing = executing
        self.backend = backend
        self.exec = ActionExecutor(backend)
        self.cursor = CursorController(cursor_backend or backend, mapping.cursor)
        self.logic = EventLogic(profile.classes, apply_mapping_overrides(profile.params, mapping))
        self.scorer: WindowScorer | None = None
        self.t_first = None
        self.warmup = float(mapping.safety.get("warmup_s", 6))
        self.max_epm = int(mapping.safety.get("max_events_per_min", 40))
        self.recent = deque()
        self.fired = []               # all logic events (Fired), for replay scoring
        self.executed = []            # (t, gesture, event name)
        self.n_windows = 0
        self.log_probs = log_probs
        self.state_classes = profile.classes
        state.classes = list(profile.classes)
        self._log = None
        if log_path:
            Path(log_path).parent.mkdir(exist_ok=True)
            self._log_f = open(log_path, "w", newline="", encoding="utf-8")
            self._log = csv.writer(self._log_f)
            self._log.writerow(["wall", "t", "gesture", "p", "event", "action", "executed", "paused"])
        self._head_idx = [i for i, c in enumerate(profile.classes) if c in HEAD_CLASSES]
        self.clock = (lambda: self._last_t) if fast else time.monotonic
        self._last_t = 0.0
        self._cursor_t = None

    # -- control
    def set_paused(self, paused: bool):
        self.st.paused = paused
        self.cursor.enabled = not paused
        if paused:
            self.cursor.target[:] = 0
            self.cursor.vel[:] = 0

    def toggle_pause(self):
        self.set_paused(not self.st.paused)

    # -- data in
    def feed(self, x, ts0=None):
        if self.scorer is None:
            self.t_first = float(ts0 or 0.0)
            self.scorer = WindowScorer(self.p, t0=self.t_first)
        for r in self.scorer.push(x):
            self.on_window(r)

    def on_window(self, r):
        self.n_windows += 1
        self._last_t = r.t
        st = self.st
        st.probs = r.probs
        top = int(np.argmax(r.probs))
        st.top = self.p.classes[top]
        if r.t - self.t_first < self.warmup:
            return
        now = self.clock()
        p_head = float(r.probs[self._head_idx].sum()) if self._head_idx else 0.0
        if not st.paused:
            self.cursor.update(r.cursor[0], r.cursor[1], p_head, now)
        if self.fast:                                   # replay without wall-clock: integrate per window
            self.cursor.step(now, self.p.step_s)
        for f in self.logic.step(r.t, r.probs):
            self.fired.append(f)
            self.handle(f, now)

    def handle(self, f, now):
        name, ev = self.map.event_for(f.cls)
        executed = False
        action = ev["action"] if ev else ""
        if self.report_all and not self.st.paused and (ev is None or action == "cursor"):
            self._say(f"{f.cls} detected" + (" (cursor gesture)" if ev else " (unmapped)"), f, name)
        if ev is not None and action != "cursor":
            if action == "pause_toggle":
                self.toggle_pause()
                self._say(f"{f.cls} -> {'PAUSED' if self.st.paused else 'RESUMED'}", f, name)
                executed = True
            elif not self.st.paused:
                self.recent.append(now)
                while self.recent and now - self.recent[0] > 60:
                    self.recent.popleft()
                if len(self.recent) > self.max_epm:
                    self.set_paused(True)
                    self._say(f"RATE LIMIT ({self.max_epm}/min): paused", f, name)
                else:
                    if self.executing:
                        self.exec.do(ev)
                    self.cursor.freeze(now)
                    self.executed.append((f.t, f.cls, name))
                    executed = True
                    self._say(f"{f.cls} -> {name}", f, name)
        if self._log:
            self._log.writerow([f"{time.time():.3f}", f"{f.t:.3f}", f.cls, f"{f.p:.3f}", name or "", action,
                                int(executed), int(self.st.paused)])

    def _say(self, text, f, name):
        self.st.last_event, self.st.last_event_wall = text, time.time()
        self.st.n_events += 1
        line = f"[{f.t:8.2f}s] {text}  (p={f.p:.2f})"
        print(line, flush=True)
        if self.events_f:
            self.events_f.write(time.strftime("%H:%M:%S") + "  " + line + "\n")

    def close(self):
        if self.events_f:
            self.events_f.close()
        if self._log:
            self._log_f.close()


# ------------------------------------------------------------------- sources
def run_live(engine: Engine, cfg, profile, st: State, units_to_uv):
    import pylsl
    from lsl_utils import describe_stream, effective_channel_names, print_stream_info, resolve_stream
    ecfg = cfg["eeg_stream"]
    info = resolve_stream(ecfg.get("type", "EEG"), ecfg.get("name"), ecfg.get("resolve_timeout_s", 8))
    if info is None:
        st.message = "no EEG stream found"
        raise SystemExit("No LSL EEG stream found. Start the amplifier's LSL output (or python src/synthetic_eeg.py).")
    inlet = pylsl.StreamInlet(info, max_buflen=30, processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter)
    try:
        info = inlet.info(5.0)                  # full description (channel labels / units live here)
    except Exception:
        pass
    d = describe_stream(info)
    print_stream_info(d, cfg)
    from channels import ChannelMap, detect_live
    # auto-detect which channels are running: collect ~2.5 s first, then name/select channels
    pre, n_pre, t_pre = [], 0, time.time()
    while n_pre < int(2.5 * d["srate"]) and time.time() - t_pre < 15 and not st.stop:
        x, ts = inlet.pull_chunk(timeout=0.2, max_samples=2048)
        if ts:
            pre.append((np.asarray(x, dtype=np.float64), ts[0]))
            n_pre += len(ts)
    live = detect_live(np.concatenate([p[0] for p in pre])) if pre else None
    try:
        cmap = ChannelMap(d["labels"] or [f"ch{i + 1}" for i in range(d["n_channels"])], cfg, live=live, wanted=list(profile.ch_names))
    except ValueError as e:
        raise SystemExit(str(e))
    for note in cmap.notes:
        print("Channels:", note)
    print("Channels: " + cmap.describe() + f" (source names: {', '.join(map(str, cmap.source_names))})")
    if cmap.problems:
        raise SystemExit("Electrodes not running (flat/zero): " + ", ".join(cmap.problems))
    check_stream_matches(profile, cmap.names, d["srate"])
    for x, t0 in pre:
        engine.feed(cmap.apply(x) * units_to_uv, t0)
    st.source = "EEG LSL"
    st.mode = "RUNNING"
    st.message = ""
    last_data = time.time()
    while not st.stop:
        x, ts = inlet.pull_chunk(timeout=0.1, max_samples=1024)
        if not ts and time.time() - last_data > 2.0:
            st.message = f"NO EEG DATA for {time.time() - last_data:.0f} s (stream stalled?)"
        if ts:
            last_data = time.time()
            if st.message.startswith("NO EEG"):
                st.message = ""
            engine.feed(cmap.apply(np.asarray(x, dtype=np.float64)) * units_to_uv, ts[0])
            st.lag_s = max(0.0, pylsl.local_clock() - ts[-1])
            if st.lag_s > 1.5:
                st.message = f"FALLING BEHIND ({st.lag_s:.1f}s)"
            elif st.message.startswith("FALLING"):
                st.message = ""


def run_replay(engine: Engine, rec, st: State, speed: float, fast: bool, tail: float):
    n0 = 0
    if tail < 1.0:
        n0 = int(len(rec.data) * (1 - tail))
    data, ts = rec.data[n0:], rec.ts[n0:]
    st.source = "REPLAY"
    st.mode = "RUNNING"
    chunk = int((5.0 if fast else 0.1) * rec.fs)
    t_wall0 = time.perf_counter()
    for i in range(0, len(data), chunk):
        if st.stop:
            break
        engine.feed(data[i:i + chunk].astype(np.float64), ts[i] if i == 0 else None)
        if not fast:
            due = (i + chunk) / rec.fs / speed
            sl = due - (time.perf_counter() - t_wall0)
            if sl > 0:
                time.sleep(sl)
    return n0


# ---------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="default", help="profiles/<name>.joblib")
    ap.add_argument("--mapping", default=None, help="mapping file (default config/mapping.yaml)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true", help="live EEG but never touch the OS (log only)")
    ap.add_argument("--replay", help="session folder to replay instead of live EEG")
    ap.add_argument("--eeg", help="EEG file for the replayed session (.cnt/.xdf/.npz)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (1 = real time)")
    ap.add_argument("--fast", action="store_true", help="replay as fast as possible (no pacing, no wall clock)")
    ap.add_argument("--tail", type=float, default=1.0, help="replay only the last fraction of the recording (e.g. 0.2)")
    ap.add_argument("--live-actions", action="store_true", help="DANGEROUS: let a replay drive the real mouse/keyboard")
    ap.add_argument("--no-ui", action="store_true", help="no overlay window")
    ap.add_argument("--no-hotkeys", action="store_true")
    ap.add_argument("--start-paused", action="store_true")
    ap.add_argument("--events-file", default=None, help="also write every reported event to this text file")
    ap.add_argument("--report-all", action="store_true", help="report every detected gesture, incl. cursor/unmapped ones")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop automatically after N seconds (testing)")
    ap.add_argument("--log-dir", default=str(LOG_DIR))
    args = ap.parse_args(argv)

    enable_dpi_awareness()                    # before Tk / pynput so coordinates are physical pixels
    cfg = load_config(args.config)
    profile = Profile.load(PROFILE_DIR / f"{args.profile}.joblib")
    mapping = load_mapping(args.mapping, known_gestures=set(profile.gestures))
    replay = args.replay is not None
    executing = (not args.dry_run) and ((not replay) or args.live_actions)
    st = State(dry_run=not executing)
    backend = PynputBackend() if executing else VirtualBackend()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    engine = Engine(profile, mapping, backend, st, executing, fast=args.fast and replay,
                    log_path=Path(args.log_dir) / f"rt_{stamp}.csv",
                    events_file=args.events_file, report_all=args.report_all)
    if args.start_paused:
        engine.set_paused(True)

    print(f"Profile '{args.profile}': {profile.kind}, classes {profile.classes}")
    print(f"Mode: {'LIVE OS ACTIONS' if executing else 'DRY RUN (nothing is sent to the OS)'}; "
          f"kill hotkey {mapping.safety.get('kill_hotkey')}, pause hotkey {mapping.safety.get('pause_hotkey')}")
    if profile.meta.get("data") and replay and str(profile.meta["data"]).split("@")[0].replace("\\", "/").rstrip("/") == \
            str(args.replay).replace("\\", "/").rstrip("/"):
        print("NOTE: this profile was trained on the replayed session, so the replay score is optimistic "
              "(train.py's TEST numbers are the honest ones). Replay session 1 or new data for an unseen test.")
    if executing:
        try:
            import ctypes
            print(f"Screen: {ctypes.windll.user32.GetSystemMetrics(0)}x{ctypes.windll.user32.GetSystemMetrics(1)} physical px")
        except Exception:
            pass

    def kill():
        print("\nKILL hotkey pressed - stopping.", flush=True)
        st.stop = True
        threading.Timer(2.5, lambda: os._exit(0)).start()      # failsafe if something is stuck

    listeners = []
    if not args.no_hotkeys:
        h = start_hotkeys({mapping.safety.get("kill_hotkey", "<ctrl>+<alt>+q"): kill,
                           mapping.safety.get("pause_hotkey", "<ctrl>+<alt>+p"): engine.toggle_pause})
        listeners.append(h)

    rec = None
    n0 = 0
    if replay:
        from load_data import load_recording
        rec = load_recording(args.replay, units_to_uv=cfg["eeg_stream"].get("units_to_uv", 1.0),
                             eeg_file=args.eeg, verbose=False)
        from channels import apply_to_recording
        apply_to_recording(rec, cfg)
        check_stream_matches(profile, rec.ch_names, rec.fs)

    def cursor_loop():                           # ~100 Hz cursor integration (wall clock) for live / paced replay
        last = time.monotonic()
        while not st.stop:
            time.sleep(0.01)
            now = time.monotonic()
            engine.cursor.step(now, now - last)
            last = now

    def worker():
        nonlocal n0
        try:
            if replay:
                n0 = run_replay(engine, rec, st, args.speed, args.fast, args.tail)
            else:
                run_live(engine, cfg, profile, st, cfg["eeg_stream"].get("units_to_uv", 1.0))
        except SystemExit as e:
            print(e)
        except Exception as e:
            import traceback
            traceback.print_exc()
            st.message = repr(e)[:60]
        finally:
            if replay:
                st.stop = True                   # replay finished: close the overlay

    if not (replay and args.fast):
        threading.Thread(target=cursor_loop, daemon=True).start()
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    if args.max_seconds:
        threading.Timer(args.max_seconds, lambda: setattr(st, "stop", True)).start()
    try:
        if args.no_ui:
            while th.is_alive() and not st.stop:
                time.sleep(0.1)
        else:
            Overlay(st, show_bars=True).run()
            st.stop = True
    except KeyboardInterrupt:
        st.stop = True
    th.join(timeout=3)
    st.mode = "STOPPED"
    for h in listeners:
        try:
            h and h.stop()
        except Exception:
            pass
    engine.close()

    from collections import Counter
    cnt = Counter(f.cls for f in engine.fired)
    print("\nDetected gestures: " + (", ".join(f"{k} x{v}" for k, v in sorted(cnt.items())) or "none"))
    print(f"\nWindows scored: {engine.n_windows}; events fired: {len(engine.fired)}; executed: {len(engine.executed)}")
    if isinstance(backend, VirtualBackend):
        print(f"Virtual cursor moved to ({backend.x:.0f}, {backend.y:.0f}) px; actions: "
              f"{len(backend.actions())}")
    if replay and rec is not None and engine.fired:
        ev = [e for e in rec.events]
        if n0:
            tcut = rec.ts[n0]
            ev = [e for e in ev if e["t"] >= tcut]
        gest = [g for g in profile.gestures]
        sc = score_events(engine.fired, ev, gest)
        print("\n" + format_score(sc, "=== Replay scored against the cue markers (all gestures, before mapping) ==="))
    print(f"Log: {Path(args.log_dir) / ('rt_' + stamp + '.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

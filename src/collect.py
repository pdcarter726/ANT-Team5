"""Cue-driven data collection app (Tkinter).

  python src/collect.py --session 1
  python src/collect.py --session 1 --dry-run          # print schedule + duration estimate only
  python src/collect.py --session 1 --no-eeg           # UI only, no LSL EEG needed
  python src/collect.py --session 1 --cue-only         # EEG recorded on another device (adds sync clenches)

Flow: signal check -> gate (start LabRecorder) -> countdown -> trials -> done.
Outputs in data/s<N>_<timestamp>/ : markers.csv, eeg_backup.npz, metadata.json
(LabRecorder's .xdf is saved by LabRecorder; put it in the same folder).
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import tkinter as tk
import traceback
from datetime import datetime
from pathlib import Path

from common import DATA_DIR, enable_dpi_awareness, load_config, session_cfg
from protocol import (restrict_classes, SYNC_FLASH_S, SYNC_PULSES, SYNC_READY_S, SYNC_TOTAL_S, build_trials,
                      estimate_duration, warn_if_long)

COL = dict(gap="#17191d", cue="#0b4f9c", brk="#3d2e00", pause="#3d1212", done="#0f3d1a", pre="#101216")
FONT = "Segoe UI"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Cue-driven EEG data collection")
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--trials", type=int, default=None, help="override trials_per_class")
    ap.add_argument("--classes", default=None,
                    help="comma list: cue only these gestures (plus rest), e.g. jaw_clench,double_blink (targeted recalibration)")
    ap.add_argument("--seed", type=int, default=None, help="shuffle seed (default: random, stored in metadata)")
    ap.add_argument("--subject", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--cap", default="", help="cap/electrode comments")
    ap.add_argument("--out", default=None, help="output folder (default data/s<N>_<time>)")
    ap.add_argument("--no-eeg", action="store_true", help="run the cue UI without an EEG stream")
    ap.add_argument("--cue-only", action="store_true",
                    help="EEG is recorded on ANOTHER device: just show cues, log times, add sync clenches "
                         "at start and end (implies --no-eeg, no LSL)")
    ap.add_argument("--lsl-markers", action="store_true",
                    help="with --cue-only: also broadcast CueMarkers on LSL (for a networked LabRecorder)")
    ap.add_argument("--skip-check", action="store_true", help="skip the signal-check screen")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--auto-start", action="store_true", help="(testing) skip the ENTER at the gate screen")
    ap.add_argument("--topmost", action="store_true", help="(testing) keep window above others")
    ap.add_argument("--cue-s", type=float, default=None, help="(testing) override cue_s")
    ap.add_argument("--rest-s", type=float, default=None, help="(testing) override rest_s")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def fmt_t(s: float) -> str:
    s = max(0, int(round(s)))
    return f"{s // 60:d}:{s % 60:02d}"


class Collector:
    def __init__(self, args, cfg):
        self.args, self.cfg = args, cfg
        self.s = session_cfg(cfg, args.session)
        self.seed = args.seed if args.seed is not None else int(time.time()) % 100000
        self.trials = build_trials(cfg, args.session, seed=self.seed, trials_override=args.trials)
        self.cue_only = bool(args.cue_only)
        self.est = estimate_duration(self.trials, self.s, self.cue_only)
        self.n_blocks = self.est["n_blocks"]
        self.sync_kind = "start"
        self.sync_k = 0
        self.cd_next = "sync" if self.cue_only else "gap"    # what follows the countdown phase
        self.cd_msg = ""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out = Path(args.out) if args.out else DATA_DIR / f"s{args.session}_{ts}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.started_iso = datetime.now().isoformat(timespec="seconds")
        self.stream_desc = None
        self.ch_names: list[str] = []
        self.rec = None
        self.status = "created"
        self.ti = 0
        self.phase = "pre"
        self.phase_end = 0.0
        self.phase_start = 0.0
        self.pause_req = False
        self.go = False
        self.esc_t = 0.0
        self.t_session = 0.0
        self.done_trials = 0
        self.rec_result: dict = {}
        self.log_errors: list[str] = []

    # ---------------------------------------------------------------- setup
    def connect_eeg(self):
        from lsl_utils import (EegRecorder, describe_stream, effective_channel_names,
                               print_stream_info, resolve_stream)
        e = self.cfg["eeg_stream"]
        print(f"Looking for LSL stream type={e['type']} name={e.get('name')} ...")
        info = resolve_stream(e["type"], e.get("name"), e.get("resolve_timeout_s", 8))
        if info is None:
            raise SystemExit("No LSL EEG stream found. Enable LSL in the eego software, or run "
                             "`python src/synthetic_eeg.py`, or use --no-eeg for a UI-only run.")
        self.stream_desc = describe_stream(info)
        self.stream_warnings = print_stream_info(self.stream_desc, self.cfg)
        self.ch_names = effective_channel_names(self.stream_desc, self.cfg)
        self.rec = EegRecorder(info, units_to_uv=e.get("units_to_uv", 1.0), segment_dir=self.out)
        self.rec.start()

    def write_meta(self, status=None):
        if status:
            self.status = status
        a = self.args
        cfg_copy = {k: v for k, v in self.cfg.items() if k != "_path"}
        meta = dict(
            session=a.session, status=self.status, cue_only=self.cue_only, subject=a.subject, notes=a.notes, cap_comments=a.cap,
            started=self.started_iso, written=datetime.now().isoformat(timespec="seconds"),
            seed=self.seed, estimate=self.est, trials_done=self.done_trials, n_trials=len(self.trials),
            config_file=self.cfg.get("_path"), config=cfg_copy, session_config=self.s,
            stream=self.stream_desc, channel_names=self.ch_names, recording=self.rec_result,
            trials=[t.to_dict() for t in self.trials], argv=sys.argv,
            python=sys.version, platform=platform.platform(),
            files=dict(markers="markers.csv", eeg_backup="eeg_backup.npz", xdf="(saved by LabRecorder)"),
            errors=self.log_errors)
        tmp = self.out / "metadata.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        tmp.replace(self.out / "metadata.json")

    # ------------------------------------------------------------------ UI
    def build_ui(self):
        enable_dpi_awareness()
        self.root = tk.Tk()
        self.root.title(f"Collect - session {self.args.session}")
        self.root.geometry("1280x760")
        if self.args.fullscreen:
            self.root.attributes("-fullscreen", True)
        if self.args.topmost:
            self.root.attributes("-topmost", True)
        self.root.configure(bg=COL["pre"])
        self.cv = tk.Canvas(self.root, bg=COL["pre"], highlightthickness=0)
        self.hdr = self.cv.create_text(0, 0, anchor="n", fill="#cfd3da")
        self.big = self.cv.create_text(0, 0, anchor="center", fill="white", justify="center")
        self.sub = self.cv.create_text(0, 0, anchor="center", fill="#cfd3da", justify="center")
        self.nxt = self.cv.create_text(0, 0, anchor="center", fill="#ffd24d", justify="center")
        self.bar_bg = self.cv.create_rectangle(0, 0, 0, 0, fill="#2a2d33", outline="")
        self.bar = self.cv.create_rectangle(0, 0, 0, 0, fill="#ffd24d", outline="")
        self.foot = self.cv.create_text(0, 0, anchor="s", fill="#8a8f98")
        self.cv.bind("<Configure>", self.on_resize)
        self.root.bind("<Key>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", lambda: self.finish("aborted"))
        self.root.bind("<F11>", lambda e: self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen")))
        self.check = None
        if self.rec is not None and not self.args.skip_check:
            from signal_check import SignalCheck
            self.check = SignalCheck(self.root, self.rec, self.ch_names, self.cfg, on_done=self.to_gate)
            self.check.pack(fill="both", expand=True)
            self.check.start()
            self.phase = "check"
        else:
            self.to_gate()
        self.tick()

    def layout(self):
        W, H = max(self.cv.winfo_width(), 300), max(self.cv.winfo_height(), 200)
        self.W, self.H = W, H
        self.cv.coords(self.hdr, W / 2, H * 0.03)
        self.cv.itemconfigure(self.hdr, font=(FONT, -int(H * 0.035)), width=W * 0.95)
        self.cv.coords(self.big, W / 2, H * 0.40)
        self.cv.itemconfigure(self.big, font=(FONT, -int(H * 0.115), "bold"), width=W * 0.92)
        self.cv.coords(self.sub, W / 2, H * 0.60)
        self.cv.itemconfigure(self.sub, font=(FONT, -int(H * 0.04)), width=W * 0.85)
        self.cv.coords(self.nxt, W / 2, H * 0.72)
        self.cv.itemconfigure(self.nxt, font=(FONT, -int(H * 0.05), "bold"), width=W * 0.9)
        self.cv.coords(self.foot, W / 2, H * 0.985)
        self.cv.itemconfigure(self.foot, font=(FONT, -int(H * 0.028)), width=W * 0.95)
        self.bar_geom = (W * 0.2, H * 0.83, W * 0.8, H * 0.86)

    def show(self, bg, big="", sub="", nxt="", foot="", frac=None, big_color="white"):
        self.cv.configure(bg=bg)
        self.cv.itemconfigure(self.big, text=big, fill=big_color)
        self.cv.itemconfigure(self.sub, text=sub)
        self.cv.itemconfigure(self.nxt, text=nxt)
        self.cv.itemconfigure(self.foot, text=foot)
        x0, y0, x1, y1 = self.bar_geom
        if frac is None:
            self.cv.coords(self.bar_bg, 0, 0, 0, 0)
            self.cv.coords(self.bar, 0, 0, 0, 0)
        else:
            self.cv.coords(self.bar_bg, x0, y0, x1, y1)
            self.cv.coords(self.bar, x0, y0, x0 + (x1 - x0) * max(0.0, min(1.0, frac)), y1)

    def header(self):
        t = self.trials[min(self.ti, len(self.trials) - 1)]
        el = time.perf_counter() - self.t_session if self.t_session else 0
        rem_trials = sum(x.cue_s + x.rest_s for x in self.trials[self.ti:])
        txt = (f"Trial {min(self.ti + 1, len(self.trials))}/{len(self.trials)}    "
               f"Block {t.block + 1}/{self.n_blocks}    Elapsed {fmt_t(el)}    ~{fmt_t(rem_trials)} left")
        if self.rec is not None and self.rec.recording and time.time() - self.rec.last_sample_wall > 1.0:
            txt = "!!! NO EEG DATA FOR >1 s !!!    " + txt
        self.cv.itemconfigure(self.hdr, text=txt, fill="#ff6b6b" if txt.startswith("!!!") else "#cfd3da")

    # ---------------------------------------------------------------- phases
    def to_gate(self):
        if self.check:
            self.check.stop()
            self.check.pack_forget()
            self.check = None
        self.cv.pack(fill="both", expand=True)
        self.root.focus_force()
        self.phase = "gate"
        self.root.update_idletasks()
        self.draw_gate()
        if self.args.auto_start:
            self.root.after(500, self.begin_session)

    def on_resize(self, _e=None):
        """The first layout can run before the window has a real size; redo it on every resize."""
        self.layout()
        if self.phase == "gate":
            self.draw_gate()

    def draw_gate(self):
        self.layout()
        warn = warn_if_long(self.est, self.s)
        if self.cue_only:
            how = ("\n\nSTART RECORDING ON THE OTHER DEVICE FIRST, then press ENTER.\n"
                   "You will be asked to clench your jaw hard on 4 flashes at the start and at the end\n"
                   "(used to line up the two clocks). Afterwards copy the recording into:\n" + str(self.out))
        else:
            how = ("\n\nIn LabRecorder: click Update, tick the EEG stream and CueMarkers, press Start."
                   f"\nSave the .xdf into  {self.out}")
        self.show(COL["pre"], f"SESSION {self.args.session}",
                  f"{len(self.trials)} trials, about {self.est['total_min']:.1f} min."
                  + (f"\n{warn}" if warn else "") + how,
                  "", "ENTER = begin countdown     ESC x2 = quit")
        self.cv.itemconfigure(self.big, font=(FONT, -int(self.H * 0.08), "bold"))
        self.cv.itemconfigure(self.hdr, text="")

    def begin_session(self):
        from lsl_utils import MarkerSender
        if self.rec is not None:
            self.rec.start_recording()
        self.t_session = time.perf_counter()
        self.markers.reset_clock()
        self.markers.send("session_start", label="", notes=self.args.notes)
        self.write_meta("running")
        self.set_phase("countdown", float(self.s.get("start_countdown_s", 5)))

    def set_phase(self, name, dur=None):
        self.phase = name
        self.phase_start = time.perf_counter()
        self.phase_end = self.phase_start + (dur or 0)
        self.go = False
        self.layout()
        self.cv.itemconfigure(self.big, font=(FONT, -int(self.H * 0.115), "bold"))

    def start_gap(self):
        self.set_phase("gap", self.trials[self.ti].rest_s)

    def start_sync(self, kind):
        self.sync_kind, self.sync_k = kind, 0
        self.set_phase("sync", SYNC_TOTAL_S)

    def start_cue(self):
        t = self.trials[self.ti]
        self.set_phase("cue", t.cue_s)
        self.show(COL["cue"], t.display, t.hint, "", self.footer(), 1.0)
        self.header()
        self.root.update_idletasks()
        self.markers.send("cue_on", t)          # marker right after the screen is drawn

    def end_cue(self):
        t = self.trials[self.ti]
        self.markers.send("cue_off", t)
        self.done_trials += 1
        self.ti += 1
        if self.ti >= len(self.trials):
            return self.start_sync("end") if self.cue_only else self.finish("complete")
        every = int(self.s.get("break_every_blocks") or 0)
        nt = self.trials[self.ti]
        if every > 0 and nt.block != t.block and nt.block % every == 0:
            self.markers.send("break_start", label="")
            self.write_meta()
            return self.set_phase("break", float(self.s.get("break_min_s", 10)))
        if self.pause_req:
            self.pause_req = False
            self.markers.send("pause", label="")
            return self.set_phase("paused")
        self.start_gap()

    def footer(self):
        base = "SPACE = pause    ESC x2 = abort    F11 = fullscreen"
        return base + ("    [pause after this trial]" if self.pause_req else "")

    def tick(self):
        try:
            if self.phase not in ("check", "closed"):
                self.step()
        except Exception:
            tb = traceback.format_exc()
            self.log_errors.append(tb)
            print(tb, file=sys.stderr)
        if self.phase != "closed":
            self.root.after(8, self.tick)

    def step(self):
        now = time.perf_counter()
        ph = self.phase
        if ph == "gate" or ph == "done":
            return
        rem = self.phase_end - now
        dur = max(self.phase_end - self.phase_start, 1e-6)
        if ph == "countdown":
            self.show(COL["pre"], str(max(1, math.ceil(rem))),
                      self.cd_msg or "Get ready. Sit still, eyes on the screen.", "", self.footer(), rem / dur)
            self.cv.itemconfigure(self.hdr, text="Starting ...")
            if rem <= 0:
                if self.cd_next == "sync":
                    self.start_sync("start")
                else:
                    self.start_gap()
        elif ph == "sync":
            el = now - self.phase_start
            active = next((k for k, p in enumerate(SYNC_PULSES) if p <= el < p + SYNC_FLASH_S), None)
            if active is not None:
                self.show("#ffffff", "CLENCH!", "", "", "", None, big_color="#000000")
                if active >= self.sync_k:                       # new flash: log it right after drawing
                    self.root.update_idletasks()
                    self.markers.send("sync_pulse", label=f"sync_{self.sync_kind}", pulse=active)
                    self.sync_k = active + 1
            else:
                nxt = next((p for p in SYNC_PULSES if p > el), None)
                if nxt is None:
                    self.show(COL["pre"], "SYNC DONE", "Relax.", "", self.footer(), None)
                else:
                    secs = math.ceil(nxt - el)
                    self.show(COL["pre"], f"NEXT FLASH IN {secs}" if secs > 1 else "GET READY ...",
                              "Clench your jaw HARD for as long as the screen is white.\n"
                              f"({self.sync_kind} sync: lines this screen up with the other recorder)",
                              "", self.footer(), None, big_color="#ffd24d" if secs <= 1 else "white")
                self.cv.itemconfigure(self.big, font=(FONT, -int(self.H * 0.09), "bold"))
            self.cv.itemconfigure(self.hdr, text=f"Sync ({self.sync_kind}) {self.sync_k}/{len(SYNC_PULSES)}")
            if rem <= 0:
                if self.sync_kind == "start":      # breathing space before the first trial
                    self.cd_next, self.cd_msg = "gap", "Sync done. Trials start now: first cue is coming."
                    self.set_phase("countdown", SYNC_READY_S)
                else:
                    self.finish("complete")
        elif ph == "gap":
            t = self.trials[self.ti]
            self.show(COL["gap"], "RETURN TO CENTER & RELAX", "", f"Next:  {t.display}",
                      self.footer(), None, big_color="#9aa0aa")
            self.cv.itemconfigure(self.big, font=(FONT, -int(self.H * 0.07), "bold"))
            self.cv.itemconfigure(self.sub, text=t.hint)
            self.header()
            if rem <= 0:
                self.start_cue()
        elif ph == "cue":
            t = self.trials[self.ti]
            self.cv.coords(self.bar, self.bar_geom[0], self.bar_geom[1],
                           self.bar_geom[0] + (self.bar_geom[2] - self.bar_geom[0]) * max(0, rem) / dur,
                           self.bar_geom[3])
            self.cv.itemconfigure(self.foot, text=self.footer())
            self.header()
            if rem <= 0:
                self.end_cue()
        elif ph == "break":
            can = rem <= 0
            self.show(COL["brk"], "BREAK", "Relax, move, drink some water.\n"
                      + ("Press SPACE when ready to continue." if can else f"Continue available in {math.ceil(rem)} s"),
                      "", "ESC x2 = abort", None)
            self.header()
            if can and self.go:
                self.markers.send("break_end", label="")
                self.start_gap()
        elif ph == "paused":
            self.show(COL["pause"], "PAUSED", "Press SPACE to resume.", "", "ESC x2 = abort", None)
            self.header()
            if self.go:
                self.markers.send("resume", label="")
                self.start_gap()

    # ----------------------------------------------------------------- input
    def on_key(self, e):
        k = e.keysym
        if self.phase == "check":
            if self.check:
                self.check.key(e)
            if k == "Escape":
                self.finish("aborted")
            return
        if k == "Escape":
            now = time.time()
            if now - self.esc_t < 2.0:
                return self.finish("aborted")
            self.esc_t = now
            self.cv.itemconfigure(self.foot, text="Press ESC again to abort")
        elif k == "Return" and self.phase == "gate":
            self.begin_session()
        elif k == "space":
            if self.phase in ("break", "paused"):
                self.go = True
            elif self.phase == "gap":
                self.markers.send("pause", label="")
                self.set_phase("paused")
            elif self.phase in ("cue", "countdown"):
                self.pause_req = not self.pause_req
        elif self.phase == "done" and k in ("Return", "q"):
            self.close()

    # ---------------------------------------------------------------- finish
    def finish(self, status):
        if self.phase in ("done", "closed"):
            return self.close() if self.phase == "done" else None
        was_running = self.phase not in ("pre", "check", "gate")
        try:
            if self.check:
                self.check.stop()
            if was_running:
                self.markers.send("session_end", label="", status=status)
        except Exception:
            self.log_errors.append(traceback.format_exc())
        msg = ""
        try:
            if self.rec is not None:
                if was_running:
                    self.rec_result = self.rec.finalize(self.out / "eeg_backup.npz", self.ch_names,
                                                        dict(session=self.args.session))
                self.rec.stop()
                if self.rec.error:
                    self.log_errors.append("recorder: " + self.rec.error)
            self.markers.close()
        except Exception:
            self.log_errors.append(traceback.format_exc())
            msg = "\n(saving raised an error - see metadata.json 'errors'; eeg_part_*.npz files are kept)"
        try:
            self.write_meta(status if was_running else "aborted_before_start")
        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
        self.phase = "done"
        if self.check is None and getattr(self, "cv", None) is not None:
            self.cv.pack(fill="both", expand=True)
        n = self.rec_result.get("n_samples")
        self.layout()
        self.show(COL["done"], "DONE" if status == "complete" else status.upper(),
                  f"{self.done_trials}/{len(self.trials)} trials. "
                  + (f"{n} EEG samples ({self.rec_result.get('measured_srate', 0):.1f} Hz measured)." if n else "")
                  + f"\nSaved to {self.out}\n"
                  + ("Stop the recording on the other device, then copy it (e.g. the .xdf) into that folder."
                     if self.cue_only else "Stop the LabRecorder recording now.") + msg,
                  "", "ENTER = close")
        self.cv.itemconfigure(self.hdr, text="")
        print(f"[collect] finished: {status}; saved to {self.out}")
        if self.args.auto_start:
            self.root.after(800, self.close)

    def close(self):
        self.phase = "closed"
        try:
            self.root.destroy()
        except Exception:
            pass

    # ------------------------------------------------------------------- run
    def run(self):
        from lsl_utils import MarkerSender
        mk = self.cfg.get("markers", {})
        self.markers = MarkerSender(self.out / "markers.csv", self.args.session,
                                    mk.get("name", "CueMarkers"), mk.get("type", "Markers"),
                                    use_lsl=(not self.cue_only) or self.args.lsl_markers)
        if not self.args.no_eeg:
            self.connect_eeg()
        self.write_meta("created")
        self.build_ui()
        self.root.mainloop()


def main(argv=None):
    args = parse_args(argv)
    if args.cue_only:
        args.no_eeg = True
    cfg = load_config(args.config)
    if args.classes:
        restrict_classes(cfg, args.session, args.classes.split(","))
    s = session_cfg(cfg, args.session)
    if args.cue_s is not None:
        s["cue_s"] = args.cue_s
    if args.rest_s is not None:
        s["rest_s"] = args.rest_s
    seed = args.seed if args.seed is not None else 0
    trials = build_trials(cfg, args.session, seed=seed, trials_override=args.trials)
    est = estimate_duration(trials, s, args.cue_only)
    print(f"Session {args.session}: {est['n_trials']} trials in {est['n_blocks']} blocks, "
          f"{est['n_breaks']} breaks. Estimated duration {est['total_min']:.1f} min "
          f"(trials {est['trial_s'] / 60:.1f} + breaks {est['break_s'] / 60:.1f} + start {est['start_s']:.0f} s).")
    w = warn_if_long(est, s)
    if w:
        print(w)
    if args.dry_run:
        for t in trials[:30]:
            print(f"  {t.idx:4d} block {t.block:2d} {t.label:<18} cue {t.cue_s}s rest {t.rest_s}s")
        print("  ...")
        return 0
    Collector(args, cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

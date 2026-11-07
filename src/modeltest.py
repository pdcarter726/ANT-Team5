"""python modeltest : calibrate (about 5 min) on the live EEG stream, then listen and write detected events.

  python modeltest                       # calibrate with session 4 cues, then listen (dry run: nothing is sent to the OS)
  python modeltest --skip-calibration    # listen straight away with the profile given by --base
  python modeltest --synthetic           # use a fake EEG stream (pipeline check, not a model check)

Steps
  1. finds the LSL EEG stream (fails early with a clear message if there is none)
  2. calibration: the cue app (collect.py --session 4, ~5 min: 6 trials per gesture, 2 of each confounder), signal check first
  3. refits the base profile with the calibration data (up-weighted) -> profiles/modeltest.joblib (base is untouched)
  4. listens live: every detected gesture is printed and written to logs/modeltest_<time>_events.txt
     (plus the full CSV log). Nothing is sent to the mouse/keyboard. Stop with Ctrl+Alt+Q or Ctrl+C.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from common import DATA_DIR, ROOT, load_config

LOG_DIR = ROOT / "logs"
PROFILE_DIR = ROOT / "profiles"


def find_stream(cfg, timeout=None):
    from lsl_utils import resolve_stream
    e = cfg["eeg_stream"]
    return resolve_stream(e.get("type", "EEG"), e.get("name"), timeout or e.get("resolve_timeout_s", 8))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="default", help="trained profile to calibrate (profiles/<name>.joblib)")
    ap.add_argument("--name", default="modeltest", help="calibrated profile name to save")
    ap.add_argument("--skip-calibration", action="store_true", help="listen with --base as is")
    ap.add_argument("--weight", type=float, default=2.0, help="weight of the calibration windows vs the base data")
    ap.add_argument("--synthetic", action="store_true", help="start a fake EEG stream (pipeline test)")
    ap.add_argument("--events-file", default=None)
    ap.add_argument("--no-ui", action="store_true", help="no overlay window")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop listening after N seconds (0 = until killed)")
    ap.add_argument("--collect-args", nargs=argparse.REMAINDER, default=[],
                    help="extra arguments for collect.py (put last), e.g. --collect-args --trials 2")
    a = ap.parse_args(argv)
    cfg = load_config()

    if not (PROFILE_DIR / f"{a.base}.joblib").exists():
        raise SystemExit(f"No trained profile profiles/{a.base}.joblib. Train one first:\n"
                         f"  .venv\\Scripts\\python.exe src\\train.py --data data\\s2_<timestamp>@data\\Session2Recording.cnt")

    synth = None
    try:
        if a.synthetic:
            print("Starting a fake EEG stream (this checks the pipeline, not the model).")
            synth = subprocess.Popen([sys.executable, str(ROOT / "src" / "synthetic_eeg.py"), "--seed", "1"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2)
        print("Looking for the LSL EEG stream ...")
        info = find_stream(cfg)
        if info is None:
            raise SystemExit("No LSL EEG stream found. Start the amplifier's LSL output first "
                             "(or run with --synthetic to test the pipeline).")
        print(f"Found '{info.name()}': {info.channel_count()} channels at {info.nominal_srate()} Hz.")

        profile = a.base
        if not a.skip_calibration:
            from calibrate import main as calibrate_main
            extra = list(a.collect_args)
            if a.synthetic and "--auto-start" not in extra:
                extra = ["--auto-start", "--skip-check"] + extra
            print("\n=== CALIBRATION (about 5 minutes) ===\n"
                  "Follow the cues on screen: each gesture 6 times, plus short talking / typing / chewing / idle / natural blinking.\n"
                  "Do the gestures the way you will in the demo. Keep the same posture and cap.\n")
            calibrate_main(["--session", "4", "--base", a.base, "--name", a.name, "--weight", str(a.weight),
                            "--subject", "modeltest"] + (["--collect-args"] + extra if extra else []))
            profile = a.name

        print(f"\n=== LISTENING with profile '{profile}' (dry run, nothing is sent to the OS) ===")
        LOG_DIR.mkdir(exist_ok=True)
        events_file = a.events_file or str(LOG_DIR / f"modeltest_{time.strftime('%Y%m%d_%H%M%S')}_events.txt")
        print(f"Events are printed here and written to {events_file}\nStop: Ctrl+Alt+Q (or Ctrl+C).\n")
        from realtime import main as realtime_main
        rt = ["--profile", profile, "--dry-run", "--report-all", "--events-file", events_file]
        if a.no_ui:
            rt.append("--no-ui")
        if a.max_seconds:
            rt += ["--max-seconds", str(a.max_seconds)]
        return realtime_main(rt)
    finally:
        if synth is not None:
            synth.terminate()


if __name__ == "__main__":
    sys.exit(main())

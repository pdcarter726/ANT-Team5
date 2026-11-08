"""Per-session calibration, started from the dashboard's "Start session" button (runs as its own process).

  python -m controller.session_calibrate [--subject NAME] [--session 5]

1. Records the cued calibration with the team's cue app (src/collect.py --session 5, about 10 minutes,
   signal check first). The EEG is read from the live LSL stream by the cue app itself.
2. Fits the model for this person: refits profiles/default.joblib with the new data up-weighted
   (src/train.py adapt) if a base profile exists, otherwise trains a profile from this session alone
   (src/train.py train).
3. Saves profiles/session.joblib, which the dashboard then loads.

Progress goes to stdout as lines "@@STATUS <phase> <message>" (phase: record, train, done, error), which the
dashboard reads. Exit code 0 = new profile saved; anything else = no change.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
PROFILES = REPO / "profiles"
OUT_NAME = "session"


def status(phase, msg):
    print(f"@@STATUS {phase} {msg}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", default="5", help="config session with the calibration cues (5 = about 10 min)")
    ap.add_argument("--subject", default="session")
    ap.add_argument("--base", default="default", help="base profile to adapt, if it exists")
    ap.add_argument("--weight", type=float, default=2.0, help="weight of this session vs the base data")
    ap.add_argument("--trials", type=int, help="override trials_per_class for the selected calibration duration")
    ap.add_argument("--from-folder", help="skip recording; use this finished session folder")
    ap.add_argument("--collect-args", nargs=argparse.REMAINDER, default=[], help="extra args for collect.py (put last)")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(SRC))
    t_start = time.time()

    if a.from_folder:
        folder = Path(a.from_folder)
    else:
        folder = REPO / "data" / f"s{a.session}_{time.strftime('%Y%m%d_%H%M%S')}"
        status("record", f"cue app started, saving to {folder.name}")
        # --auto-start skips the LabRecorder gate: the cue app records the EEG itself (eeg_backup.npz)
        cmd = [sys.executable, str(SRC / "collect.py"), "--session", str(a.session), "--subject", a.subject,
               "--out", str(folder), "--auto-start"] + list(a.collect_args)
        if a.trials is not None:
            cmd.extend(["--trials", str(a.trials)])
        rc = subprocess.call(cmd, cwd=str(REPO))
        meta = folder / "metadata.json"
        state = json.loads(meta.read_text(encoding="utf-8")).get("status") if meta.exists() else None
        if rc != 0 or state != "complete":
            status("error", f"calibration not finished (cue app status: {state or 'none'}); model unchanged")
            return 2
        if not (folder / "eeg_backup.npz").exists():
            status("error", "the cue app saved no EEG (is the LSL stream running?); model unchanged")
            return 3

    import argparse as _ap
    import train
    base = PROFILES / f"{a.base}.joblib"
    try:
        if base.exists() and (PROFILES / f"{a.base}_train.npz").exists():
            status("train", f"adapting profile '{a.base}' to this session")
            train.adapt(_ap.Namespace(adapt=str(folder), base=a.base, name=OUT_NAME,
                                      adapt_weight=a.weight, config=None))
        else:
            status("train", "training a new profile from this session (no base profile found)")
            train.train(_ap.Namespace(data=str(folder), extra=None, name=OUT_NAME, models=None, config=None))
    except SystemExit as e:
        status("error", f"training failed: {e}")
        return 4
    except Exception as e:
        status("error", f"training failed: {type(e).__name__}: {e}")
        return 4
    out = PROFILES / f"{OUT_NAME}.joblib"
    if not out.exists() or out.stat().st_mtime < t_start:
        status("error", "training finished but no profile was written")
        return 5
    status("done", f"profiles/{OUT_NAME}.joblib saved from {folder.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

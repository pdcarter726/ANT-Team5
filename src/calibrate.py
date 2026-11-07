"""~2 minute per-user calibration: record a short cued session from the live EEG stream, then refit the profile.

  python src/calibrate.py --base default --name default       # record (collect.py session 3) + adapt
  python src/calibrate.py --base default --from-folder data/s3_XXXX   # adapt from an existing recording only

The wearer follows the same cues as in training (3 trials per gesture + short talking / natural blinking).
The base profile's training windows are kept; the new windows are up-weighted (default x2) and the model and
cursor estimators are refit. Thresholds are kept from the base profile. EEG must be an LSL stream on this PC
(the cue app records it directly, so no sync clenches are needed).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import DATA_DIR, ROOT
import train


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="default")
    ap.add_argument("--session", default="3", help="config session with the calibration cues (3 = 2 min, 4 = 5 min)")
    ap.add_argument("--name", default=None, help="output profile name (default: overwrite --base)")
    ap.add_argument("--from-folder", help="skip recording; adapt from this session folder[@eegfile]")
    ap.add_argument("--subject", default="")
    ap.add_argument("--weight", type=float, default=2.0)
    ap.add_argument("--collect-args", nargs=argparse.REMAINDER, default=[], help="extra args passed to collect.py")
    a = ap.parse_args(argv)
    folder = a.from_folder
    if not folder:
        before = set(p.name for p in DATA_DIR.glob(f"s{a.session}_*"))
        cmd = [sys.executable, str(ROOT / "src" / "collect.py"), "--session", str(a.session), "--subject", a.subject or "calib"] + a.collect_args
        print("Starting the calibration cues. Keep the EEG stream running.")
        rc = subprocess.call(cmd)
        new = sorted(p for p in DATA_DIR.glob(f"s{a.session}_*") if p.name not in before)
        if rc != 0 or not new:
            raise SystemExit("Calibration recording did not complete; profile unchanged.")
        folder = str(new[-1])
    ns = argparse.Namespace(adapt=folder, base=a.base, name=a.name or a.base, adapt_weight=a.weight, config=None)
    train.adapt(ns)


if __name__ == "__main__":
    sys.exit(main())

"""Generate a full synthetic session on disk (no LSL, no UI, seconds instead of minutes).

Writes the same files collect.py + LabRecorder would: eeg_backup.npz, markers.csv, metadata.json
and (optionally) a .xdf. Use it to test the analysis / training / replay code before real data.

  python src/make_synthetic_session.py --session 1 --trials 25 --out data/synth_s1 --xdf
  python src/make_synthetic_session.py --session 2 --trials 20 --out data/synth_s2 --difficulty hard
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from common import load_config, session_cfg
from lsl_utils import CSV_FIELDS
from protocol import (SYNC_FLASH_S, SYNC_PULSES, SYNC_READY_S, SYNC_TOTAL_S, build_trials,
                      estimate_duration)
from synthetic_eeg import REACTION_S, SignalModel
from xdf_writer import write_xdf

DIFFICULTY = {"easy": (1.0, 1.0), "normal": (1.0, 0.7), "hard": (1.3, 0.45)}   # (noise, gesture scale)


def make_session(cfg, session, out, trials=None, seed=0, difficulty="normal", write_xdf_file=False,
                 fs=None, lsl_base=5000.0, cue_only=False, pre_roll=37.3, drift_ppm=0.0):
    """cue_only=True mimics collect.py --cue-only + a recording made on another device: the EEG clock is
    unrelated to the markers (t_rel = seconds since session start), the session starts `pre_roll` s into the
    recording, sync clenches are added at start and end, and the EEG clock can drift by drift_ppm."""
    s = session_cfg(cfg, session)
    fs = fs or cfg.get("sampling_rate", 500)
    noise, gscale = DIFFICULTY[difficulty]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tr = build_trials(cfg, session, seed=seed, trials_override=trials)
    est = estimate_duration(tr, s, cue_only)
    model = SignalModel(fs, seed=seed, noise_scale=noise, gesture_scale=gscale)
    rng = np.random.default_rng(seed + 1)

    t_start = pre_roll if cue_only else 2.0              # session_start, seconds since the recording began
    t = t_start + float(s.get("start_countdown_s", 5))
    every = int(s.get("break_every_blocks") or 0)
    marks = [dict(t=t_start, event="session_start", label="", trial=-1, block=-1, cue_s=0.0)]

    def sync(kind, t):
        for p in SYNC_PULSES:
            marks.append(dict(t=t + p, event="sync_pulse", label=f"sync_{kind}", trial=-1, block=-1, cue_s=0.0))
            model.schedule("jaw_clench", int(round((t + p + 0.3 + rng.uniform(-0.05, 0.1)) * fs)), SYNC_FLASH_S)
        return t + SYNC_TOTAL_S

    if cue_only:
        t = sync("start", t) + SYNC_READY_S
    for i, tt in enumerate(tr):
        if i > 0 and every and tt.block != tr[i - 1].block and tt.block % every == 0:
            t += float(s.get("break_min_s", 10)) + rng.uniform(0, 3)
        t += tt.rest_s
        marks.append(dict(t=t, event="cue_on", label=tt.label, trial=tt.idx, block=tt.block, cue_s=tt.cue_s))
        delay = REACTION_S + rng.uniform(0.0, 0.15)
        model.schedule(tt.label, int(round((t + delay) * fs)), tt.cue_s + 0.5)
        t += tt.cue_s
        marks.append(dict(t=t, event="cue_off", label=tt.label, trial=tt.idx, block=tt.block, cue_s=tt.cue_s))
    if cue_only:
        t = sync("end", t)
    t_end = t + 1.0
    marks.append(dict(t=t_end, event="session_end", label="", trial=-1, block=-1, cue_s=0.0))

    n = int(t_end * fs)
    data = np.concatenate([model.generate(min(5000, n - i)) for i in range(0, n, 5000)])
    if cue_only:      # the other device's clock: unrelated origin, optional drift
        ts = 91234.5 + np.arange(n) / fs * (1 + drift_ppm * 1e-6)
    else:
        ts = lsl_base + np.arange(n) / fs
    names = model.names
    np.savez(out / "eeg_backup.npz", data=data, timestamps=ts, ch_names=np.array(names), fs=fs, units="uV")
    wall0 = 1.7e9
    with open(out / "markers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for m in marks:
            cls = cfg["classes"].get(m["label"], {})
            w.writerow(dict(wall_unix=f"{wall0 + m['t']:.6f}",
                            wall_iso=datetime.fromtimestamp(wall0 + m["t"]).isoformat(timespec="milliseconds"),
                            lsl_time=f"{lsl_base + m['t']:.6f}", event=m["event"], session=session,
                            trial_id=m["trial"], block=m["block"], label=m["label"],
                            display=cls.get("display", ""), cue_s=m["cue_s"],
                            t_rel=f"{m['t'] - t_start:.6f}"))
    meta = dict(session=session, status="complete", synthetic=True, difficulty=difficulty, seed=seed,
                cue_only=cue_only, true_offset_s=(float(ts[0]) + t_start) if cue_only else None,   # EEG time = t_rel + offset
                estimate=est, channel_names=names, stream=dict(srate=fs), n_trials=len(tr),
                trials=[x.to_dict() for x in tr])
    (out / "metadata.json").write_text(json.dumps(meta, indent=1))
    if write_xdf_file:
        if cue_only:    # a separate device's XDF has no marker stream
            write_xdf(out / "session.xdf", data, ts, names, fs, [], [])
        else:
            strs = [json.dumps(dict(event=m["event"], session=session, trial=m["trial"], block=m["block"],
                                    label=m["label"], cue_s=m["cue_s"])) for m in marks]
            write_xdf(out / "session.xdf", data, ts, names, fs, strs, [lsl_base + m["t"] for m in marks])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", choices=list(DIFFICULTY), default="normal")
    ap.add_argument("--xdf", action="store_true", help="also write session.xdf")
    ap.add_argument("--config", default=None)
    ap.add_argument("--cue-only", action="store_true", help="simulate a recording made on another device")
    ap.add_argument("--drift-ppm", type=float, default=0.0, help="cue-only: EEG clock drift")
    a = ap.parse_args(argv)
    p = make_session(load_config(a.config), a.session, a.out, a.trials, a.seed, a.difficulty, a.xdf,
                     cue_only=a.cue_only, drift_ppm=a.drift_ppm)
    print("wrote", p)


if __name__ == "__main__":
    main()

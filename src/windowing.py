"""Per-trial windowing aligned to cue markers (pure numpy, no filtering)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Win:
    trial: int      # trial id from the marker (windows of one trial share this -> never split in CV)
    block: int
    label: str
    i0: int         # sample index range [i0, i1) into the recording
    i1: int
    t_start: float
    t_cue: float


def cue_events(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("event") == "cue_on"]


def trial_windows(ts: np.ndarray, events: list[dict], fs: float, win_s: float = 0.5,
                  step_s: float = 0.25, offset_s: float = 0.25,
                  labels: set[str] | None = None) -> tuple[list[Win], int]:
    """Sliding windows inside each cue period.

    Windows start at cue_on + offset_s (skips the reaction time) and must end before
    cue_on + cue_s. Trials whose window falls outside the recording or across a data gap
    are skipped; returns (windows, n_skipped_trials).
    """
    n = int(round(win_s * fs))
    out, skipped = [], 0
    for e in cue_events(events):
        if labels is not None and e["label"] not in labels:
            continue
        t0, cue = e["t"], float(e.get("cue_s", 1.5))
        starts = np.arange(t0 + offset_s, t0 + cue - win_s + 1e-9, step_s)
        trial_w = []
        for s in starts:
            i0 = int(np.searchsorted(ts, s))
            i1 = i0 + n
            if i1 > len(ts) or ts[i1 - 1] - ts[i0] > win_s + 3.0 / fs:
                trial_w = []
                break
            trial_w.append(Win(int(e["trial"]), int(e.get("block", -1)), e["label"], i0, i1, float(s), t0))
        if not trial_w:
            skipped += 1
        out += trial_w
    return out, skipped

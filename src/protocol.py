"""Trial schedule building and duration estimate (pure logic, no UI)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from common import session_cfg


@dataclass
class Trial:
    idx: int          # 0-based position in the whole session
    block: int        # 0-based block
    label: str
    display: str
    hint: str
    cue_s: float
    rest_s: float
    is_confounder: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def session_class_counts(cfg: dict, session: int, trials_override: int | None = None
                         ) -> list[tuple[str, int, bool]]:
    """Return [(label, n_trials, is_confounder)] for a session."""
    s = session_cfg(cfg, session)
    main = s.get("classes") or s.get("keep") or []
    if not main:
        raise ValueError(f"session {session}: no 'classes' or 'keep' list in config")
    n_main = trials_override if trials_override is not None else s["trials_per_class"]
    out = [(c, int(n_main), False) for c in main]
    n_conf = s.get("confounder_trials_per_class", 0)
    if trials_override is not None:
        n_conf = min(n_conf, trials_override)
    for c in s.get("confounders") or []:
        if n_conf > 0:
            out.append((c, int(n_conf), True))
    for c, _, _ in out:
        if c not in cfg["classes"]:
            raise KeyError(f"class '{c}' is not defined in config 'classes'")
    return out


def build_trials(cfg: dict, session: int, seed: int | None = None,
                 trials_override: int | None = None) -> list[Trial]:
    """Balanced randomized blocks.

    A block contains each main class once in random order. Confounder classes with
    fewer trials are spread evenly across blocks. A class never repeats back-to-back
    across a block boundary.
    """
    s = session_cfg(cfg, session)
    rng = np.random.default_rng(seed)
    counts = session_class_counts(cfg, session, trials_override)
    n_blocks = max(n for _, n, _ in counts)

    membership: dict[str, set[int]] = {}
    for label, n, _ in counts:
        if n >= n_blocks:
            membership[label] = set(range(n_blocks))
        else:
            membership[label] = set(np.unique(np.round(np.linspace(0, n_blocks - 1, n)).astype(int)))

    trials: list[Trial] = []
    prev_last = None
    conf = {l: c for l, _, c in counts}
    for b in range(n_blocks):
        labels = [l for l, _, _ in counts if b in membership[l]]
        rng.shuffle(labels)
        if prev_last is not None and len(labels) > 1 and labels[0] == prev_last:
            j = int(rng.integers(1, len(labels)))
            labels[0], labels[j] = labels[j], labels[0]
        for l in labels:
            cd = cfg["classes"][l]
            trials.append(Trial(
                idx=len(trials), block=b, label=l,
                display=cd.get("display", l.upper()), hint=cd.get("hint", ""),
                cue_s=float(cd.get("cue_s", s["cue_s"])), rest_s=float(s["rest_s"]),
                is_confounder=conf[l]))
        prev_last = labels[-1]
    return trials


# Cue-only mode (recording happens on another device): the wearer clenches the jaw hard while
# the screen is white. These bursts are found in the EEG afterwards to align the two clocks.
SYNC_PULSES = (3.0, 6.0, 10.0, 13.0)  # flash onsets, s from sync start: 3 s lead-in, slow and irregular
SYNC_FLASH_S = 1.0
SYNC_TAIL_S = 2.5
SYNC_TOTAL_S = SYNC_PULSES[-1] + SYNC_FLASH_S + SYNC_TAIL_S
SYNC_READY_S = 4.0                    # "sync done, trials start now" pause after the start sync


def estimate_duration(trials: list[Trial], s: dict, cue_only: bool = False) -> dict:
    """Total session time in seconds, broken down."""
    trial_s = sum(t.cue_s + t.rest_s for t in trials)
    n_blocks = (trials[-1].block + 1) if trials else 0
    every = int(s.get("break_every_blocks") or 0)
    n_breaks = max(0, (n_blocks - 1) // every) if every > 0 else 0
    break_s = n_breaks * float(s.get("break_min_s", 10))
    start_s = float(s.get("start_countdown_s", 5)) + (2 * SYNC_TOTAL_S + SYNC_READY_S if cue_only else 0.0)
    total = trial_s + break_s + start_s
    return dict(total_s=total, total_min=total / 60, trial_s=trial_s, n_trials=len(trials),
                n_blocks=n_blocks, n_breaks=n_breaks, break_s=break_s, start_s=start_s)


def warn_if_long(est: dict, s: dict) -> str | None:
    limit = float(s.get("max_minutes_warn", 22))
    if est["total_min"] > limit:
        return (f"WARNING: estimated {est['total_min']:.1f} min exceeds the {limit:.0f} min budget. "
                f"Cut trials_per_class (--trials N) or drop classes.")
    return None

"""Score a stream of fired events against the cue markers (used by train.py and realtime.py --replay).

Each trial owns a 'slot' [cue_on, next cue_on) capped at cue_on + cue_s + 2.6 s (breaks excluded).
  gesture trial   : hit if the cued class fires in [cue_on + 0.25, cue_off + 1.0]; any other class firing
                    in [cue_on, cue_off + 1.0] is a 'wrong' event; events later in the slot are false triggers
  rest/confounder : every event in the slot is a false trigger (attributed to that label)
False triggers per minute = false events / minutes of non-gesture time. Lower is better.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from detector import NULL_CLASSES

GRACE_S = 1.0
SLOT_TAIL_S = 2.6


def trial_slots(events: list[dict], trial_ids=None):
    cues = [e for e in events if e.get("event") == "cue_on"]
    if trial_ids is not None:
        keep = set(trial_ids)
    out = []
    for i, e in enumerate(cues):
        if trial_ids is not None and int(e["trial"]) not in keep:
            continue
        on, cs = e["t"], float(e.get("cue_s", 1.5))
        nxt = cues[i + 1]["t"] if i + 1 < len(cues) else on + cs + SLOT_TAIL_S
        out.append(dict(trial=int(e["trial"]), label=e["label"], on=on, off=on + cs,
                        end=min(nxt, on + cs + SLOT_TAIL_S)))
    return out


def score_events(fired, events, gestures, trial_ids=None) -> dict:
    """fired: iterable of objects with .t and .cls. gestures: classes that can be hit."""
    slots = trial_slots(events, trial_ids)
    fired = sorted(fired, key=lambda f: f.t)
    per = {g: dict(n=0, hits=0, wrong=0, dup=0, lat=[]) for g in gestures}
    wrong_detail = defaultdict(int)
    false_n = defaultdict(int)
    false_min = defaultdict(float)
    total_false, nongest_s = 0, 0.0
    wrong_total = 0
    for s in slots:
        lab = s["label"]
        evs = [f for f in fired if s["on"] <= f.t < s["end"]]
        if lab in gestures:
            r = per[lab]
            r["n"] += 1
            nongest_s += max(0.0, s["end"] - (s["off"] + GRACE_S))
            window_end = s["off"] + GRACE_S
            hit = False
            for f in evs:
                if f.t <= window_end:
                    if f.cls == lab and f.t >= s["on"] + 0.25:
                        if not hit:
                            hit = True
                            r["hits"] += 1
                            r["lat"].append(f.t - s["on"])
                        else:
                            r["dup"] += 1
                    else:
                        r["wrong"] += 1
                        wrong_total += 1
                        wrong_detail[f"{lab}->{f.cls}"] += 1
                else:
                    false_n["iti"] += 1
                    total_false += 1
            false_min["iti"] += max(0.0, s["end"] - window_end) / 60
        else:
            dur = s["end"] - s["on"]
            nongest_s += dur
            false_min[lab] += dur / 60
            for f in evs:
                false_n[lab] += 1
                total_false += 1
    minutes = nongest_s / 60
    n_g = sum(r["n"] for r in per.values())
    rec = [r["hits"] / r["n"] for r in per.values() if r["n"]]
    return dict(
        per_class={g: dict(n=r["n"], recall=(r["hits"] / r["n"] if r["n"] else float("nan")),
                           median_latency_s=(float(np.median(r["lat"])) if r["lat"] else float("nan")),
                           wrong=r["wrong"], duplicates=r["dup"]) for g, r in per.items()},
        mean_recall=float(np.mean(rec)) if rec else 0.0,
        false_events=total_false, nongesture_minutes=minutes,
        false_per_min=(total_false / minutes if minutes > 0 else 0.0),
        false_by_label={k: dict(events=false_n[k], minutes=false_min[k],
                                per_min=(false_n[k] / false_min[k] if false_min[k] > 0 else 0.0))
                        for k in sorted(set(false_n) | set(false_min))},
        wrong_per_gesture_trial=(wrong_total / n_g if n_g else 0.0), wrong_detail=dict(wrong_detail),
        gesture_trials=n_g)


def objective(m: dict, w_false=0.5, w_wrong=0.5) -> float:
    """Recall minus a heavy penalty on false triggers (false triggers matter more than recall)."""
    return m["mean_recall"] - w_false * m["false_per_min"] - w_wrong * m["wrong_per_gesture_trial"]


def format_score(m: dict, title="") -> str:
    L = []
    if title:
        L.append(title)
    L.append(f"{'class':<16}{'trials':>7}{'recall':>8}{'latency s':>11}{'wrong':>7}{'dup':>5}")
    for g, r in m["per_class"].items():
        if r["n"]:
            L.append(f"{g:<16}{r['n']:7d}{r['recall']:8.2f}{r['median_latency_s']:11.2f}{r['wrong']:7d}{r['duplicates']:5d}")
    L.append(f"mean recall {m['mean_recall']:.3f}   wrong-class events per gesture trial {m['wrong_per_gesture_trial']:.3f}")
    if m.get("wrong_detail"):
        L.append("wrong-class events (cued->fired): " + ", ".join(f"{k} x{v}" for k, v in m["wrong_detail"].items()))
    L.append(f"FALSE TRIGGERS: {m['false_events']} events in {m['nongesture_minutes']:.1f} min of non-gesture time "
             f"= {m['false_per_min']:.2f} per minute")
    for k, v in m["false_by_label"].items():
        L.append(f"   {k:<18}{v['events']:4d} events / {v['minutes']:.1f} min = {v['per_min']:.2f} per min")
    return "\n".join(L)


def run_logic(logic, times, probs, t0=-np.inf, t1=np.inf):
    """Run EventLogic over a probability stream restricted to [t0, t1]; returns the fired events."""
    logic.reset()
    out = []
    for t, p in zip(times, probs):
        if t0 <= t <= t1:
            out += logic.step(float(t), p)
    return out

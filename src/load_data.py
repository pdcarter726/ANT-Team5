"""Load a recorded session: XDF (primary, via pyxdf) with fallback to eeg_backup.npz + markers.csv.

  python src/load_data.py data/s1_20261107_142200          # folder, a .xdf, or an .npz

Time base: LabRecorder/XDF and the .npz backup both use LSL clock seconds, and markers.csv
stores `lsl_time` from the same clock, so markers and EEG align directly. If an XDF contains
a marker stream AND markers.csv exists, both are compared and the worst disagreement reported.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Recording:
    data: np.ndarray                 # (n, ch) float32, microvolts
    ts: np.ndarray                   # (n,) seconds (LSL clock)
    ch_names: list[str]
    fs: float
    events: list[dict]               # sorted by t; keys: t, event, label, trial, block, session, cue_s
    source: str = ""
    notes: list[str] = field(default_factory=list)
    alignment: dict | None = None    # set for cue-only sessions (markers aligned via sync clenches)
    wall_start: float | None = None  # unix time of the first sample, if the file records it (.cnt does)

    def class_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.events:
            if e["event"] == "cue_on":
                out[e["label"]] = out.get(e["label"], 0) + 1
        return out


# ------------------------------------------------------------------ markers
def parse_marker_json(s: str, t: float) -> dict | None:
    try:
        m = json.loads(s)
    except Exception:
        return None
    if not isinstance(m, dict) or "event" not in m:
        return None
    m["t"] = float(t)
    m.setdefault("label", "")
    m.setdefault("trial", -1)
    m.setdefault("block", -1)
    m.setdefault("cue_s", 0.0)
    return m


def load_markers_csv(path: str | Path) -> list[dict]:
    ev = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tr = r.get("t_rel")
            ev.append(dict(t=float(r["lsl_time"]), wall=float(r["wall_unix"]), event=r["event"],
                           label=r["label"], trial=int(r["trial_id"]), block=int(r["block"]),
                           session=int(r["session"]), cue_s=float(r["cue_s"]),
                           t_rel=float(tr) if tr not in (None, "") else None))
    return sorted(ev, key=lambda e: e["t"])


# --------------------------------------------------------------------- XDF
def _stream_labels(st) -> list[str]:
    try:
        chs = st["info"]["desc"][0]["channels"][0]["channel"]
        return [c["label"][0] for c in chs]
    except Exception:
        return []


def load_xdf(path: str | Path, units_to_uv: float = 1.0):
    import pyxdf
    streams, _ = pyxdf.load_xdf(str(path))
    eeg = [s for s in streams if s["info"]["type"][0].upper() == "EEG" and len(s["time_stamps"]) > 0]
    if not eeg:
        raise ValueError(f"{path}: no EEG stream with samples (found types: "
                         f"{[s['info']['type'][0] for s in streams]})")
    eeg.sort(key=lambda s: len(s["time_stamps"]), reverse=True)
    st = eeg[0]
    data = np.asarray(st["time_series"], dtype=np.float32) * units_to_uv
    ts = np.asarray(st["time_stamps"], dtype=np.float64)
    labels = _stream_labels(st)
    if len(labels) != data.shape[1]:
        labels = []
    fs = float(st["info"]["nominal_srate"][0])
    events, notes = [], []
    for s in streams:
        if s["info"]["type"][0].lower() == "markers":
            for v, t in zip(s["time_series"], s["time_stamps"]):
                m = parse_marker_json(v[0] if isinstance(v, (list, np.ndarray)) else v, t)
                if m:
                    events.append(m)
    if len(eeg) > 1:
        notes.append(f"{len(eeg)} EEG streams in the file; using the longest ({st['info']['name'][0]})")
    return data, ts, labels, fs, sorted(events, key=lambda e: e["t"]), notes


def load_cnt(path: str | Path):
    """ANT Neuro eego .cnt via MNE (needs `antio`). Returns data in uV, timestamps = seconds from file start.
    The companion .evt file only holds impedance checks and is ignored."""
    import mne
    raw = mne.io.read_raw_ant(str(path), preload=True, verbose="ERROR")
    data = (raw.get_data().T * 1e6).astype(np.float32)          # MNE stores volts
    md = raw.info["meas_date"]
    return data, np.asarray(raw.times, dtype=np.float64), list(raw.ch_names), float(raw.info["sfreq"]),         (md.timestamp() if md else None)


# ------------------------------------------------------------------ top level
def find_files(path: Path) -> dict:
    """Locate xdf / npz / markers.csv / metadata.json for a folder or a single file."""
    found = dict(xdf=None, cnt=None, npz=None, csv=None, meta=None, folder=None)
    if path.is_dir():
        found["folder"] = path
        x = sorted(path.glob("*.xdf"), key=lambda p: p.stat().st_mtime)
        found["xdf"] = x[-1] if x else None
        c = sorted(path.glob("*.cnt"), key=lambda p: p.stat().st_mtime)
        found["cnt"] = c[-1] if c else None
        if (path / "eeg_backup.npz").exists():
            found["npz"] = path / "eeg_backup.npz"
    else:
        found["folder"] = path.parent
        if path.suffix.lower() == ".xdf":
            found["xdf"] = path
        elif path.suffix.lower() == ".npz":
            found["npz"] = path
        elif path.suffix.lower() == ".cnt":
            found["cnt"] = path
        else:
            raise ValueError(f"unsupported file type: {path}")
    f = found["folder"]
    if (f / "markers.csv").exists():
        found["csv"] = f / "markers.csv"
    if (f / "metadata.json").exists():
        found["meta"] = f / "metadata.json"
    return found


def load_recording(path: str | Path, units_to_uv: float = 1.0, prefer: str = "xdf",
                   fallback_ch_names: list[str] | None = None, verbose: bool = True,
                   marker_offset: float | None = None, eeg_file: str | Path | None = None) -> Recording:
    """marker_offset: for cue-only sessions, EEG_time = t_rel + marker_offset (skips the sync search)."""
    path = Path(path)
    ff = find_files(path)
    if eeg_file:                      # EEG file kept elsewhere (e.g. copied from the other device)
        e = Path(eeg_file)
        kind = {".xdf": "xdf", ".cnt": "cnt", ".npz": "npz"}.get(e.suffix.lower())
        if kind is None:
            raise ValueError(f"unsupported EEG file type: {e.suffix}")
        ff["xdf"] = ff["cnt"] = ff["npz"] = None
        ff[kind] = e
    notes: list[str] = []
    wall_start = None
    order = ["xdf", "cnt", "npz"] if prefer == "xdf" else ["npz", "xdf", "cnt"]
    rec = None
    for kind in order:
        if ff[kind] is None:
            continue
        try:
            if kind == "xdf":
                data, ts, labels, fs, events, n = load_xdf(ff["xdf"], units_to_uv)
                notes += n
                src = f"XDF {ff['xdf'].name}"
            elif kind == "cnt":
                data, ts, labels, fs, wall_start = load_cnt(ff["cnt"])
                events, src = [], f"CNT {ff['cnt'].name}"
            else:
                z = np.load(ff["npz"], allow_pickle=False)
                data, ts, fs = z["data"], z["timestamps"], float(z["fs"])
                labels = [str(c) for c in z["ch_names"]] if "ch_names" in z.files else []
                events, src = [], f"NPZ {ff['npz'].name}"
            rec = (data, ts, labels, fs, events, src)
            break
        except Exception as e:
            notes.append(f"could not load {kind}: {e!r}; trying fallback")
    if rec is None:
        # last resort: unmerged part files left by a crashed session
        from lsl_utils import load_parts
        d, t = load_parts(ff["folder"])
        if len(t) == 0:
            raise FileNotFoundError(f"no .xdf, eeg_backup.npz or eeg_part_*.npz in {path}")
        meta = json.loads(ff["meta"].read_text()) if ff["meta"] else {}
        fs = (meta.get("stream") or {}).get("srate") or (len(t) - 1) / (t[-1] - t[0])
        rec = (d, t, meta.get("channel_names", []), float(fs), [], "NPZ parts (session did not finish)")
        notes.append("recovered from eeg_part_*.npz")
    data, ts, labels, fs, events, src = rec
    if not labels:
        labels = fallback_ch_names if fallback_ch_names and len(fallback_ch_names) == data.shape[1] \
            else [f"ch{i}" for i in range(data.shape[1])]
        notes.append("no channel labels in the recording; using fallback/generic names")

    csv_events = load_markers_csv(ff["csv"]) if ff["csv"] else []
    if events and csv_events:
        a = [e for e in events if e["event"] in ("cue_on", "cue_off")]
        b = [e for e in csv_events if e["event"] in ("cue_on", "cue_off")]
        if len(a) == len(b) and a:
            d = np.abs(np.array([x["t"] for x in a]) - np.array([x["t"] for x in b]))
            notes.append(f"XDF vs markers.csv: {len(a)} cue events, max time difference {d.max() * 1000:.1f} ms")
            if d.max() > 0.05:
                notes.append("WARNING: XDF and CSV marker times differ by >50 ms; check clock sync")
        else:
            notes.append(f"WARNING: XDF has {len(a)} cue events but markers.csv has {len(b)}")
    alignment = None
    cue_only = bool(json.loads(ff["meta"].read_text()).get("cue_only")) if ff["meta"] else False
    if not events and csv_events and cue_only:
        # EEG came from another device: markers are in seconds since session start, align to EEG clock
        ev = [dict(e, t=e["t_rel"]) for e in csv_events if e.get("t_rel") is not None]
        if marker_offset is not None:
            alignment = dict(offset=marker_offset, method="manual offset", drift_s=0.0, conf_start=None, conf_end=None)
            fn = lambda t, o=marker_offset: np.asarray(t, float) + o
        else:
            from sync_align import estimate_alignment
            alignment = estimate_alignment(data, ts, fs, ev)
            fn = alignment["fn"]
        notes.append(f"cue-only session aligned by {alignment['method']}: offset {alignment['offset']:.3f} s"
                     + (f", drift {alignment['drift_s'] * 1000:.0f} ms over the session"
                        if alignment.get("conf_end") else "")
                     + (f", confidence start {alignment['conf_start']:.1f}" if alignment.get("conf_start") else ""))
        if alignment.get("warning"):
            notes.append("WARNING: " + alignment["warning"])
        ss = next((e for e in csv_events if e["event"] == "session_start"), None)
        if wall_start is not None and ss is not None and marker_offset is None:
            # The file records its own start time. If both devices' clocks agree with the sync result
            # (within 0.5 s), the clock-based start is more precise (not limited by reaction time).
            wall_off = ss["wall"] - wall_start
            diff = alignment["offset"] - wall_off
            alignment["wall_offset"] = wall_off
            msg = (f"wall-clock cross-check: file start vs session start gives offset {wall_off:.3f} s "
                   f"(sync result differs by {diff:+.3f} s)")
            if abs(diff) < 0.5:
                base_fn, c = fn, wall_off - alignment["offset"]
                fn = lambda t, b=base_fn, c=c: b(t) + c      # keep the sync drift term, take the clock-based start
                alignment["offset"] = wall_off
                alignment["method"] += " + wall-clock start"
                msg += "; consistent, so the clock-based start is used"
            else:
                msg += "; INCONSISTENT (device clocks not in sync?), using the sync result"
            notes.append(msg)
        events = [dict(e, t=float(fn(e["t"]))) for e in ev]
    elif not events:
        events = csv_events
        if csv_events:
            notes.append("markers taken from markers.csv (lsl_time) - no marker stream in the EEG file")
        else:
            notes.append("WARNING: no markers found anywhere")
    r = Recording(data=data, ts=ts, ch_names=labels, fs=fs, events=sorted(events, key=lambda e: e["t"]),
                  source=src, notes=notes, alignment=alignment, wall_start=wall_start)
    check_quality(r)
    if verbose:
        print_summary(r)
    return r


def check_quality(r: Recording):
    n = len(r.ts)
    if n < 2:
        r.notes.append("WARNING: fewer than 2 samples")
        return
    meas = (n - 1) / (r.ts[-1] - r.ts[0])
    if abs(meas - r.fs) / r.fs > 0.02:
        r.notes.append(f"WARNING: measured rate {meas:.1f} Hz differs from nominal {r.fs} Hz")
    gaps = np.diff(r.ts) > 3.0 / r.fs
    if gaps.any():
        r.notes.append(f"WARNING: {int(gaps.sum())} timestamp gaps >3 samples (max {np.diff(r.ts).max():.3f} s)")
    if r.events:
        t0, t1 = r.ts[0], r.ts[-1]
        out = [e for e in r.events if e["event"] == "cue_on" and not (t0 <= e["t"] <= t1)]
        if out:
            r.notes.append(f"WARNING: {len(out)} cue markers fall outside the EEG time range (clock mismatch?)")


def print_summary(r: Recording):
    print(f"Source       : {r.source}")
    print(f"Samples      : {r.data.shape[0]} x {r.data.shape[1]} ch, {r.ts[-1] - r.ts[0]:.1f} s, nominal {r.fs} Hz")
    print(f"Channel names: {', '.join(r.ch_names)}")
    cc = r.class_counts()
    print(f"Cue events   : {sum(cc.values())}  " + ", ".join(f"{k}={v}" for k, v in sorted(cc.items())))
    for n in r.notes:
        print("  note:", n)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inspect a recording")
    ap.add_argument("path", help="session folder (markers.csv etc.), .xdf, .cnt or .npz")
    ap.add_argument("--eeg", default=None, help="EEG file kept elsewhere (.cnt/.xdf/.npz); path is then the session folder")
    ap.add_argument("--offset", type=float, default=None, help="cue-only: EEG time = t_rel + offset (skips sync search)")
    a = ap.parse_args()
    load_recording(a.path, marker_offset=a.offset, eeg_file=a.eeg)

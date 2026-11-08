"""Stage 4 training: build a per-user profile (model + thresholds + cursor models) from recorded sessions.

  python src/train.py --data data/s2_20261107_162913@data/Session2Recording.cnt --name default
  python src/train.py --data S2FOLDER@S2.cnt --extra S1FOLDER@S1.cnt      # extra sessions: training only
  python src/train.py --adapt CALIB_FOLDER[@eegfile] --base default --name default_cal   # 2-min calibration

Split (time ordered, last trials held out, nothing shuffled): first 60% of the main session's trials = train,
next 20% = validation (thresholds and persistence are tuned here), last 20% = TEST (reported, never tuned on).
For every model kind (threshold baseline, logistic regression, random forest) the held-out segments are
replayed window by window through the exact real-time event logic (probability threshold + persistence +
refractory), and scored as recall AND false triggers per minute on rest, confounder and between-trial time.
The kind with the best validation objective is refit on all data and saved to profiles/<name>.joblib.
Reject data: confounder cues, look_* cues and the gaps between trials (head returning to centre, etc.)
are all trained as a 'confounder' class that never produces an event.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ROOT, load_config
from detector import (HEAD_CLASSES, NULL_CLASSES, EventLogic, LogicParams, Profile, ThresholdModel,
                      WindowScorer, make_inference_ready, offline_features, offline_window_ends, score_features)
from evaluation import format_score, objective, run_logic, score_events, trial_slots
from feasibility import make_model
from features import FeatureExtractor
from load_data import load_recording
from preprocess import preprocess_offline

PROFILE_DIR = ROOT / "profiles"
KINDS = ("threshold", "lr", "rf")
WIN_S, STEP_S, OFFSET_S, ITI_GAP_S, ITI_MAX_S = 0.5, 0.125, 0.25, 0.1, 2.4


# ------------------------------------------------------------------ dataset
class Dataset:
    """All grid windows of one recording (features cached) plus labels for the usable ones."""

    def __init__(self, rec, cfg, keep, confs, sid=0, step_s=STEP_S, win_s=WIN_S):
        self.rec, self.sid = rec, sid
        fs = rec.fs
        self.win_n, self.step_n = int(round(win_s * fs)), int(round(step_s * fs))
        B = preprocess_offline(rec.data, fs, cfg.get("signal_check", {}).get("line_hz", 60.0))
        self.fe = FeatureExtractor(rec.ch_names, cfg["channel_groups"], fs)
        self.ends = offline_window_ends(len(rec.data), self.win_n, self.step_n)
        self.X = offline_features(B, self.fe, self.ends, self.win_n)
        t0 = float(rec.ts[0])
        self.times = t0 + (self.ends - 1) / fs                    # time of each window's last sample
        self.starts = self.times - (self.win_n - 1) / fs
        self.t0 = t0
        self.keep, self.confs = set(keep), set(confs)
        self._label(rec.events, win_s)

    def _label(self, events, win_s):
        n = len(self.times)
        y = np.array([""] * n, dtype=object)
        trial = np.full(n, -1)
        kind = np.array([""] * n, dtype=object)
        cues = [e for e in events if e.get("event") == "cue_on"]
        offs = {int(e["trial"]): e["t"] for e in events if e.get("event") == "cue_off"}
        other_t = np.array(sorted(e["t"] for e in events if e.get("event") != "cue_off"))
        for e in cues:
            on, cs, lab, tid = e["t"], float(e.get("cue_s", 1.5)), e["label"], int(e["trial"])
            m = (self.starts >= on + OFFSET_S) & (self.times <= on + cs)
            y[m] = lab if (lab in self.keep or lab == "rest") else "confounder"
            trial[m], kind[m] = tid + self.sid * 100000, "cue"
            off = offs.get(tid, on + cs)
            nxt = other_t[other_t > off + 1e-6]
            lim = min(off + ITI_MAX_S, nxt[0] if len(nxt) else np.inf)
            m2 = (self.starts >= off + ITI_GAP_S) & (self.times <= lim) & (y == "")
            idx = np.flatnonzero(m2)[::2]                          # every 0.25 s
            y[idx], trial[idx], kind[idx] = "confounder", tid + self.sid * 100000, "iti"
        self.y, self.trial, self.kind = y, trial, kind
        self.use = y != ""
        self.th = np.where(y == "head_right", 1.0, np.where(y == "head_left", -1.0, 0.0))
        self.tv = np.where(y == "head_down", 1.0, np.where(y == "head_up", -1.0, 0.0))

    def train_arrays(self, trial_ids=None):
        m = self.use if trial_ids is None else self.use & np.isin(self.trial, list(trial_ids))
        return self.X[m], self.y[m].astype(str), self.th[m], self.tv[m]

    def span(self, trial_ids):
        s = trial_slots(self.rec.events, [t % 100000 for t in trial_ids])
        return (min(x["on"] for x in s) - 1.0, max(x["end"] for x in s)) if s else (np.inf, -np.inf)


def parse_spec(spec: str):
    f, _, e = spec.partition("@")
    return f, (e or None)


def load_dataset(spec, cfg, keep, confs, sid):
    folder, eeg = parse_spec(spec)
    rec = load_recording(folder, units_to_uv=cfg["eeg_stream"].get("units_to_uv", 1.0), eeg_file=eeg, verbose=False)
    from channels import apply_to_recording
    cm = apply_to_recording(rec, cfg)
    print(f"  channels: {cm.describe()}")
    ds = Dataset(rec, cfg, keep, confs, sid)
    print(f"  {spec}: {len(rec.data)} samples, {len(ds.ends)} grid windows, {int(ds.use.sum())} labelled "
          f"({dict(zip(*np.unique(ds.y[ds.use], return_counts=True)))})")
    return ds


# ------------------------------------------------------------------- models
def fit_model(kind, X, y, w=None):
    if kind == "threshold":
        return ThresholdModel().fit(X, y, w)
    m = make_model(kind)
    if w is None:
        m.fit(X, y)
    else:
        m.fit(X, y, **{m.steps[-1][0] + "__sample_weight": w})
    return m


def fit_cursor(X, th, tv, w=None):
    out = {}
    for k, t in (("h", th), ("v", tv)):
        m = make_pipeline(StandardScaler(), Ridge(alpha=30.0))
        m.fit(X, t, **({} if w is None else {"ridge__sample_weight": w}))
        out[k] = m
    return out


def tune(classes, times, P, events, ids, gestures, t0, t1, verbose=False):
    """Pick persistence and thresholds that maximise recall minus heavy false-trigger penalties on `ids`."""
    def evaluate(params):
        fired = run_logic(EventLogic(classes, params), times, P, t0, t1)
        m = score_events(fired, events, gestures, ids)
        return objective(m), m

    best, bp = -1e9, None
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        for pers in (2, 3, 4, 6):
            p = LogicParams(default_thr=thr, default_persist=pers)
            j, _ = evaluate(p)
            if j > best:
                best, bp = j, p
    grid = (0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)
    for _ in range(2):                                    # per-class thresholds, coordinate descent
        for g in gestures:
            for thr in grid:
                p = bp.copy()
                p.thr[g] = thr
                j, _ = evaluate(p)
                if j > best + 1e-9:
                    best, bp = j, p
            for pers in (1, 2, 3, 4, 6, 8):                 # brief gestures (blinks) may need fewer windows
                p = bp.copy()
                p.persist[g] = pers
                j, _ = evaluate(p)
                if j > best + 1e-9:
                    best, bp = j, p
    return bp, best, evaluate(bp)[1]


def cursor_eval(P, H, V, classes, times, starts, events, ids, dz=0.3, gate=0.5):
    """Held-out cursor quality: sign accuracy while a head direction is held; false movement elsewhere."""
    hi = [i for i, c in enumerate(classes) if c in HEAD_CLASSES]
    pg = P[:, hi].sum(axis=1)
    est = {"head_right": (H, 1), "head_left": (H, -1), "head_down": (V, 1), "head_up": (V, -1)}
    rows, still = {}, dict(n=0, moving=0)
    for s in trial_slots(events, [i % 100000 for i in ids]):
        m = (starts >= s["on"] + OFFSET_S) & (times <= s["off"])
        if not m.any():
            continue
        if s["label"] in est:
            e, sg = est[s["label"]]
            v = e[m] * sg
            on = (pg[m] >= gate) & (v > dz)
            r = rows.setdefault(s["label"], dict(n=0, moving=0, right=0))
            r["n"] += int(m.sum())
            r["moving"] += int(((pg[m] >= gate) & (np.abs(e[m]) > dz)).sum())
            r["right"] += int(on.sum())
        else:
            mv = (pg[m] >= gate) & ((np.abs(H[m]) > dz) | (np.abs(V[m]) > dz))
            still["n"] += int(m.sum())
            still["moving"] += int(mv.sum())
    return rows, still


# --------------------------------------------------------------------- main
def build_profile(kind, model, cursor, ds, cfg, params, meta):
    return Profile(kind=kind, classes=[str(c) for c in model.classes_], model=make_inference_ready(model),
                   cursor_models=cursor, ch_names=list(ds.rec.ch_names), fs=float(ds.rec.fs), win_s=WIN_S,
                   step_s=STEP_S, groups=cfg["channel_groups"],
                   line_hz=cfg.get("signal_check", {}).get("line_hz", 60.0), params=params, meta=meta)


def save_profile(prof, name, X, y, th, tv, w, report):
    PROFILE_DIR.mkdir(exist_ok=True)
    prof.save(PROFILE_DIR / f"{name}.joblib")
    np.savez_compressed(PROFILE_DIR / f"{name}_train.npz", X=X, y=y, th=th, tv=tv, w=w)
    (PROFILE_DIR / f"{name}_report.txt").write_text(report, encoding="utf-8")
    summ = dict(name=name, kind=prof.kind, classes=prof.classes, fs=prof.fs, ch_names=prof.ch_names,
                params=dict(default_thr=prof.params.default_thr, default_persist=prof.params.default_persist,
                            thr=prof.params.thr, persist=prof.params.persist), meta=prof.meta)
    (PROFILE_DIR / f"{name}.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")


def train(args):
    cfg = load_config(args.config)
    if getattr(args, "channels", None):
        cfg.setdefault("model", {})["channels"] = [c.strip() for c in args.channels.split(",") if c.strip()]
        print("Training for channels:", cfg["model"]["channels"])
    s2 = cfg["sessions"][2]
    keep = list(s2["keep"])
    confs = sorted({c for sc in cfg["sessions"].values() for c in (sc.get("confounders") or [])})
    gestures = [c for c in keep if c != "rest"]
    out = []
    P_ = lambda *a: (print(*a), out.append(" ".join(str(x) for x in a)))[0]

    print("Loading and featurising:")
    main = load_dataset(args.data, cfg, keep, confs, 0)
    extra = [load_dataset(s, cfg, keep, confs, i + 1) for i, s in enumerate(args.extra or [])]
    ids = np.unique(main.trial[main.use])
    n = len(ids)
    tr_ids, va_ids, te_ids = ids[: int(n * 0.6)], ids[int(n * 0.6): int(n * 0.8)], ids[int(n * 0.8):]
    P_(f"\nMain session {n} trials (time ordered): train {len(tr_ids)}, validation {len(va_ids)}, TEST {len(te_ids)} (held out)")
    ex = [d.train_arrays() for d in extra]

    def stack(parts):
        return [np.concatenate(z) for z in zip(*parts)]

    def fit_on(kind, id_sets):
        parts = [main.train_arrays(np.concatenate(id_sets))] + ex
        X, y, th, tv = stack(parts)
        return fit_model(kind, X, y), fit_cursor(X, th, tv)

    res = {}
    for kind in (args.models or KINDS):
        t_s = time.time()
        m_tr, _ = fit_on(kind, [tr_ids])
        classes = [str(c) for c in m_tr.classes_]
        vt0, vt1 = main.span(va_ids)
        vm = (main.times >= vt0) & (main.times <= vt1)
        Pv = m_tr.predict_proba(main.X[vm])
        params, jv, mv = tune(classes, main.times[vm], Pv, main.rec.events, va_ids, gestures, vt0, vt1)
        m_trva, cur_trva = fit_on(kind, [tr_ids, va_ids])
        tt0, tt1 = main.span(te_ids)
        tm = (main.times >= tt0) & (main.times <= tt1)
        Pt = m_trva.predict_proba(main.X[tm])
        fired = run_logic(EventLogic(classes, params), main.times[tm], Pt, tt0, tt1)
        mt = score_events(fired, main.rec.events, gestures, te_ids)
        # same parameters, untuned defaults, for contrast
        mt0 = score_events(run_logic(EventLogic(classes, LogicParams()), main.times[tm], Pt, tt0, tt1),
                           main.rec.events, gestures, te_ids)
        H = cur_trva["h"].predict(main.X[tm])
        V = cur_trva["v"].predict(main.X[tm])
        crow, still = cursor_eval(Pt, H, V, classes, main.times[tm], main.starts[tm], main.rec.events, te_ids)
        res[kind] = dict(params=params, val_j=jv, val=mv, test=mt, test_default=mt0, cursor=(crow, still))
        P_(f"\n===== {kind.upper()}  (fit+tune {time.time() - t_s:.0f}s)  tuned: default thr {params.default_thr}, "
           f"persist {params.default_persist} windows ({params.default_persist * STEP_S:.2f}s), per-class thr {params.thr}")
        P_(f"validation objective {jv:.3f}  (recall {mv['mean_recall']:.2f}, false {mv['false_per_min']:.2f}/min)")
        P_(format_score(mt, "--- TEST (held-out last 20%), tuned parameters ---"))
        P_(f"--- TEST with untuned defaults (thr 0.7, persist 3): recall {mt0['mean_recall']:.2f}, "
           f"false {mt0['false_per_min']:.2f}/min")

    best = max(res, key=lambda k: res[k]["val_j"])
    P_("\n=== Model comparison on the held-out TEST segment (selection used validation only) ===")
    P_(f"{'model':<11}{'val obj':>8}{'test recall':>12}{'test false/min':>15}{'wrong/trial':>12}{'  untuned recall / false':>26}")
    for k, r in res.items():
        t, t0_ = r["test"], r["test_default"]
        P_(f"{k:<11}{r['val_j']:8.3f}{t['mean_recall']:12.2f}{t['false_per_min']:15.2f}{t['wrong_per_gesture_trial']:12.3f}"
           f"{t0_['mean_recall']:14.2f} / {t0_['false_per_min']:.2f}" + ("   <- selected" if k == best else ""))
    crow, still = res[best]["cursor"]
    P_(f"\n=== Cursor direction estimate, {best.upper()}, TEST windows (gate: P(head_*) >= 0.5, dead zone 0.3) ===")
    for lab, r in crow.items():
        P_(f"  {lab:<11} windows {r['n']:4d}  moving {r['moving'] / r['n']:.0%}  moving in the RIGHT direction {r['right'] / r['n']:.0%}")
    P_(f"  non-head cue windows (rest, other gestures, confounders): falsely moving {still['moving'] / max(still['n'], 1):.1%} "
       f"of {still['n']} windows")

    # final fit on everything (all main trials + extras) with the selected kind and tuned parameters
    X, y, th, tv = stack([main.train_arrays()] + ex)
    model = fit_model(best, X, y)
    cursor = fit_cursor(X, th, tv)
    meta = dict(trained=time.strftime("%Y-%m-%d %H:%M:%S"), data=args.data, extra=args.extra or [],
                n_windows=int(len(y)), n_train_trials=int(n), test=dict(recall=res[best]["test"]["mean_recall"],
                false_per_min=res[best]["test"]["false_per_min"]), compare={k: dict(
                    test_recall=r["test"]["mean_recall"], test_false_per_min=r["test"]["false_per_min"]) for k, r in res.items()})
    prof = build_profile(best, model, cursor, main, cfg, res[best]["params"], meta)
    save_profile(prof, args.name, X, y, th, tv, np.ones(len(y)), "\n".join(out))
    P_(f"\nSaved profiles/{args.name}.joblib (+ .json, _train.npz, _report.txt). Selected model: {best}.")
    return prof, res


def adapt(args):
    """~2 min calibration: refit the saved model on base data + the new (up-weighted) calibration session."""
    cfg = load_config(args.config)
    base = Profile.load(PROFILE_DIR / f"{args.base}.joblib")
    z = np.load(PROFILE_DIR / f"{args.base}_train.npz", allow_pickle=True)
    keep = list(cfg["sessions"][2]["keep"])
    confs = sorted({c for sc in cfg["sessions"].values() for c in (sc.get("confounders") or [])})
    ds = load_dataset(args.adapt, cfg, keep, confs, 9)
    Xn, yn, thn, tvn = ds.train_arrays()
    if len(yn) == 0:
        raise SystemExit("No labelled windows in the calibration session.")
    X = np.concatenate([z["X"], Xn])
    y = np.concatenate([z["y"].astype(str), yn])
    th, tv = np.concatenate([z["th"], thn]), np.concatenate([z["tv"], tvn])
    only = [c for c in (getattr(args, "adapt_classes", None) or []) if c]
    w_old = z["w"].astype(float)
    y_old = z["y"].astype(str)
    if only:
        bad = [c for c in only if c not in set(y_old) | set(yn)]
        if bad:
            raise SystemExit(f"unknown gesture(s) to recalibrate: {', '.join(bad)}")
        # targeted: the re-recorded gestures count more; rest/confounder windows of the new session keep base weight
        w_new = np.where(np.isin(yn, only), float(getattr(args, "class_weight", 4.0)), 1.0)
        if getattr(args, "replace", False):
            n_nd = int(np.isin(yn, only).sum())
            if n_nd == 0:
                raise SystemExit("--replace: the new session has no windows of " + ", ".join(only))
            keep_old = ~np.isin(y_old, only)
            X0, y0, th0, tv0, w0 = z["X"][keep_old], y_old[keep_old], z["th"][keep_old], z["tv"][keep_old], w_old[keep_old]
        else:
            X0, y0, th0, tv0, w0 = z["X"], y_old, z["th"], z["tv"], w_old
        X, y = np.concatenate([X0, Xn]), np.concatenate([y0, yn])
        th, tv, w = np.concatenate([th0, thn]), np.concatenate([tv0, tvn]), np.concatenate([w0, w_new])
    else:
        w = np.concatenate([w_old, np.full(len(yn), args.adapt_weight)])
    model = fit_model(base.kind, X, y, w)
    cursor = fit_cursor(X, th, tv, w)
    base.model, base.cursor_models = make_inference_ready(model), cursor
    base.classes = [str(c) for c in model.classes_]
    base.meta = dict(base.meta, adapted=time.strftime("%Y-%m-%d %H:%M:%S"), adapt_data=args.adapt,
                     adapt_windows=int(len(yn)), adapt_weight=args.adapt_weight,
                     adapt_classes=only or None)
    save_profile(base, args.name, X, y, th, tv, w, f"adapted from {args.base} with {args.adapt}")
    how = (f"{', '.join(only)} x weight {getattr(args, 'class_weight', 4.0)}" if only else f"all x weight {args.adapt_weight}")
    print(f"Adapted profile saved as profiles/{args.name}.joblib ({len(yn)} new windows; {how}).")
    return base


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="main session: FOLDER[@EEGFILE] (split train/val/test in time order)")
    ap.add_argument("--extra", nargs="*", help="additional sessions FOLDER[@EEGFILE], training only")
    ap.add_argument("--name", default="default")
    ap.add_argument("--models", nargs="*", choices=KINDS, help="subset of models to compare (default all)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--channels", help="comma list: train for ONLY these canonical channels (when the headset lacks some)")
    ap.add_argument("--adapt", help="calibration session FOLDER[@EEGFILE]: refit --base with it up-weighted")
    ap.add_argument("--base", default="default")
    ap.add_argument("--adapt-weight", type=float, default=2.0)
    ap.add_argument("--adapt-classes", nargs="*", help="with --adapt: only these gestures were re-recorded; they get --class-weight")
    ap.add_argument("--class-weight", type=float, default=4.0, help="weight of the re-recorded gestures' new windows")
    ap.add_argument("--replace", action="store_true", help="drop the OLD windows of the re-recorded gestures")
    args = ap.parse_args(argv)
    if args.adapt:
        adapt(args)
    elif args.data:
        train(args)
    else:
        ap.error("give --data (train) or --adapt (calibrate)")


if __name__ == "__main__":
    sys.exit(main())

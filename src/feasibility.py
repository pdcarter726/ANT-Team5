"""Stage 2 feasibility analysis: which candidate inputs are separable from EEG?

  python src/feasibility.py data/s1_20261107_142200
  python src/feasibility.py data/synth_s1 --folds 5

Steps: load (XDF or npz+csv) -> causal preprocessing (CAR, 60 Hz notch, 1-40 / 0.1-10 / 30-100 Hz
branches) -> sliding windows inside each cue period -> features -> time-ordered block
cross-validation (contiguous blocks of trials, all windows of a trial stay together, never random).
Outputs (printed, and saved to <session>/analysis/): feasibility_report.txt, keep_list.yaml, results.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import load_config
from features import FeatureExtractor
from load_data import load_recording
from preprocess import preprocess_offline
from windowing import trial_windows

HEAD = ["head_left", "head_right", "head_up", "head_down"]
EYE = ["look_left", "look_right", "look_up", "look_down"]


# ------------------------------------------------------------------ CV
def block_folds(trial_ids: np.ndarray, k: int) -> list[np.ndarray]:
    """Contiguous time-ordered blocks of trials -> list of arrays of trial ids (test sets)."""
    u = np.unique(trial_ids)          # trial ids increase with time
    k = max(2, min(k, len(u)))
    return [np.asarray(a) for a in np.array_split(u, k)]


def make_model(kind: str):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"))
    return make_pipeline(StandardScaler(), RandomForestClassifier(300, min_samples_leaf=2, n_jobs=-1,
                                                                  random_state=0, class_weight="balanced"))


def cv_predict(X, y, trials, k, kind="lr", regress=False):
    """Out-of-fold predictions. Windows of one trial are always in the same fold."""
    out = None
    classes = None if regress else np.unique(y)
    pred = np.zeros(len(y)) if regress else np.zeros((len(y), len(classes)))
    for test_trials in block_folds(trials, k):
        te = np.isin(trials, test_trials)
        tr = ~te
        assert not set(trials[tr]) & set(trials[te])       # no trial straddles train/test
        if regress:
            m = make_pipeline(StandardScaler(), Ridge(alpha=30.0)).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        else:
            m = make_model(kind).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])
            for j, c in enumerate(m.classes_):
                pred[te, list(classes).index(c)] = p[:, j]
    return pred if regress else (classes, pred)


def trial_level(classes, proba, y, trials):
    """Mean probability over each trial's windows -> one decision per trial."""
    tids = np.unique(trials)
    yt, pt = [], []
    for t in tids:
        m = trials == t
        yt.append(y[m][0])
        pt.append(classes[np.argmax(proba[m].mean(axis=0))])
    return np.array(yt), np.array(pt)


def prf(y, p, classes):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=classes, zero_division=0)
    return pr, rc, f1, sup


def fmt_cm(cm, classes, width=6):
    short = [c[:width] for c in classes]
    lines = [" " * 16 + " ".join(f"{s:>{width}}" for s in short)]
    for i, c in enumerate(classes):
        lines.append(f"{c:<16}" + " ".join(f"{v:>{width}d}" for v in cm[i]))
    return "\n".join(lines)


# ------------------------------------------------------------- analysis
def group_accuracy(X, y, trials, k, subset, kind="lr", mapping=None):
    """Trial-level accuracy for a sub-problem (optionally merging labels via mapping)."""
    m = np.isin(y, subset)
    if m.sum() == 0:
        return None
    yy = np.array([mapping.get(v, v) if mapping else v for v in y[m]])
    classes, proba = cv_predict(X[m], yy, trials[m], k, kind)
    yt, pt = trial_level(classes, proba, yy, trials[m])
    return float(np.mean(yt == pt)), 1.0 / len(classes), len(np.unique(yt))


def direction_signal(X, y, trials, k, pos, neg, zero="rest", dead=0.3):
    """Continuous 1-D estimate (ridge, held-out) for an axis: pos=+1, neg=-1, zero class=0."""
    present = [c for c in (pos, neg, zero) if c in set(y)]
    if pos not in present or neg not in present:
        return None
    m = np.isin(y, [pos, neg, zero])
    tgt = np.where(y[m] == pos, 1.0, np.where(y[m] == neg, -1.0, 0.0))
    est = cv_predict(X[m], tgt, trials[m], k, regress=True)
    nz = tgt != 0
    r = float(np.corrcoef(est[nz], tgt[nz])[0, 1]) if nz.sum() > 2 else 0.0
    moving = np.abs(est[nz]) > dead
    sign_acc = float(np.mean(np.sign(est[nz][moving]) == tgt[nz][moving])) if moving.any() else 0.0
    hit = float(moving.mean())
    z = tgt == 0
    false_move = float(np.mean(np.abs(est[z]) > dead)) if z.any() else float("nan")
    ok = r >= 0.6 and sign_acc >= 0.85 and (np.isnan(false_move) or false_move <= 0.10)
    marg = r >= 0.4 and sign_acc >= 0.7
    return dict(r=r, sign_acc=sign_acc, hit_rate=hit, false_move=false_move,
                verdict="YES" if ok else ("MARGINAL" if marg else "NO"))


def decide(classes, pr, rc, f1, cm, rec_thr, prec_thr):
    """KEEP / MAYBE / DROP per class, resolving mutual confusion in favour of the better class."""
    n = cm.sum(axis=1)
    order = np.argsort(-f1)
    status, reason, kept = {}, {}, []
    for i in order:
        c = classes[i]
        confused = [(classes[j], cm[i, j] / n[i]) for j in range(len(classes)) if j != i and n[i] and cm[i, j] / n[i] >= 0.15]
        confused += [(classes[j], cm[j, i] / n[j]) for j in range(len(classes)) if j != i and n[j] and cm[j, i] / n[j] >= 0.15
                     and classes[j] not in [x[0] for x in confused]]
        if c == "rest":
            status[c] = "KEEP"
            reason[c] = (f"null class, always collected; recall {rc[i]:.2f} -> "
                         f"{1 - rc[i]:.0%} of rest trials misread as a gesture")
            kept.append(c)
        elif rc[i] >= rec_thr and pr[i] >= prec_thr:
            clash = [x for x in confused if x[0] in kept and x[0] != "rest"]
            if clash:
                status[c], reason[c] = "DROP", f"confused with kept {clash[0][0]} ({clash[0][1]:.0%})"
            else:
                status[c], reason[c] = "KEEP", f"recall {rc[i]:.2f}, precision {pr[i]:.2f}"
                kept.append(c)
        elif rc[i] >= rec_thr - 0.15 and pr[i] >= prec_thr - 0.15:
            status[c], reason[c] = "MAYBE", "borderline" + (f"; confused with {confused[0][0]}" if confused else "")
        else:
            status[c] = "DROP"
            reason[c] = f"recall {rc[i]:.2f}, precision {pr[i]:.2f}" + (f"; confused with {confused[0][0]}" if confused else "")
    return status, reason


def run(path, cfg, k=5, win=0.5, step=0.25, offset=0.25, rec_thr=0.7, prec_thr=0.7, out=None, quick=False,
        marker_offset=None, eeg_file=None):
    rec = load_recording(path, cfg["eeg_stream"].get("units_to_uv", 1.0), fallback_ch_names=cfg.get("channels"),
                         marker_offset=marker_offset, eeg_file=eeg_file)
    L = []
    P = lambda s="": (L.append(s), print(s))
    P("")
    B = preprocess_offline(rec.data, rec.fs, cfg.get("signal_check", {}).get("line_hz", 60))
    fx = FeatureExtractor(rec.ch_names, cfg["channel_groups"], rec.fs)
    P(f"Feature extractor: {len(fx.feature_names)} features, groups={list(fx.groups)}, "
      f"L/R channel pairs={len(fx.pairs)} (config 'left' and 'right' lists must be in matching order)")
    wins, skipped = trial_windows(rec.ts, rec.events, rec.fs, win, step, offset)
    if not wins:
        raise SystemExit("no windows extracted - check markers")
    X = fx.batch(B, wins)
    y = np.array([w.label for w in wins])
    trials = np.array([w.trial for w in wins])
    P(f"Windows: {len(wins)} from {len(np.unique(trials))} trials (win {win}s, step {step}s, offset {offset}s); "
      f"skipped trials: {skipped}")
    folds = block_folds(trials, k)
    P(f"Time-ordered block CV: {len(folds)} contiguous folds of trials, windows of a trial kept together.")

    results = {}
    best = None
    for kind in (("lr",) if quick else ("lr", "rf")):
        classes, proba = cv_predict(X, y, trials, k, kind)
        yt, pt = trial_level(classes, proba, y, trials)
        wp = classes[np.argmax(proba, axis=1)]
        mf1 = float(np.mean(prf(yt, pt, classes)[2]))
        results[kind] = dict(trial_acc=float(np.mean(yt == pt)), trial_macro_f1=mf1,
                             window_acc=float(np.mean(wp == y)))
        P(f"  model {kind.upper():2s}: trial accuracy {results[kind]['trial_acc']:.3f}  trial macro-F1 {mf1:.3f}  "
          f"window accuracy {results[kind]['window_acc']:.3f}   (chance {1 / len(classes):.3f})")
        if best is None or mf1 > best[0]:
            best = (mf1, kind, classes, proba, yt, pt, wp)
    _, kind, classes, proba, yt, pt, wp = best
    classes = list(classes)
    P(f"\nRanking uses the better model: {kind.upper()}")

    pr, rc, f1, sup = prf(yt, pt, classes)
    _, wrc, _, _ = prf(y, wp, classes)
    cm = confusion_matrix(yt, pt, labels=classes)
    status, reason = decide(classes, pr, rc, f1, cm, rec_thr, prec_thr)

    P("\n=== Ranked table (trial level; window-level recall for reference) ===")
    P(f"{'rank':<5}{'class':<17}{'prec':>6}{'recall':>8}{'F1':>6}{'win-rec':>9}{'n':>4}  {'decision':<8} reason")
    for rank, i in enumerate(np.argsort(-f1), 1):
        c = classes[i]
        P(f"{rank:<5}{c:<17}{pr[i]:6.2f}{rc[i]:8.2f}{f1[i]:6.2f}{wrc[i]:9.2f}{int(sup[i]):4d}  {status[c]:<8} {reason[c]}")

    if "rest" in classes:
        ri = classes.index("rest")
        P(f"\nRest false-trigger rate (rest trials predicted as any gesture): {1 - rc[ri]:.0%} "
          f"(window level: {1 - wrc[ri]:.0%}). Argmax over all classes is the worst case; "
          "train.py adds thresholds, persistence and confounder data to push this down.")

    P("\n=== Confusion matrix (trial level, rows = true, cols = predicted) ===")
    P(fmt_cm(cm, classes))
    pairs = sorted({tuple(sorted((classes[i], classes[j]))) for i in range(len(classes)) for j in range(len(classes))
                    if i != j and cm[i].sum() and cm[i, j] / cm[i].sum() >= 0.15})
    P("\nConfused class pairs (>=15% of a class's trials predicted as the other):")
    P("  " + ("; ".join(f"{a} <-> {b}" for a, b in pairs) if pairs else "none"))

    # ---- head vs eye direction
    P("\n=== Head vs eye direction ===")
    present_h = [c for c in HEAD if c in set(y)]
    present_e = [c for c in EYE if c in set(y)]
    dir_res = {}
    if present_h and present_e:
        mp = {c: "head" for c in present_h} | {c: "eye" for c in present_e}
        r = group_accuracy(X, y, trials, k, present_h + present_e, "lr", mp)
        P(f"head vs eye (2-class, any direction): trial accuracy {r[0]:.2f} (chance {r[1]:.2f})")
        dir_res["head_vs_eye"] = r[0]
    for name, grp in (("head", present_h), ("eye", present_e)):
        if len(grp) >= 2:
            r = group_accuracy(X, y, trials, k, grp, "lr")
            P(f"{name} direction ({len(grp)}-class): trial accuracy {r[0]:.2f} (chance {r[1]:.2f})")
            dir_res[f"{name}_4class"] = r[0]
    P("\nContinuous direction estimate for cursor velocity (ridge on held-out folds; target +1/-1, rest=0):")
    P(f"{'axis':<18}{'corr r':>8}{'sign acc':>10}{'moving%':>9}{'false-move(rest)':>18}  usable for cursor velocity?")
    for name, pos, neg in (("head horizontal", "head_right", "head_left"), ("head vertical", "head_up", "head_down"),
                           ("eye horizontal", "look_right", "look_left"), ("eye vertical", "look_up", "look_down")):
        d = direction_signal(X, y, trials, k, pos, neg)
        if d is None:
            continue
        dir_res[name] = d
        P(f"{name:<18}{d['r']:8.2f}{d['sign_acc']:10.2f}{d['hit_rate'] * 100:8.0f}%{d['false_move']:18.2f}  {d['verdict']}")
    P("  (YES needs r>=0.6, sign accuracy>=0.85 and rest false-movement<=10% at dead zone 0.3; MARGINAL r>=0.4 and sign>=0.7)")
    P("  Note: trained on held ~1 s postures, so this tells whether direction is readable, not full closed-loop pointing quality.")

    # ---- keep list
    keep = ["rest"] * ("rest" in classes) + [c for c in classes if status[c] == "KEEP" and c != "rest"]
    maybe = [c for c in classes if status[c] == "MAYBE"]
    P("\n=== Keep / drop (paste into config sessions.2) ===")
    P(f"{'class':<17}{'decision':<9}reason")
    for c in classes:
        P(f"{c:<17}{status[c]:<9}{reason[c]}")
    yaml_txt = "keep: [" + ", ".join(keep) + "]\n# borderline, decide by hand: [" + ", ".join(maybe) + "]\n"
    P("\n" + yaml_txt.strip())

    out = Path(out) if out else (Path(path) if Path(path).is_dir() else Path(path).parent) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "feasibility_report.txt").write_text("\n".join(L), encoding="utf-8")
    (out / "keep_list.yaml").write_text(yaml_txt, encoding="utf-8")
    (out / "results.json").write_text(json.dumps(dict(
        models=results, classes=classes, status=status, reason=reason, direction=dir_res,
        precision=dict(zip(classes, map(float, pr))), recall=dict(zip(classes, map(float, rc)))),
        indent=1, default=str))
    P(f"\nSaved to {out}")
    return dict(status=status, results=results, direction=dir_res, classes=classes, recall=dict(zip(classes, rc)),
                precision=dict(zip(classes, pr)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="session folder, .xdf or .npz")
    ap.add_argument("--config", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--win", type=float, default=0.5)
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--offset", type=float, default=0.25, help="skip this long after cue onset (reaction time)")
    ap.add_argument("--keep-recall", type=float, default=0.7)
    ap.add_argument("--keep-precision", type=float, default=0.7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true", help="logistic regression only")
    ap.add_argument("--marker-offset", type=float, default=None,
                    help="cue-only sessions: EEG time = t_rel + this (skips the sync-clench search)")
    ap.add_argument("--eeg", default=None, help="EEG file kept elsewhere (.cnt/.xdf/.npz); path is the session folder")
    a = ap.parse_args(argv)
    run(a.path, load_config(a.config), a.folds, a.win, a.step, a.offset, a.keep_recall, a.keep_precision, a.out,
        a.quick, a.marker_offset, a.eeg)


if __name__ == "__main__":
    sys.exit(main())

import csv

import numpy as np

from common import session_cfg
from feasibility import run
from make_synthetic_session import make_session
from protocol import build_trials


def test_session2_uses_keep_list_and_confounders(cfg):
    s = session_cfg(cfg, 2)
    trials = build_trials(cfg, 2, seed=0)
    labels = {t.label for t in trials}
    assert labels == set(s["keep"]) | set(s["confounders"])
    assert "rest" in s["keep"]
    assert not (set(s["keep"]) & set(s["confounders"]))
    for t in trials:
        assert t.is_confounder == (t.label in s["confounders"])
        assert t.cue_s == (6.0 if t.is_confounder else s["cue_s"])


def test_session2_fits_its_time_budget(cfg):
    from protocol import estimate_duration, warn_if_long
    s = session_cfg(cfg, 2)
    est = estimate_duration(build_trials(cfg, 2, seed=0), s, cue_only=True)
    assert warn_if_long(est, s) is None, est["total_min"]


def test_confounders_logged_with_long_cue(tmp_path, cfg):
    make_session(cfg, 2, tmp_path, trials=3, seed=1, cue_only=True)
    rows = list(csv.DictReader(open(tmp_path / "markers.csv")))
    on = {r["trial_id"]: r for r in rows if r["event"] == "cue_on"}
    off = {r["trial_id"]: r for r in rows if r["event"] == "cue_off"}
    conf = {"talking", "chewing", "natural_blinking", "typing", "idle"}
    for tid, r in on.items():
        dur = float(off[tid]["t_rel"]) - float(r["t_rel"])
        assert abs(dur - (6.0 if r["label"] in conf else 1.5)) < 1e-6


def test_feasibility_excludes_confounders_from_keep_and_reports_false_triggers(tmp_path, cfg):
    make_session(cfg, 2, tmp_path, trials=6, seed=3)
    res = run(tmp_path, cfg, k=5, quick=True)
    conf = {"talking", "chewing", "natural_blinking", "typing", "idle"}
    assert not (set(res["classes"]) & conf)                      # never ranked as inputs
    assert "confounder" not in res["classes"]
    keep_line = (tmp_path / "analysis" / "keep_list.yaml").read_text().splitlines()[0]
    assert not any(c in keep_line for c in conf)
    import json
    r = json.loads((tmp_path / "analysis" / "results.json").read_text())
    assert set(r["confounders"]) == conf
    assert all(0 <= v["window_trigger_rate"] <= 1 and v["runs_per_min"] >= 0 for v in r["confounders"].values())
    # blinking naturally should be the one that looks like a gesture (double_blink), more than quiet talking/chewing
    assert r["confounders"]["natural_blinking"]["window_trigger_rate"] >= r["confounders"]["chewing"]["window_trigger_rate"]

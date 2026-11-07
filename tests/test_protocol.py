from collections import Counter

from common import session_cfg
from protocol import build_trials, estimate_duration, warn_if_long


def test_balanced_blocks_session1(cfg):
    trials = build_trials(cfg, 1, seed=3)
    s = session_cfg(cfg, 1)
    counts = Counter(t.label for t in trials)
    assert set(counts.values()) == {s["trials_per_class"]}
    assert len(counts) == len(s["classes"])
    # every block contains each class exactly once
    for b in range(s["trials_per_class"]):
        labs = [t.label for t in trials if t.block == b]
        assert sorted(labs) == sorted(s["classes"])


def test_no_back_to_back_repeats(cfg):
    for seed in range(20):
        trials = build_trials(cfg, 1, seed=seed, trials_override=10)
        assert all(a.label != b.label for a, b in zip(trials, trials[1:]))


def test_seed_reproducible_and_random(cfg):
    a = [t.label for t in build_trials(cfg, 1, seed=1)]
    b = [t.label for t in build_trials(cfg, 1, seed=1)]
    c = [t.label for t in build_trials(cfg, 1, seed=2)]
    assert a == b and a != c


def test_duration_estimate_session1_under_budget(cfg):
    trials = build_trials(cfg, 1, seed=0)
    est = estimate_duration(trials, session_cfg(cfg, 1))
    # 350 trials * 4.0 s = 1400 s plus breaks and countdown
    assert 1400 < est["total_s"] < 26 * 60
    assert warn_if_long(est, session_cfg(cfg, 1)) is None


def test_warning_when_too_long(cfg):
    trials = build_trials(cfg, 1, seed=0, trials_override=40)
    est = estimate_duration(trials, session_cfg(cfg, 1))
    assert est["total_min"] > 26
    assert "WARNING" in warn_if_long(est, session_cfg(cfg, 1))


def test_session2_confounders_spread_and_counts(cfg):
    s = session_cfg(cfg, 2)
    trials = build_trials(cfg, 2, seed=0)
    counts = Counter(t.label for t in trials)
    for k in s["keep"]:
        assert counts[k] == s["trials_per_class"]
    for c in s["confounders"]:
        assert counts[c] == s["confounder_trials_per_class"]
        blocks = sorted(t.block for t in trials if t.label == c)
        assert blocks[-1] - blocks[0] > 30      # spread over the session, not front-loaded
    assert all(t.cue_s == 6.0 for t in trials if t.is_confounder)

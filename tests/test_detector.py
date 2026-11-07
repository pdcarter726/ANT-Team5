import numpy as np
import pytest

from detector import (EventLogic, LogicParams, Profile, ThresholdModel, WindowScorer, offline_features,
                      offline_window_ends)
from features import FeatureExtractor
from preprocess import preprocess_offline

CLASSES = ["confounder", "double_blink", "head_left", "rest"]


class StubModel:
    classes_ = np.array(["a", "b"])

    def predict_proba(self, F):
        return np.stack([F[:, 0], F[:, 1]], axis=1)


def test_streaming_windows_match_offline_for_any_chunk_size(cfg):
    fs, names = 500.0, cfg["channels"]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6000, len(names))) * 20
    x[2000:2300, 0] += 150                                   # something blink-like
    fe = FeatureExtractor(names, cfg["channel_groups"], fs)
    win_n, step_n = 250, 63
    ends = offline_window_ends(len(x), win_n, step_n)
    ref = offline_features(preprocess_offline(x, fs), fe, ends, win_n)
    prof = Profile("stub", ["a", "b"], StubModel(), {}, names, fs, 0.5, 63 / 500, cfg["channel_groups"])
    for chunk in (37, 500, 6000):
        sc = WindowScorer(prof, keep_feats=True)
        got = [r for i in range(0, len(x), chunk) for r in sc.push(x[i:i + chunk])]
        assert [r.end for r in got] == list(ends)
        np.testing.assert_allclose(np.stack([r.feats for r in got]), ref, rtol=1e-6, atol=1e-6)


def test_window_time_is_last_sample_time(cfg):
    names, fs = cfg["channels"], 500.0

    class M:
        classes_ = np.array(["a"])

        def predict_proba(self, F):
            return np.ones((len(F), 1))

    prof = Profile("stub", ["a"], M(), {}, names, fs, 0.5, 0.25, cfg["channel_groups"])
    sc = WindowScorer(prof, t0=100.0)
    r = sc.push(np.zeros((500, len(names))))
    assert r[0].end == 250 and r[0].t == pytest.approx(100.0 + 249 / fs)


def P(**kw):
    return np.array([kw.get(c, 0.0) for c in CLASSES])


def test_persistence_refractory_and_rearm():
    lg = EventLogic(CLASSES, LogicParams(default_thr=0.7, default_persist=3, default_refractory_s=1.0,
                                         global_refractory_s=0.0, rearm_windows=2))
    t = 0.0
    fired = []

    def feed(p, n):
        nonlocal t
        for _ in range(n):
            fired.extend(lg.step(t, p))
            t += 0.125

    feed(P(double_blink=0.9), 2)
    assert not fired                                               # not persistent enough
    feed(P(rest=1.0), 1)
    feed(P(double_blink=0.9), 2)
    assert not fired                                               # streak was broken
    feed(P(double_blink=0.9), 1)
    assert [f.cls for f in fired] == ["double_blink"]              # 3 in a row
    feed(P(double_blink=0.95), 20)
    assert len(fired) == 1                                         # held gesture fires once
    feed(P(rest=1.0), 3)
    feed(P(double_blink=0.9), 3)
    assert len(fired) == 2                                         # released, re-armed, refractory elapsed


def test_refractory_blocks_quick_repeat_and_low_prob_never_fires():
    lg = EventLogic(CLASSES, LogicParams(default_thr=0.8, default_persist=1, default_refractory_s=1.0,
                                         global_refractory_s=0.0, rearm_windows=1))
    out = []
    for i in range(4):
        out += lg.step(i * 0.125, P(head_left=0.6, rest=0.4))     # below threshold
    assert out == []
    out += lg.step(0.5, P(head_left=0.9))
    out += lg.step(0.625, P(rest=1.0))
    out += lg.step(0.75, P(head_left=0.9))
    assert len(out) == 1                                           # second one inside the 1 s refractory
    out += lg.step(1.6, P(rest=1.0))
    out += lg.step(1.75, P(head_left=0.9))
    assert len(out) == 2


def test_null_classes_never_fire_and_global_refractory():
    lg = EventLogic(CLASSES, LogicParams(default_persist=1, global_refractory_s=0.5, default_refractory_s=0.0,
                                         rearm_windows=1))
    assert lg.step(0.0, P(rest=1.0)) == [] and lg.step(0.1, P(confounder=1.0)) == []
    assert len(lg.step(1.0, P(head_left=0.99))) == 1
    lg.step(1.1, P(rest=1.0))
    assert lg.step(1.2, P(double_blink=0.99)) == []                # inside the 0.5 s global lockout
    lg.step(1.3, P(rest=1.0))
    assert len(lg.step(1.6, P(double_blink=0.99))) == 1


def test_threshold_model_learns_a_separating_feature():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((600, 5))
    y = np.array(["rest"] * 300 + ["double_blink"] * 300)
    X[300:, 2] += 4
    m = ThresholdModel().fit(X, y)
    assert m.rules_["double_blink"][0] == 2
    assert (m.predict(X) == y).mean() > 0.95
    P_ = m.predict_proba(X)
    assert P_.shape == (600, 2) and (P_ >= 0).all()

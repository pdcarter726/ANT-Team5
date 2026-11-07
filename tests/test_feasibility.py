import numpy as np
import pytest

from feasibility import block_folds, cv_predict, decide, run
from make_synthetic_session import make_session
from preprocess import StreamingPreprocessor, preprocess_offline


def test_block_folds_are_contiguous_and_disjoint():
    trials = np.repeat(np.arange(40), 4)            # 4 windows per trial
    folds = block_folds(trials, 5)
    assert len(folds) == 5
    flat = np.concatenate(folds)
    assert sorted(flat) == list(range(40))
    for f in folds:                                  # each fold is a contiguous run of trial ids
        assert list(f) == list(range(f[0], f[0] + len(f)))


def test_cv_never_splits_a_trial():
    rng = np.random.default_rng(0)
    trials = np.repeat(np.arange(30), 4)
    X = rng.standard_normal((len(trials), 6))
    y = np.array(["a", "b"] * 60)[: len(trials)]
    y = np.repeat(np.array(["a", "b"] * 15), 4)
    classes, proba = cv_predict(X, y, trials, 5)     # asserts inside cv_predict on trial overlap
    assert proba.shape == (len(y), 2) and np.allclose(proba.sum(axis=1), 1)


def test_streaming_filter_matches_one_shot():
    """Chunked causal processing must equal processing the whole array (real-time parity)."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5000, 4)) * 10 + 300.0     # with a DC offset
    whole = preprocess_offline(x, 500.0)
    p = StreamingPreprocessor(500.0, 4)
    parts = [p.process(x[i:i + 137]) for i in range(0, 5000, 137)]
    for k in whole:
        assert np.allclose(np.concatenate([q[k] for q in parts]), whole[k], atol=1e-8)


def test_filters_pass_the_right_bands():
    fs = 500.0
    t = np.arange(10 * fs) / fs
    x = np.zeros((len(t), 2))
    x[:, 0] = np.sin(2 * np.pi * 8 * t)            # 8 Hz: eog branch yes, emg no
    x[:, 1] = np.sin(2 * np.pi * 60 * t)           # 60 Hz line: notched
    B = StreamingPreprocessor(fs, 2, car=False).process(x)      # CAR off: 2 channels would cancel each other
    rms = lambda a: np.sqrt(np.mean(a[2000:] ** 2))
    assert rms(B["eog"][:, 0]) > 0.6 and rms(B["emg"][:, 0]) < 0.05
    assert rms(B["emg"][:, 1]) < 0.1 and rms(B["eog"][:, 1]) < 0.05


def test_decide_resolves_mutual_confusion():
    classes = ["rest", "a", "b", "c"]
    cm = np.array([[10, 0, 0, 0], [0, 8, 2, 0], [0, 4, 6, 0], [0, 0, 0, 10]])
    pr = np.array([1, 0.67, 0.75, 1.0]); rc = np.array([1, 0.8, 0.6, 1.0]); f1 = 2 * pr * rc / (pr + rc)
    st, _ = decide(classes, pr, rc, f1, cm, 0.7, 0.7)
    assert st["rest"] == "KEEP" and st["c"] == "KEEP" and st["a"] in ("KEEP", "MAYBE") and st["b"] != "KEEP"


@pytest.fixture(scope="module")
def easy_session(tmp_path_factory, cfg):
    d = tmp_path_factory.mktemp("feas")
    make_session(cfg, 1, d, trials=10, seed=2, difficulty="normal")
    return d


def test_feasibility_end_to_end_beats_chance_and_writes_outputs(easy_session, cfg):
    res = run(easy_session, cfg, k=5, quick=True)
    assert res["results"]["lr"]["trial_acc"] > 0.5            # chance is 1/14
    assert res["recall"]["jaw_clench"] > 0.8 and res["recall"]["head_left"] > 0.8
    out = easy_session / "analysis"
    assert (out / "feasibility_report.txt").exists() and (out / "keep_list.yaml").exists()
    assert "keep: [rest" in (out / "keep_list.yaml").read_text()
    assert "head horizontal" in res["direction"]

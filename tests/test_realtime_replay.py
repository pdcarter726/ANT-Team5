"""End to end on a synthetic session: train -> profile -> replay through the real-time engine."""
import argparse

import pytest

import train
from channels import apply_to_recording
from detector import Fired, Profile
from input_layer import VirtualBackend, load_mapping
from load_data import load_recording
from make_synthetic_session import make_session
from overlay import State
from realtime import Engine, check_stream_matches


@pytest.fixture(scope="module")
def trained(tmp_path_factory, cfg):
    d = tmp_path_factory.mktemp("rt")
    make_session(cfg, 2, d / "s2", trials=14, seed=3, difficulty="easy")
    train.PROFILE_DIR = d / "profiles"
    args = argparse.Namespace(data=str(d / "s2"), extra=None, name="t", models=["lr"], config=None)
    prof, res = train.train(args)
    return d, prof, res


def engine(prof, executing=True):
    mapping = load_mapping(known_gestures=set(prof.gestures))
    b, st = VirtualBackend(), State()
    return Engine(prof, mapping, b, st, executing=executing, fast=True), b, st, mapping


def test_train_writes_profile_and_heldout_numbers_are_sane(trained):
    d, prof, res = trained
    assert (d / "profiles" / "t.joblib").exists() and (d / "profiles" / "t_train.npz").exists()
    t = res["lr"]["test"]
    assert t["mean_recall"] > 0.5
    assert t["false_per_min"] < 6
    loaded = Profile.load(d / "profiles" / "t.joblib")
    assert loaded.classes == prof.classes and "confounder" in loaded.classes


def test_replay_engine_executes_mapped_actions_in_virtual_backend(trained, cfg):
    d, prof, res = trained
    rec = load_recording(d / "s2", verbose=False)
    apply_to_recording(rec, cfg)
    eng, b, st, _ = engine(prof)
    eng.warmup = 1.0
    chunk = 2500
    for i in range(0, len(rec.data), chunk):
        eng.feed(rec.data[i:i + chunk].astype(float), rec.ts[0] if i == 0 else None)
    assert eng.n_windows > 100 and eng.fired
    assert {a[0] for a in b.actions()} & {"click", "key"}           # winks / blinks / eyebrows reached the backend
    assert any(m[0] == "move" for m in b.log)                       # head gestures moved the virtual cursor


def test_dry_run_sends_nothing_to_the_backend(trained):
    d, prof, res = trained
    eng, b, st, _ = engine(prof, executing=False)
    eng.handle(Fired(1.0, "left_wink", 0.99), 1.0)
    assert b.actions() == [] and st.n_events == 1                   # logged, not executed


def test_pause_blocks_actions_and_only_jaw_toggles(trained):
    d, prof, res = trained
    eng, b, st, _ = engine(prof)
    eng.set_paused(True)
    eng.handle(Fired(1.0, "left_wink", 0.99), 1.0)
    assert b.actions() == []
    eng.handle(Fired(2.0, "jaw_clench", 0.99), 2.0)                 # unlock
    assert not st.paused
    eng.handle(Fired(3.0, "left_wink", 0.99), 3.0)
    assert b.actions() == [("click", "left", 1)]


def test_rate_limiter_pauses_runaway(trained):
    d, prof, res = trained
    eng, b, st, mapping = engine(prof)
    for i in range(mapping.safety["max_events_per_min"] + 5):
        eng.handle(Fired(i * 0.1, "left_wink", 0.99), i * 0.1)
    assert st.paused and len(b.actions()) == mapping.safety["max_events_per_min"]


def test_stream_mismatch_is_refused(trained):
    d, prof, res = trained
    with pytest.raises(SystemExit):
        check_stream_matches(prof, list(reversed(prof.ch_names)), prof.fs)
    with pytest.raises(SystemExit):
        check_stream_matches(prof, prof.ch_names, prof.fs + 100)
    check_stream_matches(prof, [c.upper() for c in prof.ch_names], prof.fs)   # case-insensitive

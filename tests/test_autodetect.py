"""Auto-detection of running channels from an LSL stream (the amp: 25 unlabelled channels, the 25th all zeros)."""
import threading
import time

import numpy as np
import pylsl
import pytest

from channels import ChannelMap, detect_live, name_channels


def fake_eeg(n=3000, n_ch=25, dead=(24,), seed=0):
    x = np.random.default_rng(seed).standard_normal((n, n_ch)) * 20 + 1000     # big DC like the amp
    for d in dead:
        x[:, d] = 0.0
    return x


def test_detect_live_finds_zero_flat_and_nan_channels():
    x = fake_eeg(dead=(3, 24))
    x[:, 7] = 5.0                                    # constant
    x[10, 9] = np.nan
    live = detect_live(x)
    assert [i for i, v in enumerate(live) if not v] == [3, 7, 9, 24]


def test_unlabelled_first_n_named_extra_trailing_channel_ignored(cfg):
    live = detect_live(fake_eeg(dead=(24,)))
    names, notes = name_channels([f"ch{i + 1}" for i in range(25)], cfg, live)
    assert names[:24] == cfg["channels"] and names[24] == "ch25"
    assert any("extra channels not running" in n for n in notes) and not any("WARNING" in n for n in notes)


def test_live_extra_channel_triggers_a_warning(cfg):
    live = detect_live(fake_eeg(dead=()))
    _, notes = name_channels([f"ch{i + 1}" for i in range(25)], cfg, live)
    assert any("carry signal" in n for n in notes)


def test_labelled_streams_keep_their_labels(cfg):
    lab = ["A1", "B2", "C3"]
    names, notes = name_channels(lab, cfg, np.array([True, True, True]))
    assert names == lab and notes == []


def test_channelmap_reports_dead_model_channel(cfg):
    x = fake_eeg(dead=(10, 24))                      # channel 11 = T7 in the configured names is dead
    live = detect_live(x)
    cm = ChannelMap([f"ch{i + 1}" for i in range(25)], cfg, live=live)
    assert cm.problems == ["T7"]                      # a lost electrode keeps its name and is reported
    assert cm.names == cfg["model"]["channels"]


def test_wanted_subset_profile_selects_only_its_channels(cfg):
    sub = [c for c in cfg["model"]["channels"] if c not in ("T9", "T10")]
    cm = ChannelMap([f"ch{i + 1}" for i in range(25)], cfg, live=detect_live(fake_eeg()), wanted=sub)
    assert cm.names == sub and len(cm.idx) == len(sub)


def test_lslsource_autodetects_from_a_live_unlabelled_stream(cfg):
    from controller.eeg_source import LSLSource
    name = f"AutoDetectTest{int(time.time())}"
    info = pylsl.StreamInfo(name, "EEG", 25, 500.0, "float32", name)        # no channel labels at all
    out = pylsl.StreamOutlet(info)
    stop = threading.Event()

    def pump():
        rng = np.random.default_rng(1)
        while not stop.is_set():
            chunk = rng.standard_normal((50, 25)) * 20 + 1000
            chunk[:, 24] = 0.0
            out.push_chunk(chunk.tolist())
            time.sleep(0.1)

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    try:
        src = LSLSource(name=name, timeout=10)
        assert src.channels[:24] == cfg["channels"]
        assert src.live is not None and not src.live[24] and src.live[:24].all()
        assert src.n_running == 24 and src.dead_names() == ["ch25"]
        assert src.flat_channels() == ["ch25"]
        src.close()
    finally:
        stop.set()

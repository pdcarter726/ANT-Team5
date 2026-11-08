import numpy as np
import pytest

from channels import ChannelMap

OLD = ["Fp1", "Fp2", "F9", "F7", "F3", "Fz", "F4", "F8", "F10", "M1", "t7", "c3", "c4", "t8", "M2",
       "cz", "p7", "p3", "pz", "p4", "p8", "oz", "o1", "o2"]


def test_old_cap_names_map_onto_model_channels_via_aliases(cfg):
    cm = ChannelMap(OLD, cfg)
    assert cm.names == cfg["model"]["channels"]
    assert set(cm.dropped) == {"cz", "oz"}
    x = np.arange(24)[None, :] * np.ones((3, 24))
    assert cm.apply(x)[0, cm.names.index("T9")] == OLD.index("M1")      # M1 -> T9 picked the right column


def test_new_cap_stream_with_zero_25th_channel_and_no_labels(cfg):
    cm = ChannelMap([f"ch{i}" for i in range(25)], cfg)                   # what the amp's LSL stream reports
    assert cm.names == cfg["model"]["channels"]
    assert set(cm.dropped) == {"P9", "P10", "ch25"}
    x = np.zeros((2, 25))
    x[:, 9] = 9.0                                                         # column 9 is T9 in the configured names
    assert cm.apply(x)[0, cm.names.index("T9")] == 9.0


def test_new_cap_labelled_case_insensitive_and_reordered(cfg):
    names = [c.upper() for c in cfg["channels"]][::-1]
    cm = ChannelMap(names, cfg)
    assert [n.lower() for n in cm.names] == [n.lower() for n in cfg["model"]["channels"]]


def test_missing_model_channel_is_a_clear_error(cfg):
    names = [n for n in cfg["channels"] if n.lower() != "t7"]
    with pytest.raises(ValueError, match="T7"):
        ChannelMap(names, cfg)

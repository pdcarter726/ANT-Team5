from pathlib import Path

import numpy as np
import pytest

from common import ROOT
from features import FeatureExtractor
from load_data import find_files, load_cnt

CNT = ROOT / "data" / "Session1Recording.cnt"


def test_feature_groups_match_channel_names_case_insensitively(cfg):
    names = ["Fp1", "Fp2", "F9", "F7", "F3", "Fz", "F4", "F8", "F10", "M1", "t7", "c3", "c4", "t8", "M2",
             "cz", "p7", "p3", "pz", "p4", "p8", "oz", "o1", "o2"]
    fx = FeatureExtractor(names, cfg["channel_groups"], 500.0)
    assert len(fx.groups["central"]) == 3 and len(fx.groups["temporal"]) == 6     # c3/cz/c4, t7/t8/...
    assert len(fx.pairs) == 10                                                    # nothing silently dropped
    assert all(names[l].lower() != names[r].lower() for l, r in fx.pairs)


def test_config_groups_only_reference_real_channels(cfg):
    have = {c.lower() for c in cfg["channels"]}
    for g, cs in cfg["channel_groups"].items():
        assert all(c.lower() in have for c in cs), g
    assert len(cfg["channel_groups"]["left"]) == len(cfg["channel_groups"]["right"])


def test_find_files_picks_up_cnt(tmp_path):
    (tmp_path / "x.cnt").write_bytes(b"")
    assert find_files(tmp_path)["cnt"].name == "x.cnt"
    assert find_files(tmp_path / "x.cnt")["cnt"].name == "x.cnt"


@pytest.mark.skipif(not CNT.exists(), reason="real recording not present")
def test_real_cnt_loads_in_microvolts():
    data, ts, labels, fs, wall = load_cnt(CNT)
    assert data.shape[1] == 24 and fs == 500.0 and len(labels) == 24
    assert abs(len(ts) / fs - ts[-1]) < 0.01 and wall > 1.7e9
    assert np.isfinite(data).all()

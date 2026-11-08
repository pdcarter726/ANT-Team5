"""Channel naming, auto-detection and selection shared by training, replay, live use and the dashboard.

Different headsets name (and place) channels differently. Models are trained on a fixed list of canonical channels
(config `model.channels`, or a profile's own `ch_names`); `model.aliases` maps another headset's names onto them
(e.g. old M1 -> T9). Channels the model does not use are dropped BEFORE preprocessing, so common-average referencing and
features are computed over the same channel set everywhere.

Auto-detection (live streams): `detect_live(x)` finds channels that are not running (all zero / flat / NaN, e.g. the amp's
unused 25th channel). A stream without labels (ch1, ch2, ...) is named by giving the configured `channels` names, in order,
to the first channels; extra trailing channels (a dead status channel) are left unnamed and ignored.
"""
from __future__ import annotations

import re

import numpy as np

GENERIC = re.compile(r"^(ch|extra)?\s*\d+$", re.I)
FLAT_STD_UV = 0.05          # std below this over a few seconds = not running (a real electrode never gets this quiet)


def model_spec(cfg: dict):
    m = cfg.get("model") or {}
    aliases = {str(k).lower(): v for k, v in (m.get("aliases") or {}).items()}
    return m.get("channels"), aliases


def is_generic(names) -> bool:
    return bool(len(names)) and all(GENERIC.match(str(n).strip()) for n in names)


def detect_live(x, flat_std=FLAT_STD_UV) -> np.ndarray:
    """x: (samples, channels). Boolean mask, True where the channel is running (finite and not flat)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 10:
        return np.ones(x.shape[1] if x.ndim == 2 else 0, dtype=bool)
    ok = np.isfinite(x).all(axis=0)
    sd = np.where(ok, np.nanstd(np.where(np.isfinite(x), x, 0.0), axis=0), 0.0)
    return ok & (sd > flat_std)


def name_channels(names, cfg: dict, live=None):
    """Resolve display names for a stream. Labelled streams keep their labels. Unlabelled ones: the configured names go to
    the first len(config channels) channels in order; any channels after those are 'extra' (the amp's 25th status channel)
    and stay ch25, ch26, ... A named channel that is not running is a bad electrode: it keeps its name and is reported
    through `live` / ChannelMap.problems. Returns (names, notes)."""
    names = [str(n) for n in names]
    cfg_names = list(cfg.get("channels") or [])
    notes = []
    if not is_generic(names) or not cfg_names:
        return names, notes
    k = min(len(cfg_names), len(names))
    out = list(cfg_names[:k]) + [f"ch{i + 1}" for i in range(k, len(names))]
    notes.append(f"stream has no channel labels: first {k} channels named from config.yaml channels")
    extra = list(range(k, len(names)))
    if extra:
        if live is not None:
            run = [i + 1 for i in extra if live[i]]
            dead = [i + 1 for i in extra if not live[i]]
            if dead:
                notes.append("extra channels not running (flat/zero), ignored: " + ", ".join(f"#{i}" for i in dead))
            if run:
                notes.append("WARNING: extra channels " + ", ".join(f"#{i}" for i in run) + " carry signal but are not in "
                             "config.yaml channels and will be ignored")
        if len(extra) > 1:
            notes.append(f"WARNING: stream has {len(names)} channels, config lists {len(cfg_names)}: naming assumes the "
                         "first channels match the config order")
    if len(names) < len(cfg_names):
        notes.append(f"WARNING: stream has only {len(names)} channels, config lists {len(cfg_names)}")
    return out, notes


class ChannelMap:
    """names (as reported by the source) -> indices of the model's canonical channels, in model order.

    wanted: channel list to select (a profile's ch_names); default = config model.channels (or all channels).
    live:   optional running-mask from detect_live; a wanted channel that is not running is listed in `.problems`."""

    def __init__(self, names, cfg: dict, live=None, wanted=None):
        names, self.notes = name_channels(names, cfg, live)
        cfg_wanted, aliases = model_spec(cfg)
        wanted = wanted if wanted is not None else cfg_wanted
        canon = [aliases.get(str(n).lower(), n) for n in names]
        low = {}
        for i, c in enumerate(canon):
            low.setdefault(str(c).lower(), i)
        self.missing = []
        if wanted is None:
            self.idx, self.names = list(range(len(names))), canon
        else:
            self.missing = [w for w in wanted if w.lower() not in low]
            if self.missing:
                raise ValueError(f"source channels {names} lack model channels {self.missing}; "
                                 "add an alias under model.aliases in config.yaml or train a profile for the "
                                 "channels you have (train.py --channels ...)")
            self.idx = [low[w.lower()] for w in wanted]
            self.names = list(wanted)
        self.source_names = names
        keep = set(self.idx)
        self.dropped = [names[i] for i in range(len(names)) if i not in keep]
        self.problems = []
        if live is not None:
            self.problems = [self.names[k] for k, i in enumerate(self.idx) if not live[i]]

    def apply(self, x):
        return np.asarray(x)[:, self.idx]

    def describe(self) -> str:
        s = f"{len(self.idx)} model channels used; dropped: {', '.join(map(str, self.dropped)) or 'none'}"
        if self.problems:
            s += f"; NOT RUNNING: {', '.join(self.problems)}"
        return s


def apply_to_recording(rec, cfg: dict):
    """Rename/select the channels of a loaded Recording in place; returns the ChannelMap."""
    live = detect_live(rec.data[: int(rec.fs * 30)]) if is_generic(rec.ch_names) else None
    cm = ChannelMap(rec.ch_names, cfg, live=live)
    rec.data = cm.apply(rec.data)
    rec.ch_names = list(cm.names)
    return cm

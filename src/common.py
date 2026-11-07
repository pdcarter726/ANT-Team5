"""Shared helpers: paths, config loading, Windows DPI awareness."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"
DATA_DIR = ROOT / "data"


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(p)
    return cfg


def enable_dpi_awareness() -> None:
    """Make Windows report real pixel coordinates (needed before Tk/pyautogui)."""
    if sys.platform != "win32":
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def session_cfg(cfg: dict, session: int) -> dict:
    sessions = cfg["sessions"]
    if session in sessions:
        return sessions[session]
    if str(session) in sessions:
        return sessions[str(session)]
    raise KeyError(f"session {session} not defined in config 'sessions'")

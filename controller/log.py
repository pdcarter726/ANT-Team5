"""Timestamped log lines to the terminal and controller/logs/controller_<date>_<time>.log."""
import threading
import time
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
_lock = threading.Lock()
_file = None
listeners = []          # callables(line) - the dashboard shows the log too


def open_file():
    global _file
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        _file = open(LOG_DIR / time.strftime("controller_%Y%m%d_%H%M%S.log"), "w", encoding="utf-8", buffering=1)
        return _file.name
    except Exception:
        return None


def log(cat, msg, **fields):
    t = time.time()
    stamp = time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"
    extra = " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
    line = f"[{stamp}] [{cat}] {msg}" + (f" | {extra}" if extra else "")
    with _lock:
        print(line, flush=True)
        if _file:
            try:
                _file.write(line + "\n")
            except Exception:
                pass
    for fn in list(listeners):
        try:
            fn(line)
        except Exception:
            pass


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.1f}" if abs(v) >= 10 or v == 0 else f"{v:.3g}"
    if isinstance(v, (list, tuple)):
        return "(" + ",".join(_fmt(x) for x in v) + ")"
    return str(v)

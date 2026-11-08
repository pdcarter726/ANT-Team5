"""List every LSL stream this PC can see (any type, any host) - use when "no LSL stream found".

  .venv\\Scripts\\python.exe src\\lsl_scan.py            # scan 8 s
  .venv\\Scripts\\python.exe src\\lsl_scan.py --watch    # keep scanning, print when streams appear/disappear
"""
from __future__ import annotations

import argparse
import socket
import sys
import time

import pylsl


def scan(wait=8.0):
    return pylsl.resolve_streams(wait_time=wait)


def describe(streams) -> str:
    if not streams:
        return "  (no LSL streams visible at all)"
    return "\n".join(f"  name={s.name()!r}  type={s.type()!r}  channels={s.channel_count()}  "
                     f"rate={s.nominal_srate():g} Hz  host={s.hostname()}" for s in streams)


NOT_FOUND_HELP = """Nothing named/typed as configured was found. Checklist:
  1. Same network? The amp's stream may come from another PC (earlier it came from host DESKTOP-R0EKNDO, this PC is {me}).
     Both machines must be on the same subnet; guest/campus Wi-Fi often blocks LSL discovery (multicast). Try a phone hotspot or cable.
  2. Windows Firewall on BOTH machines: allow Python / the amp software on Private AND Public networks (LSL uses UDP multicast
     + TCP, ports 16571-16604). Quickest test: turn the firewall off briefly on both.
  3. Is the outlet actually running (amp software streaming, not just connected)? Restart streaming on the amp PC.
  4. Different type/name than configured? Run: python src/lsl_scan.py, then set eeg_stream: type / name in config/config.yaml."""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--wait", type=float, default=8.0)
    a = ap.parse_args(argv)
    print(f"This PC: {socket.gethostname()}   liblsl {pylsl.library_version()}")
    if not a.watch:
        print("Scanning ...")
        s = scan(a.wait)
        print(describe(s))
        return 0 if s else 1
    seen = set()
    print("Watching (Ctrl+C to stop) ...")
    try:
        while True:
            cur = {(s.name(), s.type(), s.hostname(), s.channel_count()) for s in scan(3.0)}
            for c in cur - seen:
                print(time.strftime("%H:%M:%S"), "APPEARED:", c)
            for c in seen - cur:
                print(time.strftime("%H:%M:%S"), "GONE    :", c)
            seen = cur
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())

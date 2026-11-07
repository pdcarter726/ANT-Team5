"""Minimal XDF writer (one numeric stream + one string marker stream).

pyxdf only reads XDF; this exists so the loader can be tested against a real .xdf file
and so make_synthetic_session.py can produce LabRecorder-like output without LabRecorder.
Format: https://github.com/sccn/xdf/wiki/Specifications
"""
from __future__ import annotations

import struct
from xml.sax.saxutils import escape

import numpy as np


def _varlen(n: int) -> bytes:
    if n < 256:
        return b"\x01" + bytes([n])
    if n < 2 ** 32:
        return b"\x04" + struct.pack("<I", n)
    return b"\x08" + struct.pack("<Q", n)


def _chunk(tag: int, content: bytes) -> bytes:
    body = struct.pack("<H", tag) + content
    return _varlen(len(body)) + body


def _header_xml(name, stype, nch, srate, fmt, source_id, labels=None, unit=None) -> bytes:
    ch = ""
    if labels:
        ch = "<desc><channels>" + "".join(
            f"<channel><label>{escape(l)}</label><unit>{escape(unit or '')}</unit><type>EEG</type></channel>"
            for l in labels) + "</channels></desc>"
    return (f'<?xml version="1.0"?><info><name>{escape(name)}</name><type>{escape(stype)}</type>'
            f"<channel_count>{nch}</channel_count><nominal_srate>{srate}</nominal_srate>"
            f"<channel_format>{fmt}</channel_format><source_id>{escape(source_id)}</source_id>"
            f"{ch}</info>").encode()


def write_xdf(path, data, timestamps, ch_names, fs, marker_strings, marker_times,
              eeg_name="SyntheticEEG", marker_name="CueMarkers", block=20000):
    """data: (n, ch) float32; timestamps: (n,) seconds; markers: list[str] + times."""
    data = np.ascontiguousarray(data, dtype="<f4")
    ts = np.asarray(timestamps, dtype="<f8")
    n, nch = data.shape
    out = [b"XDF:", _chunk(1, b'<?xml version="1.0"?><info><version>1.0</version></info>')]
    out.append(_chunk(2, struct.pack("<I", 1) + _header_xml(eeg_name, "EEG", nch, fs, "float32",
                                                           "synthetic-xdf", ch_names, "microvolts")))
    has_markers = len(marker_strings) > 0
    if has_markers:
        out.append(_chunk(2, struct.pack("<I", 2) + _header_xml(marker_name, "Markers", 1, 0, "string", "markers-xdf")))
    dt = np.dtype([("f", "u1"), ("t", "<f8"), ("x", "<f4", (nch,))])
    for s in range(0, n, block):
        e = min(n, s + block)
        rec = np.empty(e - s, dtype=dt)
        rec["f"], rec["t"], rec["x"] = 8, ts[s:e], data[s:e]
        out.append(_chunk(3, struct.pack("<I", 1) + _varlen(e - s) + rec.tobytes()))
    if has_markers:
        body = bytearray(struct.pack("<I", 2) + _varlen(len(marker_strings)))
        for m, t in zip(marker_strings, marker_times):
            b = m.encode()
            body += b"\x08" + struct.pack("<d", t) + _varlen(len(b)) + b
        out.append(_chunk(3, bytes(body)))
    footers = [(1, n, ts[0], ts[-1])]
    if has_markers:
        footers.append((2, len(marker_strings), min(marker_times), max(marker_times)))
    for sid, cnt, a, b in footers:
        out.append(_chunk(6, struct.pack("<I", sid) +
                          (f'<?xml version="1.0"?><info><first_timestamp>{a}</first_timestamp>'
                           f"<last_timestamp>{b}</last_timestamp><sample_count>{cnt}</sample_count></info>").encode()))
    with open(path, "wb") as f:
        f.write(b"".join(out))

"""Live signal-check view: scrolling traces, per-channel RMS with FLAT/SAT/NOISY flags,
and a band-power bar for one channel (verify blinks, jaw clench, eyes-closed alpha).

Standalone:  python src/signal_check.py            (needs an LSL EEG stream)
Embedded:    SignalCheck(parent, recorder, ch_names, cfg) inside collect.py
"""
from __future__ import annotations

import argparse
import tkinter as tk

import numpy as np

from signal_metrics import BANDS, band_powers, channel_stats

BG, FG, DIM = "#14161a", "#e8e8ea", "#8a8f98"
TRACE = "#6cb6ff"
FLAGC = {"FLAT": "#ff5c5c", "SAT": "#ff5c5c", "NOISY": "#ffb84d", "NODATA": "#8a8f98", "": "#5ad17a"}


class SignalCheck(tk.Frame):
    def __init__(self, parent, recorder, ch_names, cfg, on_done=None):
        super().__init__(parent, bg=BG)
        self.rec, self.names, self.cfg = recorder, list(ch_names), cfg
        sc = cfg.get("signal_check", {})
        self.sc = sc
        low = [n.lower() for n in self.names]
        want = [c.lower() for c in sc.get("display_channels", []) if c.lower() in low]
        self.disp = [low.index(c) for c in want] or list(range(min(8, len(self.names))))
        self.window_s = float(sc.get("window_s", 6))
        self.scale = float(sc.get("uv_per_row", 100))
        self.car = False
        self.band_ch = self.disp[0]
        self.on_done = on_done
        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._job = None
        self.running = False

    def start(self):
        self.running = True
        self.focus_set()
        self._tick()

    def stop(self):
        self.running = False
        if self._job:
            self.after_cancel(self._job)

    def key(self, e):
        k = e.keysym
        if k in ("plus", "equal", "KP_Add"):
            self.scale = max(5, self.scale / 1.5)
        elif k in ("minus", "KP_Subtract"):
            self.scale *= 1.5
        elif k == "c":
            self.car = not self.car
        elif k in ("Right", "bracketright"):
            self.band_ch = (self.band_ch + 1) % len(self.names)
        elif k in ("Left", "bracketleft"):
            self.band_ch = (self.band_ch - 1) % len(self.names)
        elif k == "Return" and self.on_done:
            self.on_done()

    def _tick(self):
        if not self.running:
            return
        try:
            self._draw()
        except Exception as ex:  # never let a drawing bug kill the app
            self.canvas.delete("all")
            self.canvas.create_text(20, 20, anchor="nw", fill="#ff5c5c", text=f"draw error: {ex!r}")
        self._job = self.after(60, self._tick)

    def _draw(self):
        c = self.canvas
        c.delete("all")
        W, H = max(c.winfo_width(), 400), max(c.winfo_height(), 300)
        fs = self.rec.fs
        data, ts = self.rec.recent(self.window_s)
        font = ("Consolas", -max(11, int(H / 55)))
        small = ("Consolas", -max(10, int(H / 65)))
        if self.rec.error:
            c.create_text(10, 10, anchor="nw", fill="#ff5c5c", font=font, text="EEG reader error: " + self.rec.error)
        if len(data) < 10:
            c.create_text(W / 2, H / 2, fill=DIM, font=font, text="waiting for EEG samples ...")
            return
        x = data.astype(np.float64)
        x = x - np.median(x, axis=0, keepdims=True)
        if self.car:
            x = x - np.mean(x, axis=0, keepdims=True)
        # ---- traces (left 62%)
        lw = int(W * 0.62)
        top, rows = 28, len(self.disp)
        rh = (H - top - 70) / rows
        c.create_text(8, 6, anchor="nw", fill=DIM, font=small,
                      text=f"+/-: scale ({self.scale:.0f} uV/half-row)   c: CAR {'ON' if self.car else 'off'}   "
                           f"left/right: band channel   Enter: continue")
        npts = min(len(x), lw - 60)
        idx = np.linspace(0, len(x) - 1, npts).astype(int)
        for r, ch in enumerate(self.disp):
            cy = top + rh * (r + 0.5)
            y = np.clip(x[idx, ch] / self.scale, -1, 1) * rh * 0.48
            pts = []
            for i, v in enumerate(y):
                pts += [50 + i * (lw - 60) / max(npts - 1, 1), cy - v]
            c.create_line(50, cy, lw - 10, cy, fill="#262a31")
            c.create_line(*pts, fill=TRACE, width=1)
            c.create_text(6, cy, anchor="w", fill=FG, font=font, text=self.names[ch])
        # ---- RMS table (right)
        rx = lw + 10
        stats = channel_stats(data.astype(np.float64)[-int(2 * fs):],
                              self.sc.get("flat_rms_uv", .3), self.sc.get("saturated_uv", 2000),
                              self.sc.get("noisy_rms_uv", 150))
        nrow = len(self.names)
        th = min((H - 130) / nrow, 22)
        c.create_text(rx, 6, anchor="nw", fill=DIM, font=small, text="RMS (uV, last 2 s)")
        maxr = max(30.0, float(np.percentile([s["rms"] for s in stats], 90)) * 1.5)
        bw = (W - rx) * 0.45
        for i, s in enumerate(stats):
            y0 = 28 + i * th
            col = FLAGC.get(s["flag"], FG)
            sel = ">" if i == self.band_ch else " "
            c.create_text(rx, y0, anchor="nw", fill=col, font=small, text=f"{sel}{self.names[i]:<4}{s['rms']:7.1f}")
            c.create_rectangle(rx + 110, y0 + 2, rx + 110 + min(1.0, s["rms"] / maxr) * bw, y0 + th - 4,
                               fill=col, outline="")
            if s["flag"]:
                c.create_text(rx + 115 + bw, y0, anchor="nw", fill=col, font=small, text=s["flag"])
        # ---- band power bar for selected channel (bottom)
        by = 28 + nrow * th + 14
        bp = band_powers(data[-int(4 * fs):, self.band_ch].astype(np.float64), fs)
        c.create_text(rx, by, anchor="nw", fill=FG, font=small,
                      text=f"Band power  {self.names[self.band_ch]}   alpha peak {bp['alpha_peak']:.1f} Hz")
        n = len(BANDS)
        bh = max(8, (H - by - 24) / n - 4)
        for i, (name, rel) in enumerate(bp["rel"].items()):
            yy = by + 20 + i * (bh + 4)
            colr = "#ffd24d" if name.startswith("alpha") else ("#ff8f6b" if name.startswith("EMG") else TRACE)
            c.create_text(rx, yy, anchor="nw", fill=DIM, font=small, text=f"{name:<11}")
            c.create_rectangle(rx + 120, yy, rx + 120 + rel * (W - rx - 190), yy + bh, fill=colr, outline="")
            c.create_text(W - 8, yy, anchor="ne", fill=FG, font=small, text=f"{rel * 100:4.0f}%")
        c.create_text(8, H - 40, anchor="nw", fill=DIM, font=small,
                      text="Checks: blink = big frontal spike (Fp1/Fp2) | jaw clench = EMG bar jumps on T7/T8 | "
                           "eyes closed = alpha bar + peak ~10 Hz on Oz")


def main():
    from common import enable_dpi_awareness, load_config
    from lsl_utils import (EegRecorder, describe_stream, effective_channel_names,
                           print_stream_info, resolve_stream)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    enable_dpi_awareness()
    cfg = load_config(args.config)
    e = cfg["eeg_stream"]
    info = resolve_stream(e["type"], e.get("name"), e.get("resolve_timeout_s", 8))
    if info is None:
        raise SystemExit("No LSL EEG stream found. Start the eego LSL stream or: python src/synthetic_eeg.py")
    d = describe_stream(info)
    print_stream_info(d, cfg)
    rec = EegRecorder(info, units_to_uv=e.get("units_to_uv", 1.0))
    rec.start()
    root = tk.Tk()
    root.title("Signal check")
    root.geometry("1280x760")
    sc = SignalCheck(root, rec, effective_channel_names(d, cfg), cfg, on_done=root.destroy)
    sc.pack(fill="both", expand=True)
    root.bind("<Key>", sc.key)
    sc.start()
    root.mainloop()
    rec.stop()


if __name__ == "__main__":
    main()

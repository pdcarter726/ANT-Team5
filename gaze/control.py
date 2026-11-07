"""Tiny UDP control channel so the EEG/ML app can talk to the gaze cursor.

From the ML app (any language can send these UDP text messages to 127.0.0.1:50555):
    from gaze import control
    control.freeze(0.8)     # hold the cursor still while a gesture happens
    control.pause()         # stop moving the cursor
    control.resume()
    control.toggle()
    control.recalibrate()   # full 25-point calibration
    control.add_pose()      # 9 extra points at the current head position
    control.recenter()      # one center dot, fixes drift
    control.status()        # -> dict with x, y, paused, calibrating, face
"""
import json
import socket
import threading

HOST, PORT = "127.0.0.1", 50555


def _send(msg, want_reply=False, timeout=0.3):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.settimeout(timeout)
        s.sendto(msg.encode(), (HOST, PORT))
        if want_reply:
            data, _ = s.recvfrom(4096)
            return json.loads(data.decode())
    except (socket.timeout, ConnectionResetError):
        return None
    finally:
        s.close()


def freeze(seconds=0.8): _send(f"freeze {seconds}")
def pause(): _send("pause")
def resume(): _send("resume")
def toggle(): _send("toggle")
def recalibrate(): _send("recalibrate")
def add_pose(): _send("addpose")
def recenter(): _send("recenter")
def status(): return _send("status", want_reply=True)


class ControlServer(threading.Thread):
    """Runs inside the gaze app. handler(cmd, arg) -> dict or None (reply for 'status')."""

    def __init__(self, handler, port=PORT):
        super().__init__(daemon=True)
        self.handler = handler
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((HOST, port))
        self.sock.settimeout(0.2)
        self.running = True

    def run(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
            except (socket.timeout, ConnectionResetError, OSError):
                continue
            parts = data.decode(errors="ignore").strip().split()
            if not parts:
                continue
            reply = self.handler(parts[0].lower(), parts[1] if len(parts) > 1 else None)
            if reply is not None:
                self.sock.sendto(json.dumps(reply).encode(), addr)

    def stop(self):
        self.running = False
        self.sock.close()

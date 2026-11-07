"""Serve the hands-free arcade locally and open it in the browser (standard library only).

  python games/serve.py                    # http://127.0.0.1:8765  (the arcade menu)
  python games/serve.py --game chess       # open one game directly: flappy, simon, chess, checkers, tictactoe, airhockey
  python games/serve.py --port 9000 --no-browser

The controller dashboard starts the same server in the background (start_server), so this script is only
needed to run the games without the dashboard.
"""
import argparse
import functools
import http.server
import threading
import urllib.request
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
GAMES = {
    "flappy": "Flappy Bird", "simon": "Simon", "chess": "Chess",
    "checkers": "Checkers", "tictactoe": "Tic Tac Toe", "airhockey": "Air Hockey",
}
PORT = 8765


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")     # edits show up on a plain reload
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def _arcade_already_at(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/shared/hf.js", timeout=0.5) as r:
            return r.status == 200
    except Exception:
        return False


def start_server(port=PORT):
    """Serve games/ from a background thread. Returns (server or None, base url).
    If the arcade is already being served on `port` (e.g. by this script), that one is reused; if the port
    is taken by something else, a free port is used (scores and settings are stored per port by the browser)."""
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=str(HERE)))
    except OSError:
        if _arcade_already_at(port):
            return None, f"http://127.0.0.1:{port}/"
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(HERE)))
    threading.Thread(target=server.serve_forever, daemon=True, name="arcade-http").start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/"


def game_url(base, game=None, embed=False):
    url = base + (f"{game}/" if game else "")
    return url + ("?embed=1" if embed else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--game", choices=list(GAMES), help="open this game directly instead of the arcade menu")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    server, base = start_server(a.port)
    url = game_url(base, a.game)
    print(f"Arcade running at {url}" + ("" if server else "  (already running in another process)"))
    print("F11 = fullscreen in the browser.  Ctrl+C here to stop.")
    if not a.no_browser:
        webbrowser.open(url)
    if server is None:
        return
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()

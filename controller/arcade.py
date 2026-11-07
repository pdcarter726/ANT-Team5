"""The hands-free arcade (games/), served from inside the controller and shown in the Games tab."""
import importlib.util
from pathlib import Path

from .log import log

GAMES_DIR = Path(__file__).resolve().parent.parent / "games"
MAPPING = "arcade"          # controller/mappings/arcade.json: eyebrow raise = Space, jaw clench = left button

_spec = importlib.util.spec_from_file_location("arcade_serve", GAMES_DIR / "serve.py")
_serve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_serve)
GAMES = _serve.GAMES

_state = {"server": None, "base": None}


def base_url():
    """Start the web server on first use (background thread) and return its base URL."""
    if _state["base"] is None:
        server, base = _serve.start_server()
        _state.update(server=server, base=base)
        log("ARCADE", "serving the games" if server else "using the arcade server that is already running", url=base)
    return _state["base"]


def url(game=None):
    """URL of the arcade menu (game=None) or one game, with ?embed=1 so in-page Arcade chrome is hidden."""
    return _serve.game_url(base_url(), game, embed=True)


def stop():
    if _state["server"]:
        _state["server"].shutdown()
        _state["server"].server_close()
        _state.update(server=None, base=None)

import subprocess
import sys
from pathlib import Path


def _relaunch_in_venv():
    """`python -m controller` from a Python without the project's packages: run again with the project .venv."""
    root = Path(__file__).resolve().parent.parent
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python"):
        py = root / rel
        if py.exists() and Path(sys.executable).resolve() != py.resolve():
            print(f"[controller] this Python lacks the project packages; relaunching with {py}")
            raise SystemExit(subprocess.call([str(py), "-m", "controller"] + sys.argv[1:], cwd=str(root)))
    raise SystemExit("PySide6 is not installed. Run:  .venv\\Scripts\\python.exe -m pip install -r requirements-controller.txt")


try:
    import PySide6  # noqa: F401
except ImportError:
    _relaunch_in_venv()

from .app import main

main()

"""`python modeltest` entry point: re-launches src/modeltest.py with the project's .venv interpreter,
so it works even when the plain `python` on PATH has none of the dependencies."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not VENV_PY.exists():
    VENV_PY = ROOT / ".venv" / "bin" / "python"
py = str(VENV_PY) if VENV_PY.exists() else sys.executable
if not VENV_PY.exists():
    print("WARNING: .venv not found, using the current interpreter (see README setup).")
sys.exit(subprocess.call([py, str(ROOT / "src" / "modeltest.py")] + sys.argv[1:]))

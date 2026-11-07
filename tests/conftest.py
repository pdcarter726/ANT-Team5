import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest

from common import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()

from pathlib import Path

import pytest


@pytest.fixture
def tiny_repo() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "tiny_repo"

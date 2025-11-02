from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from sylphy.logging import reset_logging


@pytest.fixture(autouse=True)
def clean_env_and_reset(tmp_path, monkeypatch) -> Iterator[None]:
    """Reset logger and clear environment variables between tests."""
    for key in list(os.environ.keys()):
        if key.startswith("SYLPHY_LOG_"):
            monkeypatch.delenv(key, raising=False)

    reset_logging("sylphy")
    yield
    reset_logging("sylphy")

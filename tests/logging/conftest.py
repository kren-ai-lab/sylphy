from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from sylphy.logging import reset_logging

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def clean_env_and_reset(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset logger and clear environment variables between tests."""
    for key in list(os.environ.keys()):
        if key.startswith("SYLPHY_LOG_"):
            monkeypatch.delenv(key, raising=False)

    reset_logging("sylphy")
    yield
    reset_logging("sylphy")

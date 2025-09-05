# tests/conftest.py
from __future__ import annotations

import os
from typing import Iterator

import pytest

from sylphy.logging import reset_logging

PKG_LOGGER_NAME = "protein_representation"


@pytest.fixture(autouse=True)
def clean_env_and_reset(tmp_path, monkeypatch) -> Iterator[None]:
    """
    Clean logging-related env vars and reset our logger before each test.
    This keeps tests hermetic and prevents handler duplication across tests.
    """
    # Unset all PR_LOG_* variables
    for key in list(os.environ.keys()):
        if key.startswith("PR_LOG_"):
            monkeypatch.delenv(key, raising=False)

    # Ensure no handlers are left from a previous test
    reset_logging(PKG_LOGGER_NAME)

    yield

    # Cleanup at teardown as well (paranoia)
    reset_logging(PKG_LOGGER_NAME)

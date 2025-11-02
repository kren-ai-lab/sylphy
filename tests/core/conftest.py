from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from sylphy.core.config import set_cache_root
from sylphy.core.model_registry import clear_registry


@pytest.fixture(autouse=True)
def _isolate_env_and_registry(monkeypatch, tmp_path) -> Iterator[None]:
    """Isolate environment variables, cache, and model registry for each test."""
    for key in list(os.environ.keys()):
        if key.startswith("SYLPHY_LOG_") or key.startswith("SYLPHY_MODEL_"):
            monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("SYLPHY_LOG_STDERR", "0")
    set_cache_root(tmp_path)
    clear_registry()

    yield

    clear_registry()

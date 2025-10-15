from __future__ import annotations

import os
import typing as t

import pytest

from sylphy.core.config import set_cache_root
from sylphy.core.model_registry import clear_registry

PKG_LOG_ENV_PREFIX = "PR_LOG_"
PKG_MODEL_ENV_PREFIX = "PR_MODEL_"


@pytest.fixture(autouse=True)
def _isolate_env_and_registry(monkeypatch, tmp_path) -> t.Iterator[None]:
    # Clean project-specific prefixes you used antes
    for key in list(os.environ.keys()):
        if key.startswith(PKG_LOG_ENV_PREFIX) or key.startswith(PKG_MODEL_ENV_PREFIX):
            monkeypatch.delenv(key, raising=False)

    # Silence sylphy logs to keep pytest output clean
    monkeypatch.setenv("SYLPHY_LOG_STDERR", "0")

    # Isolate cache under tmp
    set_cache_root(tmp_path)

    # Fresh registry
    clear_registry()

    yield

    # Cleanup
    clear_registry()

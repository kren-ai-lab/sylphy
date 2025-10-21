from __future__ import annotations

"""
Test bootstrap for embedding_extractor tests.

- Stubs model resolution to a temporary directory (no network).
- Keeps logging quiet and confined to tmp.

Note: Fake transformers and esm modules are now provided by the root conftest.py
"""

import os
from typing import Iterator

import pytest


# ----------------------------
# Pytest fixtures (env & resolver)
# ----------------------------


@pytest.fixture(autouse=True)
def _quiet_logs(tmp_path, monkeypatch) -> Iterator[None]:
    """Confine logging to tmp and disable any pre-existing SYLPHY_LOG_* env vars."""
    for k in list(os.environ.keys()):
        if k.startswith("SYLPHY_LOG_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SYLPHY_LOG_FILE", str(tmp_path / "embeddings.log"))
    yield


@pytest.fixture(autouse=True)
def _stub_resolve_model(tmp_path, monkeypatch) -> Iterator[None]:
    """
    Make the model resolver return a local temp directory so backends can load
    without network or real artifacts.
    """
    from sylphy.core import model_registry as reg

    local = tmp_path / "fake_model_dir"
    local.mkdir(parents=True, exist_ok=True)
    (local / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(reg, "resolve_model", lambda name: local, raising=True)
    yield

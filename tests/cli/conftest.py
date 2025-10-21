# tests/cli/conftest.py
from __future__ import annotations

import os
from typing import Iterator

import pytest


@pytest.fixture(autouse=True)
def _quiet_logs(tmp_path, monkeypatch) -> Iterator[None]:
    """Redirect logs to a temporary file and clean SYLPHY_LOG_* environment variables."""
    for k in list(os.environ.keys()):
        if k.startswith("SYLPHY_LOG_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SYLPHY_LOG_FILE", str(tmp_path / "cli.log"))
    yield


@pytest.fixture(autouse=True)
def _stub_model_registry(tmp_path, monkeypatch):
    """Point model resolution to a temporary directory to avoid network calls."""
    model_dir = tmp_path / "fake_model_dir"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    from sylphy.core import model_registry as reg

    monkeypatch.setattr(reg, "resolve_model", lambda name: model_dir, raising=True)
    yield

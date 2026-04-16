from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _quiet_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect logs to a temporary file and clean SYLPHY_LOG_* environment variables."""
    for k in list(os.environ.keys()):
        if k.startswith("SYLPHY_LOG_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SYLPHY_LOG_FILE", str(tmp_path / "embeddings.log"))


@pytest.fixture(autouse=True)
def _stub_resolve_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point model resolution to a temporary directory to avoid network calls."""
    from sylphy.core import model_registry as reg
    from sylphy.embedding_extractor import embedding_based

    local = tmp_path / "fake_model_dir"
    local.mkdir(parents=True, exist_ok=True)
    (local / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(reg, "resolve_model", lambda _: local, raising=True)
    monkeypatch.setattr(embedding_based, "resolve_model", lambda _: local, raising=True)

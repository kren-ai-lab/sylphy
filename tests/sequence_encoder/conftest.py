from __future__ import annotations

import os
from typing import TYPE_CHECKING

import polars as pl
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _quiet_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect logs to a temporary file and clean SYLPHY_LOG_* environment variables."""
    for k in list(os.environ.keys()):
        if k.startswith("SYLPHY_LOG_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SYLPHY_LOG_FILE", str(tmp_path / "seqenc.log"))


@pytest.fixture
def toy_df() -> pl.DataFrame:
    """Provide a small DataFrame with canonical sequences of varying lengths."""
    return pl.DataFrame({"sequence": ["ACD", "WYYVV", "KLMNPQ", "GGG"]})

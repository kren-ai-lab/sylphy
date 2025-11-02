from __future__ import annotations

import os
from collections.abc import Iterator

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _quiet_logs(tmp_path, monkeypatch) -> Iterator[None]:
    """Redirect logs to a temporary file and clean SYLPHY_LOG_* environment variables."""
    for k in list(os.environ.keys()):
        if k.startswith("SYLPHY_LOG_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SYLPHY_LOG_FILE", str(tmp_path / "seqenc.log"))
    yield


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """Provide a small DataFrame with canonical sequences of varying lengths."""
    return pd.DataFrame({"sequence": ["ACD", "WYYVV", "KLMNPQ", "GGG"]})

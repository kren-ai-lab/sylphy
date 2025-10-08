from __future__ import annotations

import os
from typing import Iterator

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch) -> Iterator[None]:
    """
    Keep logs inside tmp and remove any SYLPHY_LOG_* between tests to avoid
    cross-test leakage or handler duplication.
    """
    for k in list(os.environ.keys()):
        if k.startswith("SYLPHY_LOG_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SYLPHY_LOG_FILE", str(tmp_path / "seqenc.log"))
    yield


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """
    Canonical short sequences with some variability in length.
    """
    return pd.DataFrame({"sequence": ["ACD", "WYYVV", "KLMNPQ", "GGG"]})

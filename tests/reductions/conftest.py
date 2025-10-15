from __future__ import annotations

"""
Shared fixtures for reduction tests: quiet logs, small synthetic datasets.
"""

import os
from typing import Iterator

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _quiet_logs_and_clean_env(tmp_path, monkeypatch) -> Iterator[None]:
    """Silence console logs and write logs under a temp file, clearing SYLPHY_LOG_* envs."""
    for k in list(os.environ.keys()):
        if k.startswith("SYLPHY_LOG_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SYLPHY_LOG_FILE", str(tmp_path / "reductions.log"))
    yield


@pytest.fixture
def X_small() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(12, 6)).astype("float32")


@pytest.fixture
def X_nonneg() -> np.ndarray:
    """Non-negative data for NMF tests."""
    rng = np.random.default_rng(1)
    return np.abs(rng.normal(size=(10, 5))).astype("float32")


@pytest.fixture
def df_small(X_small) -> pd.DataFrame:
    cols = [f"f{i}" for i in range(X_small.shape[1])]
    return pd.DataFrame(X_small, columns=cols)

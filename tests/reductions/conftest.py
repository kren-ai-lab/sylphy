from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _quiet_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect logs to a temporary file and clean SYLPHY_LOG_* environment variables."""
    for k in list(os.environ.keys()):
        if k.startswith("SYLPHY_LOG_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SYLPHY_LOG_FILE", str(tmp_path / "reductions.log"))


@pytest.fixture
def X_small() -> np.ndarray:
    """Provide a small (12x6) random array for reduction tests."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(12, 6)).astype("float32")


@pytest.fixture
def X_nonneg() -> np.ndarray:
    """Provide non-negative (10x5) random array for NMF tests."""
    rng = np.random.default_rng(1)
    return np.abs(rng.normal(size=(10, 5))).astype("float32")


@pytest.fixture
def df_small(X_small: np.ndarray) -> pd.DataFrame:
    """Wrap X_small in a DataFrame with feature columns."""
    columns = pd.Index([f"f{i}" for i in range(X_small.shape[1])])
    return pd.DataFrame(X_small, columns=columns)

# tests/reductions/conftest.py
from __future__ import annotations

import os
from typing import Iterator

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _quiet_logs_and_clean_env(tmp_path, monkeypatch) -> Iterator[None]:
    # Silencia salidas y escribe logs en un archivo temporal
    for k in list(os.environ.keys()):
        if k.startswith("PR_LOG_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PR_LOG_FILE", str(tmp_path / "reductions.log"))
    yield


@pytest.fixture
def X_small() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(12, 6)).astype("float32")


@pytest.fixture
def X_nonneg() -> np.ndarray:
    # para NMF: datos no negativos
    rng = np.random.default_rng(1)
    return np.abs(rng.normal(size=(10, 5))).astype("float32")


@pytest.fixture
def df_small(X_small) -> pd.DataFrame:
    import pandas as pd
    cols = [f"f{i}" for i in range(X_small.shape[1])]
    return pd.DataFrame(X_small, columns=cols)

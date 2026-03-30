from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from sylphy.misc.utils_lib import UtilsLib


def test_parquet_export_missing_engine_suggests_parquet_extra(monkeypatch, tmp_path: Path):
    df = pd.DataFrame({"sequence": ["AAAA"]})

    def _boom(self: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        raise ImportError("Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom)

    with pytest.raises(ImportError, match=r"sylphy\[parquet\]"):
        UtilsLib.export_data(df, tmp_path / "out.parquet", file_format="parquet")

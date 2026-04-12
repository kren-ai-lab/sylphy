from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from sylphy.misc.utils_lib import UtilsLib

if TYPE_CHECKING:
    from pathlib import Path


def test_parquet_export_missing_engine_suggests_parquet_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    df = pd.DataFrame({"sequence": ["AAAA"]})

    def _boom(_self: pd.DataFrame, *_args: object, **_kwargs: object) -> object:
        msg = "Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'."
        raise ImportError(msg)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom)

    with pytest.raises(ImportError, match=r"sylphy\[parquet\]"):
        UtilsLib.export_data(df, tmp_path / "out.parquet", file_format="parquet")

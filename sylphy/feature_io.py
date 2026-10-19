"""Write feature tables keyed by record id (shared by the CLI and the Python API)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from sylphy.exceptions import DataError

if TYPE_CHECKING:
    import polars as pl

    from sylphy.types import FileFormat


def check_ids(df: pl.DataFrame, id_col: str) -> pl.Series:
    """Return ``df[id_col]`` as strings; raise ``DataError`` if missing, empty or duplicated."""
    import polars as pl  # noqa: PLC0415

    if id_col not in df.columns:
        msg = f"Id column '{id_col}' not found. Available: {df.columns}"
        raise DataError(msg)
    ids = df[id_col].cast(pl.String).fill_null("")
    if (ids.str.strip_chars() == "").any():
        msg = f"Empty ids in column '{id_col}'."
        raise DataError(msg)
    dups = ids.filter(ids.is_duplicated()).unique().sort().to_list()
    if dups:
        msg = f"Duplicate ids in column '{id_col}' (e.g. {dups[:3]})."
        raise DataError(msg)
    return ids


def write_features(features: pl.DataFrame, ids: pl.Series, output: Path, fmt: FileFormat) -> None:
    """Write numeric ``features`` keyed by ``ids`` (tables: ``id`` first; arrays: ids alongside)."""
    import numpy as np  # noqa: PLC0415
    import polars as pl  # noqa: PLC0415

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    id_list = ids.cast(pl.String).to_list()
    if fmt in {"csv", "parquet"}:
        table = pl.DataFrame({"id": id_list}, schema={"id": pl.String}).hstack(features)
        if fmt == "csv":
            table.write_csv(output)
        else:
            table.write_parquet(output)
        return
    values = features.to_numpy()
    if fmt == "npz":
        np.savez_compressed(
            output, values=values, columns=np.array(features.columns), ids=np.array(id_list, dtype=str)
        )
    else:
        np.save(output, values, allow_pickle=False)
        output.with_suffix(".ids.txt").write_text("".join(f"{i}\n" for i in id_list), encoding="utf-8")

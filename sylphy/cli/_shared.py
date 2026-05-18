from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, cast, get_args

import typer

from sylphy.types import FileFormat

if TYPE_CHECKING:
    import polars as pl

HELP_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

EXPORT_CHOICES: tuple[FileFormat, ...] = get_args(FileFormat)
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def level_from_str(name: str) -> int:
    """Map a string log level to the stdlib logging constant."""
    return getattr(logging, (name or "INFO").upper(), logging.INFO)


def load_dataset(input_path: Path, seq_col: str) -> pl.DataFrame:
    """Load a CSV, TSV, or Parquet file and validate the sequence column."""
    import polars as pl  # noqa: PLC0415

    if not input_path.exists():
        msg = f"Input file not found: {input_path}"
        raise typer.BadParameter(msg)

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        df = pl.read_csv(input_path)
    elif suffix == ".tsv":
        df = pl.read_csv(input_path, separator="\t")
    elif suffix == ".parquet":
        df = pl.read_parquet(input_path)
    else:
        msg = f"Unsupported input format '{suffix}'. Supported: csv, tsv, parquet"
        raise typer.BadParameter(msg)

    if seq_col not in df.columns:
        msg = f"Column '{seq_col}' not found. Available: {df.columns}"
        raise typer.BadParameter(msg)

    return df.with_columns(pl.col(seq_col).cast(pl.String).fill_null("").alias(seq_col))


def infer_format(path: Path) -> FileFormat:
    """Infer and validate output format from path extension."""
    supported = ", ".join(EXPORT_CHOICES)
    if not path.suffix:
        msg = f"Output path must include a file extension. Supported: {supported}"
        raise typer.BadParameter(msg)
    ext = path.suffix.lstrip(".").lower()
    if ext not in EXPORT_CHOICES:
        msg = f"Unsupported output format '.{ext}'. Supported: {supported}"
        raise typer.BadParameter(msg)
    return cast("FileFormat", ext)

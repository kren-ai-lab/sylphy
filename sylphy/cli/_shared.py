from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, TypeVar, get_args

import typer

from sylphy.types import FileFormat

_T = TypeVar("_T", bound=str)

if TYPE_CHECKING:
    import polars as pl

HELP_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


EXPORT_CHOICES: tuple[FileFormat, ...] = get_args(FileFormat)
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def level_from_str(name: str) -> int:
    """Map a string log level to the stdlib logging constant."""
    return getattr(logging, (name or "INFO").upper(), logging.INFO)


def validate_choice(value: str, choices: tuple[_T, ...], opt: str) -> _T:  # noqa: UP047
    """Validate a CLI option against a list of case-insensitive choices."""
    normalized = (value or "").strip().lower()
    allowed: dict[str, _T] = {choice.lower(): choice for choice in choices}
    if normalized not in allowed:
        msg = f"Invalid {opt}: {value!r}. Allowed: {', '.join(choices)}"
        raise typer.BadParameter(msg)
    return allowed[normalized]


def load_csv(input_path: Path, seq_col: str) -> pl.DataFrame:
    """Load a CSV file and validate the requested sequence column."""
    import polars as pl  # noqa: PLC0415

    if not input_path.exists():
        msg = f"Input file not found: {input_path}"
        raise typer.BadParameter(msg)
    if input_path.suffix.lower() != ".csv":
        msg = "Only CSV is supported as input."
        raise typer.BadParameter(msg)
    df = pl.scan_csv(input_path).collect()
    if seq_col not in df.columns:
        msg = f"Column '{seq_col}' not found. Available: {df.columns}"
        raise typer.BadParameter(msg)
    return df.with_columns(pl.col(seq_col).cast(pl.String).fill_null("").alias(seq_col))


def ensure_ext(path: Path, fmt: str) -> Path:
    """Append the requested extension only when the user omitted one."""
    fmt = fmt.lower().lstrip(".")
    return path if path.suffix else path.with_suffix(f".{fmt}")

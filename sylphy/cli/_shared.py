from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    import pandas as pd

HELP_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


EXPORT_CHOICES = ("csv", "npy", "npz", "parquet")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def level_from_str(name: str) -> int:
    """Map a string log level to the stdlib logging constant."""
    return getattr(logging, (name or "INFO").upper(), logging.INFO)


def validate_choice(value: str, choices: tuple[str, ...], opt: str) -> str:
    """Validate a CLI option against a list of case-insensitive choices."""
    normalized = (value or "").strip().lower()
    allowed = {choice.lower(): choice for choice in choices}
    if normalized not in allowed:
        msg = f"Invalid {opt}: {value!r}. Allowed: {', '.join(choices)}"
        raise typer.BadParameter(msg)
    return allowed[normalized]


def load_csv(input_path: Path, seq_col: str) -> pd.DataFrame:
    """Lazy-load a CSV file and validate the requested sequence column."""
    if not input_path.exists():
        msg = f"Input file not found: {input_path}"
        raise typer.BadParameter(msg)
    if input_path.suffix.lower() != ".csv":
        msg = "Only CSV is supported as input."
        raise typer.BadParameter(msg)
    try:
        import pandas as pd  # noqa: PLC0415
    except Exception as exc:
        msg = "pandas is required to read CSV input."
        raise typer.BadParameter(msg) from exc

    df = pd.read_csv(input_path)
    if seq_col not in df.columns:
        msg = f"Column '{seq_col}' not found. Available: {list(df.columns)}"
        raise typer.BadParameter(msg)
    df[seq_col] = df[seq_col].astype(str).fillna("")
    return df


def ensure_ext(path: Path, fmt: str) -> Path:
    """Append the requested extension only when the user omitted one."""
    fmt = fmt.lower().lstrip(".")
    return path if path.suffix else path.with_suffix(f".{fmt}")

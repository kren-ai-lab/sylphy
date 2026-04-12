from __future__ import annotations

import logging
from pathlib import Path

import typer

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
        raise typer.BadParameter(f"Invalid {opt}: {value!r}. Allowed: {', '.join(choices)}")
    return allowed[normalized]


def load_csv(input_path: Path, seq_col: str):
    """Lazy-load a CSV file and validate the requested sequence column."""
    if not input_path.exists():
        raise typer.BadParameter(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".csv":
        raise typer.BadParameter("Only CSV is supported as input.")
    try:
        import pandas as pd
    except Exception as exc:
        raise typer.BadParameter("pandas is required to read CSV input.") from exc

    df = pd.read_csv(input_path)
    if seq_col not in df.columns:
        raise typer.BadParameter(f"Column '{seq_col}' not found. Available: {list(df.columns)}")
    df[seq_col] = df[seq_col].astype(str).fillna("")
    return df


def ensure_ext(path: Path, fmt: str) -> Path:
    """Append the requested extension only when the user omitted one."""
    fmt = fmt.lower().lstrip(".")
    return path if path.suffix else path.with_suffix(f".{fmt}")

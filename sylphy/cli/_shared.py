from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

import typer

from sylphy import __version__
from sylphy.cli._config_file import apply_config
from sylphy.cli._provenance import write_provenance
from sylphy.exceptions import DataError
from sylphy.feature_io import check_ids
from sylphy.types import FileFormat

if TYPE_CHECKING:
    import polars as pl

HELP_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

EXPORT_CHOICES: tuple[FileFormat, ...] = get_args(FileFormat)
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


# Options that stay on the command line and are not provenance parameters.
CONFIG_EXCLUDE = frozenset({"input_path", "output", "config"})


def config_callback(ctx: typer.Context, value: Path | None) -> Path | None:
    """Eager ``--config`` callback: install the file as defaults for the other options."""
    apply_config(ctx, value, exclude=CONFIG_EXCLUDE)
    return value


CONFIG_OPTION = typer.Option(
    None,
    "--config",
    callback=config_callback,
    is_eager=True,
    dir_okay=False,
    help="YAML/JSON file with option values (keys are option names, e.g. max_length).",
)


def write_run_provenance(
    ctx: typer.Context,
    output: Path,
    *,
    command: str,
    created_at: str,
    n_rows: int,
    n_cols: int,
    packages: tuple[str, ...],
    extensions: dict[str, Any] | None = None,
) -> None:
    """Write the provenance sidecar for a finished encode/embed run."""
    # log_level doesn't change results, so it stays out of the fingerprint
    params = {
        k: (v.value if isinstance(v, Enum) else v)
        for k, v in ctx.params.items()
        if k not in CONFIG_EXCLUDE and k != "log_level"
    }
    write_provenance(
        output,
        tool="sylphy",
        version=__version__,
        command=command,
        created_at=created_at,
        parameters=params,
        inputs=[Path(ctx.params["input_path"])],
        packages=packages,
        output_info={"n_rows": n_rows, "n_cols": n_cols},
        extensions=extensions,
    )


def level_from_str(name: str) -> int:
    """Map a string log level to the stdlib logging constant."""
    return getattr(logging, (name or "INFO").upper(), logging.INFO)


FASTA_SUFFIXES = (".fa", ".fasta", ".faa")


def _read_fasta(path: Path, id_col: str, seq_col: str) -> pl.DataFrame:
    """Parse a FASTA file; the id is the first whitespace-delimited token of each header."""
    import polars as pl  # noqa: PLC0415

    ids: list[str] = []
    seqs: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith(">"):
            header = line[1:].split(maxsplit=1)
            ids.append(header[0] if header else "")
            seqs.append("")
        elif line and seqs:
            seqs[-1] += line
        elif line:
            msg = f"Invalid FASTA {path}: sequence data before the first header."
            raise DataError(msg)
    return pl.DataFrame({id_col: ids, seq_col: seqs}, schema={id_col: pl.String, seq_col: pl.String})


def load_dataset(input_path: Path, seq_col: str, id_col: str = "id") -> pl.DataFrame:
    """Load a FASTA, CSV, TSV, or Parquet file; validate the id and sequence columns.

    The id column is cast to string and must be present, non-empty and unique.
    """
    import polars as pl  # noqa: PLC0415

    if not input_path.exists():
        msg = f"Input file not found: {input_path}"
        raise DataError(msg)

    suffix = input_path.suffix.lower()
    if suffix in FASTA_SUFFIXES:
        df = _read_fasta(input_path, id_col, seq_col)
    elif suffix == ".csv":
        df = pl.read_csv(input_path, infer_schema=False)
    elif suffix == ".tsv":
        df = pl.read_csv(input_path, separator="\t", infer_schema=False)
    elif suffix == ".parquet":
        df = pl.read_parquet(input_path)
    else:
        msg = f"Unsupported input format '{suffix}'. Supported: fasta (fa/faa), csv, tsv, parquet"
        raise typer.BadParameter(msg)

    if id_col not in df.columns:
        msg = f"Id column '{id_col}' not found; set --id-col. Available: {df.columns}"
        raise typer.BadParameter(msg)
    if seq_col not in df.columns:
        msg = f"Column '{seq_col}' not found. Available: {df.columns}"
        raise typer.BadParameter(msg)

    df = df.with_columns(
        pl.col(id_col).cast(pl.String).fill_null(""),
        pl.col(seq_col).cast(pl.String).fill_null("").alias(seq_col),
    )
    try:
        check_ids(df, id_col)
    except DataError as exc:
        msg = f"{exc} Fix the input or set --id-col."
        raise DataError(msg) from exc
    return df


def snapshot_commit(model_path: str | None) -> str | None:
    """Return the commit of a HuggingFace cache path (``.../snapshots/<commit>/...``), else ``None``."""
    if not model_path:
        return None
    parts = Path(model_path).parts
    if "snapshots" in parts[:-1]:
        return parts[parts.index("snapshots") + 1]
    return None


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
    return ext

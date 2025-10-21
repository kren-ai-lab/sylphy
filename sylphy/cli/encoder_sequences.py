"""sylphy/cli/encoder_sequences.py

Unified CLI to encode protein/peptide sequences with:
- one_hot
- ordinal
- frequency
- kmers (TF-IDF)
- physicochemical (AAIndex / group_based)
- fft (pipeline: physicochemical -> FFT)

Design goals:
- Factory-first: build encoders via sylphy.sequence_encoder.factory
- Lazy imports: avoid loading heavy deps at CLI startup
- Consistent exports: ensure proper file extensions based on --format-output
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="encode-sequences",
    help="Encode sequences with one-hot, ordinal, frequency, k-mers, physicochemical (AAIndex/group_based) or FFT.",
    no_args_is_help=True,
)

# ---- Declarative option sets (keep stdlib-only at import-time) ----
ENCODER_CHOICES = ("one_hot", "ordinal", "frequency", "kmers", "physicochemical", "fft")
DESCRIPTOR_CHOICES = ("aaindex", "group_based")
EXPORT_CHOICES = ("csv", "npy", "npz", "parquet")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _level_from_str(name: str) -> int:
    """Map string log level to logging constant (lazy import)."""
    import logging

    return getattr(logging, (name or "INFO").upper(), logging.INFO)


def _validate_choice(value: str, choices: tuple[str, ...], opt: str) -> str:
    """Validate a CLI option against a list of choices (case-insensitive)."""
    v = (value or "").strip().lower()
    allowed = {c.lower(): c for c in choices}
    if v not in allowed:
        raise typer.BadParameter(f"Invalid {opt}: {value!r}. Allowed: {', '.join(choices)}")
    return allowed[v]


def _load_csv(input_path: Path, seq_col: str):
    """Lazy-load CSV via pandas and validate the sequence column."""
    if not input_path.exists():
        raise typer.BadParameter(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".csv":
        raise typer.BadParameter("Only CSV is supported as input.")
    try:
        import pandas as pd  # lazy
    except Exception as exc:
        raise typer.BadParameter("pandas is required to read CSV input.") from exc

    df = pd.read_csv(input_path)
    if seq_col not in df.columns:
        raise typer.BadParameter(f"Column '{seq_col}' not found. Available: {list(df.columns)}")
    df[seq_col] = df[seq_col].astype(str).fillna("")
    return df


def _ensure_ext(path: Path, fmt: str) -> Path:
    """Ensure output path has the correct extension based on fmt.

    Rules:
    - If path already has a suffix and it matches fmt (case-insensitive), keep it.
    - If path has a different suffix, keep user's suffix (do NOT override).
    - If path has no suffix, append .fmt
    """
    fmt = fmt.lower().lstrip(".")
    if path.suffix:
        # Keep user's explicit suffix
        return path
    return path.with_suffix(f".{fmt}")


@app.command("run")
def run(
    # encoder / pipeline
    encoder: str = typer.Option(
        "physicochemical",
        "--encoder",
        "-e",
        help=f"Encoder backend. One of: {', '.join(ENCODER_CHOICES)}.",
        show_default=True,
    ),
    # dataset options
    input_data: Path = typer.Option(..., "--input-data", "-i", help="CSV file with sequences."),
    sequence_identifier: str = typer.Option(
        "sequence", "--sequence-identifier", "-s", help="Sequence column name."
    ),
    max_length: int = typer.Option(1024, "--max-length", "-m", help="Max sequence length (when applicable)."),
    allow_extended: bool = typer.Option(
        False, "--allow-extended/--no-allow-extended", help="Enable extended alphabet (B, Z, X, U, O)."
    ),
    allow_unknown: bool = typer.Option(
        False, "--allow-unknown/--no-allow-unknown", help="Allow 'X' when extended alphabet is not enabled."
    ),
    # backend-specific
    type_descriptor: Optional[str] = typer.Option(
        "aaindex",
        "--type-descriptor",
        "-t",
        help=f"Descriptor space for physicochemical encoders (and FFT pre-step). One of: {', '.join(DESCRIPTOR_CHOICES)}.",
        show_default=True,
    ),
    name_property: str = typer.Option(
        "ANDN920101",
        "--name-property",
        "-n",
        help="Property/column name in the descriptor table (AAIndex key or group_based label).",
        show_default=True,
    ),
    size_kmer: int = typer.Option(3, "--size-kmer", "-k", help="k for TF-IDF k-mers (kmers backend)."),
    # output
    output: Path = typer.Option(..., "--output", "-o", help="Output file path (extension can be omitted)."),
    format_output: str = typer.Option(
        "csv",
        "--format-output",
        "-f",
        help=f"Output format. One of: {', '.join(EXPORT_CHOICES)}.",
        show_default=True,
    ),
    # logging
    debug: bool = typer.Option(False, "--debug/--no-debug", help="Enable verbose logs within encoders."),
    log_level: str = typer.Option(
        "INFO", "--log-level", help=f"Log level: {', '.join(LOG_LEVELS)}.", show_default=True
    ),
) -> None:
    """Encode sequences and export feature matrices using Sylphy's encoders.

    Notes
    -----
    - 'fft' runs as a pipeline: physicochemical (AAIndex/group_based) first, then FFT,
      so the signal is numerical before spectral analysis.
    - Alphabet and length validation are handled by the shared base encoder.
    """
    try:
        # Cheap validations (no heavy imports yet)
        enc_choice = _validate_choice(encoder, ENCODER_CHOICES, "encoder")
        fmt_choice = _validate_choice(format_output, EXPORT_CHOICES, "format-output")
        if enc_choice == "physicochemical" or enc_choice == "fft":
            _validate_choice(type_descriptor or "aaindex", DESCRIPTOR_CHOICES, "type-descriptor")

        level = _level_from_str(log_level)
        df = _load_csv(input_data, sequence_identifier)

        # Import factory only when the user actually runs the command
        from sylphy.sequence_encoder.factory import create_encoder

        # Compute final output path with ensured extension (fix for missing extensions)
        final_output = _ensure_ext(output, fmt_choice)

        # --- FFT pipeline (physicochemical -> FFT) ---
        if enc_choice == "fft":
            phys = create_encoder(
                "physicochemical",
                dataset=df,
                sequence_column=sequence_identifier,
                max_length=max_length,
                type_descriptor=(type_descriptor or "aaindex"),
                name_property=name_property,
                allow_extended=allow_extended,
                allow_unknown=allow_unknown,
                debug=debug,
                debug_mode=level,
            )
            phys.run_process()
            if phys.coded_dataset is None or phys.coded_dataset.empty:
                raise RuntimeError("Physicochemical step produced empty features.")

            fft = create_encoder(
                "fft",
                dataset=phys.coded_dataset,
                sequence_column=sequence_identifier,
                debug=debug,
                debug_mode=level,
            )
            fft.run_process()

            # Some implementations expect (data, path); others only (path).
            # Keep the more specific signature used for FFT if your encoder requires the dataset:
            try:
                fft.export_encoder(fft.coded_dataset, str(final_output), file_format=fmt_choice)
            except TypeError:
                fft.export_encoder(str(final_output), file_format=fmt_choice)

            typer.echo(f"[fft] Saved to: {final_output}")
            return

        # --- Single-step backends ---
        kwargs_common = dict(
            dataset=df,
            sequence_column=sequence_identifier,
            allow_extended=allow_extended,
            allow_unknown=allow_unknown,
            debug=debug,
            debug_mode=level,
        )

        if enc_choice in ("one_hot", "ordinal", "physicochemical"):
            kwargs_common["max_length"] = max_length

        if enc_choice == "physicochemical":
            kwargs_common["type_descriptor"] = type_descriptor or "aaindex"
            kwargs_common["name_property"] = name_property

        if enc_choice == "kmers":
            kwargs_common["size_kmer"] = size_kmer

        enc = create_encoder(enc_choice, **kwargs_common)
        enc.run_process()

        # Export with ensured extension; support both possible method signatures
        try:
            enc.export_encoder(str(final_output), file_format=fmt_choice)
        except TypeError:
            # Older/export variants that expect (data, path)
            enc.export_encoder(enc.coded_dataset, str(final_output), file_format=fmt_choice)

        typer.echo(f"[{enc_choice}] Saved to: {final_output}")

    except typer.BadParameter as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    except Exception as e:
        typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

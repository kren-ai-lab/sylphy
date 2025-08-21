# protein_representation/cli/encode_sequences.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Literal

import numpy as np
import pandas as pd
import typer

from protein_representation.sequence_encoder import (
    create_encoder,
    FFTEncoder,
)

app = typer.Typer(
    name="encode-sequences",
    help="Encode protein sequences using classical strategies (one-hot, ordinal, k-mers, physicochemical, frequency, FFT).",
)

def _load_csv(input_path: Path, seq_col: str) -> pd.DataFrame:
    if not input_path.exists():
        raise typer.BadParameter(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".csv":
        raise typer.BadParameter("Only CSV is supported in this command.")
    df = pd.read_csv(input_path)
    if seq_col not in df.columns:
        raise typer.BadParameter(
            f"Column '{seq_col}' not found. Available: {list(df.columns)}"
        )
    df[seq_col] = df[seq_col].astype(str).fillna("")
    return df

def _level_from_str(name: str) -> int:
    import logging
    mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return mapping.get((name or "").strip().upper(), logging.INFO)

@app.command()
def run(
    # Strategy
    encoder: Literal["onehot", "ordinal", "fft", "physicochemical", "frequency", "kmers", "kmer", "physchem"] = typer.Option(
        "onehot", "--encoder", "-e",
        help="Encoding strategy.",
        case_sensitive=False,
    ),
    # IO
    input_data: Path = typer.Option(
        ..., "--input-data", "-i",
        help="Input CSV path with sequences.",
    ),
    output: Path = typer.Option(
        ..., "--output", "-o",
        help="Output file path (CSV or NPY).",
    ),
    format_output: Literal["csv", "npy"] = typer.Option(
        "csv", "--format-output", "-f",
        help="Export format.",
        case_sensitive=False,
    ),
    sequence_identifier: str = typer.Option(
        "sequence", "--sequence-identifier", "-s",
        help="Column in CSV that contains amino acid sequences.",
    ),
    # Encoder params
    max_length: int = typer.Option(
        1024, "--max-length", "-L", min=1,
        help="Max length per sequence (padding/truncation, encoder dependent).",
    ),
    size_kmer: int = typer.Option(
        3, "--size-kmer", "-k", min=2,
        help="K-mer length for k-mers encoder.",
    ),
    type_descriptor: Literal["aaindex", "group_based"] = typer.Option(
        "aaindex", "--type-descriptor", "-t",
        help="Descriptor type for physicochemical encoder.",
        case_sensitive=False,
    ),
    name_property: str = typer.Option(
        "ANDN920101", "--name-property", "-n",
        help="Descriptor name (AAindex or group_based).",
    ),
    # Logging
    debug: bool = typer.Option(
        False, "--debug/--no-debug",
        help="Enable verbose logs for this command.",
    ),
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = typer.Option(
        "INFO", "--log-level",
        help="Library log level.",
        case_sensitive=False,
    ),
):
    """
    Examples
    --------
    \b
    protein_representation encode-sequences run \\
      --encoder kmers \\
      --input-data data/sequences.csv \\
      --output results/kmers_k3.csv \\
      --sequence-identifier sequence \\
      --size-kmer 3
    \b
    protein_representation encode-sequences run \\
      --encoder physicochemical \\
      --input-data data/sequences.csv \\
      --output results/physchem.csv \\
      --name-property ANDN920101
    \b
    # 'fft' applies FFT on top of a physicochemical vector
    protein_representation encode-sequences run \\
      --encoder fft \\
      --input-data data/sequences.csv \\
      --output results/physchem_fft.csv \\
      --name-property ANDN920101
    """
    try:
        df = _load_csv(input_data, sequence_identifier)

        if encoder.lower() == "fft":
            # 1) Physicochemical encoding
            phys = create_encoder(
                "physicochemical",
                dataset=df,
                sequence_column=sequence_identifier,
                max_length=max_length,
                debug=debug,
                debug_mode=_level_from_str(log_level),
                type_descriptor=type_descriptor,
                name_property=name_property,
            )
            phys.run_process()

            # 2) FFT on top of the numeric matrix
            fft_encoder = FFTEncoder(
                dataset=phys.coded_dataset,
                sequence_column=sequence_identifier,
                debug=debug,
                debug_mode=_level_from_str(log_level),
            )
            fft_encoder.run_process()
            fft_encoder.export_encoder(
                df_encoder=fft_encoder.coded_dataset,
                path=str(output),
                file_format=format_output.lower(),
            )
            typer.echo(f"Encoded features (physchem+FFT) saved to: {output}")
            return

        # Other encoders via factory
        enc = create_encoder(
            encoder,
            dataset=df,
            sequence_column=sequence_identifier,
            max_length=max_length,
            debug=debug,
            debug_mode=_level_from_str(log_level),
            size_kmer=size_kmer,
            type_descriptor=type_descriptor,
            name_property=name_property,
        )
        enc.run_process()
        enc.export_encoder(path=str(output), file_format=format_output.lower())
        typer.echo(f"Encoded features saved to: {output}")

    except typer.BadParameter as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    except Exception as e:
        typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

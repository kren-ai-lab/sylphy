# protein_representation/cli/get_embeddings.py
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from protein_representation.embedding_extractor import create_embedding  # lazy alias to EmbeddingFactory
from protein_representation.constants.cli_constants import (DebugMode, ExportOption, Device, Precision,
                                                            PoolOption)

app = typer.Typer(
    name="get-embedding",
    help="Extract protein sequence embeddings using a selected pretrained model.",
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
    # Model & backend
    model: str = typer.Option(
        "facebook/esm2_t6_8M_UR50D", "--model", "-m",
        help="Model identifier (HF ref or registry key), e.g. 'facebook/esm2_t6_8M_UR50D'."
    ),
    device: Device = typer.Option(
        "cuda", "--device", "-d",
        help="Inference device.",
        case_sensitive=False,
    ),
    precision: Precision = typer.Option(
        "fp32", "--precision", "-p",
        help="Mixed precision for CUDA (ignored on CPU).",
        case_sensitive=False,
    ),
    batch_size: int = typer.Option(
        8, "--batch-size", "-b", min=1,
        help="Batch size for inference.",
    ),
    max_length: int = typer.Option(
        1024, "--max-length", "-L", min=1,
        help="Max tokens per sequence (truncation).",
    ),
    pool: PoolOption = typer.Option(
        "mean", "--pool",
        help="Pooling strategy for last hidden states.",
        case_sensitive=False,
    ),
    oom_backoff: bool = typer.Option(
        True, "--oom-backoff/--no-oom-backoff",
        help="Auto-reduce batch size on CUDA OOM and retry.",
    ),
    # IO
    input_data: Path = typer.Option(
        ..., "--input-data", "-i",
        help="Input CSV with sequences.",
    ),
    sequence_identifier: str = typer.Option(
        "sequence", "--sequence-identifier", "-s",
        help="Column in CSV that contains amino acid sequences.",
    ),
    output: Path = typer.Option(
        ..., "--output", "-o",
        help="Output file for embeddings.",
    ),
    format_output: ExportOption = typer.Option(
        "csv", "--format-output", "-f",
        help="Export format.",
        case_sensitive=False,
    ),
    # Logging
    debug: bool = typer.Option(
        False, "--debug/--no-debug",
        help="Enable verbose logs for this command.",
    ),
    log_level: DebugMode = typer.Option(
        "INFO", "--log-level",
        help="Library log level.",
        case_sensitive=False,
    ),
):
    """
    Example:
      protein_representation get-embedding run \\
        --model facebook/esm2_t6_8M_UR50D \\
        --input-data datasets/demo_amp.csv \\
        --output results/emb_esm2.csv \\
        --sequence-identifier sequence \\
        --device cuda --precision fp16 --batch-size 16 --pool mean
    """
    try:
        df = _load_csv(input_data, sequence_identifier)

        embedder = create_embedding(
            model_name=model,
            dataset=df,
            column_seq=sequence_identifier,
            name_device=device.value.lower(),
            precision=precision.value.lower(),
            oom_backoff=oom_backoff,
            debug=debug,
            debug_mode=_level_from_str(log_level),
        )

        # New unified API from EmbeddingBased
        embedder.load_hf_tokenizer_and_model()
        embedder.run_process(max_length=max_length, batch_size=batch_size, pool=pool.value.lower())
        embedder.export_encoder(path=str(output), file_format=format_output.value.lower())

        typer.echo(f"Embeddings saved to: {output}")

    except typer.BadParameter as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    except Exception as e:
        typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

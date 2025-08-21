# protein_representation/cli/get_embeddings.py
from pathlib import Path
import pandas as pd
import typer

from protein_representation.embedding_extractor import embedding_factory

app = typer.Typer(
    name="get-embedding",
    help="Extract protein sequence embeddings using a selected pretrained model."
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

@app.command()
def run(
    model: str = typer.Option(
        "facebook/esm2_t6_8M_UR50D", "--model", "-m",
        help="Model identifier (HF ref or registry key)."
    ),
    input_data: Path = typer.Option(
        ..., "--input_data", "-i",
        help="Input CSV path with sequences."
    ),
    output: Path = typer.Option(
        ..., "--output", "-o",
        help="Output file path for embeddings (CSV or NPY)."
    ),
    format_output: str = typer.Option(
        "csv", "--format_output", "-f",
        help="Export format.",
        case_sensitive=False,
        show_choices=True,
    ),
    sequence_identifier: str = typer.Option(
        "sequence", "--sequence_identifier", "-s",
        help="Column in CSV that contains amino acid sequences."
    ),
    device: str = typer.Option(
        "cuda", "--device", "-d",
        help="Inference device.",
        case_sensitive=False,
        show_choices=True,
    ),
    precision: str = typer.Option(
        "fp32", "--precision", "-p",
        help="Mixed-precision for inference (saves VRAM on CUDA).",
        case_sensitive=False,
        show_choices=True,
    ),
    oom_backoff: int = typer.Option(
        1, "--oom_backoff",
        help="Auto-reduce batch size on CUDA OOM. (e.g, 1=Activate, 0=Deactivate)"
    ),
    debug: int = typer.Option(
        0, "--debug",
        help="Enable verbose logs in the library. (e.g, 1=Activate, 0=Deactivate)"
    ),
    debug_mode: int = typer.Option(
        20, "--debug_mode",
        help="Library log level as int (e.g., 10=DEBUG, 20=INFO, 30=WARNING)."
    ),
):
    """
    Example:
      protein_representation get-embedding run \\
        --model facebook/esm2_t6_8M_UR50D \\
        --input_data datasets/demo_amp.csv \\
        --output results/emb_esm2.csv \\
        --sequence_identifier sequence \\
        --device cuda --precision fp16
    """
    
    allowed_formats = {"csv", "npy"}
    allowed_devices = {"cuda", "cpu"}
    allowed_precisions = {"fp32", "fp16", "bf16"}
    allowed_debug_options = {0, 1}
    allowed_debug_mode = {10, 20, 30}
    allowed_oom_options = {0, 1}

    if format_output.lower() not in allowed_formats:
        raise typer.BadParameter(f"--format-output must be one of {sorted(allowed_formats)}")
    if device.lower() not in allowed_devices:
        raise typer.BadParameter(f"--device must be one of {sorted(allowed_devices)}")
    if precision.lower() not in allowed_precisions:
        raise typer.BadParameter(f"--precision must be one of {sorted(allowed_precisions)}")
    if int(debug) not in allowed_debug_options:
        raise typer.BadParameter(f"--debug must be one of {sorted(allowed_debug_options)}")
    if int(debug_mode) not in allowed_debug_mode:
        raise typer.BadParameter(f"--debug_mode must be one of {sorted(allowed_debug_mode)}")
    if int(oom_backoff) not in allowed_oom_options:
        raise typer.BadParameter(f"--oom_backoff must be one of {sorted(allowed_oom_options)}")
    try:
        df = _load_csv(input_data, sequence_identifier)

        embedder = embedding_factory.EmbeddingFactory(
            model_name=model,
            dataset=df,
            column_seq=sequence_identifier,
            name_device=device.lower(),
            precision=precision.lower(),
            oom_backoff=oom_backoff,
            debug=debug,
            debug_mode=debug_mode,
        )

        embedder.load_model_tokenizer()
        df_emb = embedder.embedding_process()
        embedder.cleaning_memory()

        embedder.export_embeddings(
            df_embedding=df_emb,
            path=str(output),
            file_format=format_output.lower(),
        )

        typer.echo(f"Embeddings saved to: {output}")

    except typer.BadParameter as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    except Exception as e:
        typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

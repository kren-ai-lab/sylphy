"""sylphy/cli/get_embeddings.py

Unified CLI to extract protein/peptide embeddings from pretrained models
(ESM2, ProtT5, ProtBERT, Ankh2, Mistral-Prot, ESM-C), using Sylphy's
factory and EmbeddingBased API.

Design goals
------------
- Factory-first (no backend is imported until execution).
- Lazy imports (pandas, torch/transformers load only inside the command).
- Consistent, explicit options (model, device, precision, batch, max_length).
- Robust OOM handling (halve batch size and retry, if enabled).
- Correct output file extension based on --format-output.

Backends are selected by the factory from the model name. See:
- EmbeddingFactory: backend selection by name substring/namespace.
- EmbeddingBased: run_process(), layer selection/aggregation, pooling, export.

"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="get-embedding",
    help="Extract protein sequence embeddings using a selected pretrained model.",
    no_args_is_help=True,
)

# ---- Declarative choices (keep stdlib-only at import time) -------------------
DEVICE_CHOICES = ("cuda", "cpu")
PRECISION_CHOICES = ("fp32", "fp16", "bf16")
POOL_CHOICES = ("mean", "cls", "eos")
LAYER_AGG_CHOICES = ("mean", "sum", "concat")
EXPORT_CHOICES = ("csv", "npy", "npz", "parquet")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


# ---- Small helpers -----------------------------------------------------------
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

    Rules
    -----
    - If path already has a suffix, keep user's suffix (do NOT override).
    - If path has no suffix, append .{fmt}.
    """
    fmt = fmt.lower().lstrip(".")
    return path if path.suffix else path.with_suffix(f".{fmt}")


# ---- Command ----------------------------------------------------------------
@app.command("run")
def run(
    # Model & backend
    model: str = typer.Option(
        "facebook/esm2_t6_8M_UR50D",
        "--model",
        "-m",
        help="Model identifier (HF ref or registry key), e.g. 'facebook/esm2_t6_8M_UR50D'.",
        show_default=True,
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        "-d",
        help=f"Inference device. One of: {', '.join(DEVICE_CHOICES)}.",
        show_default=True,
    ),
    precision: str = typer.Option(
        "fp32",
        "--precision",
        "-p",
        help=f"AMP precision on CUDA. One of: {', '.join(PRECISION_CHOICES)}.",
        show_default=True,
    ),
    batch_size: int = typer.Option(
        8,
        "--batch-size",
        "-b",
        min=1,
        help="Batch size for inference.",
        show_default=True,
    ),
    max_length: int = typer.Option(
        1024,
        "--max-length",
        "-L",
        min=1,
        help="Max tokens per sequence (truncation at tokenizer level).",
        show_default=True,
    ),
    oom_backoff: bool = typer.Option(
        True,
        "--oom-backoff/--no-oom-backoff",
        help="Auto-reduce batch size on CUDA OOM and retry.",
        show_default=True,
    ),
    # Layer/Pooling controls (handled by EmbeddingBased) :contentReference[oaicite:2]{index=2}
    layers: str = typer.Option(
        "last",
        "--layers",
        help="Layer selection: 'last' | 'last4' | 'all' | an integer index. "
        "Multiple indices can be provided as comma-separated (e.g., '0,3,6').",
        show_default=True,
    ),
    layer_agg: str = typer.Option(
        "mean",
        "--layer-agg",
        help=f"Aggregation across selected layers. One of: {', '.join(LAYER_AGG_CHOICES)}.",
        show_default=True,
    ),
    pool: str = typer.Option(
        "mean",
        "--pool",
        help=f"Token pooling strategy. One of: {', '.join(POOL_CHOICES)}.",
        show_default=True,
    ),
    # IO
    input_data: Path = typer.Option(
        ...,
        "--input-data",
        "-i",
        help="Input CSV with sequences.",
    ),
    sequence_identifier: str = typer.Option(
        "sequence",
        "--sequence-identifier",
        "-s",
        help="Column in CSV that contains amino acid sequences.",
        show_default=True,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output file path for embeddings (extension can be omitted).",
    ),
    format_output: str = typer.Option(
        "csv",
        "--format-output",
        "-f",
        help=f"Export format. One of: {', '.join(EXPORT_CHOICES)}.",
        show_default=True,
    ),
    # Logging
    debug: bool = typer.Option(
        False,
        "--debug/--no-debug",
        help="Enable verbose logs for this command.",
        show_default=True,
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help=f"Library log level: {', '.join(LOG_LEVELS)}.",
        show_default=True,
    ),
) -> None:
    """Extract embeddings and export them to disk using the chosen format.

    Notes
    -----
    - Backend selection is handled by the factory from the model name
      (ESM2, ProtT5, ProtBERT, Ankh2, Mistral-Prot, ESM-C). :contentReference[oaicite:3]{index=3}
    - The EmbeddingBased pipeline manages tokenizer/model loading, AMP precision,
      OOM backoff, layer selection/aggregation, pooling, and export. :contentReference[oaicite:4]{index=4}

    Examples
    --------
    sylphy get-embedding run \\
      -m facebook/esm2_t6_8M_UR50D -i data/demo.csv -s sequence \\
      -o results/emb_esm2 -f parquet -d cuda -p fp16 -b 16 --layers last4 --layer-agg mean --pool mean

    sylphy get-embedding run \\
      -m Rostlab/prot_t5_xl_uniref50 -i data/demo.csv -o results/emb_t5.npy -f npy -d cuda -p bf16

    sylphy get-embedding run \\
      -m ElnaggarLab/ankh2-ext1 -i data/demo.csv -o results/emb_ankh -f csv --layers all --layer-agg sum --pool cls
    """
    try:
        # Cheap validations first (keep startup fast)
        device_v = _validate_choice(device, DEVICE_CHOICES, "device")
        precision_v = _validate_choice(precision, PRECISION_CHOICES, "precision")
        pool_v = _validate_choice(pool, POOL_CHOICES, "pool")
        layer_agg_v = _validate_choice(layer_agg, LAYER_AGG_CHOICES, "layer-agg")
        fmt_v = _validate_choice(format_output, EXPORT_CHOICES, "format-output")
        lvl = _level_from_str(log_level)

        # CSV → DataFrame (lazy pandas)
        df = _load_csv(input_data, sequence_identifier)

        # Parse layers spec for convenience: accept ints or CSV of ints
        layers_spec: object
        ls = (layers or "last").strip().lower()
        if ls in {"last", "last4", "all"}:
            layers_spec = ls
        else:
            # e.g., "0", "0,3,6"
            try:
                if "," in ls:
                    layers_spec = [int(x.strip()) for x in ls.split(",") if x.strip() != ""]
                else:
                    layers_spec = int(ls)
            except ValueError:
                raise typer.BadParameter(
                    "Invalid --layers. Use 'last' | 'last4' | 'all' | an integer | comma-separated integers."
                )

        # Import the factory only here (avoid heavy imports at module import time)
        # Factory chooses backend based on model name. :contentReference[oaicite:5]{index=5}
        from sylphy.embedding_extractor import (
            create_embedding,  # lazy alias to EmbeddingFactory :contentReference[oaicite:6]{index=6}
        )

        embedder = create_embedding(
            model_name=model,
            dataset=df,
            column_seq=sequence_identifier,
            name_device=device_v,
            precision=precision_v,
            oom_backoff=oom_backoff,
            debug=debug,
            debug_mode=lvl,
        )

        # Run the unified embedding pipeline (loads model/tokenizer internally if needed) :contentReference[oaicite:7]{index=7}
        embedder.run_process(
            max_length=max_length,
            batch_size=batch_size,
            layers=layers_spec,  # "last" | "last4" | "all" | int | [ints]
            layer_agg=layer_agg_v,  # "mean" | "sum" | "concat"
            pool=pool_v,  # "mean" | "cls" | "eos"
        )

        # Ensure output extension and export (supports csv/npy/npz/parquet) :contentReference[oaicite:8]{index=8}
        final_output = _ensure_ext(output, fmt_v)
        try:
            embedder.export_encoder(str(final_output), file_format=fmt_v)
        except TypeError:
            # Backward-compat signature (some older backends expect (data, path))
            embedder.export_encoder(embedder.coded_dataset, str(final_output), file_format=fmt_v)

        typer.echo(f"Embeddings saved to: {final_output}")

    except typer.BadParameter as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    except Exception as e:
        typer.secho(f"Unexpected error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

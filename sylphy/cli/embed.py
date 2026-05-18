"""sylphy embed — embedding extraction from pretrained protein models."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, cast

import typer

from sylphy.cli._enums import DeviceChoice, LayerAggChoice, PoolChoice, PrecisionChoice
from sylphy.cli._shared import HELP_CONTEXT_SETTINGS, LOG_LEVELS, infer_format, level_from_str, load_dataset

if TYPE_CHECKING:
    from sylphy.types import LayerAggType, PoolType, PrecisionType

app = typer.Typer(
    name="embed",
    context_settings=HELP_CONTEXT_SETTINGS,
    no_args_is_help=True,
)

_MODEL = typer.Option(
    "facebook/esm2_t6_8M_UR50D",
    "--model",
    "-m",
    show_default=True,
    help="Model identifier (HuggingFace ref or registry key).",
)
_DEVICE = typer.Option(DeviceChoice.cuda, "--device", "-d", show_default=True, help="Inference device.")
_PRECISION = typer.Option(
    PrecisionChoice.fp32, "--precision", "-p", show_default=True, help="AMP precision on CUDA."
)
_BATCH_SIZE = typer.Option(8, "--batch-size", "-b", min=1, show_default=True, help="Batch size.")
_MAX_LEN = typer.Option(
    1024,
    "--max-length",
    "-l",
    min=1,
    show_default=True,
    help="Max tokens per sequence (truncated at tokenizer level).",
)
_OOM_BACKOFF = typer.Option(
    True,  # noqa: FBT003
    "--oom-backoff/--no-oom-backoff",
    show_default=True,
    help="Auto-reduce batch size on CUDA OOM and retry.",
)
_LAYERS = typer.Option(
    "last",
    "--layers",
    show_default=True,
    help="Layer selection: 'last' | 'last4' | 'all' | int | comma-separated ints.",
)
_LAYER_AGG = typer.Option(
    LayerAggChoice.mean, "--layer-agg", show_default=True, help="Aggregation across selected layers."
)
_POOL = typer.Option(PoolChoice.mean, "--pool", show_default=True, help="Token pooling strategy.")
_INPUT = typer.Option(..., "--input", "-i", help="Input file (csv/tsv/parquet).")
_SEQ_COL = typer.Option(
    "sequence", "--seq-col", "-s", show_default=True, help="Column with amino acid sequences."
)
_OUTPUT = typer.Option(
    ...,
    "--output",
    "-o",
    help="Output file path (extension determines format: csv/parquet/npy/npz).",
)
_LOG_LEVEL = typer.Option(
    "INFO", "--log-level", show_default=True, help=f"Log level: {', '.join(LOG_LEVELS)}."
)


@app.command("embed", help="Extract per-sequence embeddings from a pretrained protein model.")
def embed(  # noqa: D103
    *,
    input_path: Path = _INPUT,
    seq_col: str = _SEQ_COL,
    output: Path = _OUTPUT,
    model: str = _MODEL,
    device: DeviceChoice = _DEVICE,
    precision: PrecisionChoice = _PRECISION,
    batch_size: int = _BATCH_SIZE,
    max_length: int = _MAX_LEN,
    oom_backoff: bool = _OOM_BACKOFF,
    layers: str = _LAYERS,
    layer_agg: LayerAggChoice = _LAYER_AGG,
    pool: PoolChoice = _POOL,
    log_level: str = _LOG_LEVEL,
) -> None:
    fmt = infer_format(output)
    df = load_dataset(input_path, seq_col)

    ls = (layers or "last").strip().lower()
    layers_spec: str | int | list[int]
    if ls in {"last", "last4", "all"}:
        layers_spec = ls
    else:
        try:
            layers_spec = [int(x.strip()) for x in ls.split(",") if x.strip()] if "," in ls else int(ls)
        except ValueError:
            msg = "Invalid --layers. Use 'last' | 'last4' | 'all' | an integer | comma-separated integers."
            raise typer.BadParameter(msg) from None

    from sylphy.embedding_extractor import create_embedding  # noqa: PLC0415

    embedder = create_embedding(
        model_name=model,
        dataset=df,
        column_seq=seq_col,
        name_device=str(device),
        precision=cast("PrecisionType", precision),
        oom_backoff=oom_backoff,
        debug_mode=level_from_str(log_level),
    )
    embedder.run_process(
        max_length=max_length,
        batch_size=batch_size,
        layers=layers_spec,
        layer_agg=cast("LayerAggType", layer_agg),
        pool=cast("PoolType", pool),
    )
    embedder.export_encoder(str(output), file_format=fmt)
    typer.echo(f"Embeddings saved to: {output}")

"""sylphy embed — embedding extraction from pretrained protein models."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, cast

import typer

from sylphy.cli._enums import DeviceChoice, LayerAggChoice, PoolChoice, PrecisionChoice
from sylphy.cli._provenance import utc_now
from sylphy.cli._shared import (
    CONFIG_OPTION,
    HELP_CONTEXT_SETTINGS,
    LOG_LEVELS,
    infer_format,
    level_from_str,
    load_dataset,
    snapshot_commit,
    write_run_provenance,
)
from sylphy.feature_io import write_features

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
_INPUT = typer.Option(..., "--input", "-i", help="Input file (fasta/csv/tsv/parquet).")
_ID_COL = typer.Option(
    "id", "--id-col", show_default=True, help="Column with unique record ids (tables; FASTA uses the header)."
)
_REVISION = typer.Option(
    None, "--revision", help="Model revision (branch, tag or commit); default: the registry's."
)
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


@app.callback(
    invoke_without_command=True, help="Extract per-sequence embeddings from a pretrained protein model."
)
def embed(
    *,
    ctx: typer.Context,
    input_path: Path = _INPUT,
    id_col: str = _ID_COL,
    seq_col: str = _SEQ_COL,
    output: Path = _OUTPUT,
    model: str = _MODEL,
    revision: str | None = _REVISION,
    device: DeviceChoice = _DEVICE,
    precision: PrecisionChoice = _PRECISION,
    batch_size: int = _BATCH_SIZE,
    max_length: int = _MAX_LEN,
    oom_backoff: bool = _OOM_BACKOFF,
    layers: str = _LAYERS,
    layer_agg: LayerAggChoice = _LAYER_AGG,
    pool: PoolChoice = _POOL,
    log_level: str = _LOG_LEVEL,
    config: Path | None = CONFIG_OPTION,  # noqa: ARG001
) -> None:
    r"""Extract per-sequence embeddings from a pretrained protein model and export to disk.

    Examples:
        ESM2 on GPU, fp16, last layer, mean pool → parquet::

            sylphy embed -i data/seqs.csv -o out/esm2.parquet \
              -m facebook/esm2_t6_8M_UR50D -d cuda -p fp16 -b 16

        ProtT5 on GPU, bf16, last 4 layers averaged → npy::

            sylphy embed -i data/seqs.csv -o out/t5.npy \
              -m Rostlab/prot_t5_xl_uniref50 -d cuda -p bf16 --layers last4 --layer-agg mean

        Ankh2, all layers summed, CLS pooling → csv::

            sylphy embed -i data/seqs.csv -o out/ankh.csv \
              -m ElnaggarLab/ankh2-ext1 --layers all --layer-agg sum --pool cls

        CPU inference, parquet input, specific layer indices::

            sylphy embed -i data/seqs.parquet -o out/esm2_layers.csv \
              -m facebook/esm2_t6_8M_UR50D -d cpu --layers 0,3,6 --layer-agg concat

    """
    created_at = utc_now()
    fmt = infer_format(output)
    df = load_dataset(input_path, seq_col, id_col)

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

    if revision:
        _pin_revision(model, revision)

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
    features = embedder.coded_dataset.drop(seq_col)
    write_features(features, df[id_col], output, fmt)
    write_run_provenance(
        ctx,
        output,
        command="embed",
        created_at=created_at,
        n_rows=features.height,
        n_cols=features.width,
        packages=("sylphy", "numpy", "polars", "torch", "transformers"),
        extensions={
            "model": model,
            "revision_requested": revision,
            "revision_resolved": snapshot_commit(embedder.resolved_path),
            "device": embedder.device.type,
            "precision": precision.value,
            "batch_size_effective": embedder.effective_batch_size or batch_size,
        },
    )
    typer.echo(f"Embeddings saved to: {output}")


def _pin_revision(model: str, revision: str) -> None:
    """Register ``model`` with ``revision`` so the registry's ``snapshot_download`` call uses it."""
    from sylphy.core.model_registry import ModelNotFoundError, get_model_spec, register_model  # noqa: PLC0415
    from sylphy.core.model_spec import ModelSpec  # noqa: PLC0415

    try:
        spec = get_model_spec(model)
    except ModelNotFoundError:
        if "/" not in model:
            msg = f"--revision needs a registered model or a HuggingFace 'org/name' ref, got '{model}'."
            raise typer.BadParameter(msg) from None
        spec = ModelSpec(name=model, provider="huggingface", ref=model)
    register_model(replace(spec, revision=revision))

"""sylphy encode — classical sequence encoding (one-hot, ordinal, frequency, k-mers, physicochemical, FFT)."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Annotated, cast

import typer

from sylphy.cli._enums import DescriptorType, EncodeMethod
from sylphy.cli._provenance import utc_now
from sylphy.cli._shared import (
    CONFIG_OPTION,
    HELP_CONTEXT_SETTINGS,
    LOG_LEVELS,
    infer_format,
    level_from_str,
    load_dataset,
    write_run_provenance,
)
from sylphy.exceptions import SylphyRuntimeError
from sylphy.feature_io import write_features

if TYPE_CHECKING:
    import polars as pl

    from sylphy.sequence_encoder.encoder_base import EncoderBase
    from sylphy.sequence_encoder.fft_encoder import FFTEncoder
    from sylphy.types import FileFormat

app = typer.Typer(
    name="encode",
    context_settings=HELP_CONTEXT_SETTINGS,
    no_args_is_help=True,
)

_METHOD = typer.Option(
    EncodeMethod.physicochemical, "--method", "-e", show_default=True, help="Encoding method."
)
_INPUT = typer.Option(..., "--input", "-i", help="Input file (fasta/csv/tsv/parquet).")
_ID_COL = typer.Option(
    "id", "--id-col", show_default=True, help="Column with unique record ids (tables; FASTA uses the header)."
)
_SEQ_COL = typer.Option(
    "sequence", "--seq-col", "-s", show_default=True, help="Column with amino acid sequences."
)
_MAX_LEN = typer.Option(
    1024,
    "--max-length",
    "-l",
    show_default=True,
    help="Max sequence length (one_hot/ordinal/physicochemical).",
)
_ALLOW_EXT = typer.Option("--allow-extended", help="Enable extended alphabet (B, Z, X, U, O).")
_ALLOW_UNK = typer.Option("--allow-unknown", help="Allow 'X' without extended alphabet.")
_TYPE_DESC = typer.Option(
    DescriptorType.aaindex,
    "--type-descriptor",
    "-t",
    show_default=True,
    help="Descriptor space for physicochemical/fft.",
)
_NAME_PROP = typer.Option(
    "ANDN920101", "--name-property", "-n", show_default=True, help="AAIndex key or group_based label."
)
_SIZE_KMER = typer.Option(3, "--size-kmer", "-k", show_default=True, help="k for TF-IDF k-mers.")
_OUTPUT = typer.Option(
    ...,
    "--output",
    "-o",
    help="Output file path (extension determines format: csv/parquet/npy/npz).",
)
_LOG_LEVEL = typer.Option(
    "INFO", "--log-level", show_default=True, help=f"Log level: {', '.join(LOG_LEVELS)}."
)


@app.callback(invoke_without_command=True, help="Encode sequences into a numeric feature matrix.")
def encode(
    *,
    ctx: typer.Context,
    input_path: Path = _INPUT,
    id_col: str = _ID_COL,
    seq_col: str = _SEQ_COL,
    output: Path = _OUTPUT,
    method: EncodeMethod = _METHOD,
    max_length: int = _MAX_LEN,
    allow_extended: Annotated[bool, _ALLOW_EXT] = False,
    allow_unknown: Annotated[bool, _ALLOW_UNK] = False,
    type_descriptor: DescriptorType = _TYPE_DESC,
    name_property: str = _NAME_PROP,
    size_kmer: int = _SIZE_KMER,
    log_level: str = _LOG_LEVEL,
    config: Path | None = CONFIG_OPTION,  # noqa: ARG001
) -> None:
    r"""Encode sequences and export feature matrices.

    Examples:
        One-hot encode from CSV, output to parquet::

            sylphy encode --method one_hot -i data/seqs.csv -o out/onehot.parquet

        Ordinal encode with custom max length::

            sylphy encode --method ordinal -i data/seqs.csv -o out/ordinal.npy -l 512

        Physicochemical (AAIndex property), allow unknown residues::

            sylphy encode --method physicochemical -i data/seqs.csv -o out/phys.parquet \
              --name-property ARGP820101 --allow-unknown

        k-mers (TF-IDF) with k=4::

            sylphy encode --method kmers -i data/seqs.parquet -o out/kmers.csv -k 4

        FFT pipeline (physicochemical → FFT)::

            sylphy encode --method fft -i data/seqs.csv -o out/fft.csv \
              --type-descriptor aaindex --name-property ANDN920101

    """
    created_at = utc_now()
    fmt = infer_format(output)
    level = level_from_str(log_level)
    df = load_dataset(input_path, seq_col, id_col)

    from sylphy.sequence_encoder.factory import create_encoder  # noqa: PLC0415

    if method == EncodeMethod.fft:
        phys = create_encoder(
            "physicochemical",
            dataset=df,
            sequence_column=seq_col,
            max_length=max_length,
            type_descriptor=str(type_descriptor),
            name_property=name_property,
            allow_extended=allow_extended,
            allow_unknown=allow_unknown,
            debug_mode=level,
        )
        phys.run_process()
        if phys.coded_dataset is None or phys.coded_dataset.is_empty():
            msg = "Physicochemical step produced empty features."
            raise typer.BadParameter(msg)

        fft = cast(
            "FFTEncoder",
            create_encoder("fft", dataset=phys.coded_dataset, sequence_column=seq_col, debug_mode=level),
        )
        fft.run_process()
        _save(
            ctx,
            cast("EncoderBase", phys).dataset[id_col],
            fft.coded_dataset.drop(seq_col),
            output,
            fmt,
            created_at,
        )
        typer.echo(f"[fft] Saved to: {output}")
        return

    kwargs: dict = {
        "dataset": df,
        "sequence_column": seq_col,
        "allow_extended": allow_extended,
        "allow_unknown": allow_unknown,
        "debug_mode": level,
    }
    if method in (EncodeMethod.one_hot, EncodeMethod.ordinal, EncodeMethod.physicochemical):
        kwargs["max_length"] = max_length
    if method == EncodeMethod.physicochemical:
        kwargs["type_descriptor"] = str(type_descriptor)
        kwargs["name_property"] = name_property
    if method == EncodeMethod.kmers:
        kwargs["size_kmer"] = size_kmer

    enc = create_encoder(str(method), **kwargs)
    enc.run_process()
    _save(
        ctx,
        cast("EncoderBase", enc).dataset[id_col],
        enc.coded_dataset.drop(seq_col),
        output,
        fmt,
        created_at,
    )
    typer.echo(f"[{method}] Saved to: {output}")


def _save(
    ctx: typer.Context, ids: pl.Series, features: pl.DataFrame, output: Path, fmt: FileFormat, created_at: str
) -> None:
    """Write ``id`` + feature columns and the provenance sidecar.

    Encoders may drop rows (alphabet/length filters); ``ids`` must come from the encoder's surviving dataset.
    """
    if len(ids) != features.height:
        msg = f"Internal error: {len(ids)} ids for {features.height} encoded rows."
        raise SylphyRuntimeError(msg)
    write_features(features, ids, output, fmt)
    write_run_provenance(
        ctx,
        output,
        command="encode",
        created_at=created_at,
        n_rows=features.height,
        n_cols=features.width,
        packages=("sylphy", "numpy", "polars"),
    )

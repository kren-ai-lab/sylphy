"""sylphy encode — classical sequence encoding (one-hot, ordinal, frequency, k-mers, physicochemical, FFT)."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Annotated, cast

import typer

from sylphy.cli._enums import DescriptorType, EncodeMethod
from sylphy.cli._shared import HELP_CONTEXT_SETTINGS, LOG_LEVELS, infer_format, level_from_str, load_dataset

if TYPE_CHECKING:
    from sylphy.sequence_encoder.fft_encoder import FFTEncoder

app = typer.Typer(
    name="encode",
    context_settings=HELP_CONTEXT_SETTINGS,
    no_args_is_help=True,
)

_METHOD = typer.Option(
    EncodeMethod.physicochemical, "--method", "-e", show_default=True, help="Encoding method."
)
_INPUT = typer.Option(..., "--input", "-i", help="Input file (csv/tsv/parquet).")
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


@app.command("encode", help="Encode sequences into a numeric feature matrix.")
def encode(  # noqa: D103
    *,
    input_path: Path = _INPUT,
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
) -> None:
    fmt = infer_format(output)
    level = level_from_str(log_level)
    df = load_dataset(input_path, seq_col)

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
        fft.export_encoder(str(output), file_format=fmt, df_encoder=fft.coded_dataset)
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
    enc.export_encoder(str(output), file_format=fmt)
    typer.echo(f"[{method}] Saved to: {output}")

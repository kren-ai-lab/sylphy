# protein_representation/cli/encode_sequences.py
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import typer

from protein_representation.sequence_encoder import (
    OneHotEncoder,
    OrdinalEncoder,
    FFTEncoder,
    PhysicochemicalEncoder,
    FrequencyEncoder, 
    KMersEncoders,
)

app = typer.Typer(
    name="encode-sequences",
    help="Encode protein sequences from a CSV using various strategies (one-hot, k-mers, FFT, etc.)."
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


def _build_encoder(
        dataframe:pd.DataFrame, 
        sequence_identifier:str, 
        kind: str, 
        params: Dict[str, Any],
        debug:bool,
        debug_mode:int):

    kind = kind.lower()

    if kind == "onehot":
        return OneHotEncoder(
            dataset=dataframe,
            sequence_column=sequence_identifier,
            max_length=params.get("max_length", 1024),
            debug=debug,
            debug_mode=debug_mode
        )

    if kind == "ordinal":
        return OrdinalEncoder(
            dataset=dataframe,
            sequence_column=sequence_identifier,
            max_length=params.get("max_length", 1024),
            debug=debug,
            debug_mode=debug_mode
        )

    if kind == "physicochemical" or kind == "fft":
        return PhysicochemicalEncoder(
            dataset=dataframe,
            sequence_column=sequence_identifier,
            max_length=params.get("max_length", 1024),
            type_descriptor=params.get("type_descriptor", "aaindex"),
            name_property=params.get("name_property", "ANDN920101"),
            debug=debug,
            debug_mode=debug_mode
        )

    if kind == "frequency":
        return FrequencyEncoder(
            dataset=dataframe,
            sequence_column=sequence_identifier,
            max_length=params.get("max_length", 1024),
            debug=debug,
            debug_mode=debug_mode
        )

    if kind in ("kmers", "k-mers", "kmer"):
        return KMersEncoders(
            dataset=dataframe,
            sequence_column=sequence_identifier,
            max_length=params.get("max_length", 1024),
            size_kmer=params.get("size_kmer", 3),
            debug=debug,
            debug_mode=debug_mode
        )

    raise typer.BadParameter(
        f"Unknown encoder '{kind}'. "
        "Use one of: onehot, ordinal, fft, physicochemical, frequency, kmers."
    )

# ----------------- command ----------------- #

@app.command()
def run(

    # strategy
    encoder: str = typer.Option(
        "onehot", "--encoder", "-e",
        help="Encoding strategy: onehot | ordinal | fft | physicochemical | frequency | kmers",
    ),

    # IO
    input_data: Path = typer.Option(
        ..., "--input-data", "-i",
        help="Input CSV path with sequences."
    ),
    output: Path = typer.Option(
        ..., "--output", "-o",
        help="Output file path (CSV or NPY)."
    ),
    format_output: str = typer.Option(
        "csv", "--format-output", "-f",
        help="Export format: csv | npy",
    ),
    sequence_identifier: str = typer.Option(
        "sequence", "--sequence-identifier", "-s",
        help="Column in CSV that contains amino acid sequences."
    ),

    # Configs for encoders
    max_length: int = typer.Option(
        1024, "--max_length", "-m",
        help="Max length available per sequences"
    ),
    type_descriptor: str = typer.Option(
        "aaindex", "--type-descriptor", "-t",
        help="Type of descriptor employed in physicochemical properties. (e.g aaindex or group_based)"
    ),

    name_property: str = typer.Option(
        "ANDN920101", "--name-property", "-n",
        help="Name of descriptor to use as input for physicochemical encoders. (e.g ANDN920101 in the case of AAindex or group_0, in the case of group_based)"
    ),

    size_kmer : int = typer.Option(
        3, "--size-kmer", "-k",
        help="Size of K-Mer to use in KMers encoder strategy"
    ),

    # debug mode
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
    Example
    -------
    \b
    protein_representation encode-sequences run \\
      --encoder kmers \\
      --input-data data/sequences.csv \\
      --output results/kmers_k3.csv \\
      --sequence-identifier sequence \\
      --size-kmer 3
    """
    
    allowed_enc = {"onehot", "ordinal", "fft", "physicochemical", "frequency", "kmers"}
    allowed_debug_mode = {10, 20, 30}
    allowed_debug_options = {0, 1}
    allowed_formats = {"csv", "npy"}

    if format_output.lower() not in allowed_formats:
        raise typer.BadParameter(f"--format-output must be one of {sorted(allowed_formats)}")
    if int(debug) not in allowed_debug_options:
        raise typer.BadParameter(f"--debug must be one of {sorted(allowed_debug_options)}")
    if int(debug_mode) not in allowed_debug_mode:
        raise typer.BadParameter(f"--debug_mode must be one of {sorted(allowed_debug_mode)}")
    if encoder.lower() not in allowed_enc:
        raise typer.BadParameter(f"--encoder must be one of {sorted(allowed_enc)}")

    df = _load_csv(input_data, sequence_identifier)

    params: Dict[str, Any] = {
        "size_kmer": size_kmer,
        "name_property" : name_property,
        "type_descriptor" : type_descriptor,
        "max_length" : max_length
    }

    encoder_obj = _build_encoder(
        dataframe=df,
        sequence_identifier=sequence_identifier,
        kind=encoder,
        params=params,
        debug=debug,
        debug_mode=debug_mode)

    encoder_obj.run_process()

    if encoder != "fft":        
        encoder_obj.export_encoder(path=output, file_format=format_output)
    else:
        fft_encoder = FFTEncoder(
            dataset=encoder_obj.coded_dataset, 
            sequence_column=encoder_obj.sequence_column,
            debug=debug,
            debug_mode=debug_mode
        )
        
        fft_encoder.encoding_dataset()
        fft_encoder.export_encoder(
            df_encoder=fft_encoder.coded_dataset,
            path=output,
            file_format=format_output)

    typer.echo(f"Encoded features saved to: {output}")

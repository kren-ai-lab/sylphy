# tests/cli/test_encode_sequences_cli.py
from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from protein_representation.cli.encoder_sequences import app


def test_encode_onehot_saves_expected_shape(tmp_path):
    df = pd.DataFrame({"sequence": ["ACD", "WYYVV", "KLMNP"]})
    inp = tmp_path / "seqs.csv"
    df.to_csv(inp, index=False)
    out = tmp_path / "onehot.csv"

    runner = CliRunner()
    args = [
        "run",
        "--encoder", "onehot",
        "--input-data", str(inp),
        "--output", str(out),
        "--sequence-identifier", "sequence",
        "--format-output", "csv",
        "--max-length", "5",
        "--debug",
    ]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.stdout

    got = pd.read_csv(out)
    # onehot: max_length * 20 + 'sequence'
    assert got.shape[1] == 5 * 20 + 1
    assert got.shape[0] == 3


def test_encode_ordinal_basic(tmp_path):
    df = pd.DataFrame({"sequence": ["AAAA", "BBB"]})
    inp = tmp_path / "seq.csv"
    df.to_csv(inp, index=False)
    out = tmp_path / "ordinal.npy"

    runner = CliRunner()
    args = [
        "run",
        "--encoder", "ordinal",
        "--input-data", str(inp),
        "--output", str(out),
        "--sequence-identifier", "sequence",
        "--format-output", "npy",
        "--max-length", "4",
    ]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.stdout
    assert out.exists() and out.suffix == ".npy"

from __future__ import annotations

import pandas as pd
from typer.testing import CliRunner

from sylphy.cli.encoder_sequences import app


def test_encode_onehot_saves_expected_shape(tmp_path):
    """Verify CLI one-hot encodes sequences and saves with expected shape."""
    df = pd.DataFrame({"sequence": ["ACD", "WYYVV", "KLMNP"]})
    inp = tmp_path / "seqs.csv"
    df.to_csv(inp, index=False)
    out = tmp_path / "onehot.csv"

    res = CliRunner().invoke(
        app,
        [
            "--encoder",
            "one_hot",
            "--input-data",
            str(inp),
            "--output",
            str(out),
            "--sequence-identifier",
            "sequence",
            "--format-output",
            "csv",
            "--max-length",
            "5",
        ],
    )
    assert res.exit_code == 0, res.stdout

    got = pd.read_csv(out)
    assert got.shape == (3, 5 * 20 + 1)  # max_length * 20 AAs + sequence column


def test_encode_ordinal_basic(tmp_path):
    """Verify CLI ordinal encodes sequences and saves to .npy format."""
    df = pd.DataFrame({"sequence": ["AAAA", "BBB"]})
    inp = tmp_path / "seq.csv"
    df.to_csv(inp, index=False)
    out = tmp_path / "ordinal.npy"

    res = CliRunner().invoke(
        app,
        [
            "--encoder",
            "ordinal",
            "--input-data",
            str(inp),
            "--output",
            str(out),
            "--sequence-identifier",
            "sequence",
            "--format-output",
            "npy",
            "--max-length",
            "4",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert out.exists()
